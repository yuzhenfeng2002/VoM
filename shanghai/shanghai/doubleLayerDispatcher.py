import csv
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from math import ceil
from env2 import Environment
from params import UP, DOWN, IDLE, FIRST_STOP, LAST_STOP
import params

# Constants
BEST_BETA_UP = 1
BEST_BETA_DN = 1
BEST_LOOK_UP_FRAC = 1
BEST_LOOK_DN_FRAC = 1

# Constants
BETA_MIN = 0.05
BETA_MAX = 3.05
BETA_KNOT_STOP_SEQ = 15
BETA_THREE_PIECE_KNOT_STOP_SEQS = (9, 17)

def beta_linear_field(n_stops: int, a0: float, a1: float, beta_min=0.05, beta_max=3.05) -> np.ndarray:
    idx = np.arange(n_stops, dtype=float)
    if n_stops <= 1: x = np.zeros_like(idx)
    else: x = 2.0 * idx / (n_stops - 1) - 1.0  # Regularize stop indices to [-1, 1]
    beta = a0 + a1 * x
    beta = np.clip(beta, beta_min, beta_max)
    return beta

def beta_piecewise_linear_field(n_stops: int, a0: float, a1: float, a2: float,
                                knot_stop_seq: int = BETA_KNOT_STOP_SEQ, beta_min=0.05, beta_max=3.05) -> np.ndarray:
    """Continuous two-piece linear beta field, then clipped like the original linear field."""
    if n_stops <= 0: return np.array([], dtype=float)
    idx = np.arange(n_stops, dtype=float)
    if n_stops <= 1: x = np.zeros_like(idx)
    else: x = 2.0 * idx / (n_stops - 1) - 1.0
    knot_idx = int(np.clip(knot_stop_seq - 1, 0, n_stops - 1))
    x_knot = x[knot_idx]
    hinge = np.maximum(0.0, x - x_knot)
    beta = a0 + a1 * x + a2 * hinge
    return np.clip(beta, beta_min, beta_max)

def beta_three_piece_linear_field(n_stops: int, a0: float, a1: float, a2: float, a3: float,
                                  knot_stop_seqs=BETA_THREE_PIECE_KNOT_STOP_SEQS,
                                  beta_min=0.05, beta_max=3.05) -> np.ndarray:
    """Continuous three-piece linear beta field with knots at stop_seq=9 and stop_seq=17."""
    if n_stops <= 0: return np.array([], dtype=float)
    idx = np.arange(n_stops, dtype=float)
    if n_stops <= 1: x = np.zeros_like(idx)
    else: x = 2.0 * idx / (n_stops - 1) - 1.0

    beta = a0 + a1 * x
    for slope_delta, knot_stop_seq in zip((a2, a3), knot_stop_seqs):
        knot_idx = int(np.clip(knot_stop_seq - 1, 0, n_stops - 1))
        beta += slope_delta * np.maximum(0.0, x - x[knot_idx])
    return np.clip(beta, beta_min, beta_max)

def beta_field_from_coeffs(n_stops: int, coeffs, default_beta: float, beta_min=0.05, beta_max=3.05) -> np.ndarray:
    if coeffs is None: return np.ones(n_stops) * default_beta
    coeffs = tuple(float(c) for c in coeffs)
    if len(coeffs) == 2: return beta_linear_field(n_stops, coeffs[0], coeffs[1], beta_min=beta_min, beta_max=beta_max)
    if len(coeffs) == 3:
        return beta_piecewise_linear_field(
            n_stops, coeffs[0], coeffs[1], coeffs[2], beta_min=beta_min, beta_max=beta_max,
        )
    if len(coeffs) == 4:
        return beta_three_piece_linear_field(
            n_stops, coeffs[0], coeffs[1], coeffs[2], coeffs[3], beta_min=beta_min, beta_max=beta_max,
        )
    raise ValueError("coeffs must contain 2 -- 4 values for linear or piecewise beta.")

def param_naive_dispatch(env: Environment,
                         beta_up: float = BEST_BETA_UP,
                         beta_dn: float = BEST_BETA_DN,
                         look_up_frac: float = BEST_LOOK_UP_FRAC,
                         look_dn_frac: float = BEST_LOOK_DN_FRAC,
                         coeffs_up = None,
                         coeffs_dn = None,) -> None:
    if int(env.queueing_flow.sum() + env.onboard_flow.sum()) == 0:
        for m in env.modules:
            if m.current_stop == FIRST_STOP or m.current_stop == LAST_STOP:
                m.new_direction = IDLE
        return

    n_stops = env.n_stops
    Cap = env.capacity

    # Calculate beta fields for UP and DOWN directions based on coefficients or defaults
    beta_up_vec = beta_field_from_coeffs(n_stops, coeffs_up, beta_up, beta_min=BETA_MIN, beta_max=BETA_MAX)
    beta_dn_vec = beta_field_from_coeffs(n_stops, coeffs_dn, beta_dn, beta_min=BETA_MIN, beta_max=BETA_MAX)

    # Group modules by (stop, current direction)
    mid_per_stop_direction = {}
    for mid, m in enumerate(env.modules):
        stop = m.current_stop
        if stop not in mid_per_stop_direction:
            mid_per_stop_direction[stop] = {UP: [], DOWN: [], IDLE: []}
        mid_per_stop_direction[stop][m.new_direction].append(mid)

    # Process each stop independently
    rng = np.random.default_rng(seed=env.seed + env.t * 1000)
    for stop, mid_per_direction in mid_per_stop_direction.items():
        ### 1) Available modules at this stop: idle + "extra" beyond onboard needs
        available_mid = {}
        available_mid[IDLE] = list(mid_per_direction[IDLE])  # idle modules are always free
        for direction in (UP, DOWN):
            indices = mid_per_direction[direction]
            loads = env.onboard_flow[stop, direction]
            if loads > 0:
                need_for_onboard = ceil(loads / Cap)
            else:
                need_for_onboard = 0
            reserved_count = min(need_for_onboard, len(indices))
            extra_indices = indices[reserved_count:]
            available_mid[direction] = list(extra_indices)
        if not available_mid[IDLE] and not available_mid[UP] and not available_mid[DOWN]:
            continue  # nothing to dispatch at this stop
        ### 2) Compute directional "needs" from local queues using fractions
        # UP direction
        up_start = stop
        up_total_len = n_stops - stop
        up_len = max(1, int(up_total_len * look_up_frac + 0.9999))
        up_end = min(n_stops, up_start + up_len)
        quu_sum = float(env.queueing_flow[up_start:up_end, UP].sum())
        qud_sum = float(env.queueing_flow[up_start + 1:up_end, DOWN].sum())
        qu_sum = quu_sum + qud_sum
        # DOWN direction
        dn_end = stop + 1
        dn_total_len = dn_end
        dn_len = max(1, int(dn_total_len * look_dn_frac + 0.9999))
        dn_start = max(0, dn_end - dn_len)
        qdd_sum = float(env.queueing_flow[dn_start:dn_end, DOWN].sum())
        qdu_sum = float(env.queueing_flow[dn_start:dn_end - 1, UP].sum())
        qd_sum = qdd_sum + qdu_sum
        # Local beta values for this stop from the selected beta field
        beta_up = float(beta_up_vec[stop])
        beta_dn = float(beta_dn_vec[stop])
        # Calculate needed modules in each direction
        need_up = ceil(beta_up * qu_sum / Cap) if qu_sum > 0 else 0
        need_dn = ceil(beta_dn * qd_sum / Cap) if qd_sum > 0 else 0
        ### 3) Greedily assign free modules based on remaining needs
        while need_up + need_dn > 0:
            if not available_mid[UP] and not available_mid[DOWN] and not available_mid[IDLE]:
                break  # no more available modules
            if need_up > 0 and (need_up > need_dn or (need_up == need_dn and rng.random() < 0.5)):
                if available_mid[UP]: mid = available_mid[UP].pop()
                elif available_mid[IDLE]: mid = available_mid[IDLE].pop()
                elif available_mid[DOWN]: mid = available_mid[DOWN].pop()
                else: break
                env.modules[mid].new_direction = UP
                need_up -= 1
            elif need_dn > 0:
                if available_mid[DOWN]: mid = available_mid[DOWN].pop()
                elif available_mid[IDLE]: mid = available_mid[IDLE].pop()
                elif available_mid[UP]: mid = available_mid[UP].pop()
                else: break
                env.modules[mid].new_direction = DOWN
                need_dn -= 1
            else: break
        for direction in (UP, DOWN, IDLE):
            for mid in available_mid[direction]:
                env.modules[mid].new_direction = IDLE

class PlatoonOptimizer:
    def __init__(self, env, horizon=params.double_layer_horizon, gamma=params.best_gamma):
        self.env = env
        self.H = horizon
        self.gamma = gamma
    def solve(self, fixed_directions, coeffs_up=None, coeffs_dn=None, beta_up=BEST_BETA_UP, beta_dn=BEST_BETA_DN,):
        # Calculate beta fields for UP and DOWN directions based on coefficients or defaults
        beta_up_vec = beta_field_from_coeffs(self.env.n_stops, coeffs_up, beta_up, beta_min=BETA_MIN, beta_max=BETA_MAX)
        beta_dn_vec = beta_field_from_coeffs(self.env.n_stops, coeffs_dn, beta_dn, beta_min=BETA_MIN, beta_max=BETA_MAX)

        try:
            m = gp.Model("PlatoonDispatch")
            m.setParam('OutputFlag', 0)
            m.setParam('MIPGap', params.double_layer_mipgap)

            # --- 1. Calculate Downstream Queues ---
            Q = {}
            q_flow = self.env.queueing_flow
            for i in range(self.env.n_stops):
                q_ahead_up = q_flow[i:, UP].sum() if i < self.env.n_stops - 1 else 0
                q_ahead_up += q_flow[i+1:, DOWN].sum()
                Q[(i, UP)] = q_ahead_up * beta_up_vec[i] # Queue ahead in the UP direction that this stop can see
                q_ahead_dn = q_flow[:i+1, DOWN].sum() if i > 0 else 0
                q_ahead_dn += q_flow[:i, UP].sum()
                Q[(i, DOWN)] = q_ahead_dn * beta_dn_vec[i] # Queue ahead in the DOWN direction that this stop can see
            QSum = sum(Q.values())
            for i in range(self.env.n_stops):
                Q[(i, UP)] = Q[(i, UP)] / QSum if QSum > 0 else 0
                Q[(i, DOWN)] = Q[(i, DOWN)] / QSum if QSum > 0 else 0

            # --- 2. Variables ---
            T_steps = range(self.H)
            Stops = range(self.env.n_stops)
            Dirs = [UP, DOWN]
            Platoon_Sizes = range(self.env.platoon_max + 1)
            s = m.addVars(Stops, Dirs, T_steps, vtype=GRB.INTEGER, name="s")  # Num of modules moving from stop i in direction d at time t
            w = m.addVars(Stops, Dirs, T_steps, vtype=GRB.INTEGER, name="w")  # Num of modules waiting at stop i in direction d at time t
            o = m.addVars(Stops, Dirs, T_steps, vtype=GRB.CONTINUOUS, name="o") # Num of passengers onboard modules at stop i in direction d at time t
            n = m.addVars(Stops, Dirs, Platoon_Sizes, T_steps, vtype=GRB.INTEGER, name="n")  # Num of platoons of size p moving from stop i in direction d at time t

            # --- 3. Objective Function ---
            obj = 0
            maxW = sum([v * Q[q] for q, v in fixed_directions.items()]) * self.H
            maxP = sum([(v // self.env.platoon_max) * self.env.c_move_m(self.env.platoon_max)
                        + self.env.c_move_m(v % self.env.platoon_max) for q, v in fixed_directions.items()]) * self.H
            for t in T_steps:
                for i in Stops:
                    for d in Dirs:
                        if i == FIRST_STOP and d == DOWN: continue
                        if i == LAST_STOP and d == UP: continue
                        # Moving cost
                        for p in Platoon_Sizes:
                            obj += (1 - self.gamma) * n[i, d, p, t] * self.env.c_move_m(p) / max(1, maxP)
                        # Waiting cost
                        obj += self.gamma * w[i, d, t] * Q[(i, d)] / max(1, maxW)
                        # if i != FIRST_STOP and i != LAST_STOP:
                        #     obj += w[i, d, t] * self.env.c_idle * self.env.dt * self.gamma
            m.setObjective(obj, GRB.MINIMIZE)

            # --- 4. Constraints ---
            for i in Stops:
                for d in Dirs:
                    m.addConstr(gp.quicksum(s[i, d, t] for t in T_steps) >= fixed_directions.get((i, d), 0)) # Fixed directions must be met over the horizon
                    for t in T_steps:
                        incoming = 0
                        onboard = 0
                        if t == 0:
                            incoming = fixed_directions.get((i, d), 0) # At t=0, the incoming modules are exactly those that we have already decided to move in the first layer
                            onboard = float(self.env.onboard_flow[i, d]) # Num of passengers currently onboard at this stop and direction, which must be served by the incoming modules
                        else:
                            incoming += w[i, d, t - 1] # Num of modules that were waiting at the previous time step can now start moving
                            prev_stop = i - 1 if d == UP else i + 1 # Previous stop in the current direction
                            if 0 <= prev_stop < self.env.n_stops: # If the previous stop is valid, add the modules that were moving from there in the last time step
                                incoming += s[prev_stop, d, t - 1] # Num of modules that were moving from the previous stop in the same direction at the last time step
                                onboard += o[prev_stop, d, t - 1] # Num of passengers that were onboard those modules, which are now arriving at the current stop
                            else: # If there is no previous stop in this direction (i.e., we are at the first or last stop), then there are no incoming modules from that direction, and no passengers onboard from that direction
                                opp = UP if d == DOWN else DOWN # # Opposite direction
                                neighbor = i - 1 if opp == UP else i + 1 # Neighboring stop in the opposite direction
                                if 0 <= neighbor < self.env.n_stops:
                                    incoming += s[neighbor, opp, t - 1] # Num of modules that were moving from the neighboring stop in the opposite direction at the last time step, which can now arrive at the current stop and change direction
                        m.addConstr(s[i, d, t] + w[i, d, t] == incoming) # Flow conservation
                        m.addConstr(s[i, d, t] * self.env.capacity >= o[i, d, t]) # Onboard passengers cannot exceed the capacity of the moving modules
                        m.addConstr(o[i, d, t] == onboard) # The number of passengers onboard at the current time step is equal to the number of passengers that were onboard the incoming modules
                        m.addConstr(gp.quicksum(p * n[i, d, p, t] for p in Platoon_Sizes) == s[i, d, t]) # Platoon composition
            # --- 5. Optimize ---
            m.optimize()

            # --- 6. Extract Action for t=0 ---
            move_decisions = {}
            idle_decisions = {}
            if m.status == GRB.OPTIMAL or m.status == GRB.SUBOPTIMAL:
                for i in Stops:
                    for d in Dirs:
                        n_move = int(round(s[i, d, 0].X))
                        n_idle = int(round(w[i, d, 0].X))
                        move_decisions[(i, d)] = n_move
                        idle_decisions[(i, d)] = n_idle
                return move_decisions, idle_decisions
            else: return fixed_directions
        except gp.GurobiError as e:
            print("Gurobi error:", e)
            return fixed_directions, {}
        except Exception as e:
            import traceback
            print("Other error:", e)
            traceback.print_exc()
            return fixed_directions, {}


def make_tuned_action(env: Environment, is_naive=False, coeffs_up=None, coeffs_dn=None):
    optimizer = PlatoonOptimizer(env, horizon=params.double_layer_horizon, gamma=params.best_gamma)
    def action():
        beta_up = BEST_BETA_UP
        beta_dn = BEST_BETA_DN
        look_up_frac = BEST_LOOK_UP_FRAC
        look_dn_frac = BEST_LOOK_DN_FRAC
        param_naive_dispatch(
            env,
            beta_up=beta_up,
            beta_dn=beta_dn,
            look_up_frac=look_up_frac,
            look_dn_frac=look_dn_frac,
            coeffs_up=coeffs_up,
            coeffs_dn=coeffs_dn,
        )
        if is_naive: return # If we're doing a pure naive dispatch, we skip the optimization step and just return the decisions from the param_naive_dispatch

        # Input the first layer's decisions into the second layer's optimization problem
        layer1_counts = {}
        for m in env.modules:
            d = m.new_direction
            if d != IDLE:
                k = (m.current_stop, d)
                layer1_counts[k] = layer1_counts.get(k, 0) + 1

        if not layer1_counts:
            return
        move_decisions, idle_decisions = optimizer.solve(layer1_counts, coeffs_up=coeffs_up, coeffs_dn=coeffs_dn) # 暂时不加系数

        # Output the optimized decisions
        processed_counts = {}
        for m in env.modules:
            d = m.new_direction
            if d == IDLE: continue
            key = (m.current_stop, d)
            allowed = move_decisions.get(key, 0)
            processed = processed_counts.get(key, 0)
            if processed < allowed: processed_counts[key] = processed + 1
            else: m.new_direction = IDLE
    return action


def build_experiment_episode_specs(n_episodes: int, seed_offset: int = 1000):
    replay_arrivals = bool(getattr(params, "replay_arrivals", False))
    replay_date_ids = tuple(getattr(params, "arrival_replay_date_ids", ()))
    if replay_arrivals and not replay_date_ids:
        raise ValueError("params.replay_arrivals is True, but params.arrival_replay_date_ids is empty")

    specs = []
    for ep in range(n_episodes):
        replay_date_index = None
        replay_date_id = None
        if replay_arrivals:
            replay_date_index = ep % len(replay_date_ids)
            replay_date_id = replay_date_ids[replay_date_index]
        specs.append(
            {
                "episode": ep,
                "seed": seed_offset + ep,
                "replay_date_index": replay_date_index,
                "replay_date_id": replay_date_id,
            }
        )
    return specs


def _format_coeffs_for_csv(coeffs):
    if coeffs is None:
        return ""
    return ";".join(str(float(c)) for c in coeffs)


def write_episode_cost_results_csv(path, details):
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "date_id",
        "seed",
        "is_naive",
        "gamma",
        "demand_scale",
        "platoon_factor",
        "experiment_horizon",
        "double_layer_horizon",
        "coeffs_up",
        "coeffs_dn",
        "waits",
        "moves",
        "total",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for detail in details:
            writer.writerow({field: detail.get(field, "") for field in fieldnames})


def runDLExperiment(
    is_naive=True,
    coeffs_up=None,
    coeffs_dn=None,
    verbose=False,
    n_episodes=144,
    seed_offset=1000,
    return_details=False,
    cost_csv_path=None,
):
    import statistics
    horizon = params.experiment_horizon
    episode_specs = build_experiment_episode_specs(n_episodes=n_episodes, seed_offset=seed_offset)
    totals = []
    waits = []
    moves = []
    details = []
    if verbose: print("Running Dispatch with Parametric Layer 1 + MILP Layer 2...")
    for spec in episode_specs:
        ep = spec["episode"]
        env = Environment(seed=spec["seed"], replay_date_index=spec["replay_date_index"])
        env.random_initialize()
        action = make_tuned_action(env, is_naive=is_naive, coeffs_up=coeffs_up, coeffs_dn=coeffs_dn)
        for _ in range(horizon):
            env.step(action)
        total = float(env.total_waiting_cost + env.total_moving_cost)
        totals.append(total)
        waits.append(float(env.total_waiting_cost))
        moves.append(float(env.total_moving_cost))
        details.append(
            {
                "episode": ep,
                "seed": spec["seed"],
                "date_id": spec["replay_date_id"],
                "replay_date_id": spec["replay_date_id"],
                "is_naive": is_naive,
                "gamma": float(params.best_gamma),
                "demand_scale": float(params.demand_scale),
                "platoon_factor": float(params.platoon_factor),
                "experiment_horizon": int(horizon),
                "double_layer_horizon": int(params.double_layer_horizon),
                "coeffs_up": _format_coeffs_for_csv(coeffs_up),
                "coeffs_dn": _format_coeffs_for_csv(coeffs_dn),
                "total": total,
                "waits": float(env.total_waiting_cost),
                "moves": float(env.total_moving_cost),
                "waiting": float(env.total_waiting_cost),
                "moving": float(env.total_moving_cost),
            }
        )
        if verbose:
            date_text = f", date=2024-05-{spec['replay_date_id']}" if spec["replay_date_id"] is not None else ""
            print(f"Episode {ep}{date_text}: total={total:.1f}, wait={env.total_waiting_cost:.1f}, op={env.total_moving_cost:.1f}")
    if verbose:
        print("Average over episodes:")
        print("  total  =", statistics.mean(totals))
        print("  waiting=", statistics.mean(waits))
        print("  op     =", statistics.mean(moves))
    if cost_csv_path is not None:
        write_episode_cost_results_csv(cost_csv_path, details)
    if return_details:
        return waits, moves, details
    return waits, moves


if __name__ == "__main__":
    n_episodes = 12 * 12
    is_naive = False
    coeffs_up = (0.8316215053151956,0.08383023774930787,1.145668922345302,-0.7275011374588818)
    coeffs_dn = (1.1454242379924269,-1.0020964707344506,1.3981261320423664,-0.752704374966651)
    # coeffs_up = None
    # coeffs_dn = None
    cost_csv_path = Path(__file__).resolve().parents[1] / "outputs" / "m3_72.csv"
    runDLExperiment(
        is_naive=is_naive,
        coeffs_up=coeffs_up,
        coeffs_dn=coeffs_dn,
        verbose=True,
        n_episodes=n_episodes,
        seed_offset=0,
        cost_csv_path=cost_csv_path,
    )
    print(f"Episode cost results saved to {cost_csv_path}")
