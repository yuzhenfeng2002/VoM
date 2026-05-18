from dataclasses import dataclass
import numpy as np
import params
from params import UP, DOWN, IDLE, FIRST_STOP, LAST_STOP

@dataclass
class Passenger:
    origin: int
    destination: int
    current_stop: int
    arrival_time: int
    boarding_time: int
    departure_time: int

@dataclass
class Module:
    current_stop: int
    new_direction: int  # 0: up, 1: down, 2: idle
    old_direction: int
    history_stops: list

class Environment:
    def __init__(self, seed=42, replay_date_index=None, replay_date_id=None):
        assert params.n_stops > 1 # ensure there are at least two stops
        self.n_stops = params.n_stops # number of stops
        self.n_modules = params.n_modules # number of modules
        self.capacity = params.capacity # capacity per module
        self.dt = params.dt # time step
        self.seed = seed
        self.rng = np.random.default_rng(seed) # random number generator
        self.base_lambda = params.base_lambda # base arrival rates
        self.c_wait = params.c_wait # waiting cost per passenger per unit time
        self.platoon_max = params.platoon_max # maximum platoon size
        self.c_move_max = params.c_move_max # maximum moving cost per platoon per unit time
        self.c_move_m = params.c_move_m # moving cost function
        self.replay_arrivals = bool(getattr(params, "replay_arrivals", False))
        self.dest_choice_prob = None if self.replay_arrivals else params.dest_choice_prob
        if replay_date_id is not None:
            replay_date_index = params.replay_date_index_for_date_id(replay_date_id)
        elif replay_date_index is None and self.replay_arrivals:
            resolver = getattr(params, "replay_date_index_for_seed", None)
            if resolver is None:
                replay_date_index = getattr(params, "default_replay_date_index", 0)
            else:
                replay_date_index = resolver(seed)
        self.replay_date_index = replay_date_index
        self.reset()

    def reset(self):
        self.t = 0  # current time
        self.modules = []  # list of modules
        self.active_passengers = []  # list of active passengers = queuing + onboard
        self.finished_passengers = []  # list of finished passengers
        self.queueing_flow = np.zeros((self.n_stops, 2), dtype=int)  # queuing flow at each stop and direction
        self.onboard_flow = np.zeros((self.n_stops, 2), dtype=int)  # onboard flow at each stop and direction
        self.module_flow = np.zeros((self.n_stops, 3), dtype=int)  # module flow at each stop and direction
        self.total_waiting_cost = 0  # total waiting cost (passengers)
        self.total_moving_cost = 0

    def random_initialize(self):
        self._poisson_arrival(dt=self.dt) # initial passenger arrivals
        for n in range(self.n_modules):
            stop = self.rng.choice(self.n_stops) # random initial stop
            if stop == FIRST_STOP: direction = self.rng.choice([UP, IDLE]) # cannot go down at first stop
            elif stop == LAST_STOP: direction = self.rng.choice([DOWN, IDLE]) # cannot go up at last stop
            else: direction = self.rng.choice([UP, DOWN, IDLE]) # can go any direction
            self.modules.append(Module(stop, direction, direction, [stop])) # create module
            self.module_flow[stop, direction] += 1 # update module flow

    def _poisson_arrival(self, dt: float):
        if self.replay_arrivals:
            self._replay_arrival()
            return

        arrivals = np.zeros((self.n_stops, 2), dtype=int) # initialize arrivals
        for i in range(self.n_stops):
            for d in [UP, DOWN]:
                arrivals[i, d] += self.rng.poisson(params.time_lambda(i, d, self.t, self.base_lambda) * dt) # sample arrivals
        arrivals[FIRST_STOP, DOWN] = 0 # no arrivals going down at first stop
        arrivals[LAST_STOP, UP] = 0  # no arrivals going up at last stop
        self.queueing_flow += arrivals # update queueing flow
        for n in range(self.n_stops):
            for direction in [UP, DOWN]:
                for _ in range(arrivals[n, direction]):
                    # sample destination based on probabilities
                    destination = self.rng.choice(list(self.dest_choice_prob[(n, direction)].keys()),
                                                  p=list(self.dest_choice_prob[(n, direction)].values()))
                    # create passenger and add to active passengers
                    passenger = Passenger(
                        origin=n,
                        destination=destination,
                        current_stop=n,
                        arrival_time=self.t,
                        boarding_time=-1,
                        departure_time=-1,
                    )
                    self.active_passengers.append(passenger)

    def _replay_arrival(self):
        od_counts = np.asarray(params.arrival_od_counts(self.t, self.replay_date_index), dtype=int)
        expected_shape = (self.n_stops, self.n_stops)
        if od_counts.shape != expected_shape:
            raise ValueError(f"arrival_od_counts returned shape {od_counts.shape}, expected {expected_shape}")
        if np.any(od_counts < 0):
            raise ValueError("arrival_od_counts must be non-negative")

        for origin in range(self.n_stops):
            destinations = np.nonzero(od_counts[origin])[0]
            for destination in destinations:
                if destination == origin:
                    continue
                count = int(od_counts[origin, destination])
                direction = UP if destination > origin else DOWN
                self.queueing_flow[origin, direction] += count
                for _ in range(count):
                    passenger = Passenger(
                        origin=origin,
                        destination=int(destination),
                        current_stop=origin,
                        arrival_time=self.t,
                        boarding_time=-1,
                        departure_time=-1,
                    )
                    self.active_passengers.append(passenger)

    def _make_observation(self):
        return {
            "time": self.t,
            "queues": self.queueing_flow.copy(),
            "loads": self.onboard_flow.copy(),
            "modules": self.module_flow.copy(),
        }

    def _alighting(self):
        for p in self.active_passengers:
            if p.boarding_time == -1: continue # not boarded yet
            if p.current_stop == p.destination: # reached destination
                p.departure_time = self.t # set departure time
                direction = UP if p.destination > p.origin else DOWN # determine direction via origin and destination
                # update onboard flow
                self.onboard_flow[p.current_stop, direction] = self.onboard_flow[p.current_stop, direction] - 1
                assert self.onboard_flow[p.current_stop, direction] >= 0
                self.finished_passengers.append(p)
        self.active_passengers = [p for p in self.active_passengers if p.departure_time == -1]

    def _boarding(self):
        for n in range(self.n_stops):
            for direction in [UP, DOWN]:
                # calculate available seats
                n_seats = self.module_flow[n, direction] * self.capacity - self.onboard_flow[n, direction]
                assert n_seats >= 0
                if n_seats <= 0:
                    continue # no available seats
                # board passengers
                n_boarded = 0
                for p in self.active_passengers:
                    if n_boarded >= n_seats: break # no more seats
                    if p.boarding_time != -1: continue # already boarded
                    if p.current_stop != n: continue # not at this stop
                    if direction != (UP if p.destination > n else DOWN): continue # wrong direction
                    p.boarding_time = self.t # set boarding time
                    # update queueing and onboard flow
                    self.queueing_flow[n, direction] = self.queueing_flow[n, direction] - 1
                    assert self.queueing_flow[n, direction] >= 0
                    self.onboard_flow[n, direction] += 1
                    n_boarded += 1 # increment boarded count

    def _moving(self):
        self.module_flow[:] = 0 # reset module flow
        for m in self.modules:
            if m.new_direction == UP and m.current_stop < LAST_STOP:
                m.current_stop += 1 # move up
            elif m.new_direction == DOWN and m.current_stop > FIRST_STOP:
                m.current_stop -= 1 # move down
            if m.new_direction == UP and m.current_stop == LAST_STOP:
                m.new_direction = IDLE # idle at last stop
            elif m.new_direction == DOWN and m.current_stop == FIRST_STOP:
                m.new_direction = IDLE # idle at first stop
            m.old_direction = m.new_direction # update old direction
            m.history_stops.append(m.current_stop) # record history
            self.module_flow[m.current_stop, m.new_direction] += 1 # update module flow

        self.onboard_flow[:] = 0 # reset onboard flow
        for p in self.active_passengers:
            if p.boarding_time == -1: continue # not boarded yet
            if p.current_stop == p.destination: continue # already at destination
            step = 1 if p.destination > p.origin else -1 # determine step direction
            p.current_stop += step # move passenger
            self.onboard_flow[p.current_stop, UP if p.destination > p.origin else DOWN] += 1 # update onboard flow

    def calculate_costs(self):
        waiting_cost = self.queueing_flow.sum() * self.c_wait * self.dt # waiting cost is proportional to queueing flow
        moving_cost = 0
        moving_group = {} # group modules by (stop, direction)
        for m in self.modules:
            if m.new_direction == IDLE or (m.new_direction == DOWN and m.current_stop == FIRST_STOP) or (m.new_direction == UP and m.current_stop == LAST_STOP):
                continue
            key = (m.current_stop, m.new_direction) # key for grouping
            moving_group.setdefault(key, []).append(m) # group modules
        for _, ids in moving_group.items(): # calculate moving costs
            n_platoon, n_remain = divmod(len(ids), self.platoon_max) # platoon division
            moving_cost += n_platoon * self.c_move_m(self.platoon_max) * self.dt # cost for full platoons
            if n_remain > 0: # cost for remaining modules
                moving_cost += self.c_move_m(n_remain) * self.dt
        return waiting_cost, moving_cost

    def _update_directions(self):
        self.module_flow[:] = 0 # reset module flow
        for m in self.modules:
            self.module_flow[m.current_stop, m.new_direction] += 1 # update module flow

    def step(self, action):
        self.pre_action_step() # alight and board passengers
        action()  # apply action
        self.post_action_step()  # update directions, board again, calculate costs, move, and arrive new passengers
        return self._make_observation()

    def pre_action_step(self):
        self._alighting()  # alight passengers
        self._boarding()  # board passengers
        return self._make_observation()

    def post_action_step(self):
        self._update_directions() # update module directions
        self._boarding() # board passengers again after direction update
        waiting_cost, moving_cost = self.calculate_costs() # calculate costs
        self.total_waiting_cost += waiting_cost # update total waiting cost
        self.total_moving_cost += moving_cost
        self.t += self.dt # increment time
        self._moving() # move modules and passengers
        self._poisson_arrival(self.dt) # new passenger arrivals
        return self._make_observation()

    def print(self, width=4):
        # Print the current state of the environment
        fmt = lambda array, column: " ".join(f"{int(array[n, column]):^{width}}" for n in range(self.n_stops))
        stop_idx = " ".join(f"{n:^{width}}" for n in range(self.n_stops))
        print(f"Stops:  {stop_idx}")
        print(f"Q->:    {fmt(self.queueing_flow, UP)}")
        print(f"Q<-:    {fmt(self.queueing_flow, DOWN)}")
        print("-" * 40)
        print(f"B->:    {fmt(self.onboard_flow, UP)}")
        print(f"B<-:    {fmt(self.onboard_flow, DOWN)}")
        print("-" * 40)
        print(f"M->:    {fmt(self.module_flow, UP)}")
        print(f"M--:    {fmt(self.module_flow, IDLE)}")
        print(f"M<-:    {fmt(self.module_flow, DOWN)}")
        print(f"#Q:{int(self.queueing_flow.sum())}\t"
              f"#B:{int(self.onboard_flow.sum())}\t"
              f"#M:{int(self.module_flow.sum())}\t"
              f"#Finished:{len(self.finished_passengers)}\t"
              f"Total Cost:{int(self.total_waiting_cost + self.total_moving_cost)}" + "-"*40)
