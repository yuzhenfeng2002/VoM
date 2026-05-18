# Value of Modularity Against En-Route Demand Uncertainty

This repository contains the data and code used for the paper **"Value of Modularity Against En-Route Demand Uncertainty in Bus Services"**.

The code evaluates real-time dispatch policies for a modular bus fleet under replayed, uncertain en-route demand. The main implemented policy is a two-layer dispatcher:
1. **Layer 1:** a fast parametric directional assignment rule that uses observed stop-direction queues and aggregate onboard loads.
2. **Layer 2:** a short-horizon MILP that decides which assigned modules should depart immediately and which should wait briefly to form larger platoons.

The simulator uses only aggregate operational observations for dispatching. Individual passenger destinations are generated and tracked by the environment so the simulator can move passengers correctly, but the dispatcher does not use individual OD information when making online decisions.

## Repository Layout

```text
+-- inputs/
|   +-- arrival_replay_may15_17_20_23_27_31_env2/
|       +-- arrival_replay_env2.npz
+-- shanghai/
|   +-- doubleLayerDispatcher.py
|   +-- env2.py
|   +-- params.py
```

## Code Components

### `shanghai/env2.py`

Defines the event-based corridor simulation environment.
- `Passenger`: passenger state, including origin, destination, current stop, arrival time, boarding time, and departure time.
- `Module`: modular bus unit state, including current stop, current/new direction, previous direction, and stop history.
- `Environment`: simulates arrivals, boarding, alighting, module movement, passenger movement, and cost accumulation.

The environment state is represented by:
- `queueing_flow[n_stops, 2]`: waiting passengers by stop and direction.
- `onboard_flow[n_stops, 2]`: onboard passengers by stop and direction.
- `module_flow[n_stops, 3]`: modules by stop and status, where statuses are `UP`, `DOWN`, and `IDLE`.

Each simulation step follows this order:
1. Alight passengers who reached their destination.
2. Board passengers using modules already assigned to the correct direction.
3. Apply the dispatch action.
4. Update module directions.
5. Board again after the direction update.
6. Accumulate waiting and movement costs.
7. Move modules and onboard passengers.
8. Add the next batch of passenger arrivals.

### `shanghai/params.py`

Loads experiment constants and replayed arrival data from:
```text
inputs/arrival_replay_may15_17_20_23_27_31_env2/arrival_replay_env2.npz
```

Important parameters include:
- `n_stops = 25`
- `n_modules = 40`
- `capacity = 8`
- `experiment_horizon = 168`
- `double_layer_horizon = 4`
- `best_gamma = 0.72`
- `platoon_max = 8`
- `c_wait = 3`
- `c_move_max = 12`
- `platoon_factor = 2`

The replay artifact contains 12 out-of-sample weekdays, 168 time bins per day, and 25 stops. The data file records a 10:00-17:00 period with 2.5-minute bins. In the simulator, `dt = 1` means one decision epoch.

### `shanghai/doubleLayerDispatcher.py`

Contains the dispatch policies and experiment runner.

Main functions and classes:
- `beta_linear_field`, `beta_piecewise_linear_field`, `beta_three_piece_linear_field`: construct spatial aggressiveness weights for Layer 1.
- `beta_field_from_coeffs`: selects the spatial weight specification from coefficient length.
- `param_naive_dispatch`: Layer 1 directional assignment rule.
- `PlatoonOptimizer`: Layer 2 MILP for short-horizon platoon consolidation.
- `make_tuned_action`: builds a callable action for the simulation environment.
- `runDLExperiment`: runs repeated replay episodes and optionally writes episode-level costs to CSV.

The Shanghai experiment uses three-piece linear spatial weights with knots at stop sequences 9 and 17. Internally, stop indices are zero-based, so stop sequence 1 in the paper corresponds to index 0 in the code.

## Data

The replay input is a NumPy `.npz` archive with these arrays:

| Key | Shape | Description |
| --- | ---: | --- |
| `arrival_od_tensor` | `(12, 168, 25, 25)` | OD arrival counts by replay day, time bin, origin, and destination |
| `lambda_profile` | `(168, 25, 2)` | Directional arrival-rate profile |
| `dest_prob_matrix` | `(25, 25)` | Empirical destination probabilities |
| `date_ids` | `(12,)` | Day-of-month IDs used by the replay set |
| `horizon` | scalar | Number of time bins per episode |
| `n_stops` | scalar | Number of stops |
| `dt_minutes` | scalar | Minutes per data bin |
| `bin_seconds` | scalar | Seconds per data bin |
| `period_start` | scalar | Start time of the replay period |
| `period_end` | scalar | End time of the replay period |

The replay days currently stored in the artifact are:
```text
15, 16, 17, 20, 21, 22, 23, 27, 28, 29, 30, 31
```

## Dependencies

Use Python 3.10 or newer.

Required packages:
- `numpy`
- `gurobipy`

`gurobipy` and a working Gurobi license are required for the full two-layer dispatcher. The script imports `gurobipy` at module load time, so the package is also required before running the Layer 1-only experiments through `doubleLayerDispatcher.py`.

## Running the Main Experiment

From the repository root:
```bash
python shanghai/doubleLayerDispatcher.py
```

The current `__main__` block runs the tuned Layer 1 plus Layer 2 policy with:
- `n_episodes = 144`
- `seed_offset = 0`
- `gamma = 0.72`
- three-piece Layer 1 coefficients for both directions
It writes:
```text
outputs/m3_72.csv
```

Each episode uses one replay day selected cyclically from the 12 replay dates, so 144 episodes correspond to 12 passes over the 12 replay days with different random seeds for initialization and tie-breaking.
