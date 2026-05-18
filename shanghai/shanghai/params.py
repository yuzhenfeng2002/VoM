from __future__ import annotations

from pathlib import Path
import numpy as np


UP = 0
DOWN = 1
IDLE = 2

REPLAY_DIR = Path(__file__).resolve().parents[1] / "inputs" / "arrival_replay_may15_17_20_23_27_31_env2"
REPLAY_FILE = REPLAY_DIR / "arrival_replay_env2.npz"
if not REPLAY_FILE.exists():
    raise FileNotFoundError(
        f"Arrival replay artifact not found: {REPLAY_FILE}."
    )

_data = np.load(REPLAY_FILE, allow_pickle=False)

n_stops = int(_data["n_stops"])
FIRST_STOP = 0
LAST_STOP = n_stops - 1
dt = 1
experiment_horizon = int(_data["horizon"])
double_layer_horizon = 4
double_layer_mipgap = 0.02
best_gamma = 0.72
demand_scale = 1.0

n_modules = 40
capacity = 8

arrival_od_tensor = _data["arrival_od_tensor"]
lambda_profile = _data["lambda_profile"]
arrival_replay_date_ids = tuple(str(x) for x in _data["date_ids"].tolist())
arrival_replay_day_count = int(arrival_od_tensor.shape[0])
replay_arrivals = True
repeat_replay_arrivals = False
default_replay_date_index = 0
choice_probability_method = "exact_od_replay"
base_lambda = lambda_profile.mean(axis=0)
_zero_arrival_od_counts = np.zeros((n_stops, n_stops), dtype=arrival_od_tensor.dtype)


def replay_date_index_for_seed(seed: int) -> int:
    return int(seed) % arrival_replay_day_count


def replay_date_index_for_date_id(date_id: str | int) -> int:
    date_id_text = str(date_id).zfill(2)
    try:
        return arrival_replay_date_ids.index(date_id_text)
    except ValueError as exc:
        raise KeyError(f"Date id {date_id_text} is not in replay dates {arrival_replay_date_ids}") from exc


def arrival_od_counts(t: int, replay_date_index: int | None = None) -> np.ndarray:
    day_idx = default_replay_date_index if replay_date_index is None else int(replay_date_index)
    day_idx %= arrival_replay_day_count
    time_idx = int(t)
    if repeat_replay_arrivals:
        time_idx %= experiment_horizon
    elif time_idx < 0 or time_idx >= experiment_horizon:
        return _zero_arrival_od_counts
    return arrival_od_tensor[day_idx, time_idx]


def time_lambda(i, d, t, base_lambda=None):
    idx = int(t) % experiment_horizon
    return max(0.0, float(lambda_profile[idx, int(i), int(d)]) * demand_scale)


platoon_factor = 2
c_wait = 3
platoon_max = 8
c_move_max = 12


def c_move_m(x):
    return c_move_max * pow(x / platoon_max, 1 / platoon_factor)
