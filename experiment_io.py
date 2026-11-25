"""Shared input validation for the fresh 5,000-game experiment."""
from pathlib import Path
import pandas as pd

OUT = Path("processed_data_5000")
EXPECTED_GAMES = 5000
KEYS = ["Game_ID", "Player"]

def read_required(root, name, prerequisite, **kwargs):
    path = Path(root) / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path.resolve()}. Run {prerequisite} first, from the repository root.")
    return pd.read_csv(path, **kwargs)

def validate_players(d, name="dataset", expected_games=EXPECTED_GAMES):
    if d[KEYS].isna().any().any() or d.duplicated(KEYS).any():
        raise ValueError(f"{name}: missing or duplicate game/player keys")
    games = d.Game_ID.nunique()
    if games != expected_games or len(d) != 4 * expected_games or not d.groupby("Game_ID").size().eq(4).all():
        raise ValueError(f"{name}: found {games:,} games and {len(d):,} players; expected {expected_games:,} games and {4*expected_games:,} players. Check your input folder.")
    return d

def same_players(a, b):
    return pd.MultiIndex.from_frame(a[KEYS]).sort_values().equals(pd.MultiIndex.from_frame(b[KEYS]).sort_values())

def validate_advantage(d, expected_games=EXPECTED_GAMES):
    validate_players(d, "starting_advantage.csv", expected_games)
    required = ["Blocked", "Status", "Turns_To_50", "Prob_Turn_10", "Player_Number"]
    missing = set(required) - set(d.columns)
    if missing:
        raise ValueError(f"Missing starting advantage columns: {sorted(missing)}. Rerun 05_starting_advantage.")
    if d.Blocked.isna().any() or not pd.api.types.is_bool_dtype(d.Blocked):
        raise ValueError("Blocked must be a Boolean column. Rerun 05_starting_advantage.")
    if not d.Prob_Turn_10.between(0, 1).all() or not d.loc[d.Blocked,"Turns_To_50"].isna().all():
        raise ValueError("Invalid readiness values. Rerun 05_starting_advantage.")
    return d
