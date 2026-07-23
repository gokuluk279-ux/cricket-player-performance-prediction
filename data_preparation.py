"""Utilities for preparing ODI ball-by-ball data."""

from __future__ import annotations

from pathlib import Path
import pandas as pd


PHASE_LABELS = {
    "powerplay": "Powerplay",
    "middle": "Middle Overs",
    "death": "Death Overs",
}


def assign_phase(ball: float) -> str:
    """Return the ODI innings phase for a ball value such as 10.3."""
    if pd.isna(ball):
        return "Unknown"

    over = int(float(ball))
    if 0 <= over <= 10:
        return PHASE_LABELS["powerplay"]
    if 11 <= over <= 40:
        return PHASE_LABELS["middle"]
    if 41 <= over <= 50:
        return PHASE_LABELS["death"]
    return "Outside ODI Range"


def load_and_prepare(path: str | Path, minimum_year: int = 2014) -> pd.DataFrame:
    """Load the raw dataset and perform basic cleaning and phase assignment."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path, low_memory=False)
    required = {
        "match_id", "start_date", "venue", "innings", "ball",
        "batting_team", "bowling_team", "striker", "runs_off_bat",
        "wicket_type",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data["start_date"] = pd.to_datetime(
        data["start_date"], errors="coerce", dayfirst=True
    )
    data["runs_off_bat"] = pd.to_numeric(data["runs_off_bat"], errors="coerce").fillna(0)
    data = data[data["start_date"].dt.year >= minimum_year].copy()
    data["phase"] = data["ball"].apply(assign_phase)
    data["is_dismissal"] = data["wicket_type"].notna().astype(int)
    return data


def build_batting_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate batting performance by player and innings phase."""
    summary = (
        data.groupby(["striker", "phase"], as_index=False)
        .agg(
            total_runs=("runs_off_bat", "sum"),
            balls_faced=("ball", "count"),
            dismissals=("is_dismissal", "sum"),
            matches=("match_id", "nunique"),
        )
    )
    summary["strike_rate"] = (
        100 * summary["total_runs"] / summary["balls_faced"].clip(lower=1)
    )
    summary["batting_average"] = (
        summary["total_runs"] / summary["dismissals"].replace(0, pd.NA)
    )
    return summary
