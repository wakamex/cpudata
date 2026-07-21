import json
import subprocess
from pathlib import Path

from var_metric import VAR_METHOD, calc_auc_above_regression


OUTPUT_FILE = Path("docs/history.json")


def git(*args):
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def load_historical_snapshots():
    """Load the newest committed data snapshot for every available date."""
    snapshots = {}
    revisions = git("log", "--format=%H", "--", "docs/data.json").splitlines()
    for revision in revisions:
        data = json.loads(git("show", f"{revision}:docs/data.json"))
        date = data.get("updated")
        if (
            not date
            or date in snapshots
            or "regression" not in data
            or "brand_frontiers" not in data
        ):
            continue
        snapshots[date] = data

    # Prefer the working-tree snapshot so rerunning after today's analysis
    # preserves an entry that has not been committed yet.
    working_data = json.loads(Path("docs/data.json").read_text())
    date = working_data.get("updated")
    if date and "regression" in working_data and "brand_frontiers" in working_data:
        snapshots[date] = working_data
    return snapshots


def calculate_entry(date, data):
    regression = data["regression"]
    values = {}
    for brand in ("AMD", "Intel"):
        points = data["brand_frontiers"].get(brand, [])
        values[brand] = calc_auc_above_regression(
            [point["price"] for point in points],
            [point["score"] for point in points],
            regression["slope"],
            regression["intercept"],
        )
    return {
        "date": date,
        "amd_var": values["AMD"],
        "intel_var": values["Intel"],
        "method": VAR_METHOD,
    }


def main():
    snapshots = load_historical_snapshots()
    history = [calculate_entry(date, snapshots[date]) for date in sorted(snapshots)]
    OUTPUT_FILE.write_text(json.dumps(history, indent=2) + "\n")
    print(f"Backfilled {len(history)} entries in {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
