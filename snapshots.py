import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path


SCHEMA_VERSION = 1
SNAPSHOT_ROOT = Path("data/snapshots")
REPRODUCIBILITY_FILES = (
    Path("cpudata.py"),
    Path("snapshots.py"),
    Path("var_metric.py"),
    Path("uv.lock"),
)


def parse_price_usd_cents(raw_price):
    """Parse a PassMark price without losing cent precision."""
    normalized = raw_price.replace(",", "").replace("$", "").replace("*", "").strip()
    if normalized == "NA":
        return None
    cents = Decimal(normalized) * 100
    if cents != cents.to_integral_value():
        raise ValueError(f"price has sub-cent precision: {raw_price}")
    return int(cents)


def git_run_metadata():
    """Describe the checkout that performed the scrape."""
    git_sha = os.environ.get("GITHUB_SHA")
    if not git_sha:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout != ""

    return {
        "git_sha": git_sha,
        "git_dirty": dirty,
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "python_version": platform.python_version(),
        "file_sha256": {
            path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in REPRODUCIBILITY_FILES
        },
    }


def build_snapshot(observations, source_url, source_content, captured_at):
    """Build and validate a lossless snapshot of the scraped table."""
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    if not observations:
        raise ValueError("refusing to save an empty snapshot")

    cpu_ids = [row["cpu_id"] for row in observations]
    if any(cpu_id is None for cpu_id in cpu_ids):
        raise ValueError("every observation must have a CPU ID")
    if len(cpu_ids) != len(set(cpu_ids)):
        raise ValueError("CPU IDs must be unique within a snapshot")
    if any(row["score"] <= 0 for row in observations):
        raise ValueError("CPU scores must be positive")
    if any(
        row["price_usd_cents"] is not None and row["price_usd_cents"] < 0
        for row in observations
    ):
        raise ValueError("CPU prices cannot be negative")

    captured_at = captured_at.astimezone(timezone.utc)

    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": captured_at.isoformat().replace("+00:00", "Z"),
        "source": {
            "url": source_url,
            "content_sha256": hashlib.sha256(source_content).hexdigest(),
        },
        "run": git_run_metadata(),
        "units": {
            "score": "PassMark CPU Mark",
            "price": "USD cents",
        },
        "row_count": len(observations),
        "observations": observations,
    }


def write_snapshot(snapshot, captured_at, root=SNAPSHOT_ROOT):
    """Atomically write a timestamped snapshot and return its path and hash."""
    captured_at = captured_at.astimezone(timezone.utc)
    relative_path = Path(
        f"{captured_at:%Y/%m/%d}/{captured_at:%Y-%m-%dT%H-%M-%SZ}.json"
    )
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = (json.dumps(snapshot, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=".snapshot-",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    try:
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return path, digest
