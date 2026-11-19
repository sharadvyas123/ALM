"""Rebuild aggregate CV5 tables and figures from completed fold artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from alm.data.provenance import sha256
from scripts.train_cv5 import _aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id:
        raise ValueError("run-id must be a simple directory name")
    run_root = ROOT / "runs" / args.run_id
    results = {}
    for fold in range(1, 6):
        fold_dir = run_root / f"fold_{fold:02d}"
        status_path = fold_dir / "fold_status.json"
        if not status_path.is_file():
            raise FileNotFoundError(f"Missing fold status: {status_path}")
        status = json.loads(status_path.read_text())
        if status.get("status") != "complete":
            raise ValueError(f"Cannot aggregate incomplete fold {fold:02d}")
        for name, digest in status.get("artifact_sha256", {}).items():
            if sha256(fold_dir / name) != digest:
                raise ValueError(f"Fold artifact checksum mismatch: {fold_dir / name}")
        results[f"fold_{fold:02d}"] = json.loads((fold_dir / "outer_metrics.json").read_text())
    result = _aggregate(run_root, results)
    print(json.dumps({"status": result["status"], "aggregate": str(run_root / "aggregate/summary.json")}, indent=2))


if __name__ == "__main__":
    main()
