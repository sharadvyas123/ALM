"""Generate a new version, dry-run planning, or verify a completed version."""

import argparse
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alm.data.generation import generate, load_config, prepare, verify_existing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/data/mad_source_disjoint_v1.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Compute source and mixture plans without writing files")
    mode.add_argument("--verify-existing", action="store_true", help="Read-only deterministic replay and complete audio verification")
    args = parser.parse_args()
    config, paths = load_config(args.config, ROOT)
    if args.dry_run or args.verify_existing:
        sources, assigned, recipes, summary = prepare(config, paths)
        if args.verify_existing:
            summary = verify_existing(config, paths, sources, assigned, recipes, summary)
    else:
        summary, report = generate(config, paths, ROOT, args.config, shlex.join([sys.executable, *sys.argv]))
        summary["audit"] = {k: report[k] for k in ("verdict", "split_counts", "overlapping_source_count",
                                                "overlapping_group_count", "audio_files_checked", "waveforms_reconstructed")}
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
