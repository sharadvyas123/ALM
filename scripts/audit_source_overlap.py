"""Read-only legacy/new MAD provenance audit; optional exclusive JSON export."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alm.data.auditing import audit_dataset, require_disjoint
from alm.data.provenance import load_sources, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, default=ROOT / "data/processed_mad/metadata.csv")
    parser.add_argument("--source-audio", type=Path, default=ROOT / "data/processed_mad/audio")
    parser.add_argument("--raw-training", type=Path, default=ROOT / "data/raw/MAD_dataset/training.csv")
    parser.add_argument("--raw-test", type=Path, default=ROOT / "data/raw/MAD_dataset/test.csv")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--mixture-manifest", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--verify-waveforms", action="store_true")
    parser.add_argument("--require-disjoint", action="store_true")
    parser.add_argument("--output", type=Path, help="New JSON report path; existing reports are never overwritten")
    args = parser.parse_args()
    if args.verify_waveforms and (not args.audio_dir or not args.mixture_manifest or not args.source_manifest):
        parser.error("--verify-waveforms requires --audio-dir, --mixture-manifest, and --source-manifest")
    try:
        sources, provenance = load_sources(args.source_csv, args.raw_training, args.raw_test)
        report = audit_dataset(
            args.metadata, sources, args.source_audio, args.source_manifest,
            args.mixture_manifest, args.audio_dir, args.verify_waveforms,
            {"sample_rate": 16000, "duration_seconds": 10},
        )
        report["provenance"] = provenance
        report["input_sha256"] = {str(p): sha256(p) for p in (args.metadata, args.source_csv, args.raw_training, args.raw_test)}
    except (ValueError, KeyError, OSError) as exc:
        report = {"complete": False, "verdict": "Inconclusive — insufficient or unreliable source metadata",
                  "error_count": 1, "errors": [str(exc)]}
    if args.output:
        write_json(args.output, report)
    omitted = {"affected_example_files", "overlapping_sources", "overlapping_groups"}
    print(json.dumps({k: v for k, v in report.items() if k not in omitted}, indent=2, sort_keys=True))
    if not report["complete"]:
        return 2
    if args.require_disjoint:
        try:
            require_disjoint(report)
        except ValueError:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
