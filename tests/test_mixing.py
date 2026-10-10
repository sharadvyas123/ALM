import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm.data.auditing import audit_dataset, audit_rows
from alm.data.generation import generate
from alm.data.mixing import metadata_rows, plan_mixtures, render_dataset, render_recipe
from alm.data.provenance import METADATA_FIELDS, sha256, write_csv
from _mad_fixtures import assigned_sources, mixing_config


class MixtureTests(unittest.TestCase):
    def setUp(self):
        self.sources, self.assigned = assigned_sources()
        self.config = mixing_config()
        self.recipes = plan_mixtures(self.assigned, self.config)

    def test_components_stay_in_pool_and_label_union_is_correct(self):
        lookup = {r["file"]: r for r in self.assigned}
        for row in self.recipes:
            refs = row["source_files"].split("|")
            self.assertTrue(all(lookup[r]["split"] == row["split"] for r in refs))
            labels = {lookup[r]["label"] for r in refs}
            self.assertEqual(list(map(int, row["label_vector"].split(","))), [int(c in labels) for c in range(7)])
            gains = list(map(float, row["mix_weights"].split("|")))
            self.assertAlmostEqual(sum(gains), 1.0)
            if row["mix_type"] == "3-class":
                self.assertGreaterEqual(min(gains), 0.2)
        self.assertEqual(self.recipes, plan_mixtures(list(reversed(self.assigned)), self.config))

    def test_clean_sources_unique_and_streams_independent(self):
        for split in ("train", "val"):
            clean = [r["source_files"] for r in self.recipes if r["split"] == split and r["mix_type"] == "clean"]
            self.assertEqual(len(clean), len(set(clean)))
        config = copy.deepcopy(self.config)
        config["counts_by_split"]["train"]["2-class"] += 1
        changed = plan_mixtures(self.assigned, config)
        # IDs shift when train size changes, but validation choices and gains do not.
        fields = ("source_files", "mix_weights", "mix_type")
        choices = lambda rs: [[r[k] for k in fields] for r in rs if r["split"] == "val"]
        self.assertEqual(choices(self.recipes), choices(changed))

    def test_known_signal_render_and_exclusive_output(self):
        with tempfile.TemporaryDirectory(dir="/tmp/omnirush") as directory:
            root = Path(directory)
            signals = {"a.wav": np.full(160000, 10000, dtype=np.int16),
                       "b.wav": np.full(160000, -20000, dtype=np.int16)}
            for name, audio in signals.items():
                sf.write(root / name, audio, 16000, subtype="PCM_16")
            recipe = {"file": "mixed.wav", "source_files": "a.wav|b.wav",
                      "mix_weights": "0.3|0.7", "mix_type": "2-class"}
            actual, scale = render_recipe(recipe, root, self.config)
            self.assertTrue(np.all(actual == -11000))
            self.assertEqual(scale, 1.0)
            render_dataset([recipe], root, root, self.config)
            before = (root / "mixed.wav").read_bytes()
            with self.assertRaises(FileExistsError):
                render_dataset([recipe], root, root, self.config)
            self.assertEqual((root / "mixed.wav").read_bytes(), before)
            sf.write(root / "bad.wav", signals["a.wav"], 8000, subtype="PCM_16")
            with self.assertRaises(ValueError):
                render_recipe(dict(recipe, source_files="a.wav|bad.wav"), root, self.config)

    def test_generation_refuses_existing_directory_before_writes(self):
        with tempfile.TemporaryDirectory(dir="/tmp/omnirush") as directory:
            root = Path(directory)
            marker = root / "marker"
            marker.write_text("keep")
            with self.assertRaises(FileExistsError):
                generate({}, {"output_dir": root, "manifest_dir": root / "new"}, root, None, "test")
            self.assertEqual(marker.read_text(), "keep")
            self.assertFalse((root / "new").exists())

    def test_complete_audio_audit_detects_tampering(self):
        with tempfile.TemporaryDirectory(dir="/tmp/omnirush") as directory:
            root = Path(directory)
            audio = root / "audio"
            audio.mkdir()
            for source in self.sources:
                signal = np.full(160000, (source["label"] + 1) * 1000, dtype=np.int16)
                sf.write(root / source["file"], signal, 16000, subtype="PCM_16")
            rendered = render_dataset(self.recipes, root, audio, self.config)
            write_csv(root / "metadata.csv", metadata_rows(self.recipes), METADATA_FIELDS)
            write_csv(root / "source_splits.csv", [dict(r, audio_sha256=sha256(root / r["file"])) for r in self.assigned])
            write_csv(root / "mixtures.csv", rendered)
            def audit():
                return audit_dataset(root / "metadata.csv", self.sources, root, root / "source_splits.csv",
                                     root / "mixtures.csv", audio, True, self.config)
            report = audit()
            self.assertTrue(report["complete"], report["errors"])
            self.assertEqual(report["waveforms_reconstructed"], len(rendered))
            self.assertEqual(report["source_audio_hashes_checked"], len(self.assigned))
            sf.write(audio / rendered[0]["file"], np.zeros(160000, dtype=np.int16), 16000, subtype="PCM_16")
            report = audit()
            self.assertFalse(report["complete"])
            self.assertTrue(any("checksum mismatch" in e for e in report["errors"]))
            sf.write(root / self.sources[0]["file"], np.zeros(160000, dtype=np.int16), 16000, subtype="PCM_16")
            report = audit()
            self.assertTrue(any("Source audio checksum mismatch" in e for e in report["errors"]))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.sources, assigned = assigned_sources()
        self.rows = metadata_rows(plan_mixtures(assigned, mixing_config()))
        self.audio = Path("/tmp/omnirush")

    def test_exact_and_parent_overlap_are_separate(self):
        # Move a clean row to validation; its folder mates remain in training.
        row = next(r for r in self.rows if r["split"] == "train" and r["mix_type"] == "clean")
        original = dict(row)
        row["split"] = "val"
        self.rows.append(dict(original, id="999", file="extra.wav"))
        report = audit_rows(self.rows, self.sources, self.audio)
        self.assertTrue(report["complete"], report["errors"])
        self.assertGreaterEqual(report["overlapping_source_count"], 1)
        self.assertGreaterEqual(report["overlapping_group_count"], 1)
        self.assertGreaterEqual(report["affected_examples"]["val"], 1)

    def test_absolute_alias_not_counted_as_another_source(self):
        row = next(r for r in self.rows if r["split"] == "train" and r["mix_type"] == "clean")
        alias = dict(row, id="999", file="extra.wav", split="val",
                     source_files=str(self.audio / row["source_files"]))
        self.rows.append(alias)
        report = audit_rows(self.rows, self.sources, self.audio)
        self.assertTrue(report["complete"], report["errors"])
        self.assertEqual(report["overlapping_source_count"], 1)

    def test_malformed_missing_unknown_and_bad_label_rows_fail(self):
        for change in ({"source_files": ""}, {"source_files": "missing.wav"},
                       {"source_files": self.rows[0]["source_files"] + "|"},
                       {"split": "unknown"}, {"label_vector": "0,0,0,0,0,0,0"},
                       {"mix_weights": "nan"}):
            with self.subTest(change=change):
                rows = copy.deepcopy(self.rows)
                rows[0].update(change)
                report = audit_rows(rows, self.sources, self.audio)
                self.assertFalse(report["complete"])
                self.assertTrue(report["verdict"].startswith("Inconclusive"))


if __name__ == "__main__":
    unittest.main()
