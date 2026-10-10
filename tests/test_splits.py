import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alm.data.provenance import build_sources, reference_id, relative_id, youtube_id
from alm.data.splitting import split_sources
from _mad_fixtures import fixture_rows


class SourceIdentityTests(unittest.TestCase):
    def test_url_identity_preserves_case_and_ignores_timestamp(self):
        self.assertEqual(youtube_id("https://youtu.be/AbCdEf123_-?t=20"), "AbCdEf123_-")
        self.assertEqual(youtube_id("https://www.youtube.com/watch?v=AbCdEf123_-&t=1s"), "AbCdEf123_-")
        self.assertEqual(youtube_id("https://www.youtube.com/watch?v=too_long_123456"), "")
        self.assertEqual(youtube_id(""), "")
        self.assertEqual(youtube_id("https://www.youtube.com/watch?v=AbCdEf123_-&v=XXXXXXXXXXX"), "")

    def test_paths_keep_full_identity(self):
        root = Path("/tmp/omnirush")
        self.assertEqual(reference_id("./sub/../Example.wav", root), "Example.wav")
        self.assertEqual(reference_id(str(root / "Example.wav"), root), "Example.wav")
        self.assertEqual(relative_id("training\\001\\1.wav"), "training/001/1.wav")
        self.assertNotEqual(relative_id("training/001/1.wav"), relative_id("training/002/1.wav"))
        self.assertNotEqual(reference_id("Example.wav", root), reference_id("example.wav", root))
        for value in ("", "..", "../outside.wav", "C:\\elsewhere\\1.wav"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reference_id(value, root)

    def test_transitive_grouping_and_test_isolation(self):
        processed, raw = fixture_rows(4)
        # Folder 0 contains two videos, linking folder 1 to folder 0.
        raw[1]["youtube url"] = raw[7]["youtube url"]
        # Folder 2 links to folder 1's video too, and an official test clip.
        raw[14]["youtube url"] = raw[7]["youtube url"]
        raw.append({"path": "test/900/0.wav", "label": "0", "youtube url": raw[7]["youtube url"]})
        sources, _ = build_sources(processed, raw)
        linked = [r for r in sources if r["folder"] != "training/003"]
        self.assertEqual(len({r["group_id"] for r in linked}), 1)
        self.assertTrue(all(r["exclusion_reason"] == "official_test_linked_group" for r in linked))
        self.assertTrue(all(r["split"] == "eligible" for r in sources if r["folder"] == "training/003"))

    def test_missing_urls_are_not_one_shared_recording(self):
        processed, raw = fixture_rows(3)
        for row in raw[:14]:
            row["youtube url"] = ""
        # One bad clip in a verified folder may inherit that folder's group.
        raw[14]["youtube url"] = "invalid"
        sources, summary = build_sources(processed, raw)
        unknown = [r for r in sources if r["split"] == "excluded"]
        self.assertEqual(len(unknown), 14)
        self.assertEqual(len({r["group_id"] for r in unknown}), 2)
        self.assertEqual(sources[14]["provenance_status"], "linked_via_folder")
        self.assertEqual(summary["eligibility_counts"]["eligible"], 7)

    def test_rejects_ambiguous_source_mapping(self):
        processed, raw = fixture_rows(2)
        with self.assertRaises(ValueError):
            build_sources(processed, raw + [raw[0]])
        duplicate = copy.deepcopy(processed)
        duplicate[1]["source_file"] = duplicate[0]["source_file"]
        with self.assertRaises(ValueError):
            build_sources(duplicate, raw)


class SourceSplitTests(unittest.TestCase):
    def test_split_is_group_disjoint_covers_classes_and_replays(self):
        processed, raw = fixture_rows()
        sources, _ = build_sources(processed, raw)
        assigned, summary = split_sources(sources, 42, 0.25, 64, {"train": 7, "val": 7})
        again, _ = split_sources(sources, 42, 0.25, 64, {"train": 7, "val": 7})
        self.assertEqual(assigned, again)
        train = {r["group_id"] for r in assigned if r["split"] == "train"}
        val = {r["group_id"] for r in assigned if r["split"] == "val"}
        self.assertFalse(train & val)
        self.assertEqual(len(train) + len(val), 8)
        self.assertTrue(all(len(summary["pools"][s]["class_counts"]) == 7 for s in ("train", "val")))
        shuffled_sources, _ = build_sources(list(reversed(processed)), list(reversed(raw)))
        self.assertEqual(split_sources(shuffled_sources, 42, 0.25, 64, {"train": 7, "val": 7})[0], assigned)

    def test_infeasible_clean_capacity_fails(self):
        processed, raw = fixture_rows()
        sources, _ = build_sources(processed, raw)
        with self.assertRaisesRegex(ValueError, "No feasible"):
            split_sources(sources, 42, 0.25, 4, {"train": 1000, "val": 1000})


if __name__ == "__main__":
    unittest.main()
