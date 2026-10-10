"""Strict metadata IO and conservative parent-recording identities."""

import csv
import hashlib
import json
import posixpath
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlsplit


CLASS_NAMES = (
    "Communication", "Gunshot", "Footsteps", "Shelling", "Vehicle",
    "Helicopter", "Fighter",
)
METADATA_FIELDS = (
    "id", "file", "labels", "label_names", "label_vector", "mix_type",
    "source_files", "mix_weights", "split",
)


def read_csv(path, required=()):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields) or not set(required) <= set(fields):
            raise ValueError(f"Invalid CSV schema: {path}")
        rows = list(reader)
        if any(None in row or any(v is None for v in row.values()) for row in rows):
            raise ValueError(f"Malformed CSV record: {path}")
    return rows


def write_csv(path, rows, fields=None):
    """Create a new artifact; never replace a previous artifact."""
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def relative_id(value):
    """Normalize path syntax, retaining case and the full directory identity."""
    if not isinstance(value, str) or not value.strip() or "\x00" in value or "|" in value:
        raise ValueError(f"Invalid source path: {value!r}")
    value = value.strip().replace("\\", "/")
    normalized = posixpath.normpath(value)
    if (normalized in {".", ".."} or normalized.startswith(("/", "../"))
            or re.match(r"^[A-Za-z]:", normalized)):
        raise ValueError(f"Expected a dataset-relative path: {value!r}")
    return normalized


def reference_id(value, audio_root):
    """Resolve absolute/relative aliases against the same explicit audio root."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing source reference")
    spelling = value.strip().replace("\\", "/")
    if re.match(r"^[A-Za-z]:", spelling):
        raise ValueError("Foreign drive path requires an explicit relocation map")
    root = Path(audio_root).resolve()
    path = Path(spelling)
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        return relative_id(resolved.relative_to(root).as_posix())
    except ValueError as exc:
        raise ValueError(f"Source outside the configured audio root: {value!r}") from exc


def youtube_id(url):
    """Ignore playback parameters; never guess or repair malformed video IDs."""
    try:
        parsed = urlsplit(url.strip())
        host = (parsed.hostname or "").lower()
        parts = parsed.path.strip("/").split("/")
        value = None
        if host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
            if parsed.path == "/watch":
                values = set(parse_qs(parsed.query).get("v", []))
                value = next(iter(values)) if len(values) == 1 else None
            elif len(parts) == 2 and parts[0] in {"embed", "shorts", "live"}:
                value = parts[1]
        elif host in {"youtu.be", "www.youtu.be"} and len(parts) == 1:
            value = parts[0]
        return value if value and re.fullmatch(r"[A-Za-z0-9_-]{11}", value) else ""
    except ValueError:
        return ""


def build_sources(processed_rows, raw_rows):
    """Link full folders and validated video IDs transitively across BOTH official splits.

    Missing URLs do not share an identity. A missing URL can inherit only its
    own folder's group. All folders in a component must remain in one split.
    """
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    raw_by_path = {}
    folder_videos = defaultdict(set)
    video_folders = defaultdict(set)
    invalid_urls = 0
    for row in raw_rows:
        path = relative_id(row["path"])
        if path in raw_by_path:
            raise ValueError(f"Duplicate raw source path: {path}")
        split = PurePosixPath(path).parts[0]
        if split not in {"training", "test"}:
            raise ValueError(f"Unexpected raw split: {path}")
        raw_by_path[path] = row
        folder = str(PurePosixPath(path).parent)
        find("folder:" + folder)
        folder_videos[folder]
        vid = youtube_id(row["youtube url"])
        if vid:
            folder_videos[folder].add(vid)
            video_folders[vid].add(folder)
            union("folder:" + folder, "youtube:" + vid)
        else:
            invalid_urls += 1

    # Use raw test membership even if a raw test clip was not preprocessed.
    test_groups = {
        find("folder:" + str(PurePosixPath(path).parent))
        for path in raw_by_path if PurePosixPath(path).parts[0] == "test"
    }
    sources = []
    seen_files, seen_paths, seen_ids = set(), set(), set()
    for row in processed_rows:
        file = relative_id(row["file"])
        path = relative_id(row["source_file"])
        if file in seen_files or path in seen_paths or row["id"] in seen_ids:
            raise ValueError("Duplicate processed file, source path, or ID")
        seen_files.add(file)
        seen_paths.add(path)
        seen_ids.add(row["id"])
        if path not in raw_by_path:
            raise ValueError(f"Unresolved processed source: {path}")
        original = raw_by_path[path]
        label = int(row["label"])
        if not 0 <= label < len(CLASS_NAMES) or CLASS_NAMES[label] != row["label_name"]:
            raise ValueError(f"Noncanonical label mapping: {file}")
        official_split = PurePosixPath(path).parts[0]
        if original["label"] != row["label"] or official_split != row["split"]:
            raise ValueError(f"Raw/processed label or split mismatch: {file}")
        folder = str(PurePosixPath(path).parent)
        group = find("folder:" + folder)
        vid = youtube_id(original["youtube url"])
        verified = bool(folder_videos[folder])
        if official_split == "test":
            assignment, reason = "official_test", "official_test"
        elif group in test_groups:
            assignment, reason = "excluded", "official_test_linked_group"
        elif not verified:
            assignment, reason = "excluded", "unverified_video_identity"
        else:
            assignment, reason = "eligible", ""
        sources.append({
            "source_id": path, "processed_id": row["id"], "file": file,
            "source_file": path, "label": label, "label_name": CLASS_NAMES[label],
            "official_split": official_split, "folder": folder,
            "video_id": vid, "group_id": group,
            "provenance_status": "verified_video" if vid else (
                "linked_via_folder" if verified else "unverified_folder"),
            "split": assignment, "exclusion_reason": reason,
        })
    sources.sort(key=lambda r: r["file"])
    summary = {
        "raw_rows": len(raw_rows), "processed_rows": len(sources),
        "invalid_or_missing_url_rows": invalid_urls,
        "folders_with_multiple_valid_video_ids": sum(len(v) > 1 for v in folder_videos.values()),
        "video_ids_in_multiple_folders": sum(len(v) > 1 for v in video_folders.values()),
        "eligibility_counts": dict(Counter(r["split"] for r in sources)),
        "exclusion_counts": dict(Counter(r["exclusion_reason"] for r in sources if r["split"] == "excluded")),
        "unverified_official_test_clips": sum(
            r["split"] == "official_test" and r["provenance_status"] == "unverified_folder"
            for r in sources
        ),
    }
    return sources, summary


def load_sources(processed_csv, training_csv, test_csv):
    processed = read_csv(processed_csv, ("id", "file", "source_file", "label", "label_name", "split"))
    required = ("path", "label", "youtube url")
    raw = []
    for path, split in ((training_csv, "training"), (test_csv, "test")):
        rows = read_csv(path, required)
        if any(PurePosixPath(relative_id(r["path"])).parts[0] != split for r in rows):
            raise ValueError(f"Raw CSV split mismatch: {path}")
        raw.extend(rows)
    return build_sources(processed, raw)
