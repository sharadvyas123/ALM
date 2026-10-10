from alm.data.provenance import CLASS_NAMES, build_sources


def fixture_rows(group_count=8):
    processed, raw = [], []
    for group in range(group_count):
        for label, name in enumerate(CLASS_NAMES):
            index = len(processed) + 1
            path = f"training/{group:03d}/{label}.wav"
            processed.append({"id": str(index), "file": f"source_{index:03d}.wav",
                              "source_file": path, "label": str(label),
                              "label_name": name, "split": "training"})
            raw.append({"path": path, "label": str(label),
                        "youtube url": f"https://www.youtube.com/watch?v=V{group:010d}&t=1s"})
    return processed, raw


def assigned_sources():
    processed, raw = fixture_rows(4)
    sources, _ = build_sources(processed, raw)
    assigned = [dict(r, split="train" if r["folder"] in {"training/000", "training/001"} else "val") for r in sources]
    return sources, assigned


def mixing_config():
    return {"seed": 42, "sample_rate": 16000, "duration_seconds": 10,
            "alpha_range": [0.3, 0.7], "dirichlet_concentration": [1, 1, 1],
            "three_class_weight_floor": 0.2,
            "counts_by_split": {s: {"clean": 14, "2-class": 3, "3-class": 3} for s in ("train", "val")}}
