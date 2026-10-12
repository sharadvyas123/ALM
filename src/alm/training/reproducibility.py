"""Seeds and environment provenance for every experiment."""
from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import random
import subprocess
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if torch.cuda.is_available() and deterministic:
        # PyTorch's flash-attention backward can remain nondeterministic even
        # when deterministic algorithms are requested. Use the deterministic
        # math SDP path for reproducible research runs.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def worker_seed(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def environment_record(root: Path | None = None) -> dict:
    """Return JSON-safe Python, package, CUDA and git provenance."""
    packages = {}
    for name in ("numpy", "torch", "torchaudio", "soundfile", "pandas", "scikit-learn", "tqdm", "matplotlib"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "pid": os.getpid(), "packages": packages, "torch": torch.__version__,
              "cuda_available": torch.cuda.is_available(),
              "cuda_version": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if root is not None:
        try:
            result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            result["git_status"] = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            result["git_commit"], result["git_status"] = None, None
    return result


def write_environment(path: Path, root: Path | None = None) -> dict:
    value = environment_record(root)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return value
