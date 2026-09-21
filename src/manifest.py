"""Reproducibility metadata, written beside every measured number.

A result whose software versions, seed, quantization and decoding settings are unrecorded
cannot be defended a month later, and a benchmark comparing two configs is only meaningful
if the reader can see what else differed between them. Cheap to write, impossible to
reconstruct afterwards.
"""

from __future__ import annotations

import json
import platform
from importlib import metadata
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

_PACKAGES = (
    "torch", "transformers", "datasets", "peft", "trl",
    "bitsandbytes", "vllm", "accelerate", "pydantic",
)


def versions() -> dict[str, str]:
    """Installed versions of the packages that can move a number."""
    found = {}
    for package in _PACKAGES:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            found[package] = "not installed"
    return found


def gpu() -> str | None:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        return None


def write(name: str, **details) -> dict:
    """Write artifacts/<name>_manifest.json and return it."""
    manifest = {
        **details,
        "gpu": gpu(),
        "python": platform.python_version(),
        "packages": versions(),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"{name}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
