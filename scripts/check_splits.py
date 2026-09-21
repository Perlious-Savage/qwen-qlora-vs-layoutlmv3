"""Check that no test document also appears in the training split.

CORD's splits are separate files, so they are disjoint by construction. That is not the
risk. The risk is repeated receipts: CORD was collected from a limited set of restaurants,
and the same receipt layout - sometimes the same transaction - can appear more than once.
A fine-tune that has memorised a test receipt reports a score that is partly recall of
training data.

If overlap exists it does not invalidate the Project 1 comparison, because Project 1
trained on the identical split and inherited the identical overlap. It does have to be
reported rather than discovered by a reader.

Documents are compared by their word sequence, which is what both models actually see.

    python scripts/check_splits.py

Needs all three CORD-v2 splits (~2.3GB on first run).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import DATASET_ID, parse_example  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def fingerprints(dataset) -> dict[str, list[int]]:
    """Map word-sequence hash -> indices, so duplicates within a split show up too."""
    seen: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(dataset):
        image = record["image"]
        words = parse_example(record["ground_truth"], image.width, image.height)["words"]
        digest = hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest()
        seen[digest].append(index)
    return seen


def main() -> None:
    from datasets import load_dataset

    print(f"Loading {DATASET_ID} ...")
    splits = {name: load_dataset(DATASET_ID, split=name) for name in ("train", "validation", "test")}
    prints = {name: fingerprints(dataset) for name, dataset in splits.items()}

    train_test = sorted(set(prints["train"]) & set(prints["test"]))
    train_validation = sorted(set(prints["train"]) & set(prints["validation"]))

    report = {
        "dataset": DATASET_ID,
        "method": "sha256 of the space-joined word sequence from data.parse_example",
        "n_documents": {name: len(dataset) for name, dataset in splits.items()},
        "n_unique_documents": {name: len(p) for name, p in prints.items()},
        "duplicates_within_split": {
            name: sum(len(indices) - 1 for indices in p.values() if len(indices) > 1)
            for name, p in prints.items()
        },
        "train_test_overlap": len(train_test),
        "train_validation_overlap": len(train_validation),
        "overlapping_test_indices": sorted(
            index for digest in train_test for index in prints["test"][digest]
        ),
        "note": (
            "Any overlap is shared with Project 1, which trained on the same split. It "
            "affects both sides of the comparison equally and is reported, not corrected."
        ),
    }

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "split_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if train_test:
        print(
            f"\nWARNING: {len(train_test)} test documents have an identical word sequence "
            "in train. Report this next to the headline F1."
        )
    else:
        print("\nNo train/test overlap by word sequence.")


if __name__ == "__main__":
    main()
