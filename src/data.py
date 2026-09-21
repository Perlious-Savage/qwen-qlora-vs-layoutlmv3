"""CORD-v2 loader and label construction.

CORD-v2 (naver-clova-ix/cord-v2, CC-BY-4.0) ships a `ground_truth` JSON per receipt.
Inside it, `valid_line` entries are already annotated at word level: each carries a
category (for example `menu.nm`, `total.total_price`) and a list of words, each with
its own quadrilateral.

That matters. It means converting CORD into BIO token-classification labels is a
deterministic mapping, not a fuzzy string match against OCR output. Fuzzy alignment
would inject label noise into training and evaluation at the same time, and the two
errors would be correlated, so they would hide rather than show up.

The official dataset is used directly rather than a community-preprocessed copy,
because several of those rely on dataset loading scripts that current versions of
`datasets` refuse to execute.
"""

from __future__ import annotations

import json
from typing import Any

DATASET_ID = "naver-clova-ix/cord-v2"


def quad_to_bbox(quad: dict[str, Any]) -> list[int]:
    """Collapse CORD's 4-point polygon into an axis-aligned box."""
    xs = [quad["x1"], quad["x2"], quad["x3"], quad["x4"]]
    ys = [quad["y1"], quad["y2"], quad["y3"], quad["y4"]]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    """LayoutLMv3 expects coordinates on a 0-1000 scale, not raw pixels."""
    x0, y0, x1, y1 = bbox
    scaled = [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]
    # Clamp: a stray annotation outside the page will otherwise fail the embedding lookup.
    return [max(0, min(1000, v)) for v in scaled]


def parse_example(ground_truth: str, width: int, height: int) -> dict[str, list]:
    """Turn one CORD record into words, normalized boxes and BIO tags."""
    gt = json.loads(ground_truth)
    words: list[str] = []
    bboxes: list[list[int]] = []
    tags: list[str] = []

    for line in gt.get("valid_line", []):
        category = line.get("category", "other")
        for position, word in enumerate(line.get("words", [])):
            text = (word.get("text") or "").strip()
            if not text:
                continue
            words.append(text)
            bboxes.append(normalize_bbox(quad_to_bbox(word["quad"]), width, height))
            tags.append(f"{'B' if position == 0 else 'I'}-{category}")

    return {"words": words, "bboxes": bboxes, "ner_tags": tags}


def build_label_list(dataset) -> list[str]:
    """Collect every BIO tag present, with O first so it takes index 0."""
    labels: set[str] = set()
    for split in dataset:
        for record in dataset[split]:
            gt = json.loads(record["ground_truth"])
            for line in gt.get("valid_line", []):
                category = line.get("category", "other")
                labels.add(f"B-{category}")
                labels.add(f"I-{category}")
    return ["O"] + sorted(labels)


def load_cord(split: str | None = None):
    """Load CORD-v2 from the Hub. Requires `datasets` (installed on Colab)."""
    from datasets import load_dataset

    return load_dataset(DATASET_ID, split=split)


def to_token_classification(dataset, label2id: dict[str, int]):
    """Map the raw dataset into the columns LayoutLMv3 training expects."""

    def convert(record):
        image = record["image"]
        parsed = parse_example(record["ground_truth"], image.width, image.height)
        return {
            "image": image.convert("RGB"),
            "words": parsed["words"],
            "bboxes": parsed["bboxes"],
            "labels": [label2id.get(tag, 0) for tag in parsed["ner_tags"]],
        }

    return dataset.map(convert, remove_columns=dataset.column_names)
