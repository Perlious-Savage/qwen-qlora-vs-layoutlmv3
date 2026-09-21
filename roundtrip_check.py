"""Measure the evaluation harness's ceiling before any model is trained.

Converts the *gold* JSON back into entity spans and scores it against the gold BIO tags
with the same `metrics.py` that produced Project 1's 0.948. The answer must be exactly
1.000. Anything less is a bug in the converter, and every model number this harness later
produces would carry that loss silently - looking like a weaker model rather than a
broken ruler.

It also measures the ceiling of the alternative target format, CORD's nested `gt_parse`,
which is the reason that format was not used. That decision should rest on a number, not
on an argument.

    python roundtrip_check.py

Needs the CORD-v2 test split (~235MB on first run).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.convert import FieldSpan, fields_to_tags, gold_fields
from src.data import DATASET_ID, parse_example
from src.metrics import precision_recall_f1
from src.sft_data import verify_categories

ARTIFACTS = Path(__file__).parent / "artifacts"


def gt_parse_fields(ground_truth: str) -> list[FieldSpan]:
    """Flatten CORD's nested `gt_parse` view into fields, for the comparison ceiling.

    `gt_parse` stores values without the printed label word, so 'TOTAL 60.000' comes back
    as '60.000'. The gold span covers both words, so this representation cannot recover it.
    """
    parsed = json.loads(ground_truth).get("gt_parse", {})
    fields: list[FieldSpan] = []

    for section, content in parsed.items():
        groups = content if isinstance(content, list) else [content]
        for group in groups:
            if not isinstance(group, dict):
                continue
            for key, value in group.items():
                values = value if isinstance(value, list) else [value]
                fields.extend(
                    FieldSpan(type=f"{section}.{key}", text=str(v)) for v in values
                )

    return fields


def main() -> None:
    from datasets import load_dataset

    print(f"Loading {DATASET_ID} test split ...")
    dataset = load_dataset(DATASET_ID, split="test")

    # Checked before the scoring loop so a vocabulary mismatch surfaces on its own rather
    # than hiding behind an F1 that was computed and then thrown away.
    categories = verify_categories(dataset)

    references: list[list[str]] = []
    flat_predictions: list[list[str]] = []
    nested_predictions: list[list[str]] = []
    flat_unlocated = nested_unlocated = 0
    imperfect: list[int] = []

    for index, record in enumerate(dataset):
        image = record["image"]
        parsed = parse_example(record["ground_truth"], image.width, image.height)
        words, reference = parsed["words"], parsed["ner_tags"]
        references.append(reference)

        flat, flat_stats = fields_to_tags(words, gold_fields(record["ground_truth"]))
        flat_predictions.append(flat)
        flat_unlocated += flat_stats.unlocated
        if precision_recall_f1([reference], [flat])[2] != 1.0:
            imperfect.append(index)

        nested, nested_stats = fields_to_tags(words, gt_parse_fields(record["ground_truth"]))
        nested_predictions.append(nested)
        nested_unlocated += nested_stats.unlocated

    flat_scores = precision_recall_f1(references, flat_predictions)
    nested_scores = precision_recall_f1(references, nested_predictions)

    report = {
        "dataset": DATASET_ID,
        "split": "test",
        "n_documents": len(dataset),
        "flat_entity_list": {
            "precision": flat_scores[0],
            "recall": flat_scores[1],
            "f1": flat_scores[2],
            "unlocated_fields": flat_unlocated,
            "note": "The format used. Exact by construction; this run is the proof.",
        },
        "nested_gt_parse": {
            "precision": nested_scores[0],
            "recall": nested_scores[1],
            "f1": nested_scores[2],
            "unlocated_fields": nested_unlocated,
            "note": (
                "Not used. gt_parse stores values without the printed label word, so it "
                "cannot recover spans that cover both. This is the ceiling a model "
                "trained on that format could never exceed, whatever its quality."
            ),
        },
        "categories": categories,
    }

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "converter_ceiling.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, indent=2))

    if flat_scores[2] != 1.0:
        raise SystemExit(
            f"Converter ceiling is {flat_scores[2]:.4f}, not 1.000. "
            f"{len(imperfect)} documents affected, first at index {imperfect[0]}. "
            "Fix this before training anything: every model score from this harness "
            "would inherit the loss."
        )

    print("\nCeiling is exactly 1.000. The harness adds no error of its own.")


if __name__ == "__main__":
    main()
