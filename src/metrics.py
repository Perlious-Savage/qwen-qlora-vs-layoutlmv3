"""Entity-level scoring for BIO sequence labelling.

Written out rather than pulled from seqeval, whose setup.py fails to build on current
Python. Forty lines is a better trade than a dependency that breaks the install, and
it means the metric is inspectable: an interviewer asking "what exactly does your F1
count?" gets an answer from this file.

Scoring is entity-level, not token-level, which is the stricter and more meaningful
choice. A predicted entity counts as correct only if its type and its full span both
match the reference exactly. Getting three tokens of a four-token merchant name right
scores zero, which is the right call — a partially extracted field is not a usable one.
"""

from __future__ import annotations

from collections import defaultdict


def extract_entities(tags: list[str]) -> set[tuple[str, int, int]]:
    """Pull (type, start, end) spans out of a BIO tag sequence.

    Tolerant of malformed sequences: an I- tag with no preceding B- of the same type
    opens a new entity rather than being dropped. Models do emit these, and silently
    discarding them would flatter the score.
    """
    entities: set[tuple[str, int, int]] = set()
    current_type: str | None = None
    start = 0

    for index, tag in enumerate(tags + ["O"]):
        prefix = tag[:1]
        entity_type = tag[2:] if len(tag) > 2 and tag[1] == "-" else None

        if prefix == "B" or (prefix == "I" and entity_type != current_type):
            if current_type is not None:
                entities.add((current_type, start, index))
            current_type, start = entity_type, index
        elif prefix == "O":
            if current_type is not None:
                entities.add((current_type, start, index))
            current_type = None

    return entities


def _counts(references: list[list[str]], predictions: list[list[str]]):
    true_positive = 0
    predicted = 0
    actual = 0
    per_type: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])

    for reference, prediction in zip(references, predictions):
        reference_entities = extract_entities(reference)
        predicted_entities = extract_entities(prediction)
        overlap = reference_entities & predicted_entities

        true_positive += len(overlap)
        predicted += len(predicted_entities)
        actual += len(reference_entities)

        for entity in overlap:
            per_type[entity[0]][0] += 1
        for entity in predicted_entities:
            per_type[entity[0]][1] += 1
        for entity in reference_entities:
            per_type[entity[0]][2] += 1

    return true_positive, predicted, actual, per_type


def _prf(tp: int, pred: int, act: int) -> tuple[float, float, float]:
    precision = tp / pred if pred else 0.0
    recall = tp / act if act else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def precision_recall_f1(
    references: list[list[str]], predictions: list[list[str]]
) -> tuple[float, float, float]:
    tp, pred, act, _ = _counts(references, predictions)
    return _prf(tp, pred, act)


def classification_report(
    references: list[list[str]], predictions: list[list[str]]
) -> str:
    tp, pred, act, per_type = _counts(references, predictions)
    lines = [f"{'field':<28}{'prec':>8}{'rec':>8}{'f1':>8}{'support':>9}", "-" * 61]
    for entity_type in sorted(per_type):
        t, p, a = per_type[entity_type]
        precision, recall, f1 = _prf(t, p, a)
        lines.append(f"{entity_type:<28}{precision:>8.3f}{recall:>8.3f}{f1:>8.3f}{a:>9}")
    precision, recall, f1 = _prf(tp, pred, act)
    lines.append("-" * 61)
    lines.append(f"{'micro avg':<28}{precision:>8.3f}{recall:>8.3f}{f1:>8.3f}{act:>9}")
    return "\n".join(lines)
