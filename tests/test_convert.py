"""Tests for the JSON-to-spans converter.

Written before the fine-tune, on purpose. This module decides what every model in the
comparison scores, and its failure mode is silent: a converter bug reads as a worse model.
The cases below are the ones where a plausible implementation gets it wrong in a way that
still produces a number.
"""

import json
from decimal import Decimal

from src.convert import (
    FieldSpan,
    build_completion,
    fields_to_tags,
    gold_fields,
    parse_prediction,
    prediction_to_tags,
    to_receipt,
)
from src.data import parse_example
from src.metrics import extract_entities, precision_recall_f1
from src.tools import check_total_reconciles


def make_ground_truth(lines: list[tuple[str, list[str]]]) -> str:
    """Build a CORD-shaped record. Quads are dummies; only word order matters here."""
    quad = {f"{axis}{corner}": 0 for axis in "xy" for corner in range(1, 5)}
    return json.dumps(
        {
            "valid_line": [
                {"category": category, "words": [{"text": w, "quad": quad} for w in words]}
                for category, words in lines
            ]
        }
    )


RECEIPT = [
    ("menu.nm", ["ES", "TEH"]),
    ("menu.price", ["8.000"]),
    # 'TEH' on its own, after 'TEH' has already appeared inside the first entity. A
    # converter that searches from the left without honouring claimed positions puts this
    # at index 1 and silently mislabels two entities at once.
    ("menu.nm", ["TEH"]),
    ("menu.price", ["5.000"]),
    # Identical text and identical type, twice. Each occurrence must get its own span.
    ("menu.nm", ["KOPI"]),
    ("menu.price", ["9.000"]),
    ("menu.nm", ["KOPI"]),
    ("menu.price", ["9.000"]),
    ("sub_total.subtotal_price", ["SUBTOTAL", "31.000"]),
    ("total.total_price", ["TOTAL", "31.000"]),
]


def test_gold_json_round_trips_to_the_exact_same_spans():
    """The ceiling. If this is not exact, no model number from this harness is meaningful."""
    ground_truth = make_ground_truth(RECEIPT)
    parsed = parse_example(ground_truth, width=1000, height=1000)

    tags, stats = fields_to_tags(parsed["words"], gold_fields(ground_truth))

    assert stats.unlocated == 0
    assert tags == parsed["ner_tags"]
    assert extract_entities(tags) == extract_entities(parsed["ner_tags"])
    assert precision_recall_f1([parsed["ner_tags"]], [tags]) == (1.0, 1.0, 1.0)
    # Every word in CORD belongs to some entity, so a stray 'O' means a dropped field.
    assert "O" not in tags


def test_gold_fields_matches_the_word_list_data_py_builds():
    """`gold_fields` and `parse_example` must filter words identically or the round trip rots."""
    ground_truth = make_ground_truth(
        [("menu.nm", ["  ES  ", "", "TEH"]), ("menu.price", ["8.000"])]
    )
    parsed = parse_example(ground_truth, width=1000, height=1000)

    assert [f.text for f in gold_fields(ground_truth)] == ["ES TEH", "8.000"]
    assert parsed["words"] == ["ES", "TEH", "8.000"]


def test_a_line_with_no_usable_words_is_dropped_by_both_sides():
    ground_truth = make_ground_truth([("menu.etc", ["  ", ""]), ("menu.nm", ["KOPI"])])
    parsed = parse_example(ground_truth, width=1000, height=1000)

    tags, _ = fields_to_tags(parsed["words"], gold_fields(ground_truth))
    assert tags == parsed["ner_tags"] == ["B-menu.nm"]


# --- unparseable output scores zero, never crashes, is never skipped ------------------

UNPARSEABLE = [
    None,
    "",
    "   ",
    "I could not read this receipt.",
    "{",
    '{"fields": [{"type": "menu.nm", "text": "ES',  # truncated mid-generation
    '{"fields": [{"type": "menu.nm"}]}',  # missing required key
    '{"fields": "not a list"}',
    '{"items": []}',  # right idea, wrong contract
    "[]",
]


def test_unparseable_output_returns_none_without_raising():
    for raw in UNPARSEABLE:
        assert parse_prediction(raw) is None, raw


def test_unparseable_output_scores_zero_rather_than_being_skipped():
    words = ["TOTAL", "31.000"]
    reference = ["B-total.total_price", "I-total.total_price"]

    for raw in UNPARSEABLE:
        tags, stats = prediction_to_tags(words, raw)
        assert stats.parsed is False
        assert stats.predicted == 0
        assert extract_entities(tags) == set()
        # Recall takes the hit; precision is untouched because nothing was predicted.
        assert precision_recall_f1([reference], [tags]) == (0.0, 0.0, 0.0)


def test_fences_and_surrounding_prose_are_tolerated():
    payload = '{"fields": [{"type": "menu.nm", "text": "KOPI"}]}'
    for raw in (
        payload,
        f"```json\n{payload}\n```",
        f"```\n{payload}\n```",
        f"Here is the extraction:\n{payload}\nLet me know if you need more.",
    ):
        extraction = parse_prediction(raw)
        assert extraction is not None and extraction.fields[0].text == "KOPI"


def test_empty_field_list_parses_but_predicts_nothing():
    """Distinct from unparseable: the model followed the contract and found nothing."""
    tags, stats = prediction_to_tags(["TOTAL", "31.000"], '{"fields": []}')
    assert stats.parsed is True
    assert stats.predicted == 0
    assert extract_entities(tags) == set()


# --- predictions that do not exist in the source ------------------------------------


def test_unlocatable_prediction_costs_precision_instead_of_vanishing():
    words = ["TOTAL", "31.000"]
    reference = ["B-total.total_price", "I-total.total_price"]
    fields = [
        FieldSpan(type="total.total_price", text="TOTAL 31.000"),
        FieldSpan(type="menu.nm", text="NASI GORENG"),  # never printed on this receipt
    ]

    tags, stats = fields_to_tags(words, fields)

    assert (stats.located, stats.unlocated) == (1, 1)
    precision, recall, _ = precision_recall_f1([reference], [tags])
    assert recall == 1.0
    assert precision == 0.5  # two predictions, one right - the invention was counted


def test_invented_field_type_is_a_false_positive_not_a_rejected_document():
    words = ["TOTAL", "31.000"]
    reference = ["B-total.total_price", "I-total.total_price"]
    fields = [
        FieldSpan(type="total.total_price", text="TOTAL 31.000"),
        FieldSpan(type="receipt.vibe", text="TOTAL"),
    ]

    tags, stats = fields_to_tags(words, fields)

    assert stats.predicted == 2
    precision, recall, _ = precision_recall_f1([reference], [tags])
    assert recall == 1.0  # the good field still scored
    assert precision == 0.5


def test_empty_text_is_one_wrong_answer_not_a_free_pass():
    words = ["TOTAL", "31.000"]
    _, stats = fields_to_tags(words, [FieldSpan(type="menu.nm", text="")])
    assert (stats.located, stats.unlocated) == (0, 1)


def test_phantom_spans_stay_distinct_from_each_other():
    words = ["TOTAL"]
    fields = [FieldSpan(type="menu.nm", text=f"GHOST {i}") for i in range(3)]
    tags, stats = fields_to_tags(words, fields)

    assert stats.unlocated == 3
    assert len(extract_entities(tags)) == 3  # not collapsed into one


# --- claiming and casefolding -------------------------------------------------------


def test_duplicate_predictions_do_not_both_claim_one_span():
    words = ["TOTAL", "31.000"]
    reference = ["B-total.total_price", "I-total.total_price"]
    duplicate = FieldSpan(type="total.total_price", text="TOTAL 31.000")

    tags, stats = fields_to_tags(words, [duplicate, duplicate])

    assert (stats.located, stats.unlocated) == (1, 1)
    precision, recall, _ = precision_recall_f1([reference], [tags])
    assert (precision, recall) == (0.5, 1.0)


def test_casefolded_text_still_locates_the_right_span():
    """Position defines the span, so case drift must not be punished as a wrong answer."""
    words = ["TOTAL", "31.000"]
    reference = ["B-total.total_price", "I-total.total_price"]

    tags, stats = fields_to_tags(
        words, [FieldSpan(type="total.total_price", text="total 31.000")]
    )

    assert stats.located == 1
    assert precision_recall_f1([reference], [tags]) == (1.0, 1.0, 1.0)


def test_exact_matches_are_assigned_before_casefolded_ones():
    """Per-field exact-then-fold would let the first field steal the second field's span."""
    words = ["KOPI", "kopi"]
    fields = [FieldSpan(type="menu.nm", text="kopi"), FieldSpan(type="menu.etc", text="KOPI")]

    tags, stats = fields_to_tags(words, fields)

    assert stats.unlocated == 0
    assert tags == ["B-menu.etc", "B-menu.nm"]


# --- the business view --------------------------------------------------------------


def test_build_completion_round_trips_through_parse_prediction():
    fields = gold_fields(make_ground_truth(RECEIPT))
    reparsed = parse_prediction(build_completion(fields))

    assert reparsed is not None
    assert [(f.type, f.text) for f in reparsed.fields] == [(f.type, f.text) for f in fields]


def test_to_receipt_takes_the_value_not_the_label_word():
    receipt = to_receipt("doc-1", gold_fields(make_ground_truth(RECEIPT)))

    assert receipt.subtotal == Decimal("31000")
    assert receipt.total == Decimal("31000")
    assert [item.name for item in receipt.line_items] == ["ES TEH", "TEH", "KOPI", "KOPI"]
    assert [item.total_price for item in receipt.line_items] == [
        Decimal("8000"),
        Decimal("5000"),
        Decimal("9000"),
        Decimal("9000"),
    ]


def test_extracted_amounts_flow_through_the_deterministic_checks():
    """The model states values; Python decides whether they reconcile. Same as Project 1."""
    receipt = to_receipt("doc-1", gold_fields(make_ground_truth(RECEIPT)))
    finding = check_total_reconciles(
        receipt.subtotal, receipt.tax, receipt.service_charge, receipt.discount, receipt.total
    )
    assert finding.passed

    receipt.total = Decimal("99000")
    assert not check_total_reconciles(
        receipt.subtotal, receipt.tax, receipt.service_charge, receipt.discount, receipt.total
    ).passed
