"""Conversion between the LLM's JSON output and the entity spans `metrics.py` scores.

This is the load-bearing module of the whole project. LayoutLMv3 emitted one label per
word, so scoring it was direct. Qwen emits JSON, and every point of the comparison
depends on turning that JSON back into the *same* spans without flattering or penalising
it by accident. Silent bugs here look like model quality.

The field contract, chosen so the round trip is exact rather than approximate:

    {"fields": [{"type": "total.total_price", "text": "TOTAL 60.000"}, ...]}

`text` is the entity's word sequence exactly as it appears in the source, *including the
printed label word*. CORD annotates `total.total_price` as ['TOTAL', '60.000'], not
['60.000']. CORD's other view, the nested `gt_parse`, stores only the value, so recovering
the gold span from it needs fuzzy alignment - and that guesswork would sit inside the
headline number. `roundtrip_check.py` measures both ceilings so this choice is backed by a
measurement rather than by this paragraph.

Why the round trip is exact, not merely usually right. `data.parse_example` builds the word
list by walking `valid_line` in order, so word order is entity order, and every word belongs
to exactly one entity. Placing fields left to right and refusing to reuse a claimed position
means that when field N is placed, every position before its true start is already claimed
by fields 1..N-1. Its leftmost unclaimed match is therefore its true start, by induction
from the first field at position 0. `roundtrip_check.py` asserts F1 == 1.000 over the real
test split rather than trusting the argument.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, ValidationError

from .schemas import ExtractedReceipt, LineItem
from .tools import parse_money


class FieldSpan(BaseModel):
    """One extracted entity.

    `type` is deliberately *not* validated against the CORD label vocabulary. A model that
    invents a field name has made a false positive, and a false positive should cost
    precision. Rejecting the document instead would convert one bad field into a zero for
    every field on the receipt, which overstates the failure.
    """

    type: str
    text: str


class Extraction(BaseModel):
    fields: list[FieldSpan]


@dataclass
class ConversionStats:
    """Why a document scored what it scored.

    F1 alone cannot distinguish "could not follow the output contract" from "could not read
    the receipt", and for a base model prompted zero-shot those are the two candidate
    explanations for the same low number. Aggregated into parse_rate and locate_rate, these
    counters separate them.
    """

    parsed: bool
    located: int = 0
    unlocated: int = 0

    @property
    def predicted(self) -> int:
        return self.located + self.unlocated


_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")
_NUMBER = re.compile(r"\d[\d.,]*")


def parse_prediction(raw: str | None) -> Extraction | None:
    """Parse raw model output, or return None if it cannot be parsed.

    None means the document scores zero: the caller emits no predicted entities for it.
    It must never raise and must never be skipped - a skipped document silently shrinks
    the test set to whatever the model happened to handle.

    Prose around the JSON is tolerated (first brace to last brace) because the question
    being measured is whether the model found the fields, not whether it suppressed a
    preamble. Malformed JSON is not tolerated, and a truncated completion has no closing
    brace, so it lands here rather than being half-salvaged into a better-looking score.
    """
    if not raw:
        return None

    text = _FENCE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None

    try:
        payload = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None

    try:
        return Extraction.model_validate(payload)
    except ValidationError:
        return None


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _locate(words: list[str], tags: list[str], target: str) -> tuple[int, int] | None:
    """Leftmost unclaimed run of words whose joined text equals `target`.

    Matching is on the joined string rather than on a list of whitespace-split tokens,
    because a CORD word can itself contain a space - '( L' is one annotated word in the
    test split. Splitting the field text on whitespace invents a boundary that is not
    there, and the entity then matches nothing. That cost 0.026 F1 of pure harness error
    until `roundtrip_check.py` caught it.
    """
    if not target:
        return None

    for start in range(len(words)):
        if tags[start] != "O":
            continue
        accumulated = ""
        for end in range(start, len(words)):
            if tags[end] != "O":
                break
            accumulated = words[end] if not accumulated else f"{accumulated} {words[end]}"
            if len(accumulated) > len(target):
                break
            if accumulated == target:
                return start, end - start + 1
    return None


def fields_to_tags(
    words: list[str], fields: list[FieldSpan]
) -> tuple[list[str], ConversionStats]:
    """Convert predicted fields into a BIO tag sequence over `words`.

    Matching runs in two passes over all fields - exact first, then casefolded - rather than
    both passes per field. Per-field order would let an early field take a casefolded match
    at a position a later field needed exactly. Case affects only *locating* a span; the
    score depends on position, so a casefolded match is a correct placement, not a lenient one.

    Predictions that match nowhere become phantom spans appended past the end of `words`.
    They can never equal a gold span, so they cost precision exactly as a wrong answer
    should. Dropping them would quietly raise the score of the model that hallucinates most.
    `metrics.py` compares documents pairwise, not tokens, so the longer sequence is safe and
    `metrics.py` stays byte-identical to Project 1's.
    """
    tags = ["O"] * len(words)
    exact = [_normalize(word) for word in words]
    folded = [word.casefold() for word in exact]
    placed = [False] * len(fields)
    located = 0

    for haystack, transform in ((exact, _normalize), (folded, lambda t: _normalize(t).casefold())):
        for index, field in enumerate(fields):
            if placed[index]:
                continue
            found = _locate(haystack, tags, transform(field.text))
            if found is None:
                continue
            start, width = found
            tags[start] = f"B-{field.type}"
            for position in range(start + 1, start + width):
                tags[position] = f"I-{field.type}"
            placed[index] = True
            located += 1

    phantom: list[str] = []
    for index, field in enumerate(fields):
        if placed[index]:
            continue
        # An empty `text` also lands here: it locates nowhere, so it is one wrong answer.
        width = max(1, len(fields[index].text.split()))
        phantom.append(f"B-{field.type}")
        phantom.extend([f"I-{field.type}"] * (width - 1))

    stats = ConversionStats(
        parsed=True, located=located, unlocated=sum(1 for p in placed if not p)
    )
    return tags + phantom, stats


def prediction_to_tags(
    words: list[str], raw: str | None
) -> tuple[list[str], ConversionStats]:
    """Raw model output to scoreable tags. The one path every config goes through."""
    extraction = parse_prediction(raw)
    if extraction is None:
        return ["O"] * len(words), ConversionStats(parsed=False)
    return fields_to_tags(words, extraction.fields)


def gold_fields(ground_truth: str) -> list[FieldSpan]:
    """Build the target field list from CORD's `valid_line`.

    Word filtering mirrors `data.parse_example` exactly - same strip, same skip-if-empty.
    If the two ever diverge the round trip stops being exact, which is precisely the silent
    bug `roundtrip_check.py` exists to catch.
    """
    parsed = json.loads(ground_truth)
    fields: list[FieldSpan] = []

    for line in parsed.get("valid_line", []):
        texts = [(word.get("text") or "").strip() for word in line.get("words", [])]
        texts = [text for text in texts if text]
        if texts:
            fields.append(
                FieldSpan(type=line.get("category", "other"), text=" ".join(texts))
            )

    return fields


def build_completion(fields: list[FieldSpan]) -> str:
    """The SFT target string. Inverse of `parse_prediction`."""
    return json.dumps(
        {"fields": [{"type": f.type, "text": f.text} for f in fields]},
        ensure_ascii=False,
    )


# --- the business view, so the deterministic checks still apply ----------------------

_AMOUNT_FIELDS = {
    "sub_total.subtotal_price": "subtotal",
    "sub_total.tax_price": "tax",
    "sub_total.service_price": "service_charge",
    "sub_total.discount_price": "discount",
    "total.total_price": "total",
}


def _amount(text: str) -> Decimal | None:
    """Pull the value out of a label+value span.

    The last numeric token, not the whole string: receipts print the label first, and
    `parse_money` on 'ITEM 2 X 10.000' would otherwise concatenate digits into 210000.
    Parsing is still `tools.parse_money`, so the Indonesian thousands-separator rule and
    the Decimal guarantee are the ones Project 1 used.
    """
    numbers = _NUMBER.findall(text)
    return parse_money(numbers[-1]) if numbers else None


def to_receipt(doc_id: str, fields: list[FieldSpan]) -> ExtractedReceipt:
    """Map extracted fields onto the Project 1 data contract.

    The point is that the numbers then flow through `tools.py`: the model states values,
    and Python decides whether they reconcile. No total is ever computed by the model, in
    this project any more than in the last one.

    Line items are grouped by walking the fields in order - a `menu.nm` opens an item and
    subsequent quantity/price fields attach to it. That is how receipts are laid out and
    how CORD orders `valid_line`.
    """
    receipt = ExtractedReceipt(doc_id=doc_id)
    current: LineItem | None = None

    for field in fields:
        if field.type in _AMOUNT_FIELDS:
            setattr(receipt, _AMOUNT_FIELDS[field.type], _amount(field.text))
        elif field.type == "menu.nm":
            current = LineItem(name=field.text)
            receipt.line_items.append(current)
        elif current is None:
            continue
        elif field.type == "menu.cnt":
            count = _amount(field.text)
            if count is not None:
                current.quantity = count
        elif field.type == "menu.unitprice":
            current.unit_price = _amount(field.text)
        elif field.type == "menu.price":
            current.total_price = _amount(field.text)

    return receipt
