"""Keyword and position based extraction, with no model at all.

This exists to make the transformer justify itself. "I fine-tuned LayoutLMv3" is not
an achievement unless something cheaper was tried and lost, and on structured documents
regex baselines are often embarrassingly competitive. Whatever the gap turns out to be
gets reported honestly, including if it is small.

One detail drives the whole design. In CORD an annotated entity spans the printed label
*and* its value together: `total.total_price` is `['TOTAL', '60.000']`, not `['60.000']`.
Tagging only the amount makes every span mismatch and scores exactly zero, which looks
like a terrible baseline rather than the labelling bug it is.

Receipts are Indonesian, so the trigger list carries both languages: cash is also
'tunai', change is also 'kembali'.
"""

from __future__ import annotations

import re

AMOUNT = re.compile(r"^-?[\d][\d.,]*$")

# Longest phrase wins. 'TOTAL DISC' must beat 'TOTAL', or every discount line is
# mislabelled as the grand total.
PHRASES: list[tuple[str, str]] = [
    ("total disc", "sub_total.discount_price"),
    ("sub total", "sub_total.subtotal_price"),
    ("subtotal", "sub_total.subtotal_price"),
    ("service charge", "sub_total.service_price"),
    ("service", "sub_total.service_price"),
    ("discount", "sub_total.discount_price"),
    ("disc", "sub_total.discount_price"),
    ("tax", "sub_total.tax_price"),
    ("ppn", "sub_total.tax_price"),
    ("pb1", "sub_total.tax_price"),
    ("cash", "total.cashprice"),
    ("tunai", "total.cashprice"),
    ("change", "total.changeprice"),
    ("kembali", "total.changeprice"),
    ("kembalian", "total.changeprice"),
    ("edc", "total.creditcardprice"),
    ("debit", "total.creditcardprice"),
    ("credit", "total.creditcardprice"),
    ("qty", "total.menuqty_cnt"),
    ("total", "total.total_price"),
]
MAX_PHRASE_WORDS = 2


def is_amount(token: str) -> bool:
    return bool(AMOUNT.match(token)) and any(c.isdigit() for c in token)


def _normalize(token: str) -> str:
    return token.lower().strip(":.-$()")


def _match_phrase(words: list[str], index: int) -> tuple[str, int] | None:
    """Longest matching trigger phrase starting at index -> (category, word count)."""
    for length in range(MAX_PHRASE_WORDS, 0, -1):
        if index + length > len(words):
            continue
        phrase = " ".join(_normalize(w) for w in words[index : index + length])
        for trigger, category in PHRASES:
            # Exact equality only. Allowing a prefix match here lets a two-word window
            # satisfy a one-word trigger, which makes the entity swallow the next line.
            if phrase == trigger:
                return category, length
    return None


def predict_tags(words: list[str], lookahead: int = 4) -> list[str]:
    """Predict a BIO tag per word from keywords and adjacency alone.

    Output shape matches the model's exactly, so both are scored by the same metric
    and the comparison is like for like.
    """
    tags = ["O"] * len(words)
    index = 0

    while index < len(words):
        if tags[index] != "O":
            index += 1
            continue
        match = _match_phrase(words, index)
        if match is None:
            index += 1
            continue

        category, phrase_length = match
        # Extend from the label through the first amount that follows it, because the
        # annotated entity covers both.
        end = None
        for offset in range(phrase_length, phrase_length + lookahead):
            position = index + offset
            if position >= len(words):
                break
            if is_amount(words[position]):
                end = position
                break

        if end is None:
            index += 1
            continue

        tags[index] = f"B-{category}"
        for position in range(index + 1, end + 1):
            tags[position] = f"I-{category}"
        index = end + 1

    # Remaining bare amounts preceded by words are line-item prices; the words before
    # them are the item name.
    for position, word in enumerate(words):
        if tags[position] != "O" or not is_amount(word):
            continue
        start = position - 1
        while start >= 0 and tags[start] == "O" and not is_amount(words[start]):
            start -= 1
        start += 1
        if start < position:
            tags[position] = "B-menu.price"
            tags[start] = "B-menu.nm"
            for inner in range(start + 1, position):
                tags[inner] = "I-menu.nm"

    return tags
