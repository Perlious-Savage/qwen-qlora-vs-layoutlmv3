"""Deterministic financial computation.

Every monetary quantity in this pipeline is computed here, in Python, using Decimal.
No model is ever asked to do arithmetic. Language models produce plausible-looking
numbers rather than correct ones, and the failure is silent: a wrong total looks
exactly like a right one. Once the computation lives in code, that error class is
gone by construction rather than by prompting.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .schemas import Finding, LineItem, Severity

CENTS = Decimal("0.01")

# Receipts routinely round to the nearest unit, and OCR drops the occasional digit.
# One cent of slack avoids flagging every receipt while still catching real mismatches.
DEFAULT_TOLERANCE = Decimal("0.01")

_CURRENCY = re.compile(r"[^\d.,\-()]")


def quantize(value: Decimal) -> Decimal:
    """Round to two decimal places, the unit money is actually reported in."""
    return value.quantize(CENTS)


def parse_money(text: str | Decimal | int | float | None) -> Decimal | None:
    """Parse a monetary string into a Decimal, or None if it isn't a number.

    Separator handling is genuinely ambiguous and worth being explicit about.
    CORD is a corpus of Indonesian receipts, where '12.000' means twelve thousand,
    not twelve. The heuristic: a separator followed by exactly three digits is a
    thousands separator; one followed by one or two digits is a decimal point.
    '1.234' is therefore read as 1234, which is correct for this corpus and wrong
    for a corpus quoting three decimal places. Documented rather than hidden.

    Floats are accepted but routed through str() so the binary representation never
    reaches the Decimal.
    """
    if text is None:
        return None
    if isinstance(text, Decimal):
        return text
    if isinstance(text, (int, float)):
        return Decimal(str(text))

    cleaned = _CURRENCY.sub("", str(text)).strip()
    if not cleaned:
        return None

    negative = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("()")
    if cleaned.startswith("-"):
        negative = True
        cleaned = cleaned[1:]
    if not cleaned:
        return None

    if "," in cleaned and "." in cleaned:
        # Whichever separator comes last is the decimal point.
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned or "." in cleaned:
        sep = "," if "," in cleaned else "."
        head, _, tail = cleaned.rpartition(sep)
        if len(tail) == 3 and head:
            cleaned = head.replace(sep, "") + tail  # thousands separator
        else:
            cleaned = cleaned.replace(sep, ".")

    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    return -value if negative else value


def line_item_total(item: LineItem) -> Decimal | None:
    """quantity x unit_price, when both are present."""
    if item.unit_price is None:
        return None
    return quantize(item.quantity * item.unit_price)


def sum_line_items(items: list[LineItem]) -> Decimal:
    """Sum the stated total of each line item.

    Long sums are exactly where language models fail most often, which is the whole
    reason this is a function rather than a prompt.
    """
    total = Decimal("0")
    for item in items:
        value = item.total_price if item.total_price is not None else line_item_total(item)
        if value is not None:
            total += value
    return quantize(total)


def expected_total(
    subtotal: Decimal | None,
    tax: Decimal | None = None,
    service_charge: Decimal | None = None,
    discount: Decimal | None = None,
) -> Decimal | None:
    """subtotal + tax + service - discount."""
    if subtotal is None:
        return None
    total = subtotal
    for addition in (tax, service_charge):
        if addition is not None:
            total += addition
    if discount is not None:
        total -= abs(discount)
    return quantize(total)


def within_tolerance(
    a: Decimal | None, b: Decimal | None, tolerance: Decimal = DEFAULT_TOLERANCE
) -> bool:
    if a is None or b is None:
        return False
    return abs(quantize(a) - quantize(b)) <= tolerance


# --- checks, each returning a structured Finding ---------------------------------


def check_line_items_sum_to_subtotal(
    items: list[LineItem],
    subtotal: Decimal | None,
    tolerance: Decimal = DEFAULT_TOLERANCE,
) -> Finding:
    if not items or subtotal is None:
        return Finding(
            check_id="line_items_sum_to_subtotal",
            passed=True,
            severity=Severity.INFO,
            message="Not enough data to check; skipped rather than guessed.",
        )
    computed = sum_line_items(items)
    ok = within_tolerance(computed, subtotal, tolerance)
    return Finding(
        check_id="line_items_sum_to_subtotal",
        passed=ok,
        severity=Severity.INFO if ok else Severity.ERROR,
        message=(
            "Line items sum to the stated subtotal."
            if ok
            else f"Line items sum to {computed}, but the receipt states {quantize(subtotal)}."
        ),
        computed={"sum_of_line_items": str(computed), "stated_subtotal": str(quantize(subtotal))},
    )


def check_total_reconciles(
    subtotal: Decimal | None,
    tax: Decimal | None,
    service_charge: Decimal | None,
    discount: Decimal | None,
    total: Decimal | None,
    tolerance: Decimal = DEFAULT_TOLERANCE,
) -> Finding:
    expected = expected_total(subtotal, tax, service_charge, discount)
    if expected is None or total is None:
        return Finding(
            check_id="total_reconciles",
            passed=True,
            severity=Severity.INFO,
            message="Not enough data to check; skipped rather than guessed.",
        )
    ok = within_tolerance(expected, total, tolerance)
    return Finding(
        check_id="total_reconciles",
        passed=ok,
        severity=Severity.INFO if ok else Severity.ERROR,
        message=(
            "Stated total reconciles with subtotal, tax and charges."
            if ok
            else f"Components imply a total of {expected}, but the receipt states {quantize(total)}."
        ),
        computed={"expected_total": str(expected), "stated_total": str(quantize(total))},
    )


def check_line_item_arithmetic(items: list[LineItem]) -> Finding:
    """Flag any line where quantity x unit_price disagrees with the stated line total."""
    mismatches: list[str] = []
    for index, item in enumerate(items):
        computed = line_item_total(item)
        if computed is None or item.total_price is None:
            continue
        if not within_tolerance(computed, item.total_price):
            mismatches.append(
                f"line {index}: {item.quantity} x {item.unit_price} = {computed}, "
                f"stated {quantize(item.total_price)}"
            )
    ok = not mismatches
    return Finding(
        check_id="line_item_arithmetic",
        passed=ok,
        severity=Severity.INFO if ok else Severity.WARNING,
        message="Line item arithmetic is consistent." if ok else "; ".join(mismatches),
        computed={"mismatch_count": str(len(mismatches))},
    )
