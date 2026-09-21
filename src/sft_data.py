"""Prompt construction and SFT dataset building for CORD-v2.

Qwen2.5-7B-Instruct has no vision tower, so it receives the word sequence and nothing
else. LayoutLMv3 received the same words plus bounding boxes plus the page image. That
asymmetry is not a flaw in the experiment, it is the thing being measured, and it belongs
in the results table rather than in a footnote. Serialising boxes into the prompt was
considered and rejected: it would make the 7B look better without making the comparison
more informative about when to reach for an LLM.

One prompt template serves base few-shot and fine-tuned evaluation alike. Two templates
would mean the base-vs-tuned gap partly measures prompt engineering.
"""

from __future__ import annotations

import json
import random

from .convert import FieldSpan, build_completion, gold_fields
from .data import DATASET_ID, parse_example

# The 29 categories CORD-v2 actually uses, enumerated from the dataset and then frozen
# here so the prompt is identical offline, on Colab and in CI. `verify_categories` checks
# it back against the data: a category present in the data but missing here would be a
# field the model is never told to emit, and it would lose those entities silently.
#
# 29 categories is 59 BIO labels, which is what Project 1's label list came to - the two
# projects are working from the same vocabulary.
#
# No category occurs in test that does not also occur in train, so nothing here is derived
# from the test split.
CATEGORIES = [
    "menu.nm", "menu.num", "menu.unitprice", "menu.cnt", "menu.discountprice",
    "menu.price", "menu.itemsubtotal", "menu.vatyn", "menu.etc",
    "menu.sub.nm", "menu.sub.unitprice", "menu.sub.cnt", "menu.sub.price",
    "void_menu.nm", "void_menu.price",
    "sub_total.subtotal_price", "sub_total.discount_price", "sub_total.service_price",
    "sub_total.othersvc_price", "sub_total.tax_price", "sub_total.etc",
    "total.total_price", "total.total_etc", "total.cashprice", "total.changeprice",
    "total.creditcardprice", "total.emoneyprice", "total.menutype_cnt", "total.menuqty_cnt",
]

SYSTEM_PROMPT = (
    "You extract labelled fields from Indonesian receipt text.\n\n"
    "Reply with JSON only, in exactly this shape:\n"
    '{"fields": [{"type": "<field type>", "text": "<exact words from the receipt>"}]}\n\n'
    "Rules:\n"
    "- `text` must be copied verbatim from the receipt, including the printed label word. "
    'A total printed as "TOTAL 60.000" has text "TOTAL 60.000", not "60.000".\n'
    "- List fields in the order they appear on the receipt.\n"
    "- Repeat a type as often as it occurs; a receipt has many menu.nm fields.\n"
    "- Omit fields that are not present. Do not invent values.\n\n"
    "Valid field types:\n" + "\n".join(CATEGORIES)
)


def build_prompt(words: list[str]) -> str:
    return "Receipt text:\n" + " ".join(words)


def words_of(record) -> list[str]:
    """The same word sequence the model is scored against, from `data.parse_example`."""
    image = record["image"]
    return parse_example(record["ground_truth"], image.width, image.height)["words"]


def build_messages(words: list[str], fields: list[FieldSpan] | None = None) -> dict:
    """Conversational prompt/completion pair, the format TRL masks the prompt in."""
    example = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(words)},
        ]
    }
    if fields is not None:
        example["completion"] = [
            {"role": "assistant", "content": build_completion(fields)}
        ]
    return example


def build_fewshot(dataset, split: str, k: int = 2, seed: int = 0) -> list[dict]:
    """Fixed few-shot exemplars for the base model, drawn from the training split only.

    Fixed, not per-document: sampling a fresh exemplar per test receipt would make the
    base-model number depend on draw luck. The split assertion is the whole point of the
    function - a few-shot example from validation or test is leakage that no later check
    would catch.
    """
    if split != "train":
        raise ValueError(f"few-shot exemplars must come from train, got {split!r}")

    indices = random.Random(seed).sample(range(len(dataset)), k)
    messages: list[dict] = []
    for index in indices:
        record = dataset[index]
        words = words_of(record)
        messages.append({"role": "user", "content": build_prompt(words)})
        messages.append(
            {
                "role": "assistant",
                "content": build_completion(gold_fields(record["ground_truth"])),
            }
        )
    return messages


def build_sft_dataset(dataset):
    """Map a CORD split into TRL prompt/completion pairs."""

    def convert(record):
        return build_messages(
            words_of(record), gold_fields(record["ground_truth"])
        )

    return dataset.map(convert, remove_columns=dataset.column_names)


def verify_categories(dataset) -> dict:
    """Confirm CATEGORIES covers every category the data actually uses."""
    seen: set[str] = set()
    for record in dataset:
        parsed = json.loads(record["ground_truth"])
        for line in parsed.get("valid_line", []):
            seen.add(line.get("category", "other"))

    missing = sorted(seen - set(CATEGORIES))
    if missing:
        raise AssertionError(
            f"categories present in the data but absent from the prompt: {missing}"
        )
    return {
        "dataset": DATASET_ID,
        "categories_in_prompt": len(CATEGORIES),
        "categories_in_data": len(seen),
        "unused_in_data": sorted(set(CATEGORIES) - seen),
    }
