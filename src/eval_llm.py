"""Evaluate a generative model on CORD-v2 with Project 1's scoring code.

Every configuration - base few-shot, QLoRA, QLoRA served 4-bit - goes through this one
function. A separate evaluation path per config is how two numbers end up differing for a
reason nobody writes down.

Three rates are reported next to F1, and they are what make a low score interpretable:

    parse_rate       fraction of outputs that were valid JSON in the contract's shape
    locate_rate      fraction of predicted fields found verbatim in the receipt text
    truncation_rate  fraction of generations that hit the token cap

A base model scoring 0.10 with parse_rate 0.20 failed to follow the output format. The
same 0.10 with parse_rate 1.00 and locate_rate 0.30 read the receipt badly. Those are
different findings and F1 alone cannot tell them apart.

    python -m src.eval_llm --config base --fewshot 2
    python -m src.eval_llm --config lora --adapter outputs/qwen-cord-lora
    python -m src.eval_llm --lengths          # measure gold completion lengths only
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from . import manifest
from .convert import build_completion, gold_fields, prediction_to_tags
from .data import DATASET_ID
from .metrics import classification_report, precision_recall_f1
from .sft_data import SYSTEM_PROMPT, build_fewshot, build_messages, words_of

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# Set from the measured gold completion length distribution - see
# artifacts/completion_lengths.json and `--lengths`. Comfortably above the longest real
# target, because a completion cut off mid-JSON is unparseable and scores zero, which is
# indistinguishable from a model that cannot read receipts unless truncation is measured
# separately. `evaluate` reports truncation_rate for exactly that reason.
MAX_NEW_TOKENS = 1536

# One decoding configuration for every model in the comparison. Greedy, so evaluation is
# deterministic and the base-vs-tuned gap carries no sampling noise.
DECODE = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}


def dtype_kwargs(dtype) -> dict:
    """`torch_dtype` was renamed to `dtype` in transformers 4.56.

    Colab's pinned version moves around, and discovering the rename after a model has
    finished downloading costs more than the four lines it takes to handle here.
    """
    import transformers

    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    return {"dtype" if (major, minor) >= (4, 56) else "torch_dtype": dtype}


def _prompts(dataset, fewshot: list[dict] | None) -> tuple[list[list[dict]], list[list[str]]]:
    prompts, words_per_document = [], []
    for record in dataset:
        words = words_of(record)
        messages = build_messages(words)["prompt"]
        if fewshot:
            messages = [messages[0], *fewshot, messages[1]]
        prompts.append(messages)
        words_per_document.append(words)
    return prompts, words_per_document


def evaluate(dataset, generate, config: str, fewshot: list[dict] | None = None) -> dict:
    """Generate, convert, score. `generate` maps message lists to (text, truncated) pairs."""
    prompts, words_per_document = _prompts(dataset, fewshot)

    print(f"Generating {len(prompts)} completions ...")
    outputs = generate(prompts)

    references: list[list[str]] = []
    predictions: list[list[str]] = []
    parsed = truncated = located = predicted = 0
    samples = []

    for words, record, (raw, hit_cap) in zip(words_per_document, dataset, outputs):
        references.append(_reference_tags(record))
        tags, stats = prediction_to_tags(words, raw)
        predictions.append(tags)

        parsed += stats.parsed
        truncated += hit_cap
        located += stats.located
        predicted += stats.predicted
        if len(samples) < 20:
            samples.append({"raw": raw, "parsed": stats.parsed, "located": stats.located})

    precision, recall, f1 = precision_recall_f1(references, predictions)
    total = len(prompts)
    metrics = {
        "config": config,
        "dataset": DATASET_ID,
        "split": "test",
        "model": MODEL_ID,
        "n_documents": total,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "parse_rate": parsed / total if total else 0.0,
        "locate_rate": located / predicted if predicted else 0.0,
        "truncation_rate": truncated / total if total else 0.0,
        "n_predicted_fields": predicted,
        "fewshot_examples": len(fewshot) // 2 if fewshot else 0,
        "decode": DECODE,
    }

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"{config}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (ARTIFACTS / f"{config}_per_field_report.txt").write_text(
        classification_report(references, predictions)
    )
    (ARTIFACTS / f"{config}_samples.json").write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest.write(
        config,
        model=MODEL_ID,
        dataset=DATASET_ID,
        split="test",
        decode=DECODE,
        fewshot_examples=metrics["fewshot_examples"],
        system_prompt_sha256=_digest(SYSTEM_PROMPT),
    )

    print(json.dumps(metrics, indent=2))
    return metrics


def _reference_tags(record) -> list[str]:
    from .data import parse_example

    image = record["image"]
    return parse_example(record["ground_truth"], image.width, image.height)["ner_tags"]


def _digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# --- backends -----------------------------------------------------------------------


def hf_generate_fn(model, tokenizer):
    """Greedy generation over already-loaded weights.

    Separate from `hf_backend` so a training run can evaluate the model it still has in
    memory rather than reloading 7B parameters from disk to score them.
    """
    import torch

    def generate(prompts: list[list[dict]]) -> list[tuple[str, bool]]:
        results = []
        for index, messages in enumerate(prompts, 1):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(
                    **inputs, pad_token_id=tokenizer.eos_token_id, **DECODE
                )
            new_tokens = output[0][inputs["input_ids"].shape[1] :]
            results.append(
                (
                    tokenizer.decode(new_tokens, skip_special_tokens=True),
                    len(new_tokens) >= MAX_NEW_TOKENS,
                )
            )
            if index % 10 == 0:
                print(f"  {index}/{len(prompts)}")
        return results

    return generate


def hf_backend(adapter: str | None = None, load_in_4bit: bool = False):
    """Transformers generation. Used for correctness; `bench.py` measures speed."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    kwargs: dict = {**dtype_kwargs(torch.bfloat16), "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return hf_generate_fn(model, tokenizer)


def vllm_backend(adapter: str | None = None, quantization: str | None = None):
    """vLLM generation. Batches the whole split, which is why serving uses it."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=MODEL_ID,
        enable_lora=adapter is not None,
        quantization=quantization,
        max_model_len=8192,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    lora = LoRARequest("cord", 1, adapter) if adapter else None

    def generate(prompts: list[list[dict]]) -> list[tuple[str, bool]]:
        outputs = llm.chat(prompts, sampling, lora_request=lora)
        return [
            (o.outputs[0].text, o.outputs[0].finish_reason == "length") for o in outputs
        ]

    return generate


# --- gold completion lengths, which set MAX_NEW_TOKENS --------------------------------


def measure_lengths(dataset) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    lengths = [
        len(tokenizer(build_completion(gold_fields(record["ground_truth"])))["input_ids"])
        for record in dataset
    ]
    lengths.sort()

    report = {
        "model": MODEL_ID,
        "split": "test",
        "n_documents": len(lengths),
        "min": lengths[0],
        "median": statistics.median(lengths),
        "p95": lengths[int(0.95 * (len(lengths) - 1))],
        "max": lengths[-1],
        "max_new_tokens": MAX_NEW_TOKENS,
        "headroom_over_max": MAX_NEW_TOKENS - lengths[-1],
        "note": (
            "MAX_NEW_TOKENS must exceed max. A truncated completion is unparseable and "
            "scores zero, which reads as a model failure rather than a config failure."
        ),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "completion_lengths.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if lengths[-1] >= MAX_NEW_TOKENS:
        raise SystemExit(
            f"MAX_NEW_TOKENS={MAX_NEW_TOKENS} is below the longest gold completion "
            f"({lengths[-1]}). Raise it before evaluating anything."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="base", help="name for the artifacts this run writes")
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--adapter", default=None, help="path to a trained LoRA adapter")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--quantization", default=None, help="vLLM quantization, e.g. bitsandbytes")
    parser.add_argument("--fewshot", type=int, default=0, help="exemplars, from train only")
    parser.add_argument("--limit", type=int, default=0, help="0 = the whole test split")
    parser.add_argument("--lengths", action="store_true", help="measure gold lengths and stop")
    args = parser.parse_args()

    from datasets import load_dataset

    test = load_dataset(DATASET_ID, split="test")
    if args.limit:
        test = test.select(range(min(args.limit, len(test))))

    if args.lengths:
        measure_lengths(test)
        return

    fewshot = None
    if args.fewshot:
        fewshot = build_fewshot(
            load_dataset(DATASET_ID, split="train"), split="train", k=args.fewshot
        )

    generate = (
        vllm_backend(args.adapter, args.quantization)
        if args.backend == "vllm"
        else hf_backend(args.adapter, args.load_in_4bit)
    )
    evaluate(test, generate, args.config, fewshot)


if __name__ == "__main__":
    main()
