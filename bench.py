"""Measure what the 7B actually costs to serve, against the 125M encoder.

Three quantities that routinely get reported as one, separated here because they answer
different questions:

    load_seconds        cold start. Matters for autoscaling, not for steady state.
    warm_latency_ms     one document, batch size 1, after warmup. What a caller waits.
    tokens_per_second   sustained generation. What capacity planning uses.

The LayoutLMv3 row is not the same workload and the table says so: it is a single forward
pass over 512 tokens, while Qwen generates several hundred tokens autoregressively. They
are comparable as cost per document and in no other way. That row uses the base
`microsoft/layoutlmv3-base` checkpoint rather than Project 1's fine-tuned weights, which
is legitimate here - latency, memory and parameter count depend on the architecture, not
on what the weights were trained to.

All configurations run in one process on one GPU so the numbers are mutually comparable;
peak memory is reset between them and the previous model is freed. Numbers from separate
runs, or from different GPUs, are not comparable and should not be put in one table.

    python bench.py --adapter outputs/qwen-cord-lora
    python bench.py --adapter outputs/qwen-cord-lora --engine vllm
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

from src import manifest
from src.data import DATASET_ID
from src.eval_llm import DECODE, MAX_NEW_TOKENS, MODEL_ID
from src.sft_data import build_messages, words_of

ARTIFACTS = Path(__file__).parent / "artifacts"

WARMUP = 2
LAYOUTLMV3_ID = "microsoft/layoutlmv3-base"


def _free() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peak_gb() -> float:
    import torch

    return round(torch.cuda.max_memory_allocated() / 1024**3, 2)


def _directory_bytes(path: str | Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def _summarise(name: str, workload: str, load_seconds: float, latencies_ms: list[float],
               generated_tokens: int, generate_seconds: float, **extra) -> dict:
    latencies_ms.sort()
    return {
        "config": name,
        "workload": workload,
        "load_seconds": round(load_seconds, 1),
        "warm_latency_p50_ms": round(statistics.median(latencies_ms)),
        "warm_latency_p95_ms": round(latencies_ms[int(0.95 * (len(latencies_ms) - 1))]),
        "tokens_per_second": round(generated_tokens / generate_seconds, 1)
        if generate_seconds
        else None,
        "peak_vram_gb": _peak_gb(),
        **extra,
    }


def bench_hf(name: str, documents, adapter: str | None, load_in_4bit: bool) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _free()
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    kwargs: dict = {"dtype": torch.bfloat16, "device_map": "auto"}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)

    merged = False
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        if not load_in_4bit:
            # Merging folds the adapter into the base weights, removing the per-layer
            # LoRA matmuls from every forward pass. It is not possible in 4-bit without
            # dequantising, which is why that config keeps the adapter separate.
            model = model.merge_and_unload()
            merged = True
    model.eval()
    load_seconds = time.time() - started

    def one(messages) -> tuple[float, int]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        clock = time.perf_counter()
        with torch.no_grad():
            output = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **DECODE)
        torch.cuda.synchronize()
        return (time.perf_counter() - clock) * 1000, output.shape[1] - inputs["input_ids"].shape[1]

    prompts = [build_messages(words)["prompt"] for words in documents]
    for messages in prompts[:WARMUP]:
        one(messages)

    latencies, tokens = [], 0
    clock = time.perf_counter()
    for messages in prompts:
        latency_ms, new_tokens = one(messages)
        latencies.append(latency_ms)
        tokens += new_tokens
    elapsed = time.perf_counter() - clock

    result = _summarise(
        name,
        "autoregressive generation, batch size 1",
        load_seconds,
        latencies,
        tokens,
        elapsed,
        engine="transformers",
        adapter_merged=merged,
        quantization="nf4" if load_in_4bit else "bf16",
        mean_generated_tokens=round(tokens / len(prompts), 1),
        on_disk_gb=round(_directory_bytes(adapter) / 1024**3, 4) if adapter else None,
    )
    del model
    _free()
    return result


def bench_vllm(name: str, documents, adapter: str | None, quantization: str | None) -> dict:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    _free()
    started = time.time()
    llm = LLM(
        model=MODEL_ID,
        enable_lora=adapter is not None,
        quantization=quantization,
        max_model_len=8192,
    )
    load_seconds = time.time() - started

    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    lora = LoRARequest("cord", 1, adapter) if adapter else None
    prompts = [build_messages(words)["prompt"] for words in documents]

    llm.chat(prompts[:WARMUP], sampling, lora_request=lora)

    # Latency: one at a time, the way a single caller experiences it.
    latencies = []
    for messages in prompts:
        clock = time.perf_counter()
        llm.chat([messages], sampling, lora_request=lora)
        latencies.append((time.perf_counter() - clock) * 1000)

    # Throughput: the whole batch at once, which is the reason to run vLLM at all.
    clock = time.perf_counter()
    outputs = llm.chat(prompts, sampling, lora_request=lora)
    elapsed = time.perf_counter() - clock
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    result = _summarise(
        name,
        "autoregressive generation; latency at batch 1, throughput batched",
        load_seconds,
        latencies,
        tokens,
        elapsed,
        engine="vllm",
        quantization=quantization or "bf16",
        batch_size_for_throughput=len(prompts),
        mean_generated_tokens=round(tokens / len(prompts), 1),
        on_disk_gb=round(_directory_bytes(adapter) / 1024**3, 4) if adapter else None,
    )
    del llm
    _free()
    return result


def bench_layoutlmv3(documents) -> dict:
    """The other side of the comparison, measured on the same GPU in the same run."""
    import torch
    from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

    _free()
    started = time.time()
    processor = AutoProcessor.from_pretrained(LAYOUTLMV3_ID, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(LAYOUTLMV3_ID).to("cuda").eval()
    load_seconds = time.time() - started

    from PIL import Image

    blank = Image.new("RGB", (1000, 1000))

    def one(words) -> float:
        encoding = processor(
            blank,
            words,
            boxes=[[0, 0, 100, 100]] * len(words),
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).to("cuda")
        torch.cuda.synchronize()
        clock = time.perf_counter()
        with torch.no_grad():
            model(**encoding)
        torch.cuda.synchronize()
        return (time.perf_counter() - clock) * 1000

    for words in documents[:WARMUP]:
        one(words)

    latencies = [one(words) for words in documents]
    result = {
        "config": "layoutlmv3-base",
        "workload": "single forward pass, 512 tokens - NOT the same work as generation",
        "load_seconds": round(load_seconds, 1),
        "warm_latency_p50_ms": round(statistics.median(sorted(latencies))),
        "warm_latency_p95_ms": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))]),
        "tokens_per_second": None,
        "peak_vram_gb": _peak_gb(),
        "engine": "transformers",
        "quantization": "fp32",
        "parameters": sum(p.numel() for p in model.parameters()),
        "note": (
            "Base checkpoint, not Project 1's fine-tuned weights: latency, memory and size "
            "depend on the architecture, not on what the weights learned."
        ),
    }
    del model
    _free()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/qwen-cord-lora")
    parser.add_argument("--engine", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--documents", type=int, default=20)
    parser.add_argument("--skip-encoder", action="store_true")
    args = parser.parse_args()

    from datasets import load_dataset

    test = load_dataset(DATASET_ID, split="test").select(range(args.documents))
    documents = [words_of(record) for record in test]

    results = []
    if args.engine == "hf":
        results.append(bench_hf("qwen-base-bf16", documents, None, False))
        results.append(bench_hf("qwen-lora-merged-bf16", documents, args.adapter, False))
        results.append(bench_hf("qwen-lora-4bit", documents, args.adapter, True))
    else:
        results.append(bench_vllm("qwen-base-bf16", documents, None, None))
        results.append(bench_vllm("qwen-lora-bf16", documents, args.adapter, None))
        results.append(bench_vllm("qwen-lora-4bit", documents, args.adapter, "bitsandbytes"))

    if not args.skip_encoder:
        results.append(bench_layoutlmv3(documents))

    report = {
        "engine": args.engine,
        "n_documents": args.documents,
        "dataset": DATASET_ID,
        "results": results,
        "note": (
            "One process, one GPU, peak memory reset between configurations. Rows using "
            "different engines or different workloads are not directly comparable; the "
            "workload field says which is which."
        ),
    }

    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / f"bench_{args.engine}.json").write_text(json.dumps(report, indent=2))
    manifest.write(f"bench_{args.engine}", engine=args.engine, n_documents=args.documents)

    rows = [
        "| config | workload | load s | p50 ms | p95 ms | tok/s | peak VRAM GB |",
        "|---|---|---:|---:|---:|---:|---:|",
    ] + [
        f"| {r['config']} | {r['workload']} | {r['load_seconds']} | "
        f"{r['warm_latency_p50_ms']} | {r['warm_latency_p95_ms']} | "
        f"{r['tokens_per_second'] or '-'} | {r['peak_vram_gb']} |"
        for r in results
    ]
    table = "\n".join(rows)
    (ARTIFACTS / f"bench_{args.engine}_table.md").write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
