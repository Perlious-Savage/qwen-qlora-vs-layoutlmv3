# 7B generative vs 125M encoder: structured extraction on CORD-v2

A QLoRA fine-tune of Qwen2.5-7B-Instruct for receipt field extraction, measured against a
LayoutLMv3-base encoder fine-tuned on the identical dataset and split, scored by
byte-identical code.

**The 125M encoder wins, at roughly 1/300th the latency and 1/23rd the memory.** The 7B
model gets close after fine-tuning and is not obviously worse at the task — it is worse at
the price.

## Accuracy

Entity-level F1 on the official CORD-v2 test split (100 documents). A field counts as
correct only when its type *and* its full span match exactly; three words right out of four
scores zero.

| model | params | inputs | F1 | precision | recall | parse rate | locate rate |
|---|---:|---|---:|---:|---:|---:|---:|
| keyword + position rules | 0 | words | 0.509 | 0.557 | 0.468 | – | – |
| **LayoutLMv3-base** (Project 1) | 125M | words + boxes + page image | **0.948** | 0.942 | 0.955 | – | – |
| Qwen2.5-7B-Instruct, 2-shot | 7.6B | words only | 0.674 | 0.739 | 0.619 | 0.97 | 0.99 |
| Qwen2.5-7B-Instruct, QLoRA | 7.6B | words only | **0.908** | 0.910 | 0.906 | 1.00 | 1.00 |

The `inputs` column is the load-bearing caveat, and it is deliberately not a footnote.
LayoutLMv3 is multimodal: it sees where each word sits on the page, and the page itself.
Qwen2.5-7B-Instruct has no vision tower, so it sees the word sequence and nothing else.
Serialising bounding boxes into the prompt would narrow the gap and was rejected — it would
make the 7B look better without making the result more useful for deciding which model to
reach for.

### What moved, and what it means

**The fine-tune bought recall, not formatting.** The base model already had
`parse_rate 0.97` and `locate_rate 0.99` — it understood the output contract from two
examples and copied spans verbatim rather than paraphrasing. Its problem was omission: it
emitted 1096 fields against 1309 in the gold, and recall sat at 0.619. After fine-tuning it
emits 1304 — within five of the true count — and recall reaches 0.906.

So the +0.234 F1 is the model learning *which* spans are fields, not learning to produce
JSON. That distinction is only visible because parse rate and locate rate are reported
beside F1; with F1 alone, "it can't follow the format" and "it can't read receipts" are the
same number.

**It still loses to the encoder, by 0.041.** That gap is small enough to be honest about:
100 test documents, one training seed per model, no confidence intervals on either side. It
is evidence that the encoder is ahead, not proof. What is *not* marginal is the cost.

## Efficiency

Measured by `bench.py` on one A100, all configurations in one process with peak memory
reset between them.

### transformers, batch size 1

| config | workload | load s | p50 ms | p95 ms | tok/s | peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| Qwen base, bf16 | autoregressive generation | 8.0 | 5422 | 7774 | 29.0 | 14.27 GB |
| **Qwen LoRA merged, bf16** | autoregressive generation | 9.1 | **6416** | 9188 | 28.8 | **14.85 GB** |
| Qwen LoRA, 4-bit NF4 | autoregressive generation | 7.9 | 18978 | 27114 | 9.8 | 5.53 GB |
| **LayoutLMv3-base** | single forward pass, 512 tokens | 11.0 | **20** | 23 | – | **0.63 GB** |

### vLLM

A separate Colab session — vLLM cannot share a runtime with Colab's preinstalled torch
stack. Rows here are comparable to each other, **not** to the table above.

| config | workload | load s | p50 ms | p95 ms | tok/s | peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| Qwen base, bf16 | latency at batch 1, throughput batched | 230.0 | 1601 | 2271 | 665.5 | n/a |
| Qwen LoRA, bf16 | latency at batch 1, throughput batched | 112.0 | 2278 | 3254 | 608.7 | n/a |
| Qwen LoRA, 4-bit | **not measured** — this vLLM build rejects `bitsandbytes` quantization | – | – | – | – | – |

Peak VRAM is `n/a` for vLLM, not zero: vLLM allocates in a separate worker process, so
`torch.cuda.max_memory_allocated()` in the parent sees nothing. Reporting the 0.0 it
returned would have been a false measurement.

### The headline ratio

Comparing the two fine-tuned models on the same engine, same GPU, same run:

| | LayoutLMv3 | Qwen QLoRA | ratio |
|---|---:|---:|---:|
| F1 | 0.948 | 0.908 | −0.041 |
| p50 latency | 20 ms | 6416 ms | **321×** |
| peak VRAM | 0.63 GB | 14.85 GB | **23.6×** |
| parameters | 125M | 7.6B | 61× |

Three secondary findings worth stating plainly:

- **4-bit trades latency for memory, and the trade is steep.** NF4 cut peak VRAM 2.7× but
  made generation 3.0× *slower* (18978 ms vs 6416 ms). 4-bit quantization is a way to fit a
  model on a smaller card, not a way to make it faster.
- **vLLM is worth it for throughput, not for a single caller.** 665 tok/s batched against
  29 tok/s from transformers, but a 230-second cold start.
- **Merging the LoRA adapter cost 1 GB and 1 second.** Merged bf16 was slightly slower than
  the base model (6416 vs 5422 ms) because it generated more tokens per document, not
  because the merge is expensive.

## Pipeline

```
Qwen2.5-7B-Instruct
        │
        ├─ 2-shot baseline ────────────┐   0.674
        │                              │
   QLoRA fine-tune (4-bit NF4)         │   0.908
        │                              │
        ├─ merged bf16 ────────────────┤
        └─ 4-bit served ───────────────┤
                                       ▼
                             vLLM  ──►  benchmark
                                        F1 · latency · throughput · VRAM
```

## How the comparison is kept honest

The result rests on both models being scored by the same ruler, so the ruler is pinned
rather than described.

**The scoring code is byte-identical.** `src/metrics.py`, `src/data.py`, `src/schemas.py`,
`src/tools.py` and `src/baseline.py` are copied unchanged from
[financial-document-intelligence](https://github.com/Perlious-Savage/financial-document-intelligence)
at commit `befda04`. Their sha256 hashes are in `VENDORED.sha256` and CI runs
`sha256sum -c` on every push. If one byte of the metric changes, the build fails.

**The converter has a measured ceiling.** LayoutLMv3 emits one label per word; Qwen emits
JSON. Turning that JSON back into entity spans is where a comparison like this quietly
breaks. `roundtrip_check.py` converts the *gold* JSON back into spans and scores it against
the gold tags:

| target format | ceiling F1 | unlocatable fields | used |
|---|---:|---:|---|
| flat entity list, label word included | **1.000** | 0 | yes |
| nested `gt_parse`, value only | **0.627** | 36 | no |

A model trained to emit CORD's nested `gt_parse` could not have scored above 0.627 on this
metric however well it read receipts, because that view stores `"60.000"` where the gold
entity is `['TOTAL', '60.000']`. Comparing such a number against 0.948 would have measured
the annotation format, not the model.

The 1.000 was not free. The first run returned 0.974, and the missing 0.026 was the
converter splitting field text on whitespace when CORD annotates `'( L'` as a *single*
word. Without this check that loss would have been attributed to Qwen.

**Unparseable output scores zero.** Never skipped, never repaired. A document the model
failed on stays in the denominator.

**Invented fields cost precision.** A predicted field that appears nowhere in the receipt
becomes a span that cannot match any gold span, rather than being discarded. Discarding it
would reward the model that hallucinates most.

**One decoding configuration for every model.** Greedy, same prompt template, same token
cap, defined once in `src/eval_llm.py`, so evaluation is deterministic and the base-vs-tuned
gap carries no sampling noise. The cap is 1536 against a longest gold completion of 804
tokens, and truncation rate was 0.00% for both models.

**Few-shot exemplars come from the training split only**, fixed by seed, identical for
every test document.

**Contamination is measured, and it is not zero.** `scripts/check_splits.py` hashes each
document's word sequence:

| check | result |
|---|---:|
| test documents with an identical twin in train | **7 / 100** |
| validation documents with an identical twin in train | 11 / 100 |
| duplicate pairs within train itself | 24 |

Seven per cent of the test split is memorisable from training data. This is a property of
CORD-v2, not of either model, and Project 1 trained on the same split — both sides are
inflated by the same amount, so the *difference* between them stands while the absolute
numbers, 0.948 included, are optimistic. Indices are in `artifacts/split_check.json`.

**The model never does arithmetic.** Extracted amounts are parsed with `Decimal` and
reconciled by `src/tools.py`, exactly as in Project 1. The model reads; code adds up.

## Limitations

- **Text-only input for the 7B, against a multimodal 125M.** The single largest confound,
  and the reason to read this as a comparison of *approaches* rather than of model families.
- **The 0.041 F1 gap is within what one seed on 100 documents can support.** Neither number
  has a confidence interval. Treat the ordering as evidence, not proof. The cost ratios are
  large enough that they do not depend on it.
- **7% of the test split is duplicated in train.** Measured above; inflates both models.
- **One training seed, one epoch schedule, no LoRA-rank sweep.** `sweep.py` exists and was
  not run.
- **Two benchmark sessions.** The transformers and vLLM tables come from different Colab
  runtimes and are not comparable across tables.
- **`artifacts/base_metrics.json` and `artifacts/training_config.json` are transcriptions**
  of their runs' printed JSON — that runtime was destroyed by a vLLM install before the
  files were retrieved. Both carry a `provenance` field saying so. Every other artifact was
  written by the script that produced it.
- **The LayoutLMv3 benchmark row uses the base checkpoint**, not Project 1's fine-tuned
  weights: latency, memory and parameter count depend on the architecture, not on what the
  weights learned. Its *accuracy* row is Project 1's real measured result.
- CORD-v2 is Indonesian restaurant receipts. None of this generalises without re-measuring.

## Layout

```
src/convert.py        JSON ↔ entity spans. The load-bearing module; tested first.
src/sft_data.py       prompt template, SFT pairs, few-shot exemplars
src/eval_llm.py       one evaluation path for every configuration
src/train_qlora.py    QLoRA fine-tune (PEFT + TRL + bitsandbytes)
src/api.py            FastAPI service, MODEL_BACKEND = stub | hf | vllm
src/manifest.py       reproducibility metadata written beside every number
roundtrip_check.py    proves the harness adds no error of its own
bench.py              latency, throughput, VRAM, model size
compare.py            builds the tables above from the artifacts
sweep.py              LoRA rank / learning-rate grid (written, not run)
scripts/check_splits.py   train/test contamination check
src/metrics.py …      vendored from Project 1, hash-pinned, never edited
artifacts/project1/   Project 1's measured results, kept separate from this project's
```

## Reproducing

Local, no GPU:

```bash
pip install -r requirements.txt
pytest tests/ -q                 # 24 tests: converter contract and API
sha256sum -c VENDORED.sha256     # scoring code unchanged
```

With the dataset (~2.3 GB on first run):

```bash
pip install -r requirements-train.txt
pip uninstall -y torchao         # PEFT rejects Colab's older torchao
python roundtrip_check.py        # must print a ceiling of exactly 1.000
python scripts/check_splits.py
python -m src.eval_llm --lengths
```

On a GPU — see `notebooks/run_colab.ipynb`, which runs this as two passes because vLLM
cannot share a runtime with the rest:

```bash
python -m src.train_qlora --max-train 40 --epochs 1 --output outputs/smoke   # smoke first
python -m src.train_qlora --epochs 3                                         # ~15 min, A100
python -m src.eval_llm --config base --fewshot 2
python -m src.eval_llm --config qlora --adapter outputs/qwen-cord-lora
python bench.py --adapter outputs/qwen-cord-lora --documents 10
python compare.py
```

Serving:

```bash
docker build -t qwen-cord-extraction .
docker run -p 8000:8000 qwen-cord-extraction     # MODEL_BACKEND=stub, no GPU
```

## Training configuration

Qwen2.5-7B-Instruct, 4-bit NF4 with double quantization, LoRA rank 16 / alpha 32 /
dropout 0.05 on all attention and MLP projections — 40.4M trainable parameters, a 92 MB
adapter. 3 epochs over 800 receipts, lr 2e-4 cosine, effective batch 16, seed 0. 888
seconds on a Colab A100. Full configuration in `artifacts/training_config.json`; package
versions and GPU in the `*_manifest.json` files.

## Dataset

[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (naver-clova-ix,
CC-BY-4.0), 800 train / 100 validation / 100 test.
