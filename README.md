# 7B generative vs 125M encoder: structured extraction on CORD-v2

A QLoRA fine-tune of Qwen2.5-7B-Instruct for receipt field extraction, measured against a
LayoutLMv3-base encoder fine-tuned on the identical dataset, the identical split, and
scored by byte-identical code.

The point of this repository is the comparison, not the fine-tune. Knowing when a 7B
generative model is the wrong tool is worth more than knowing how to train one, and that
judgement is only credible with numbers attached.

## Accuracy

Entity-level F1 on the official CORD-v2 test split (100 documents). A field counts as
correct only when its type *and* its full span match exactly; three words right out of four
scores zero.

| model | params | inputs | F1 | parse rate | locate rate |
|---|---:|---|---:|---:|---:|
| keyword + position rules | 0 | words | **0.509** | – | – |
| LayoutLMv3-base (Project 1) | 125M | words + boxes + page image | **0.948** | – | – |
| Qwen2.5-7B-Instruct, few-shot | 7B | words only | TBD | TBD | TBD |
| Qwen2.5-7B-Instruct, QLoRA | 7B | words only | TBD | TBD | TBD |

The `inputs` column is the load-bearing caveat and is deliberately not a footnote.
LayoutLMv3 is multimodal: it sees where each word sits on the page and the page itself.
Qwen2.5-7B-Instruct has no vision tower, so it sees the word sequence and nothing else.
Serialising bounding boxes into the prompt would narrow that gap and was rejected — it
would make the 7B look better without making the result more useful for deciding which
model to reach for.

**parse rate** is the fraction of generations that were valid JSON in the required shape.
**locate rate** is the fraction of predicted fields found verbatim in the receipt text.
Together they separate "could not follow the output contract" from "could not read the
receipt", which F1 alone cannot distinguish — and which is usually the entire story behind
a base model's score.

## Efficiency

Measured by `bench.py`, all configurations in one process on one GPU so the rows are
mutually comparable.

| config | workload | load s | p50 ms | p95 ms | tok/s | peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| Qwen base, bf16 | autoregressive generation | TBD | TBD | TBD | TBD | TBD |
| Qwen LoRA merged, bf16 | autoregressive generation | TBD | TBD | TBD | TBD | TBD |
| Qwen LoRA, 4-bit NF4 | autoregressive generation | TBD | TBD | TBD | TBD | TBD |
| LayoutLMv3-base | single forward pass, 512 tokens | TBD | TBD | – | – | TBD |

Cold load, warm latency and sustained throughput are reported separately because they
answer different questions and get conflated constantly. The LayoutLMv3 row is a different
workload — one forward pass versus generating several hundred tokens — and the two are
comparable as cost per document and in no other way.

No claim about cost reduction appears anywhere in this repository until `bench.py` has
produced the number behind it.

## Pipeline

```
Qwen2.5-7B-Instruct
        │
        ├─ few-shot baseline ──────────┐
        │                              │
   QLoRA fine-tune (4-bit NF4)         │
        │                              │
        ├─ merged bf16 ────────────────┤
        └─ 4-bit served ───────────────┤
                                       ▼
                             vLLM  ──►  benchmark
                                        F1 · latency · throughput · VRAM
```

## How the comparison is kept honest

The whole result rests on both models being scored by the same ruler, so the ruler is
pinned rather than described.

**The scoring code is byte-identical.** `src/metrics.py`, `src/data.py`, `src/schemas.py`,
`src/tools.py` and `src/baseline.py` are copied unchanged from
[financial-document-intelligence](https://github.com/Perlious-Savage/financial-document-intelligence)
at commit `befda04`. Their sha256 hashes are in `VENDORED.sha256` and CI runs
`sha256sum -c` on every push. If one byte of the metric changes, the build fails.

**The converter has a measured ceiling.** LayoutLMv3 emits one label per word; Qwen emits
JSON. Turning that JSON back into entity spans is where a comparison like this quietly
breaks. `roundtrip_check.py` converts the *gold* JSON back into spans and scores it against
the gold tags — the answer must be exactly 1.000, or the harness is adding error that would
read as model weakness. It also measures the ceiling of the obvious alternative target
format, CORD's nested `gt_parse`, which is why that format was not used:

| target format | ceiling F1 | used |
|---|---:|---|
| flat entity list, label word included | TBD | yes |
| nested `gt_parse`, value only | TBD | no |

**Unparseable output scores zero.** It is never skipped and never repaired. A document the
model failed on stays in the denominator; silently dropping it would shrink the test set to
whatever the model happened to handle.

**Invented fields cost precision.** A predicted field that appears nowhere in the receipt
becomes a span that cannot match any gold span, rather than being discarded. Discarding it
would reward the model that hallucinates most.

**One decoding configuration for every model.** Greedy, same prompt template, same token
cap, defined once in `src/eval_llm.py`. Evaluation is therefore deterministic and the
base-vs-tuned gap carries no sampling noise. Truncation rate is reported, because a
completion cut off mid-JSON is unparseable and would otherwise read as a model failure.

**Few-shot exemplars come from the training split only**, fixed by seed, identical for
every test document.

**Contamination is measured, not assumed.** `scripts/check_splits.py` hashes each
document's word sequence and reports train/test overlap. Any overlap is shared with
Project 1, which trained on the same split, so it affects both sides equally — it is
reported rather than quietly corrected.

**The model never does arithmetic.** Extracted amounts are parsed with `Decimal` and
reconciled by `src/tools.py`, exactly as in Project 1. The model reads; code adds up.

## Limitations

- Text-only input for the 7B, against a multimodal 125M. Stated above; it is the main
  reason to read the two F1 numbers as a comparison of *approaches*, not of model families.
- 100 test documents. Differences of a point or two are not distinguishable from noise.
- Single training seed unless the F1 gap turns out narrower than 0.05, in which case the
  fine-tune is repeated across three seeds and reported as mean ± spread.
- One GPU, one session. All latency and memory numbers are from that session and are not
  portable to other hardware.
- CORD-v2 is Indonesian restaurant receipts. None of this generalises to other document
  types without re-measuring.

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
sweep.py              LoRA rank / learning-rate grid with sensitivity report
scripts/check_splits.py   train/test contamination check
src/metrics.py …      vendored from Project 1, hash-pinned, never edited
artifacts/project1/   Project 1's measured results, kept separate from this project's
```

## Running it

Local, no GPU needed:

```bash
pip install -r requirements.txt
pytest tests/ -q                 # converter and API tests
sha256sum -c VENDORED.sha256     # scoring code unchanged
```

With the dataset (~2.3GB on first run):

```bash
pip install -r requirements-train.txt
python roundtrip_check.py                  # must print a ceiling of exactly 1.000
python scripts/check_splits.py
python -m src.eval_llm --lengths           # sets the generation token cap
```

On a GPU (see `notebooks/run_colab.ipynb`):

```bash
python -m src.train_qlora --max-train 40 --epochs 1      # smoke test first
python -m src.train_qlora --epochs 3
python -m src.eval_llm --config base --fewshot 2
python -m src.eval_llm --config qlora --adapter outputs/qwen-cord-lora
python bench.py --adapter outputs/qwen-cord-lora
```

Serving:

```bash
docker build -t qwen-cord-extraction .
docker run -p 8000:8000 qwen-cord-extraction     # MODEL_BACKEND=stub, no GPU
```

## Dataset

[CORD-v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (naver-clova-ix,
CC-BY-4.0), 800 train / 100 validation / 100 test.
