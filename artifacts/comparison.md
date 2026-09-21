## Accuracy

| model | params | inputs | F1 | parse rate | locate rate |
|---|---:|---|---:|---:|---:|
| keyword + position rules | 0 | words | **0.509** | - | - |
| LayoutLMv3-base (Project 1) | 125M | words + boxes + page image | **0.948** | - | - |
| Qwen2.5-7B-Instruct, few-shot | 7B | words only | **0.674** | 0.97 | 0.99 |
| Qwen2.5-7B-Instruct, QLoRA | 7B | words only | **0.908** | 1.00 | 1.00 |

## Efficiency

### transformers, batch size 1

| config | workload | load s | p50 ms | p95 ms | tok/s | peak VRAM GB |
|---|---|---:|---:|---:|---:|---:|
| qwen-base-bf16 | autoregressive generation, batch size 1 | 8.0 | 5422 | 7774 | 29.0 | 14.27 |
| qwen-lora-merged-bf16 | autoregressive generation, batch size 1 | 9.1 | 6416 | 9188 | 28.8 | 14.85 |
| qwen-lora-4bit | autoregressive generation, batch size 1 | 7.9 | 18978 | 27114 | 9.8 | 5.53 |
| layoutlmv3-base | single forward pass, 512 tokens - NOT the same work as generation | 11.0 | 20 | 23 | - | 0.63 |

### vLLM (separate session - compare within this table, not against the one above)

| config | workload | load s | p50 ms | p95 ms | tok/s | peak VRAM GB |
|---|---|---:|---:|---:|---:|---:|
| qwen-base-bf16 | autoregressive generation; latency at batch 1, throughput batched | 230.0 | 1601 | 2271 | 665.5 | n/a |
| qwen-lora-bf16 | autoregressive generation; latency at batch 1, throughput batched | 112.0 | 2278 | 3254 | 608.7 | n/a |
| qwen-lora-4bit | **not measured** — ValidationError: 1 validation error for ModelConfig | - | - | - | - | - |

## Converter ceiling

| target format | ceiling F1 | used |
|---|---:|---|
| flat entity list, label word included | 1.000 | yes |
| nested `gt_parse`, value only | 0.627 | no |

## Checks

- train/test word-sequence overlap: 7
- truncated generations (QLoRA): 0.00%