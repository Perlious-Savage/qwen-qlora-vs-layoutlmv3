## Accuracy

| model | params | inputs | F1 | parse rate | locate rate |
|---|---:|---|---:|---:|---:|
| keyword + position rules | 0 | words | **0.509** | - | - |
| LayoutLMv3-base (Project 1) | 125M | words + boxes + page image | **0.948** | - | - |
| Qwen2.5-7B-Instruct, few-shot | 7B | words only | **TBD** | - | - |
| Qwen2.5-7B-Instruct, QLoRA | 7B | words only | **TBD** | - | - |

## Efficiency

| config | workload | load s | p50 ms | p95 ms | tok/s | peak VRAM GB |
|---|---|---:|---:|---:|---:|---:|
| _not yet measured_ | - | TBD | TBD | TBD | TBD | TBD |

## Converter ceiling

| target format | ceiling F1 | used |
|---|---:|---|
| flat entity list, label word included | 1.000 | yes |
| nested `gt_parse`, value only | 0.627 | no |

## Checks

- train/test word-sequence overlap: 7
- truncated generations (QLoRA): TBD