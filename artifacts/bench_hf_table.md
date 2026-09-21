| config | workload | load s | p50 ms | p95 ms | tok/s | peak VRAM GB |
|---|---|---:|---:|---:|---:|---:|
| qwen-base-bf16 | autoregressive generation, batch size 1 | 8.0 | 5422 | 7774 | 29.0 | 14.27 |
| qwen-lora-merged-bf16 | autoregressive generation, batch size 1 | 9.1 | 6416 | 9188 | 28.8 | 14.85 |
| qwen-lora-4bit | autoregressive generation, batch size 1 | 7.9 | 18978 | 27114 | 9.8 | 5.53 |
| layoutlmv3-base | single forward pass, 512 tokens - NOT the same work as generation | 11.0 | 20 | 23 | - | 0.63 |