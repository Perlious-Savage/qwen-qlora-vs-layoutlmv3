| config | workload | load s | p50 ms | p95 ms | tok/s | peak VRAM GB |
|---|---|---:|---:|---:|---:|---:|
| qwen-base-bf16 | autoregressive generation; latency at batch 1, throughput batched | 230.0 | 1601 | 2271 | 665.5 | 0.0 |
| qwen-lora-bf16 | autoregressive generation; latency at batch 1, throughput batched | 112.0 | 2278 | 3254 | 608.7 | 0.0 |
| qwen-lora-4bit | _ValidationError: 1 validation error for ModelConfig
  Value error, Unknown quantization method: bitsandbytes. Must be one of ['awq', 'auto_awq', 'fp8', 'fbgemm_fp8', 'fp_quant', 'modelopt', 'modelopt_fp4', 'modelopt_mxfp8', 'modelopt_mixed', 'auto_gptq', 'gptq', 'gptq_marlin', 'awq_marlin', 'humming', 'compressed-tensors', 'experts_int8', 'quark', 'moe_wna16', 'torchao', 'inc', 'mxfp4', 'gpt_oss_mxfp4', 'deepseek_v4_fp8', 'online', 'fp8_per_tensor', 'fp8_per_block', 'fp8_per_channel', 'int8_per_channel_weight_only', 'nvfp4_per_token', 'mxfp8']. [type=value_error, input_value=ArgsKwargs((), {'model': ...nderer_num_workers': 1}), input_type=ArgsKwargs]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error_ | - | - | - | - | - |