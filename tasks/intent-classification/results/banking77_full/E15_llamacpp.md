# E15: the same adapter and head served by llama.cpp on a CPU (2026-11-05)

Setup: Docker container limited to 4 CPUs and 5 GB on an Apple M3 Pro (the container is
arm64 Linux), llama.cpp server image `ghcr.io/ggml-org/llama.cpp:server` (release 0.9.0 of
2026-11-04), base `unsloth/Qwen3.5-4B-GGUF` `Qwen3.5-4B-Q4_K_M.gguf` (2,614 MB), the
all-rows Banking77 adapter converted to a LoRA GGUF with llama.cpp's `convert_lora_to_gguf.py`
(248 tensors, 16 MB, f16) and applied with `--lora` at run time. Prompt: one user message
in the chat template with thinking off, 24 tokens on average.

| | llama.cpp, 4 vCPU | MLX, the training engine |
|---|---|---|
| Head on layer 32 (fitted on this engine's post-norm states, 10,003 training rows, C 0.03, CV 93.1), 3,080 test rows | **93.60** | 93.57 |
| Generated answer, first 500 test rows (temperature 0, 24 max tokens), format valid 500 of 500 | 93.4 | 93.0 on the same rows |
| SGR at 5%, delta 0.05, seeds 42 / 7 / 2026: kept, realised error on the untouched third | 88.7% 2.5%, 81.6% 1.4%, 90.4% 1.9% | 88.8% 2.6%, 90.1% 2.3%, 90.6% 2.3% |
| Decision latency, `/embeddings` pooling last, 50 requests one at a time, p50 / p95 | 337 / 471 ms (329 / 463 without the adapter) | 63 / 107 ms on the Mac's GPU |
| Head folded into the GGUF as `cls.output` (`truncate_gguf.py --head`), served, against the Python head on 100 rows | 100 of 100 labels equal, largest logit difference 2.3e-6, 306 ms p50 | |

Engine settings that mattered: `--cache-ram 0` (the server's prompt cache otherwise grows
to 8 GB and the container is OOM-killed on the chat path), default weight repacking for a
decision-only server (337 ms), `--no-repack` for a server that also writes answers (decode
42 ms per token instead of 275; the decision then costs 618 ms).

Commands (from `routing/`, model files beside them):

```
docker run -d --cpus 4 --memory 5g -p 8080:8080 <image with the GGUF and adapter baked in> -t 4 -c 512 --cache-ram 0 --lora /models/adapter.gguf
python capture_llama.py --rows ../data/banking77/train_full.jsonl --stem lora32_trainfull --layers 32
python capture_llama.py --rows ../data/banking77/test.jsonl --stem lora32_test --layers 32
python classifier_tier.py --train lora32_trainfull --train-rows train_full.jsonl --test lora32_test --test-rows test.jsonl --out lora32_classifier_conf.jsonl
python export_head.py --train lora32_trainfull --train-rows train_full.jsonl --test lora32_test --test-rows test.jsonl --adapter <adapter dir> --out head_L32_llamacpp.npz
python score_llama.py --rows test_first500.jsonl --stem lora32_test500_scores     # server started with --no-repack
python latency_bench.py --mode embed --n 50
python truncate_gguf.py Qwen3.5-4B-Q4_K_M.gguf head32.gguf 32 --head head_L32_llamacpp.npz && python check_head_in_file.py --head head_L32_llamacpp.npz --states lora32_test --n 100
```

The image is built from `k8s/Dockerfile` (the GGUF copied into llama.cpp's server image; on
the machine used, a bind mount of the model folder did not start). `k8s/decision.yaml` runs
the same image as a pod with CPU limits and probes; through a kind cluster on the same Mac
the 8-layer early-exit file answered in 88 ms p50, the same as the bare container.
