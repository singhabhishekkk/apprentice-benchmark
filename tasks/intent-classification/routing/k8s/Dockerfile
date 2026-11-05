# The decision server as one image: llama.cpp's server plus the 4-bit model baked in, so a
# kind node (or any cluster) runs it without a mounted volume. Build from spike3/:
#   docker build -f k8s/Dockerfile -t apprentice-decision:qwen35-4b-q4 hf/gguf
# An early-exit copy (truncate_gguf.py) builds the same way with --build-arg MODEL=<file>.
FROM ghcr.io/ggml-org/llama.cpp:server
ARG MODEL=Qwen3.5-4B-Q4_K_M.gguf
COPY ${MODEL} /models/model.gguf
ENTRYPOINT ["/app/llama-server", "-m", "/models/model.gguf", "--host", "0.0.0.0", "--port", "8080", \
            "-c", "1024", "--parallel", "1", "--embeddings", "--pooling", "last", "--jinja"]
