# Qwen3.5-9B on one GB10

This campaign compares three ways of running Qwen3.5-9B on one DGX Spark (GB10). It follows the setup asked for in [ggml-org/llama.cpp discussion #31092](https://github.com/ggml-org/llama.cpp/discussions/31092).

1. Our Omega engine (`aien-serve-http --engine omega`) with no NVIDIA user-space CUDA libraries.
2. Stock llama.cpp b11913 on NVIDIA CUDA.
3. Stock llama.cpp b11913 with our GPU backend ([aien-dev/ggml-omega](https://github.com/aien-dev/ggml-omega)) in place of libggml-cuda, again with no NVIDIA user-space CUDA libraries.

- [PROTOCOL.md](PROTOCOL.md): frozen and committed before any timed run.
- [results/WEIGHTS.md](results/WEIGHTS.md): checks that every arm runs the same weights.
- [results/ref/](results/ref/): correctness references.
  - `HF-ORACLE.jsonl`: Hugging Face transformers, BF16, greedy.
  - `LLAMA16-REF.jsonl`: llama.cpp CUDA, BF16, greedy.
  - `tie-*.jsonl`: for every point where an arm's output differs from the oracle, the logit gap between the two tokens as the oracle sees it.
- [tools/](tools/): drivers and checks. These are host measurement tools, not product code.

No timed numbers exist yet. Every result lands here as raw rows first, and every arm that was not run is listed as NOT_RUN.
