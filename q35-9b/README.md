# Qwen3.5-9B on one GB10

This campaign compares three ways of running Qwen3.5-9B on one DGX Spark (GB10). It follows the setup asked for in [ggml-org/llama.cpp discussion #31092](https://github.com/ggml-org/llama.cpp/discussions/31092).

1. Our Omega engine (`aien-serve-http --engine omega`) with no NVIDIA user-space CUDA libraries.
2. Stock llama.cpp b11913 on NVIDIA CUDA.
3. Stock llama.cpp b11913 with our GPU backend ([aien-dev/ggml-omega](https://github.com/aien-dev/ggml-omega)) in place of libggml-cuda, again with no NVIDIA user-space CUDA libraries.

- [PROTOCOL.md](PROTOCOL.md): original protocol and dated resumption amendments.
- [ORDERED-WEIGHTS.md](ORDERED-WEIGHTS.md): exact checks of 427 named text tensors at the recorded pins. The original [weight report](results/WEIGHTS.md) remains historical evidence and must be read with its [correction](results/WEIGHTS.md.correction.json).
- [results/ref/](results/ref/): correctness references.
  - `HF-ORACLE.jsonl`: Hugging Face transformers, BF16, greedy.
  - `LLAMA16-REF.jsonl`: llama.cpp CUDA, BF16, greedy.
  - `tie-*.jsonl`: for every point where an arm's output differs from the oracle, the logit gap between the two tokens as the oracle sees it.
- [tools/](tools/): drivers and checks. These are host measurement tools, not product code.

The resumed timing campaign is running; no complete repeated comparison or final speed result is accepted yet. Failed attempts and historical results remain separate. The independent first-divergence observations cover the current five-arm corpus; they do not establish full-model numerical equivalence.

Publication chronology: the original protocol was already published, but these resumption amendments and corrections reached this draft after resumed timing began. The local campaign manifest records the frozen recipe before the first session. This late publication is disclosed rather than described as prospective remote registration.

`resumption.json` is the initial historical checkpoint. Its adjacent correction records later observations and remaining work. Native Intel CPU qualification and available-GPU investigation are additional pending work; no emulator result qualifies them.
