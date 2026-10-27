# Same weights on both sides (BF16 headline pair)

Source: HF Qwen/Qwen3.5-9B rev c202236235762e1c871ad0ccb60c8ee5ba337b9a. All files pass sha256 against `qwen3.5-9b-c2022362.sha256`.
- OURS16 reads the safetensors directly.
- LLAMA16 reads `q35-9b-bf16.gguf` (sha256 5d6fe481317c2570d8e136d6d6da7b33c97f5de4064f96a14b4ea46b8b80a9bb), made by `convert_hf_to_gguf.py --outtype bf16` from llama.cpp tree 7aca0d2ea.

Checked tensor by tensor with cmp_weights.py, cmp_weights2.py and cmp_ssm_a.py, which run in nvcr.io/nvidia/vllm:26.09-py3 with gguf-py taken from the same llama.cpp tree.

The GGUF has 442 tensors. 15 belong to the MTP head (blk.32.*, nextn.*); the HF checkpoint has that head too (mtp.*, excluded here), and neither engine uses it for plain decoding. The other 427 match the 427 text tensors of the safetensors one-to-one (130 + 120 + 72 + 81 + 24 = 427 below).

| GGUF tensors | Relation to the HF tensor | Count |
|---|---|---|
| ffn_up/gate/down, full-attention q/k/v/output, token_embd, output | byte-identical | 130 |
| linear-attention attn_qkv, attn_gate, ssm_alpha, ssm_beta, ssm_out (BF16) | same values, rows reordered by the converter (value-head order) | 120 |
| ssm_conv1d, ssm_dt.bias, ssm_norm (F32) | same values, widened exactly to F32, reordered | 72 |
| attn_norm, post_attention_norm, q_norm, k_norm, output_norm (F32) | exactly 1 + w (Qwen3.5 stores zero-centred RMSNorm weights; llama.cpp folds the +1 in) | 81 |
| ssm_a (F32) | exactly −exp(A_log), 0 ULP difference after sorting, reordered | 24 |

"Same values" means the sorted multiset of float32 values is identical. Both engines therefore compute with the same numbers. Only the storage layout differs.

Raw output: weights-cmp.json, weights-cmp2.txt, weights-cmp-ssm_a.txt.
