## Llama related kernels

> default: internlm2-1_8b   **hidden_size**：2048  **vocab_size**：92544

#### Kernels

##### linear INT8 




#### Shape inference

- `token_embd`  weight   `[92544, 2048]` ` [vocab_size, hidden_dim]` , `[seq_len]` -> `[seq_len, hidden_dim]`

| Layer/Op           | Input Data Shape  | Output Data Shape  | 说明                                                         |
| ------------------ | ----------------- | ------------------ | ------------------------------------------------------------ |
| `token_embd`       | `[seq_len]`       | `[seq_len, 2048]`  | 输入 token IDs → 嵌入为 2048 维向量                          |
| `blk.0.attn_norm`  | `[seq_len, 2048]` | `[seq_len, 2048]`  | LayerNorm：保持形状不变                                      |
| `blk.0.attn`       | `[seq_len, 2048]` | `[seq_len, 2048]`  | 多头注意力：输入输出同维度（Q/K/V 内部计算维度压缩，但最终合并还原） |
| `blk.0.ffn_norm`   | `[seq_len, 2048]` | `[seq_len, 2048]`  | LayerNorm：保持形状不变                                      |
| `blk.0.ffn`        | `[seq_len, 2048]` | `[seq_len, 2048]`  | SwiGLU FFN：2048 → 8192 → 2048（内部扩展 4 倍后投影还原）    |
| `...`              | `...`             | `...`              | 重复 `blk.0` 至 `blk.23`（共 24 层）                         |
| `output_norm`      | `[seq_len, 2048]` | `[seq_len, 2048]`  | 最终 LayerNorm                                               |
| `output` (lm_head) | `[seq_len, 2048]` | `[seq_len, 92544]` | 投影到词汇表维度（预测下一个 token 的 logits）               |

#### INT8 QUANT

```bash
INFO:hf-to-gguf:Loading model: internlm2-1_8b
INFO:gguf.gguf_writer:gguf: This GGUF file is for Little Endian only
INFO:hf-to-gguf:Exporting model...
INFO:hf-to-gguf:gguf: loading model part 'pytorch_model.bin'
INFO:hf-to-gguf:token_embd.weight,           torch.bfloat16 --> Q8_0, shape = {2048, 92544}
INFO:hf-to-gguf:blk.0.attn_q.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.0.attn_k.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.0.attn_v.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.0.attn_output.weight,    torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.0.ffn_gate.weight,       torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.0.ffn_up.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.0.ffn_down.weight,       torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.0.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.0.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.1.attn_q.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.1.attn_k.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.1.attn_v.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.1.attn_output.weight,    torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.1.ffn_gate.weight,       torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.1.ffn_up.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.1.ffn_down.weight,       torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.1.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.1.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.2.attn_q.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.2.attn_k.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.2.attn_v.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.2.attn_output.weight,    torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.2.ffn_gate.weight,       torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.2.ffn_up.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.2.ffn_down.weight,       torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.2.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.2.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.3.attn_q.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.3.attn_k.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.3.attn_v.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.3.attn_output.weight,    torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.3.ffn_gate.weight,       torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.3.ffn_up.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.3.ffn_down.weight,       torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.3.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.3.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.4.attn_q.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.4.attn_k.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.4.attn_v.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.4.attn_output.weight,    torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.4.ffn_gate.weight,       torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.4.ffn_up.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.4.ffn_down.weight,       torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.4.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.4.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.5.attn_q.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.5.attn_k.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.5.attn_v.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.5.attn_output.weight,    torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.5.ffn_gate.weight,       torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.5.ffn_up.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.5.ffn_down.weight,       torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.5.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.5.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.6.attn_q.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.6.attn_k.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.6.attn_v.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.6.attn_output.weight,    torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.6.ffn_gate.weight,       torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.6.ffn_up.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.6.ffn_down.weight,       torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.6.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.6.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.7.attn_q.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.7.attn_k.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.7.attn_v.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.7.attn_output.weight,    torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.7.ffn_gate.weight,       torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.7.ffn_up.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.7.ffn_down.weight,       torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.7.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.7.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.8.attn_q.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.8.attn_k.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.8.attn_v.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.8.attn_output.weight,    torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.8.ffn_gate.weight,       torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.8.ffn_up.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.8.ffn_down.weight,       torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.8.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.8.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.9.attn_q.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.9.attn_k.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.9.attn_v.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.9.attn_output.weight,    torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.9.ffn_gate.weight,       torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.9.ffn_up.weight,         torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.9.ffn_down.weight,       torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.9.attn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.9.ffn_norm.weight,       torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.10.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.10.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.10.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.10.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.10.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.10.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.10.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.10.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.10.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.11.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.11.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.11.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.11.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.11.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.11.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.11.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.11.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.11.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.12.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.12.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.12.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.12.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.12.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.12.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.12.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.12.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.12.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.13.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.13.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.13.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.13.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.13.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.13.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.13.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.13.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.13.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.14.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.14.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.14.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.14.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.14.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.14.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.14.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.14.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.14.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.15.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.15.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.15.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.15.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.15.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.15.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.15.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.15.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.15.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.16.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.16.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.16.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.16.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.16.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.16.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.16.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.16.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.16.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.17.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.17.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.17.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.17.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.17.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.17.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.17.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.17.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.17.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.18.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.18.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.18.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.18.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.18.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.18.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.18.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.18.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.18.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.19.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.19.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.19.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.19.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.19.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.19.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.19.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.19.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.19.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.20.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.20.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.20.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.20.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.20.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.20.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.20.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.20.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.20.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.21.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.21.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.21.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.21.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.21.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.21.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.21.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.21.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.21.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.22.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.22.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.22.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.22.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.22.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.22.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.22.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.22.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.22.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.23.attn_q.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.23.attn_k.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.23.attn_v.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 1024}
INFO:hf-to-gguf:blk.23.attn_output.weight,   torch.bfloat16 --> Q8_0, shape = {2048, 2048}
INFO:hf-to-gguf:blk.23.ffn_gate.weight,      torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.23.ffn_up.weight,        torch.bfloat16 --> Q8_0, shape = {2048, 8192}
INFO:hf-to-gguf:blk.23.ffn_down.weight,      torch.bfloat16 --> Q8_0, shape = {8192, 2048}
INFO:hf-to-gguf:blk.23.attn_norm.weight,     torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:blk.23.ffn_norm.weight,      torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:output_norm.weight,          torch.bfloat16 --> F32, shape = {2048}
INFO:hf-to-gguf:output.weight,               torch.bfloat16 --> Q8_0, shape = {2048, 92544}
```



##### References

1. [llama.cpp 源码解析-- CUDA版本流程与逐算子详解](https://www.bilibili.com/video/BV1Ez4y1w7fc/?spm_id_from=333.1007.0.0&vd_source=bbc0bd6d50c9a37a05c8cb4791842c0f)
2. [Llama 2 模型结构解析](https://www.bilibili.com/video/BV12h4y1N7C8/?spm_id_from=333.337.search-card.all.click&vd_source=d99fb874fa9e85fe5793ec3fa65ab064)
3. [llama 8bit](https://www.bilibili.com/video/BV1NU9wY4ENo/?spm_id_from=333.337.search-card.all.click&vd_source=d99fb874fa9e85fe5793ec3fa65ab064)
4. [llama.cpp CUDA版本的源码解析](https://www.zhihu.com/question/589100471/answer/3276334273)