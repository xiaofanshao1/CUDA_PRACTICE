# Linear

- cublass调用linear

- cuda core INT8

- tensor core INT8

  通过IMMA指令

  ```cpp
  // Ampere INT8示例（16x16x32矩阵乘）
  asm volatile(
    "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%4,%5};"
    : "=r"(d[0]), "=r"(d[1]) : "r"(a[0]), "r"(b[0]), "r"(0), "r"(0)
  );
  
  ```

  

##### llama当中涉及linear的部分

- Linear_Q_K_V
- linear_out
- Linear_up/linear_down/linear_gate
- ...

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

##### reference

- [cuBlas API](https://docs.nvidia.com/cuda/cublas/index.html) `cublas<t>gemmEx()`

- [【CUDA进阶】Tensor Core实战教程](https://www.bilibili.com/video/BV1zPZuY5E1G?spm_id_from=333.788.videopod.episodes&vd_source=d99fb874fa9e85fe5793ec3fa65ab064&p=3)

- [【大模型部署】8bit量化算子解析](https://www.bilibili.com/video/BV1NU9wY4ENo/?spm_id_from=333.1387.homepage.video_card.click&vd_source=d99fb874fa9e85fe5793ec3fa65ab064)

  

