## Llama related kernels

> default: internlm2-1_8b   **hidden_size**：2048  **vocab_size**：92544

#### Kernels

- concat_past_kv

- Repeat_kv
- topK


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
    


##### References

1. [llama.cpp 源码解析-- CUDA版本流程与逐算子详解](https://www.bilibili.com/video/BV1Ez4y1w7fc/?spm_id_from=333.1007.0.0&vd_source=bbc0bd6d50c9a37a05c8cb4791842c0f)
2. [Llama 2 模型结构解析](https://www.bilibili.com/video/BV12h4y1N7C8/?spm_id_from=333.337.search-card.all.click&vd_source=d99fb874fa9e85fe5793ec3fa65ab064)
3. [llama 8bit](https://www.bilibili.com/video/BV1NU9wY4ENo/?spm_id_from=333.337.search-card.all.click&vd_source=d99fb874fa9e85fe5793ec3fa65ab064)
4. [llama.cpp CUDA版本的源码解析](https://www.zhihu.com/question/589100471/answer/3276334273)
5. [LLM-engineer](https://github.com/RussWong/LLM-engineering)
6. [使用CUDA解决和加速TopK问题](https://www.bilibili.com/video/BV1nF411D7Fh/?spm_id_from=333.337.search-card.all.click&vd_source=d99fb874fa9e85fe5793ec3fa65ab064)
7. llama.cpp源码解读--cgraph计算图与sched后端调度机制详解 https://zhuanlan.zhihu.com/p/1893801096918585567
8. ggml   https://zhuanlan.zhihu.com/p/19968327329