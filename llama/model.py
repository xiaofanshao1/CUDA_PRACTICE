import torch
import torch.nn as nn
from torch.nn import functional as F

class InternLM2Config:
    def __init__(self):
        self.vocab_size = 92544      # 词汇表大小
        self.hidden_size = 2048      # 隐藏层维度 
        self.num_attention_heads = 16 # 注意力头数
        self.num_hidden_layers = 24   # Transformer 层数
        self.intermediate_size = 8192 # FFN 中间层维度
        self.norm_eps = 1e-6         # LayerNorm  epsilon
        
class InternLM2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Q/K/V 投影层
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self, x):
        # 简化的注意力计算 (实际需包含mask处理等)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn_output = F.scaled_dot_product_attention(q, k, v)  # 使用 PyTorch 2.0 SDPA
        return self.o_proj(attn_output)

class InternLM2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, x):
        # SwiGLU 激活函数
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class InternLM2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn = InternLM2Attention(config)
        self.post_attn_layernorm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = InternLM2MLP(config)
        
    def forward(self, x):
        # 残差连接 + LayerNorm
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attn_layernorm(x))
        return x

class InternLM2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([InternLM2Block(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        
    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# 使用示例
if __name__ == "__main__":
    config = InternLM2Config()               # 加载配置
    model = InternLM2Model(config)           # 初始化模型
    input_ids = torch.randint(0, config.vocab_size, (1, 128))  # 模拟输入（batch=1, seq=128）
    output = model(input_ids)                # 前向传播
    print(f"输入形状: {input_ids.shape} → 输出形状: {output.shape}")  # [1, 128, 2048]