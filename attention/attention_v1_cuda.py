import math
import torch
import torch.nn as nn
from einops import rearrange

class SimplifiedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 4096
        self.num_heads = 32
        self.num_key_value_heads = 32
        self.head_dim = self.hidden_size // self.num_heads
        self.rope_theta = 10000
        self.max_position_embeddings = 2048
        
        assert (self.head_dim * self.num_heads) == self.hidden_size

        self.wqkv = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        )
        
        self.wo = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.rotary_emb = self.RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=self.max_position_embeddings,
            base=self.rope_theta
        )
        
    class RotaryEmbedding(nn.Module):
        def __init__(self, dim, max_seq_len=2048, base=10000):
            super().__init__()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备感知初始化
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))  # 在目标设备创建
            t = torch.arange(max_seq_len, device=device)
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos", emb.cos())
            self.register_buffer("sin", emb.sin())
            
        def forward(self, x, position_ids):
            seq_len = position_ids.shape[-1]
            cos = self.cos[position_ids].unsqueeze(2)
            sin = self.sin[position_ids].unsqueeze(2)
            return cos, sin

    def forward(self, hidden_states, position_ids):
        batch_size, seq_len, _ = hidden_states.shape
        
        qkv = self.wqkv(hidden_states)
        
        qkv = rearrange(
            qkv,
            "b s (h g d) -> b s h g d",
            h=self.num_heads,
            g=3,
            d=self.head_dim
        )
        
        queries = qkv[:, :, :, 0, :]
        keys = qkv[:, :, :, 1, :]
        values = qkv[:, :, :, 2, :]
        
        cos, sin = self.rotary_emb(values, position_ids)
        
        def apply_rotary(x, cos, sin):
            x_rot = x * cos + torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1) * sin
            return x_rot
            
        queries = apply_rotary(queries, cos, sin)
        keys = apply_rotary(keys, cos, sin)
        
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1).bool()  # 直接在目标设备创建掩码
        scores = scores.masked_fill(mask[None, None, :, :], float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        return self.wo(attn_output)

if __name__ == "__main__":
    attn = SimplifiedAttention().cuda()  # 模型移动到CUDA
    x = torch.randn(1, 10, 4096).cuda()  # 输入数据移动到CUDA
    pos_ids = torch.arange(10).unsqueeze(0).cuda()  # 位置编码移动到CUDA
    
    output = attn(x, pos_ids)
    print(f"Input shape: {x.shape} → Output shape: {output.shape}")