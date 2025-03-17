import math
import torch
import torch.nn as nn
from einops import rearrange

class SimplifiedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        # 基础参数配置（根据forward中的假设设定）
        self.hidden_size = 4096        # 隐含层维度
        self.num_heads = 32            # 注意力头数
        self.num_key_value_heads = 32  # 键值头数（与查询头数相同）
        self.head_dim = self.hidden_size // self.num_heads  # 每个头的维度 4096/32=128
        self.rope_theta = 10000        # RoPE的base参数
        self.max_position_embeddings = 2048  # 最大序列长度
        
        # 校验参数合理性
        assert (self.head_dim * self.num_heads) == self.hidden_size, "维度不匹配"
        
        # 定义线性投影层（合并Q/K/V投影）
        self.wqkv = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim
        )
        
        # 定义输出投影层
        self.wo = nn.Linear(self.hidden_size, self.hidden_size)
        
        # 初始化旋转位置编码（RoPE）
        self.rotary_emb = self.RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=self.max_position_embeddings,
            base=self.rope_theta
        )
        
    class RotaryEmbedding(nn.Module):
        """简化的旋转位置编码实现"""
        def __init__(self, dim, max_seq_len=2048, base=10000):
            super().__init__()
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            t = torch.arange(max_seq_len)
            freqs = torch.einsum('i,j->ij', t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("cos", emb.cos())
            self.register_buffer("sin", emb.sin())
            
        def forward(self, x, position_ids):
            # 根据位置索引获取对应的cos/sin值
            seq_len = position_ids.shape[-1]
            cos = self.cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
            sin = self.sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
            return cos, sin

    def forward(self, hidden_states, position_ids):
        # ------ 步骤1：线性投影得到Q/K/V ------
        batch_size, seq_len, _ = hidden_states.shape  # [1, 10, 4096]
        
        # 合并投影 [1, 10, (32+2*32)*128] = [1,10,12288]
        qkv = self.wqkv(hidden_states)
        
        # ------ 步骤2：拆分多头结构 ------
        # 重组形状 [1,10,32,3,128] (3对应Q/K/V)
        qkv = rearrange(
            qkv,
            "b s (h g d) -> b s h g d",
            h=self.num_heads,
            g=3,  # Q/K/V三组
            d=self.head_dim
        )
        
        # 分离Q/K/V [均保持 (1,10,32,128)]
        queries = qkv[:, :, :, 0, :]  # 取Q部分
        keys = qkv[:, :, :, 1, :]     # 取K部分
        values = qkv[:, :, :, 2, :]   # 取V部分
        
        # ------ 步骤3：应用旋转位置编码 ------
        cos, sin = self.rotary_emb(values, position_ids)
        
        # 对Q/K应用RoPE（简化实现）
        def apply_rotary(x, cos, sin):
            x_rot = x * cos + torch.cat([-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]], dim=-1) * sin
            return x_rot
            
        queries = apply_rotary(queries, cos, sin)  # [1,10,32,128]
        keys = apply_rotary(keys, cos, sin)        # [1,10,32,128]
        
        # ------ 步骤4：计算注意力权重 ------
        # 调整维度顺序 [1,32,10,128]
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # 计算QK^T / sqrt(d)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用因果掩码（防止看到未来信息）
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask[None, None, :, :].to(scores.device), float('-inf'))
        
        # Softmax归一化
        attn_weights = torch.softmax(scores, dim=-1)
        
        # ------ 步骤5：加权求和并重组输出 ------
        attn_output = torch.matmul(attn_weights, values)  # [1,32,10,128]
        attn_output = attn_output.transpose(1, 2)          # [1,10,32,128]
        attn_output = attn_output.reshape(batch_size, seq_len, -1)  # [1,10,4096]
        
        # 最终线性投影
        return self.wo(attn_output)

# 测试用例
if __name__ == "__main__":
    attn = SimplifiedAttention()
    x = torch.randn(1, 10, 4096)  # 输入序列
    pos_ids = torch.arange(10).unsqueeze(0)  # 位置编码
    
    output = attn(x, pos_ids)
    print(f"输入形状: {x.shape} → 输出形状: {output.shape}")  # 应该保持形状不变