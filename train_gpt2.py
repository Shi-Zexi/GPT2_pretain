import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 线性层，用于一次性计算所有头的键、查询和值投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 记录头数和嵌入维度
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 注册下三角蒙版，控制因果注意力
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # B: 批量大小，T: 序列长度，C: 嵌入维度
        # 计算 q, k, v 并按头数分割
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # 重塑并换轴，得到 (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # 计算注意力分数，并缩放
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 应用因果蒙版，屏蔽未来位置
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # 归一化注意力权重
        att = F.softmax(att, dim=-1)
        # 加权值向量并聚合
        y = att @ v  # (B, nh, T, hs)
        # 恢复维度顺序并拼接所有头
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # 输出线性投影
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 前馈层，将嵌入维度扩展 4 倍
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        # GELU 激活
        self.gelu   = nn.GELU(approximate='tanh')
        # 输出投影回原始维度
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        # 第 1 层归一化
        self.ln_1 = nn.LayerNorm(config.n_embd)
        # 自注意力模块
        self.attn = CausalSelfAttention(config)
        # 第 2 层归一化
        self.ln_2 = nn.LayerNorm(config.n_embd)
        # 前馈网络
        self.mlp  = MLP(config)

    def forward(self, x):
        # 残差连接 + 自注意力
        x = x + self.attn(self.ln_1(x))
        # 残差连接 + 前馈网络
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    # 最大序列长度
    block_size: int = 1024
    # 词表大小
    vocab_size: int = 50257
    # 堆叠层数
    n_layer: int = 12
    # 注意力头数
    n_head: int = 12
    # 嵌入维度
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        # Transformer 模块
        self.transformer = nn.ModuleDict(dict(
            # 词嵌入
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # 位置嵌入
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # 多层 Block
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            # 最终归一化
            ln_f= nn.LayerNorm(config.n_embd),
        ))
        # 语言模型头，用于预测下一个 token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_type):
        """从 Huggingface 加载预训练 GPT-2 权重"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print(f"正在加载预训练模型: {model_type}")

        # 根据模型类型设置层、头和维度
        config_args_map = {
            'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280),
            'gpt2-xl':     dict(n_layer=48, n_head=25, n_embd=1600),
        }
        config_args = config_args_map[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        # 初始化自定义 GPT
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        # 排除蒙版缓冲
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        # 加载 Huggingface 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # 排除 HF 中的蒙版缓冲
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        # 需要转置的权重名称
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"键数量不匹配: {len(sd_keys_hf)} != {len(sd_keys)}"

        # 逐个复制参数
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 转置 Conv1D 权重
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 直接复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

# -----------------------------------------------------------------------------
# 测试加载 GPT-2 模型
model = GPT.from_pretrained('gpt2')
print("模型加载完成，没有崩溃！")