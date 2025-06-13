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

def forward(self, idx):
    # 输入idx的形状为(B, T)，B是批量大小，T是序列长度
    B, T = idx.size()
    # 保证输入序列长度不超过模型的最大块大小
    assert T <= self.config.block_size, f"输入序列长度为 {T}，超过了模型支持的最大长度 {self.config.block_size}"
    # 生成位置索引（形状为 (T,)），用于获取位置嵌入
    pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
    # 获取位置嵌入，形状为 (T, n_embd);词嵌入，形状为 (B, T, n_embd)
    pos_emb = self.transformer.wpe(pos)
    tok_emb = self.transformer.wte(idx)
    # 将词嵌入与位置嵌入相加，得到最终的输入嵌入 (B, T, n_embd)
    x = tok_emb + pos_emb
    # 依次通过每个Transformer的Block模块进行前向传播
    for block in self.transformer.h:
        x = block(x)
    # 通过最后一层LayerNorm进行归一化
    x = self.transformer.ln_f(x)
    # 使用语言模型头进行分类，得到每个位置上的词表概率分布 (B, T, vocab_size)
    logits = self.lm_head(x)
    return logits

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

# 自动检测运行设备（支持CUDA、MPS、CPU）
device = "cpu"  # 默认设备
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# 强制指定设备为CPU（用于测试）
device = "cpu"  # OVERRIDE

# 加载tokenizer
import tiktoken
enc = tiktoken.get_encoding('gpt2')
# 从input.txt文件中读取文本内容，并截取前1000个字符
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]  # 只取前1000个字符作为样例
# 编码为token序列
tokens = enc.encode(text)
# 构造形状为 (B=4, T=32) 的批次输入
B, T = 4, 32  # 批量大小4，每个序列长度32
buf = torch.tensor(tokens[:B*T + 1])  # 取出B*T+1个token（用于构造输入和标签）
x = buf[:-1].view(B, T)  # 输入x：从第0到倒数第二个token，形状为(B, T)
y = buf[1:].view(B, T)   # 标签y：从第1个到最后一个token，形状也为(B, T)

model = GPT(GPTConfig())             # 使用默认配置初始化GPT模型
model.to(device)                     # 将模型移动到自动检测的设备上

logits = model(x)  # 输出logits，形状为 (B, T, vocab_size)

print(logits.shape)  # 打印logits的维度确认输出是否正确
import sys; sys.exit(0)  

# 文本生成逻辑（已存在但未运行）
model.eval()  # 设置模型为评估模式
num_return_sequences = 5  # 要生成的序列数量
max_length = 30           # 每个生成序列的最大长度

tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)              # (8,)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)                                          

# 设置随机种子
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# 开始生成过程
while x.size(1) < max_length:
    # 前向传播，获取logits
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
    # 取出当前序列最后一个位置的logits
    logits = logits[:, -1, :]  # (B, vocab_size)
    # softmax转为概率
    probs = F.softmax(logits, dim=-1)
    # 进行Top-k采样（k=50）
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    ix = torch.multinomial(topk_probs, 1)  # (B, 1)
    # 从索引中查出采样得到的token
    xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
    # 拼接到原序列后面
    x = torch.cat((x, xcol), dim=1)  # 更新x (B, T+1)

# 打印生成的文本结果
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
