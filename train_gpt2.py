import math
import os
import time
from dataclasses import dataclass
import inspect
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
        self.c_proj.NANOGPT_SCALE_INIT = 1
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
        # 引入的高性能 Flash Attention 接口
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
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
        self.c_proj.NANOGPT_SCALE_INIT = 1

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

        # 添加了 embedding 层与输出层之间的权重共享机制（weight tying）
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):  #线性层初始化
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):  #嵌入层初始化
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
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
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

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

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # 获取所有需要梯度更新的参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # 构建优化器参数组
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},  # 应用权重衰减
            {'params': nodecay_params, 'weight_decay': 0.0}           # 不应用权重衰减
        ]

        # 打印参数统计信息
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # 检查是否支持 fused AdamW 加速版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")

        # 创建 AdamW 优化器，若支持则启用 fused 加速
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused
        )
        return optimizer

# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B  # 批大小
        self.T = T  # 序列长度
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {'train', 'val'}

        # 获取分片文件名
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()
    def reset(self):
        # state, at shard zero初始化
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # 初始化当前读取位置
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        # 获取当前批次的 token 段
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        # 构造输入 x 和目标 y
        x = (buf[:-1]).view(B, T) 
        y = (buf[1:]).view(B, T) 
        # 更新当前位置，如果越界则重置
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# simple launch:单机运行命令
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:多卡 DDP 模式运行命令（例如使用 8 个 GPU）
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# 运行训练循环
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# 建立DDP
# 若存在 RANK，则视为 DDP 模式运行
ddp = int(os.environ.get('RANK', -1)) != -1 # 是否处于 DDP 模式
if ddp:
    # 使用DDP需要CUDA, 根据等级适当设置设备
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # 这个进程将做日志记录、检查点等。
else:
    # vanilla, non-DDP
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # 自动检测当前设备
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# 初始化轻量数据加载器
total_batch_size = 524288 
B = 64   #micro batch size，即单个step实际加载的样本数量
T = 1024   # 每个样本的序列长度（token数）

assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
torch.set_float32_matmul_precision('high') 

# create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)                     # 将模型移动到自动检测的设备上
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073
# 自定义学习率调度函数，传入当前迭代次数 it
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) 
    return min_lr + coeff * (max_lr - min_lr)


# 初始化优化器，这里使用AdamW优化器
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    # 每隔 100 步执行一次验证
    if step % 100 == 0:
        model.eval()    # 切换为评估模式
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20        # 评估20个batch的平均损失
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()    # 累加验证损失
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")

    # 训练过程
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        
        loss = loss / grad_accum_steps  # 缩放 loss
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()   # 累积梯度
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # 获取当前步的学习率（含warmup和余弦衰减）
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr # 动态设置当前步的学习率
    optimizer.step()
    torch.cuda.synchronize() # 等待GPU完成工作
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    # 打印日志
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
if ddp:
    destroy_process_group()

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
