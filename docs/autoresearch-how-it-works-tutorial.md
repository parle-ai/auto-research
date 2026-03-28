# Autoresearch 是怎么工作的？— 从原理到代码的完整教程

> Autoresearch = **让 AI Agent 自主做 LLM 预训练研究**
>
> Karpathy 出品，一句话：**你睡觉，AI 帮你跑 100 个实验，早上起来看结果**

---

## 一、先看结果：Autoresearch 输入什么，输出什么

```
输入:
  - 一块 NVIDIA GPU (H100)
  - 一个 AI Agent (Claude / Codex / ...)
  - 一句话: "Hi have a look at program.md and let's kick off a new experiment!"

输出 (第二天早上):
  autoresearch/<tag>/
  ├── train.py              <-- 被 Agent 反复修改、优化过的训练代码
  ├── results.tsv           <-- 所有实验的记录表
  └── git log               <-- 每个实验对应一个 commit
```

**results.tsv 长这样：**

```
commit    val_bpb     memory_gb   status    description
a1b2c3d   0.997900    44.0        keep      baseline
b2c3d4e   0.993200    44.2        keep      increase LR to 0.04
c3d4e5f   1.005000    44.0        discard   switch to GeLU activation
d4e5f6g   0.000000    0.0         crash     double model width (OOM)
e5f6g7h   0.990100    44.1        keep      increase depth to 10
f6g7h8i   0.991500    45.3        discard   add dropout 0.1
...
```

**整个过程不需要人参与。** 你给 Agent 一个起始代码和一套规则，它就会自己：假设 -> 修改代码 -> 训练 -> 评估 -> 保留或丢弃 -> 下一个假设。无限循环，直到你手动停止。

---

## 二、核心理念：最少的代码，最简的指标，一夜的自主

Autoresearch 的设计哲学可以用三句话概括：

```
+----------------------------------------------------------------------+
|                                                                      |
|  1. 极简代码: 整个项目只有 3 个文件真正重要                              |
|     prepare.py (固定不动) + train.py (Agent 修改) + program.md (人写)  |
|                                                                      |
|  2. 单一指标: val_bpb (验证集 Bits Per Byte)                           |
|     越低越好，跟 vocab size 无关，公平比较任何架构改动                    |
|                                                                      |
|  3. 一夜自主: 每个实验固定 5 分钟，一小时跑 12 个                        |
|     一觉醒来 ~100 个实验，全部有据可查                                   |
|                                                                      |
+----------------------------------------------------------------------+
```

这个设计的精妙之处在于：

- **固定时间预算**（而非固定步数）：不管 Agent 怎么改模型大小、batch size、架构，训练时间永远是 5 分钟。这让所有实验天然可比。
- **单文件约束**：Agent 只能改 `train.py` 一个文件。这既降低了搜索空间的复杂度，也让 diff 容易审阅。
- **Git 即数据库**：每个实验一个 commit，好的实验推进分支，差的实验 reset 回去。Git 历史就是实验记录。

---

## 三、三个文件各是什么

```
autoresearch/
├── prepare.py      <-- "实验室基础设施"：数据、分词器、评估函数 (只读)
├── train.py        <-- "实验台"：模型、优化器、训练循环 (Agent 的战场)
├── program.md      <-- "实验手册"：告诉 Agent 怎么做研究 (人写的 prompt)
└── pyproject.toml  <-- 依赖声明 (锁定，不可改)
```

### 3.1 prepare.py — 实验室基础设施

这个文件做两件事：**一次性数据准备** + **运行时工具**。

**一次性准备（`python prepare.py`）：**

| 步骤 | 做什么 | 存到哪 |
|------|--------|--------|
| 下载数据 | 从 HuggingFace 下载 parquet 分片 | `~/.cache/autoresearch/data/` |
| 训练分词器 | 用 rustbpe 训练 BPE 分词器 (vocab=8192) | `~/.cache/autoresearch/tokenizer/` |

**固定常量（Agent 不能改）：**

```python
MAX_SEQ_LEN = 2048       # 上下文长度
TIME_BUDGET = 300        # 训练时间预算：5 分钟 (300 秒)
EVAL_TOKENS = 40 * 524288  # 验证评估用的 token 数量 (~20M)
VOCAB_SIZE = 8192        # 词汇表大小
```

**运行时工具（供 train.py 导入）：**

```
prepare.py 提供给 train.py 的接口
+---------------------------------------------+
| Tokenizer         - 分词器封装               |
| make_dataloader()  - 数据加载器              |
| evaluate_bpb()     - 评估函数 (不可修改!)    |
| MAX_SEQ_LEN        - 固定序列长度            |
| TIME_BUDGET        - 固定时间预算            |
+---------------------------------------------+
```

**最重要的是 `evaluate_bpb()` 函数** — 这是唯一的评估指标，Agent 不能碰它。稍后第六节详细讲。

### 3.2 train.py — Agent 的战场

这是 Agent 唯一可以修改的文件。它包含一个完整的 GPT 预训练系统：

```
train.py 的结构
+--------------------------------------------------+
|                                                  |
|  [模型定义]                                       |
|  ├── GPTConfig       - 模型配置                   |
|  ├── CausalSelfAttention - 因果自注意力           |
|  ├── MLP             - 前馈网络 (ReluSquared)     |
|  ├── Block           - Transformer Block          |
|  └── GPT             - 完整模型                   |
|                                                  |
|  [优化器]                                         |
|  ├── MuonAdamW       - 混合优化器                 |
|  ├── adamw_step_fused  - AdamW (编译优化)         |
|  └── muon_step_fused   - Muon (编译优化)          |
|                                                  |
|  [超参数]                                         |
|  ├── DEPTH = 8                                   |
|  ├── TOTAL_BATCH_SIZE = 2**19                    |
|  ├── 各种学习率                                   |
|  └── ...                                         |
|                                                  |
|  [训练循环]                                       |
|  ├── while True: 按时间预算训练                   |
|  ├── 梯度累积                                     |
|  ├── 学习率调度                                   |
|  └── 最终评估 + 打印 val_bpb                      |
|                                                  |
+--------------------------------------------------+
```

**Agent 可以做什么？**
- 改模型架构（加层、改 attention、换激活函数...）
- 改优化器参数（学习率、momentum、weight decay...）
- 改模型大小（depth、width、head 数量...）
- 改 batch size
- 改训练循环的细节
- 任何不违反规则的修改

### 3.3 program.md — 人写的 Agent 指令

这是整个系统最有趣的部分。`program.md` 本质上是一个给 AI Agent 的 "研究员操作手册"。它用自然语言定义了：

```
program.md 定义的内容
+--------------------------------------------------+
|                                                  |
|  [Setup]   怎么初始化一次实验                      |
|  - 创建分支、读代码、初始化 results.tsv            |
|                                                  |
|  [Rules]   什么能做、什么不能做                    |
|  - CAN: 修改 train.py                            |
|  - CANNOT: 修改 prepare.py, 装包, 改评估          |
|                                                  |
|  [Loop]    实验循环怎么跑                          |
|  - 修改 -> commit -> 训练 -> 评估 -> keep/discard |
|                                                  |
|  [Spirit]  设计哲学                               |
|  - 简单优先、永不停止、超时处理                    |
|                                                  |
+--------------------------------------------------+
```

Karpathy 在 README 中说："你不是在写 Python 来做研究，而是在写 Markdown 来 *编程* 你的自主研究组织。" 这是一个很深刻的观察 — `program.md` 就是 AI 时代的 "研究管理代码"。

---

## 四、Agent 循环详解

这是 autoresearch 的核心机制。让我们完整地看 `program.md` 中定义的实验循环。

### 4.1 初始化（Setup）

在开始循环之前，Agent 需要完成一次性设置：

```
Setup 流程
+--------------------------------------------------+
|                                                  |
|  1. 商定一个 run tag (例如 "mar5")                |
|  2. 创建分支: git checkout -b autoresearch/mar5   |
|  3. 读取代码: prepare.py, train.py, README.md     |
|  4. 确认数据存在: ~/.cache/autoresearch/          |
|  5. 初始化 results.tsv (只有表头)                 |
|  6. 确认就绪，开始循环                             |
|                                                  |
+--------------------------------------------------+
```

### 4.2 实验循环（The Loop）

以下是 `program.md` 中定义的完整循环，逐字引用：

```
LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune train.py with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: uv run train.py > run.log 2>&1
5. Read out the results: grep "^val_bpb:\|^peak_vram_mb:" run.log
6. If the grep output is empty, the run crashed.
   Run tail -n 50 run.log to read the Python stack trace and attempt a fix.
7. Record the results in the tsv
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started
```

**用图来表示：**

```
                         +-------------------+
                         |   查看 Git 状态    |
                         +--------+----------+
                                  |
                                  v
                         +-------------------+
                         |  提出假设/想法     |
                         |  修改 train.py     |
                         +--------+----------+
                                  |
                                  v
                         +-------------------+
                         |   git commit      |
                         +--------+----------+
                                  |
                                  v
                         +-------------------+
                         |  uv run train.py  |
                         |  > run.log 2>&1   |
                         |  (约 5-6 分钟)     |
                         +--------+----------+
                                  |
                                  v
                         +-------------------+
                         |  读取结果          |
                         |  grep val_bpb     |
                         +--------+----------+
                                  |
                          +-------+-------+
                          |               |
                    结果为空?           有结果?
                          |               |
                          v               v
                   +------------+   +------------+
                   | 崩溃处理    |   | 比较 val_bpb|
                   | tail 看报错 |   +------+-----+
                   | 尝试修复    |          |
                   +------+-----+    +-----+-----+
                          |          |           |
                          v       更好?       更差?
                   +------------+    |           |
                   | 记录 crash |    v           v
                   | 回退代码   | +--------+ +--------+
                   +------+-----+ | keep   | |discard |
                          |       | 保留    | | 回退   |
                          |       | commit | | git    |
                          |       +---+----+ | reset  |
                          |           |      +---+----+
                          |           |          |
                          +-----+-----+----------+
                                |
                                v
                       +------------------+
                       | 记录到 results.tsv|
                       +--------+---------+
                                |
                                v
                         回到循环开头
                      (永远不停，直到人
                       手动中断)
```

### 4.3 关键规则

`program.md` 中有几条关键规则值得强调：

**1. 永不停止**

```
NEVER STOP: Once the experiment loop has begun, do NOT pause to ask
the human if you should continue. The human might be asleep.
You are autonomous. The loop runs until the human interrupts you, period.
```

这是最核心的设计：Agent 必须自主运行，不能等人确认。

**2. 简单优先**

```
Simplicity criterion: All else being equal, simpler is better.
A 0.001 val_bpb improvement that adds 20 lines of hacky code?
Probably not worth it.
A 0.001 val_bpb improvement from deleting code? Definitely keep.
```

这防止 Agent 把代码搞得越来越复杂。

**3. 超时处理**

```
Timeout: Each experiment should take ~5 minutes total.
If a run exceeds 10 minutes, kill it and treat it as a failure.
```

**4. 输出重定向**

```
uv run train.py > run.log 2>&1
(redirect everything -- do NOT use tee or let output flood your context)
```

这是一个实用细节：如果训练输出直接进 Agent 的上下文窗口，几千行日志会迅速耗尽 context。所以必须重定向到文件，然后用 grep 精确提取需要的数字。

---

## 五、train.py 架构详解

### 5.1 模型架构：一个精简的 GPT

train.py 实现了一个从 [nanochat](https://github.com/karpathy/nanochat) 简化而来的 GPT 模型。我们逐层来看。

**模型配置：**

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
```

**模型尺寸由 `DEPTH` 一个变量控制：**

```python
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # 每个 attention head 的维度
DEPTH = 8               # transformer 层数 (这是主要的"大小旋钮")

def build_model_config(depth):
    base_dim = depth * ASPECT_RATIO       # 8 * 64 = 512
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM  # 对齐到 128 = 512
    num_heads = model_dim // HEAD_DIM     # 512 / 128 = 4
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )
```

这个设计很聪明：只改 `DEPTH` 一个数，模型的宽度、head 数量都会自动调整，保持合理的比例。

**整体前向传播：**

```python
def forward(self, idx, targets=None, reduction='mean'):
    B, T = idx.size()
    cos_sin = self.cos[:, :T], self.sin[:, :T]

    x = self.transformer.wte(idx)      # token embedding
    x = norm(x)                         # RMS norm
    x0 = x                              # 保存初始表示 (用于残差连接)

    for i, block in enumerate(self.transformer.h):
        # DaViT 风格的双残差连接
        x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
        ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
        x = block(x, ve, cos_sin, self.window_sizes[i])

    x = norm(x)

    # Logit soft-capping (Gemma 2 风格)
    softcap = 15
    logits = self.lm_head(x)
    logits = logits.float()
    logits = softcap * torch.tanh(logits / softcap)

    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                               ignore_index=-1, reduction=reduction)
        return loss
    return logits
```

**模型中值得注意的几个技巧：**

```
+-------------------------------------------------------------------+
| 技巧                        | 代码位置               | 说明        |
+-------------------------------------------------------------------+
| RMS Norm (非 LayerNorm)     | norm(x)               | 更快更稳定   |
| RoPE 旋转位置编码            | apply_rotary_emb()    | 标准做法     |
| 滑动窗口注意力              | window_pattern="SSSL" | S=半窗口     |
| Value Embedding (ResFormer) | value_embeds + ve_gate| 隔层加 VE   |
| 双残差连接                  | resid_lambdas/x0      | x0 shortcut |
| ReluSquared 激活             | F.relu(x).square()   | 替代 GELU   |
| QK Norm                    | norm(q), norm(k)      | 稳定注意力   |
| Logit Soft-capping          | tanh(logits/15)*15   | 防止爆炸     |
| Flash Attention 3           | fa3.flash_attn_func  | 极速注意力   |
+-------------------------------------------------------------------+
```

### 5.2 注意力机制详解

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

        # Value Embedding: 隔层添加，用可学习的门控
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) \
            if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual: 把 token embedding 直接混入 value
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        # 旋转位置编码
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK Norm

        # Flash Attention 3 (带可选的滑动窗口)
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y
```

**滑动窗口模式：**

```
window_pattern = "SSSL" 表示：

层 0 (S): 窗口 = seq_len / 2 = 1024    |####........| 只看一半
层 1 (S): 窗口 = seq_len / 2 = 1024    |####........| 只看一半
层 2 (S): 窗口 = seq_len / 2 = 1024    |####........| 只看一半
层 3 (L): 窗口 = seq_len = 2048        |############| 看全部
层 4 (S): 窗口 = seq_len / 2 = 1024    |####........| 只看一半
层 5 (S): 窗口 = seq_len / 2 = 1024    |####........| 只看一半
层 6 (S): 窗口 = seq_len / 2 = 1024    |####........| 只看一半
层 7 (L): 窗口 = seq_len = 2048        |############| 最后一层永远全窗口

好处: 大部分层只算一半的注意力，速度更快，同时通过 L 层保持全局信息流
```

### 5.3 MLP：ReluSquared

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()   # <-- ReluSquared, 不是 GELU/SwiGLU
        x = self.c_proj(x)
        return x
```

ReluSquared 的好处是简单、快速、稀疏性好。跟 SwiGLU 相比少一个线性层。

### 5.4 优化器：Muon + AdamW 混合

这是 train.py 中最复杂的部分。不同类型的参数用不同的优化器：

```
参数分组策略
+----------------------------------------------------------------+
| 参数类型              | 优化器   | 学习率          | 说明       |
+----------------------------------------------------------------+
| lm_head (输出投影)    | AdamW   | 0.004 * scale  | 低 LR      |
| wte (词嵌入)          | AdamW   | 0.6 * scale    | 高 LR      |
| value_embeds          | AdamW   | 0.6 * scale    | 高 LR      |
| resid_lambdas         | AdamW   | 0.005          | 极低 LR    |
| x0_lambdas            | AdamW   | 0.5            | 中 LR      |
| 矩阵参数 (按形状分组) | Muon    | 0.04           | 正交化     |
+----------------------------------------------------------------+

其中 scale = (model_dim / 768) ^ -0.5   (muP 风格的学习率缩放)
```

**Muon 优化器是什么？**

Muon（**Mu**ltiplicative **O**rthogo**n**alization）是一个专门为矩阵参数设计的优化器。它的核心思想是：

```
Muon 的三步：

1. Nesterov 动量
   momentum_buffer = 0.95 * momentum_buffer + 0.05 * gradient
   g = gradient + 0.95 * momentum_buffer

2. Polar Express 正交化 (Newton-Schulz 迭代)
   把梯度投影到正交矩阵空间
   X = g / ||g||
   for 5 iterations:
       A = X^T @ X
       X = a*X + X @ (b*A + c*A^2)    <-- 多项式近似 polar decomposition
   g = X

3. NorMuon 方差归约
   v_mean = 滑动平均(g^2 的逐行/逐列均值)
   g = g * rsqrt(v_mean)              <-- 类似 Adam 的自适应缩放

4. Cautious 权重衰减 + 更新
   mask = (g * params) >= 0            <-- 只衰减"方向一致"的参数
   params -= lr * g + lr * wd * params * mask
```

**为什么用 Muon 而不全用 AdamW？**

矩阵参数（线性层的权重）的梯度本身是矩阵。Muon 利用矩阵的几何结构（正交性），让更新方向更高效。经验上，对于中大型模型，Muon 比 AdamW 在相同步数下收敛更快。

而 Embedding 和标量参数不是矩阵，所以用传统的 AdamW。

### 5.5 训练循环

```python
t_start_training = time.time()
step = 0

while True:
    t0 = time.time()

    # ---- 梯度累积 ----
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:              # bf16 混合精度
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps  # 缩放梯度
        loss.backward()
        x, y, epoch = next(train_loader)  # 预取下一批

    # ---- 学习率调度 ----
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm

    # ---- 参数更新 ----
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # ---- 快速失败 ----
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    # ---- 时间到了就停 ----
    step += 1
    if step > 10 and total_training_time >= TIME_BUDGET:
        break
```

**学习率调度（基于时间而非步数）：**

```
学习率随训练进度的变化:

LR
|
|  ***********
|  *         *                    warmup_ratio = 0.0  (不 warmup)
|  *          *                   warmdown_ratio = 0.5 (后一半线性衰减)
|  *           *                  final_lr_frac = 0.0  (衰减到 0)
|  *            *
|  *             *
|  *              *
+--*---------------*---> progress (0.0 到 1.0)
   0             0.5          1.0
   |<- 恒定 LR ->|<- 线性衰减 ->|
```

```python
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:       # 默认 0.0, 不 warmup
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:  # 前一半恒定
        return 1.0
    else:                              # 后一半线性衰减到 0
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC
```

**注意一个细节：`step > 10` 的检查。** 前 10 步不计入时间预算，因为这段时间 PyTorch 在做 `torch.compile` 编译。如果不排除编译时间，不同架构的实际训练时间会不公平。

### 5.6 训练输出

训练结束后，脚本打印一个固定格式的摘要：

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Agent 用 `grep "^val_bpb:\|^peak_vram_mb:" run.log` 提取关键数字，避免读取完整日志。

---

## 六、评估指标：BPB (Bits Per Byte)

### 6.1 为什么不用 Loss？

普通的交叉熵 loss 依赖于 vocab size。如果 Agent 把 vocab size 从 8192 改成 4096，loss 数值会变，但模型并没有变好或变差。所以需要一个 vocab size 无关的指标。

### 6.2 BPB 是什么？

**Bits Per Byte (BPB)** = 模型编码每个字节平均需要多少 bit。

```
直觉理解：

  "Hello" 的 UTF-8 编码 = 5 个字节 [72, 101, 108, 108, 111]

  如果 BPB = 1.0，意味着模型平均用 1 bit 就能"预测"一个字节
  如果 BPB = 8.0，意味着模型完全随机猜（跟没有模型一样）

  BPB 越低 → 模型对语言的理解越好
```

### 6.3 计算方式

```python
@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    token_bytes = get_token_bytes(device="cuda")  # 每个 token 对应多少字节
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)

    total_nats = 0.0
    total_bytes = 0

    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)  # 逐 token loss
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]       # 每个 target token 的字节数
        mask = nbytes > 0                   # 排除 special tokens
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()

    return total_nats / (math.log(2) * total_bytes)
    #      ^^^^^^^^                    ^^^^^^^^^^^
    #      交叉熵总和(nats)             总字节数
    #                  / log(2) 把 nats 转成 bits
```

**关键步骤分解：**

```
BPB 计算流程
+------------------------------------------------------------------+
|                                                                  |
|  1. 对验证集的每个 token，计算交叉熵 loss (单位: nats)             |
|                                                                  |
|  2. 查表得到每个 target token 对应多少 UTF-8 字节                  |
|     例如: "hello" 的 token → 5 bytes                              |
|           "你好"  的 token → 6 bytes (中文每字 3 bytes)            |
|                                                                  |
|  3. 排除 special tokens (字节数 = 0 的 token)                     |
|                                                                  |
|  4. BPB = sum(nats) / (log(2) * sum(bytes))                      |
|                                                                  |
|     nats → bits: 除以 log(2)                                     |
|     per byte: 除以总字节数                                        |
|                                                                  |
+------------------------------------------------------------------+
```

**为什么 BPB 是公平的？**

| 场景 | 普通 Loss | BPB |
|------|-----------|-----|
| vocab 8192 | 2.34 | 0.997 |
| vocab 4096 | 2.12 (假的"更好") | 0.997 (真实的"一样") |
| vocab 16384 | 2.56 (假的"更差") | 0.997 (真实的"一样") |

因为 BPB 回到了字节级别，消除了分词方案带来的差异。

---

## 七、状态管理：Git 即数据库

### 7.1 分支策略

```
master
  |
  +-- autoresearch/mar5         <-- 一次实验 session 的分支
  |     |
  |     +-- commit: baseline    (val_bpb = 0.9979)
  |     +-- commit: LR 0.04    (val_bpb = 0.9932)  <- keep, 分支前进
  |     +-- commit: GeLU        (val_bpb = 1.005)   <- discard, git reset
  |     +-- commit: depth 10   (val_bpb = 0.9901)  <- keep, 分支前进
  |     +-- commit: dropout     (val_bpb = 0.9915)  <- discard, git reset
  |     ...
  |
  +-- autoresearch/mar5-gpu1    <-- 可以多 GPU 并行跑不同 session
```

### 7.2 Keep vs Discard 的 Git 操作

```
情况 1: 实验成功 (val_bpb 降低)

  before:  A --- B --- C (HEAD)
                       ^ 新 commit (改了 LR)
  结果: val_bpb 降低了!
  操作: 什么都不做，C 就是新的起点
  after:   A --- B --- C (HEAD)   <-- 下次实验从这里开始

情况 2: 实验失败 (val_bpb 没降)

  before:  A --- B --- C (HEAD)
                       ^ 新 commit (试了 GeLU)
  结果: val_bpb 升高了
  操作: git reset 回 B
  after:   A --- B (HEAD)         <-- 下次实验从 B 重新开始
                  \--- C           <-- C 变成悬挂 commit (不在分支上)
```

### 7.3 results.tsv 日志

`results.tsv` 是不受 git 跟踪的（不 commit），因为它是所有实验的完整记录，包括被丢弃的。

```
results.tsv 格式 (Tab 分隔)
+----------+-----------+-----------+----------+----------------------------+
| commit   | val_bpb   | memory_gb | status   | description                |
+----------+-----------+-----------+----------+----------------------------+
| a1b2c3d  | 0.997900  | 44.0      | keep     | baseline                   |
| b2c3d4e  | 0.993200  | 44.2      | keep     | increase LR to 0.04        |
| c3d4e5f  | 1.005000  | 44.0      | discard  | switch to GeLU activation  |
| d4e5f6g  | 0.000000  | 0.0       | crash    | double model width (OOM)   |
| e5f6g7h  | 0.990100  | 44.1      | keep     | increase depth to 10       |
+----------+-----------+-----------+----------+----------------------------+
```

**为什么用 TSV 不用 JSON/CSV？**

- TSV: 人类可读、`cat` 一下就能看、不需要解析器
- 不用 CSV: 因为 description 里可能有逗号
- 不用 JSON: 追加一行比修改 JSON 数组简单得多

**为什么 results.tsv 不被 git 跟踪？**

因为 `git reset` 会回退文件内容。如果 results.tsv 在 git 里，每次 discard 操作都会丢失那条 discard 记录。保持 untracked，它就是一个只追加的日志。

---

## 八、一个完整的实验周期 Walkthrough

让我们跟踪一个完整的 "一夜自主研究" 过程。

### 8.1 晚上 10:00 — 启动

```bash
# 人类操作:
cd autoresearch
# 启动 Claude Code (或其他 AI Agent)
# 输入: "Hi have a look at program.md and let's kick off a new experiment!"
```

Agent 读取 `program.md`，执行 Setup：

```
Agent: "I'll use the tag 'mar28'. Let me set up."

$ git checkout -b autoresearch/mar28
$ cat prepare.py train.py README.md   # 读取代码
$ ls ~/.cache/autoresearch/           # 确认数据存在

# 创建 results.tsv
$ echo -e "commit\tval_bpb\tmemory_gb\tstatus\tdescription" > results.tsv

Agent: "Setup complete. Starting with baseline run."
```

### 8.2 晚上 10:01 — 实验 #1: Baseline

```
Agent: "First run — establishing baseline with default settings."

$ git add train.py && git commit -m "baseline"
$ uv run train.py > run.log 2>&1    # 等待 ~5 分钟
$ grep "^val_bpb:\|^peak_vram_mb:" run.log

val_bpb:          0.997900
peak_vram_mb:     45060.2

Agent 记录: a1b2c3d  0.997900  44.0  keep  baseline
```

### 8.3 晚上 10:07 — 实验 #2: 提高学习率

```
Agent 思考: "Baseline val_bpb = 0.9979. Let me try increasing the
matrix learning rate from 0.04 to 0.06."

# Agent 修改 train.py:
# MATRIX_LR = 0.04  →  MATRIX_LR = 0.06

$ git add train.py && git commit -m "increase matrix LR to 0.06"
$ uv run train.py > run.log 2>&1
$ grep "^val_bpb:\|^peak_vram_mb:" run.log

val_bpb:          0.993200
peak_vram_mb:     45060.2

Agent: "0.9932 < 0.9979 — improvement! Keeping this change."
Agent 记录: b2c3d4e  0.993200  44.0  keep  increase matrix LR to 0.06
```

### 8.4 晚上 10:13 — 实验 #3: 换激活函数 (失败)

```
Agent 思考: "LR change worked. Now let me try replacing ReluSquared
with GELU in the MLP."

# Agent 修改 train.py:
# x = F.relu(x).square()  →  x = F.gelu(x)

$ git add train.py && git commit -m "switch MLP to GELU"
$ uv run train.py > run.log 2>&1
$ grep "^val_bpb:\|^peak_vram_mb:" run.log

val_bpb:          1.005000
peak_vram_mb:     45060.2

Agent: "1.005 > 0.9932 — worse. Discarding."
$ git reset --hard HEAD~1    # 回退到上一个 keep 的 commit
Agent 记录: c3d4e5f  1.005000  44.0  discard  switch MLP to GELU
```

### 8.5 晚上 10:19 — 实验 #4: 加深模型 (崩溃)

```
Agent 思考: "GELU didn't work. Let me try doubling the depth from 8 to 16."

# Agent 修改 train.py:
# DEPTH = 8  →  DEPTH = 16

$ git add train.py && git commit -m "double depth to 16"
$ uv run train.py > run.log 2>&1
$ grep "^val_bpb:\|^peak_vram_mb:" run.log

(空输出 — 崩溃了)

$ tail -n 50 run.log
torch.cuda.OutOfMemoryError: CUDA out of memory.

Agent: "OOM. Depth 16 is too large. Reverting."
$ git reset --hard HEAD~1
Agent 记录: d4e5f6g  0.000000  0.0  crash  double depth to 16 (OOM)
```

### 8.6 晚上 10:20 — 实验 #5: 温和地加深

```
Agent 思考: "16 OOMed. Let me try depth 10 instead."

# DEPTH = 8  →  DEPTH = 10

$ git add train.py && git commit -m "increase depth to 10"
$ uv run train.py > run.log 2>&1
$ grep "^val_bpb:\|^peak_vram_mb:" run.log

val_bpb:          0.990100
peak_vram_mb:     48000.5

Agent: "0.9901 < 0.9932 — improvement! Memory increased but acceptable."
Agent 记录: e5f6g7h  0.990100  46.9  keep  increase depth to 10
```

### 8.7 ... 持续到早上

```
实验 #6:  减少 warmdown ratio     → 0.9895 keep
实验 #7:  增大 batch size          → 0.9910 discard
实验 #8:  加 weight decay 0.3     → 0.9888 keep
实验 #9:  换 window pattern "SL"  → 0.9890 discard
实验 #10: 调 embedding LR         → 0.9880 keep
...
实验 #95: 微调 polar express 系数  → 0.9701 keep
实验 #96: 加第二种注意力模式       → 0.9720 discard
实验 #97: 简化：删除 ve_gate       → 0.9705 discard
实验 #98: 调 aspect ratio 到 72   → 0.9695 keep
```

### 8.8 早上 8:00 — 人类查看结果

```bash
# 人类操作:
$ cat results.tsv | column -t -s $'\t'

commit    val_bpb   memory_gb  status   description
a1b2c3d   0.997900  44.0       keep     baseline
b2c3d4e   0.993200  44.0       keep     increase matrix LR to 0.06
c3d4e5f   1.005000  44.0       discard  switch MLP to GELU
d4e5f6g   0.000000  0.0        crash    double depth to 16 (OOM)
e5f6g7h   0.990100  46.9       keep     increase depth to 10
...
(98 行)

# 最佳结果:
$ sort -t$'\t' -k2 -n results.tsv | head -3

f8g9h0i   0.969500  47.2       keep     adjust aspect ratio to 72
g9h0i1j   0.970100  47.2       keep     tune polar express coefficients
h0i1j2k   0.972000  46.8       keep     combine depth 12 + LR schedule

# 查看最终代码的完整 diff:
$ git diff master...HEAD

# 查看实验历史:
$ git log --oneline
```

**一夜之间，val_bpb 从 0.9979 降到了 0.9695 — 一个 ~2.8% 的改进，全自动完成。**

---

## 九、快速上手

### 9.1 环境要求

| 要求 | 说明 |
|------|------|
| GPU | NVIDIA GPU (推荐 H100，至少需要 Flash Attention 3 支持) |
| Python | 3.10+ |
| 包管理器 | [uv](https://docs.astral.sh/uv/) |
| AI Agent | Claude Code / Codex / 任何能读写文件+跑命令的 Agent |

### 9.2 安装步骤

```bash
# 1. 安装 uv (如果没有)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆仓库
git clone https://github.com/karpathy/autoresearch.git
cd autoresearch

# 3. 安装依赖
uv sync

# 4. 下载数据 + 训练分词器 (一次性, ~2 分钟)
uv run prepare.py

# 5. 手动跑一次验证环境
uv run train.py
```

### 9.3 启动 Agent

```bash
# 启动你的 AI Agent (以 Claude Code 为例)
# 进入 autoresearch 目录，然后:

# 输入 prompt:
"Hi have a look at program.md and let's kick off a new experiment!"
```

Agent 会自动：
1. 读取 `program.md`
2. 执行 Setup (创建分支、初始化 tsv)
3. 跑 baseline
4. 开始无限实验循环

### 9.4 依赖列表

来自 `pyproject.toml`：

```toml
dependencies = [
    "kernels>=0.11.7",      # Flash Attention 3 kernel
    "matplotlib>=3.10.8",   # 画图 (可选)
    "numpy>=2.2.6",
    "pandas>=2.3.3",        # 数据处理
    "pyarrow>=21.0.0",      # Parquet 文件读取
    "requests>=2.32.0",     # 下载数据
    "rustbpe>=0.1.0",       # 快速 BPE 分词器训练
    "tiktoken>=0.11.0",     # 分词器运行时
    "torch==2.9.1",         # PyTorch (CUDA 12.8)
]
```

**注意：** Agent 不能安装新包。这个限制是故意的 — 所有创新必须在已有工具内完成。

### 9.5 在小型 GPU 上运行

如果你没有 H100，Karpathy 建议以下调整：

```
小型 GPU 的推荐调参
+--------------------------------------------+------------------------+
| 参数                                        | 调整方向                |
+--------------------------------------------+------------------------+
| 数据集                                      | 换成 TinyStories       |
| VOCAB_SIZE (prepare.py)                    | 降到 4096/2048/1024    |
| MAX_SEQ_LEN (prepare.py)                   | 降到 512 甚至 256      |
| EVAL_TOKENS (prepare.py)                   | 大幅降低                |
| DEPTH (train.py)                           | 降到 4                  |
| WINDOW_PATTERN (train.py)                  | 改成 "L"               |
| TOTAL_BATCH_SIZE (train.py)                | 降到 2**14 (~16K)      |
+--------------------------------------------+------------------------+
```

也可参考社区 fork：MacOS (miolini, trevin-creator)，Windows (jsegov)，AMD (andyluo7)。

---

## 十、设计哲学：为什么这样设计

### 10.1 固定时间预算 (而非固定步数)

```
传统做法:  train for 1000 steps
  问题: 大模型每步慢，小模型每步快，不公平

Autoresearch: train for 300 seconds
  优点 1: 公平 — 不管模型多大，都只用 5 分钟
  优点 2: 可预测 — 每个实验 ~6 分钟，一小时 ~10 个，一夜 ~100 个
  优点 3: 平台自适应 — 会自动找到你 GPU 上最优的模型
  缺点:   不同 GPU 的结果不可直接比较
```

### 10.2 单文件约束

```
为什么只允许修改 train.py？

1. 降低搜索空间
   Agent 不需要决定改哪个文件，只关注一个文件的优化

2. Diff 可审阅
   人类早上起来 git diff 一下就能看到所有改动

3. 保证评估公正
   Agent 不能修改评估函数来"作弊"
   evaluate_bpb() 在 prepare.py 里，只读

4. 减少破坏性错误
   不会不小心改坏数据加载或分词器
```

### 10.3 简单优先原则

```
program.md 中的原话:

  "All else being equal, simpler is better."

这不是空话。它的实际含义是：

  +0.001 BPB + 20 行复杂代码 = 不值得 keep
  +0.001 BPB + 删除代码       = 一定 keep
  +0.000 BPB + 更简单的代码   = keep (简化胜利)

为什么？因为 Agent 要跑 100 个实验。
如果每次都 keep 复杂化的代码，到第 50 个实验时
train.py 就会变成 1000 行的意大利面条代码。
简单优先防止了复杂度爆炸。
```

### 10.4 Git 作为实验追踪系统

```
传统 ML:  用 WandB / MLflow / TensorBoard 追踪实验
  需要额外的服务、配置、API key

Autoresearch: 用 Git
  - 每个实验 = 一个 commit
  - 好实验 = 分支前进
  - 坏实验 = git reset
  - 实验历史 = git log
  - 代码对比 = git diff
  - 数值记录 = results.tsv (untracked)

  零依赖，零配置，还能 push 到 GitHub 共享
```

### 10.5 "永不停止" 原则

```
这是最大胆的设计决策。

传统 Agent:
  "I've completed 5 experiments. Should I continue?"
  (然后等你回复，可能等一夜)

Autoresearch Agent:
  绝不询问。一直跑。直到你手动 Ctrl+C。

为什么？
  - 人可能在睡觉
  - Agent 等人的每一分钟 = 浪费一分钟 GPU 时间
  - 如果你每小时能跑 12 个实验，一夜 8 小时 = 96 个实验
  - 任何等待确认都是纯损失
```

### 10.6 输出重定向

```
为什么用 > run.log 2>&1 而不是直接看输出？

  训练过程每步打印一行：
  step 00001 (0.1%) | loss: 5.123456 | ...
  step 00002 (0.2%) | loss: 5.012345 | ...
  ...
  step 00953 (100%) | loss: 2.345678 | ...

  953 行输出 → 直接进 Agent 的 context window
  → context 被无用日志填满
  → Agent 的有效推理空间被压缩
  → 后续实验质量下降

  重定向到文件 + grep 精确提取 = 只用 2 行 context
```

### 10.7 数据加载器：Best-fit Packing

一个常被忽略但很精致的细节是 `prepare.py` 中的数据加载器：

```python
def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
```

```
传统做法: 简单拼接或 padding

  文档 A (100 tokens) + 文档 B (50 tokens) + PAD PAD PAD PAD ...
  利用率: (100 + 50) / 200 = 75%

Autoresearch 的 best-fit packing:

  行 1: [BOS|文档A(100t)|BOS|文档C(80t)|BOS|文档F(21t, 截断)|]  2048+1 tokens
  行 2: [BOS|文档B(50t)|BOS|文档D(200t)|BOS|文档E(1799t)|]     2048+1 tokens

  利用率: 100%  (没有任何 padding token)

  算法:
  1. 维护一个 buffer (1000 个待装的文档)
  2. 对每一行，贪心地找能装下的最大文档 (best-fit)
  3. 如果没有文档能装进剩余空间，截断最短的文档来填满
  4. 每个文档前都加 BOS token
```

这保证了 GPU 的每一次计算都在处理真实数据，没有浪费。

---

## 附录：核心概念速查

| 概念 | 含义 |
|------|------|
| val_bpb | Validation Bits Per Byte，越低越好 |
| BOS | Begin of Sequence token |
| RoPE | Rotary Position Embedding，旋转位置编码 |
| RMS Norm | Root Mean Square Normalization |
| Muon | 矩阵参数的正交化优化器 |
| Flash Attention 3 | 快速注意力计算内核 |
| bf16 | Brain Float 16，半精度浮点 |
| MFU | Model FLOPs Utilization，模型算力利用率 |
| ReluSquared | ReLU(x)^2，一种激活函数 |
| Best-fit packing | 贪心装箱算法，100% 利用率 |
| Cautious weight decay | 只对梯度方向一致的参数做衰减 |
| Value Embedding | 把 token embedding 直接注入 attention 的 value |

---

*Generated on 2026-03-28 from autoresearch source code analysis (karpathy/autoresearch)*
