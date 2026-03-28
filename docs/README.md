# Auto Researcher 系统架构模式深度解析

> 基于 karpathy/autoresearch (59k)、bytedance/deer-flow (51k)、stanford-oval/storm (28k)、SakanaAI/AI-Scientist (12.8k)、microsoft/RD-Agent (12.1k) 五个顶级项目的对比分析

---

## 一、五个项目速览

| 维度 | autoresearch | DeerFlow | STORM | AI-Scientist | RD-Agent |
|------|-------------|----------|-------|-------------|---------|
| **定位** | 单 GPU 自主 ML 实验 | 通用超级 Agent 框架 | 知识策展 & 长文生成 | **全自动科学家：假设→实验→论文→审稿** | **R&D 双循环：假设→代码→评估→进化** |
| **核心代码量** | ~1000 行 (3 文件) | ~30,000 行 | ~15,000 行 | ~5,000 行 | ~40,000 行 |
| **Agent 模式** | 单 Agent 无限循环 | Lead Agent + 子 Agent 并行 | 多 Persona 模拟对话 | 5 阶段顺序流水线 | Research + Development 双循环 |
| **框架** | 无 (纯 PyTorch + Git) | LangGraph + LangChain | DSPy + LiteLLM | Aider + Semantic Scholar | 自研框架 + CoSTEER |
| **状态管理** | Git 分支 + TSV 日志 | ThreadState + 中间件链 | InformationTable + ArticleTree | 文件系统 + LaTeX 项目 | Trace DAG + Workspace |
| **输出** | 优化后的训练代码 | 研究报告 / 任意任务 | 带引用的维基百科级文章 | **LaTeX 论文 + 审稿意见** | **进化后的代码 + 实验结果** |
| **原创性** | 有 (新实验结果) | 无 (整理已有信息) | 无 (综述已有信息) | **最高 (论文被 ICLR 接收)** | 有 (代码持续进化) |

### 研究真实度排序

```
AI-Scientist  ████████████  真跑实验、写 LaTeX、模拟 peer review、论文被接收
autoresearch  ██████████    真跑 GPU 训练、优化指标，但不写论文
RD-Agent      ████████      R&D 循环进化代码，MLE-Bench 最强，偏工程优化
STORM         ████          整理资料写综述，零原创
DeerFlow      ██            通用框架，本身不做研究
```

---

## TL;DR — 六大共性模式速查

| # | 模式 | 一句话 |
|---|------|--------|
| 1 | **循环驱动** | 所有系统本质都是 Hypothesize→Execute→Evaluate→Decide 循环 |
| 2 | **多阶段流水线** | 规划→采集→执行→综合，阶段间解耦 |
| 3 | **异构模型分配** | 便宜模型做检索，贵模型做综合 |
| 4 | **激进并行化** | 多视角/多子任务并行，缩短 3-5x 时间 |
| 5 | **可审计状态追踪** | Git/日志/中间产物，每步可回溯 |
| 6 | **约束即创造力** | 限制越明确，Agent 行为越聚焦 |

> autoresearch 只有 630 行代码却 59k star，证明**简单 + 清晰约束 + 明确指标**比复杂框架更有力量。

---

## 二、六大共性模式 (Core Patterns)

### Pattern 1: 循环驱动 (Loop-Driven Autonomy)

**这是最核心的共性：所有自动研究系统的本质都是一个循环。**

```
┌──────────────────────────────────────────────────────────┐
│                    THE RESEARCH LOOP                      │
│                                                          │
│   Hypothesize → Execute → Evaluate → Decide → Repeat    │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**五个项目的循环实现：**

```python
# autoresearch: 最纯粹的循环 — 直接跑到人类醒来
while True:
    idea = agent.think("下一个实验是什么?")
    modify(train_py, idea)
    git_commit(idea.description)
    result = run("uv run train.py", timeout=300)
    if result.val_bpb < best_val_bpb:
        best_val_bpb = result.val_bpb  # keep
    else:
        git_reset_hard()                # discard

# AI-Scientist: 5 阶段顺序流水线 — 最接近真实科研
for idea in generate_ideas(seed_ideas, num_ideas=50):
    if not check_novelty(idea, semantic_scholar):  # 查重
        continue
    code_ok = perform_experiments(idea, template)   # 用 Aider 改代码跑实验
    if not code_ok:
        continue
    paper = perform_writeup(idea, template)         # 写 LaTeX 论文
    review = perform_review(paper)                  # 模拟 peer review
    save(idea, paper, review)

# RD-Agent: Research + Development 双循环
while not converged:
    # Research 循环: 提假设
    hypothesis = researcher.propose(trace_history)
    # Development 循环: 实现假设 → 代码 → 测试 → 进化
    for step in range(max_steps):
        code = developer.implement(hypothesis)
        result = evaluator.run(code)
        if result.passes:
            trace_history.add(hypothesis, result)
            break
        code = developer.debug(code, result.error)

# DeerFlow: Agent Loop 由 LangGraph 驱动
while not done:
    action = lead_agent.think(state.messages)
    if action.type == "tool_call":
        result = execute_tool(action)
        state.messages.append(result)
    elif action.type == "delegate":
        subagent_results = parallel_execute(action.subtasks)
        state.messages.append(synthesize(subagent_results))
    elif action.type == "respond":
        done = True

# STORM: 多轮对话循环 × 多 Persona
for persona in personas:          # 外循环: 遍历视角
    for turn in range(max_turn):  # 内循环: 对话轮次
        question = writer.ask(persona, history)
        search_results = retriever.search(question)
        answer = expert.answer(question, search_results)
        history.append((question, answer))
```

**关键洞察：** 循环的复杂度不同，但模式一致：
- autoresearch = **贪心循环** (每步判断 keep/discard)
- AI-Scientist = **流水线循环** (idea→实验→论文→审稿，逐个处理)
- RD-Agent = **双循环嵌套** (外层提假设，内层写代码+调试)
- DeerFlow = **反应式循环** (根据中间结果动态决策)
- STORM = **结构化循环** (预定义阶段，循环发生在阶段内)

---

### Pattern 2: 多阶段流水线 (Multi-Stage Pipeline)

**所有系统都将研究分解为明确的阶段，每阶段有专门的策略。**

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  Planning    │ → │  Gathering  │ → │  Executing  │ → │ Synthesizing│
│  (规划)      │   │  (采集)      │   │  (执行)      │   │  (综合)      │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
```

**各项目的阶段对照：**

| 阶段 | autoresearch | AI-Scientist | RD-Agent | DeerFlow | STORM |
|------|-------------|-------------|---------|----------|-------|
| **规划** | 读 program.md，确定实验方向 | 生成 idea + Semantic Scholar 查重 | Research 循环提假设 | 分析用户问题 | 生成 Persona 视角 |
| **采集** | 读代码 + 结果历史 | 读模板代码 + 相关论文 | 读历史 Trace + 实验结果 | web_search + 子 Agent | 多 Persona 对话 + 搜索 |
| **执行** | `uv run train.py` (5分钟) | Aider 改代码 + subprocess 跑实验 | CoSTEER 生成代码 + 评估 | sandbox 执行 | 语义检索 + 分段写作 |
| **综合** | 对比 val_bpb，keep/discard | 写 LaTeX 论文 + AI 审稿 | 反馈注入下一轮假设 | 综合子任务 + 写报告 | 润色 + 去重 + 引用整理 |

**核心代码模式：**

```python
# 抽象的研究流水线
class ResearchPipeline:
    def run(self, topic):
        # 阶段 1: 规划
        plan = self.planner.plan(topic)

        # 阶段 2: 信息采集 (可并行)
        evidence = parallel_map(self.gather, plan.subtasks)

        # 阶段 3: 执行/实验
        results = self.execute(plan, evidence)

        # 阶段 4: 综合输出
        output = self.synthesize(results)

        return output
```

---

### Pattern 3: 不同任务用不同模型 (Heterogeneous Model Assignment)

**三个项目都体现了"用对的模型做对的事"这一思路。**

```python
# STORM: 最显式的多模型配置
lm_configs = STORMWikiLMConfigs()
lm_configs.set_conv_simulator_lm(LM(model='gpt-4o-mini'))    # 便宜、快
lm_configs.set_question_asker_lm(LM(model='gpt-4o-mini'))    # 便宜、快
lm_configs.set_outline_gen_lm(LM(model='gpt-4o'))            # 中等
lm_configs.set_article_gen_lm(LM(model='gpt-4o'))            # 最强
lm_configs.set_article_polish_lm(LM(model='gpt-4o'))         # 最强

# DeerFlow: 运行时可切模型
config.yaml:
  models:
    - name: deepseek-v3        # 推理任务
      supports_thinking: true
    - name: gpt-4o             # 通用任务
      supports_vision: true
    - name: claude-sonnet       # 代码任务

# autoresearch: 不调 LLM API — Agent 本身就是 LLM
# 但训练脚本中区分了不同参数组的优化器:
# Muon 用于 2D 矩阵参数 (更高效)
# AdamW 用于 1D 参数 (更稳定)
```

**设计原则：**

| 任务类型 | 推荐模型选择 | 原因 |
|---------|-------------|------|
| 信息检索/搜索 query 生成 | 小模型 (mini/haiku) | 快、便宜、足够 |
| 多轮对话/问答 | 中等模型 | 平衡质量和成本 |
| 最终综合/写作 | 最强模型 | 质量关键 |
| 推理/规划 | 支持 thinking 的模型 | 需要深度推理 |

---

### Pattern 4: 并行化一切可以并行的 (Aggressive Parallelization)

**三个项目都最大化并行度以缩短研究周期。**

```python
# autoresearch: 实验串行 (GPU 约束)，但 Agent 决策是即时的
# 瓶颈是 GPU 训练时间，不是 Agent 思考时间

# DeerFlow: 子 Agent 并行 (最多 3 个同时)
class SubagentExecutor:
    _scheduler_pool = ThreadPoolExecutor(max_workers=3)
    _execution_pool = ThreadPoolExecutor(max_workers=3)

    # 用户问 "比较 5 个云服务商"
    # Turn 1: 并行启动 3 个子 Agent (AWS, Azure, GCP)
    # Turn 2: 并行启动 2 个子 Agent (阿里云, Oracle)
    # Turn 3: 综合所有结果

# STORM: Persona 对话并行
with ThreadPoolExecutor(max_workers=max_thread_num) as executor:
    futures = []
    for persona in personas:
        future = executor.submit(
            run_conversation, topic, persona, max_turn=3
        )
        futures.append(future)

    # 3 个 Persona 同时研究，3x 速度提升
    conversations = [f.result() for f in futures]

# STORM: 文章各节并行写作
with ThreadPoolExecutor(max_workers=max_thread_num) as executor:
    for section in outline.sections:
        executor.submit(write_section, section, information_table)
```

**并行模式总结：**

```
                    串行                          并行
autoresearch:   实验必须串行          ←→    多个 autoresearch 实例跑不同方向
AI-Scientist:   单 idea 串行处理      ←→    多个 idea 可并行 (多 GPU)
RD-Agent:       R&D 循环串行          ←→    CoSTEER 内部多方案并行进化
DeerFlow:       Lead Agent 串行      ←→    子 Agent 并行 (3 路)
STORM:          Pipeline 阶段串行    ←→    Persona 对话并行 + 章节写作并行
```

---

### Pattern 5: 可审计的状态追踪 (Auditable State Tracking)

**每个系统都确保"做过什么"可追溯、可回滚。**

```python
# autoresearch: Git 即状态机
# 每次实验 = 一个 commit
# 成功 = commit 保留
# 失败 = git reset --hard
# 历史 = results.tsv (未跟踪，但持久化)
#
# 优雅之处: git reflog 保留所有操作，即使 reset 了也能恢复

# DeerFlow: ThreadState + 中间件链
class ThreadState(AgentState):
    messages: list              # 完整消息历史
    artifacts: list[str]        # 输出文件列表
    sandbox: SandboxState       # 沙箱状态
    todos: list                 # 任务追踪

# 中间件提供横切关注:
# - SummarizationMiddleware: 上下文太长时自动压缩
# - MemoryMiddleware: 提取事实到长期记忆
# - DanglingToolCallMiddleware: 修复中断的工具调用

# STORM: 完整的中间产物链
output_dir/
├── conversation_log.json       # 所有对话记录
├── raw_search_results.json     # 所有搜索结果
├── storm_gen_outline.txt       # 大纲
├── storm_gen_article.txt       # 未润色文章
├── storm_gen_article_polished.txt  # 最终文章
├── url_to_info.json            # 引用信息
└── llm_call_history.jsonl      # 所有 LLM 调用日志
```

**共同原则：**
1. **每个阶段的输入输出都被序列化** — 可以从任意阶段重跑
2. **决策有据可查** — 不是黑箱
3. **LLM 调用有日志** — 成本和质量可追踪
4. **失败可回滚** — 不会丢失好的进度

---

### Pattern 6: 约束即创造力 (Constraints as Creative Force)

**三个项目都通过精心设计的约束来驱动 Agent 行为。**

```python
# autoresearch: 极致约束
CONSTRAINTS = {
    "time_budget": "5 minutes per experiment",
    "files_editable": ["train.py"],      # 只能改一个文件
    "metric": "val_bpb (lower is better)", # 单一指标
    "gpu": "single GPU",                  # 硬件约束
    "rule": "NEVER STOP",                 # 不准停
    "simplicity": "complexity cost must justify improvement",
}
# 约束反而让 Agent 更有创造力：
# - 不能加文件 → 必须在 train.py 里创新
# - 5 分钟 → 必须优化效率而不是堆算力
# - 单指标 → 不会纠结于多目标

# DeerFlow: 中间件约束
CONSTRAINTS = {
    "max_concurrent_subagents": 3,        # 最多 3 个并行子任务
    "subagent_timeout": "15 minutes",     # 子任务超时
    "clarification_interrupt": True,       # 不确定就问用户
    "guardrails": "pre-tool authorization", # 工具调用授权
}

# AI-Scientist: 质量门控约束
CONSTRAINTS = {
    "novelty_check": True,                 # Semantic Scholar 查重，不新颖就跳过
    "max_experiment_retries": 4,           # 实验最多重试 4 次
    "max_latex_retries": 5,                # LaTeX 编译最多重试 5 次
    "review_threshold": 5,                 # 审稿分低于 5 就标记低质量
    "template_constraint": True,           # 只能在模板代码基础上改
}

# RD-Agent: 进化约束
CONSTRAINTS = {
    "max_loop_iterations": 10,             # R&D 循环最多 10 轮
    "hypothesis_must_be_testable": True,   # 假设必须可验证
    "code_must_pass_eval": True,           # 代码必须通过评估才纳入
    "trace_dag": True,                     # 所有决策记录在 DAG 中
}

# STORM: 结构化约束
CONSTRAINTS = {
    "max_perspective": 3,                  # 最多 3 个视角
    "max_conv_turn": 3,                    # 每个视角最多 3 轮对话
    "search_top_k": 3,                     # 每次检索 top-3
    "exclude_ground_truth": True,          # 评估时排除答案来源
}
```

---

## 三、架构决策树 (Which Pattern When?)

当你要构建自己的 Auto Researcher 时，根据场景选择模式：

```
你的研究任务是什么?
│
├─ 做原创研究、写论文 ──→ AI-Scientist 模式
│  │ 特点: idea 生成 + 实验 + LaTeX + peer review
│  │ 适用: ML 研究、科学发现、自动写论文
│  └─ 关键: 模板系统 + 新颖性检查 + 审稿反馈
│
├─ 优化实验 (有明确指标) ──→ autoresearch 模式
│  │ 特点: 单 Agent + 贪心循环 + Git 状态
│  │ 适用: ML 训练、超参搜索、代码优化
│  └─ 关键: 固定评估指标 + 固定时间预算
│
├─ 数据科学/量化金融 ──→ RD-Agent 模式
│  │ 特点: Research 提假设 + Development 写代码 + 进化循环
│  │ 适用: Kaggle、因子挖掘、模型进化
│  └─ 关键: 假设可验证 + 代码自动进化 + Trace DAG
│
├─ 综合研究 (需要多视角) ──→ STORM 模式
│  │ 特点: 多 Persona + 对话模拟 + 二阶段检索
│  │ 适用: 文献综述、报告生成、知识整理
│  └─ 关键: Persona 生成 + 信息表聚合
│
└─ 通用复杂任务 ──→ DeerFlow 模式
   │ 特点: Lead Agent + 子 Agent 并行 + 中间件
   │ 适用: 对比分析、多源调查、代码生成
   └─ 关键: 任务分解 + 并发控制
```

---

## 四、从零构建你的 Auto Researcher — 参考实现

基于三个项目的共性模式，这是一个最小可行的 Auto Researcher 骨架：

```python
"""
Minimal Auto Researcher — 融合三大项目的核心模式
"""
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# ============================================================
# Pattern 1: 状态追踪 (from autoresearch + STORM)
# ============================================================
@dataclass
class ResearchState:
    topic: str
    hypotheses: list[str] = field(default_factory=list)
    evidence: dict[str, list[str]] = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)
    current_best: dict | None = None

    def log(self, entry: dict):
        """可审计的日志 — autoresearch 的 results.tsv 思想"""
        self.results.append(entry)
        with open("research_log.jsonl", "a") as f:
            f.write(json.dumps(entry) + "\n")

# ============================================================
# Pattern 2: 多阶段流水线 (from all three)
# ============================================================
class ResearchStage(ABC):
    @abstractmethod
    def run(self, state: ResearchState) -> ResearchState:
        pass

class PlanningStage(ResearchStage):
    """阶段 1: 规划 — 生成研究假设/视角"""
    def __init__(self, llm, num_perspectives: int = 3):
        self.llm = llm
        self.num_perspectives = num_perspectives

    def run(self, state: ResearchState) -> ResearchState:
        # STORM 模式: 生成多个研究视角
        prompt = f"""Given the topic: {state.topic}
        Generate {self.num_perspectives} different research perspectives.
        Each should explore a different angle."""

        perspectives = self.llm.generate(prompt)
        state.hypotheses = perspectives
        state.log({"stage": "planning", "perspectives": perspectives})
        return state

class GatheringStage(ResearchStage):
    """阶段 2: 信息采集 — 并行多视角研究"""
    def __init__(self, llm, retriever, max_turns: int = 3):
        self.llm = llm
        self.retriever = retriever
        self.max_turns = max_turns

    def _research_perspective(self, topic, perspective):
        """STORM 模式: 多轮对话式研究"""
        findings = []
        history = []
        for turn in range(self.max_turns):
            question = self.llm.generate(
                f"Topic: {topic}\nPerspective: {perspective}\n"
                f"History: {history}\n"
                f"Ask a specific research question."
            )
            results = self.retriever.search(question)
            answer = self.llm.generate(
                f"Question: {question}\nSources: {results}\n"
                f"Provide a grounded answer with citations."
            )
            history.append({"q": question, "a": answer})
            findings.extend(results)
        return findings

    def run(self, state: ResearchState) -> ResearchState:
        # Pattern 4: 并行化 — 多视角同时研究
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self._research_perspective, state.topic, p
                ): p for p in state.hypotheses
            }
            for future in futures:
                perspective = futures[future]
                state.evidence[perspective] = future.result()

        state.log({"stage": "gathering", "sources": len(state.evidence)})
        return state

class SynthesisStage(ResearchStage):
    """阶段 3: 综合 — 生成最终输出"""
    def __init__(self, llm):
        self.llm = llm  # Pattern 3: 用最强模型做综合

    def run(self, state: ResearchState) -> ResearchState:
        all_evidence = "\n".join(
            f"## {perspective}\n" + "\n".join(findings)
            for perspective, findings in state.evidence.items()
        )

        output = self.llm.generate(
            f"Topic: {state.topic}\n\n"
            f"Evidence from multiple perspectives:\n{all_evidence}\n\n"
            f"Write a comprehensive research report with citations."
        )

        state.current_best = {"report": output}
        state.log({"stage": "synthesis", "output_length": len(output)})
        return state

# ============================================================
# Pattern 5: 流水线编排 (Pipeline Orchestration)
# ============================================================
class AutoResearcher:
    def __init__(self, stages: list[ResearchStage]):
        self.stages = stages

    def run(self, topic: str) -> ResearchState:
        state = ResearchState(topic=topic)
        for stage in self.stages:
            state = stage.run(state)
        return state

# ============================================================
# 使用示例
# ============================================================
"""
# 初始化 (伪代码)
planner = PlanningStage(llm=cheap_model, num_perspectives=3)
gatherer = GatheringStage(llm=cheap_model, retriever=web_search, max_turns=3)
synthesizer = SynthesisStage(llm=strong_model)

researcher = AutoResearcher(stages=[planner, gatherer, synthesizer])
result = researcher.run("量子计算对密码学的影响")

print(result.current_best["report"])
"""
```

---

## 五、高级模式 (Advanced Patterns)

### 5.1 贪心 Hill-Climbing (from autoresearch)

```python
# 当你有明确的评估指标时，用 autoresearch 的模式
class HillClimbingResearcher:
    def run(self, initial_state, evaluate_fn, max_iterations=100):
        best_score = evaluate_fn(initial_state)
        best_state = initial_state

        for i in range(max_iterations):
            # 生成变体
            variant = self.mutate(best_state)
            score = evaluate_fn(variant)

            # 贪心决策: 只保留更好的
            if score > best_score:
                best_score = score
                best_state = variant
                self.log(f"Iteration {i}: KEEP (score={score})")
            else:
                self.log(f"Iteration {i}: DISCARD (score={score})")

        return best_state
```

### 5.2 中间件链 (from DeerFlow)

```python
# 当你需要横切关注点 (日志、限流、安全) 时
class Middleware(ABC):
    @abstractmethod
    def process(self, state, next_fn):
        pass

class TokenLimitMiddleware(Middleware):
    """上下文太长时自动压缩"""
    def process(self, state, next_fn):
        if count_tokens(state.messages) > self.max_tokens:
            state.messages = self.summarize(state.messages)
        return next_fn(state)

class ConcurrencyMiddleware(Middleware):
    """限制并行子任务数量"""
    def process(self, state, next_fn):
        if state.active_subtasks >= self.max_concurrent:
            state.pending_subtasks.extend(state.new_subtasks[self.max_concurrent:])
            state.new_subtasks = state.new_subtasks[:self.max_concurrent]
        return next_fn(state)

class MemoryMiddleware(Middleware):
    """从对话中提取长期记忆"""
    def process(self, state, next_fn):
        result = next_fn(state)
        # 异步提取事实，不阻塞主流程
        self.queue_memory_extraction(state.messages[-1])
        return result
```

### 5.3 二阶段检索 (from STORM)

```python
# 第一阶段: Web 搜索 (采集时)
# 第二阶段: 语义检索 (写作时) — 从已采集的信息中检索
class TwoStageRetriever:
    def __init__(self, web_searcher, encoder):
        self.web_searcher = web_searcher
        self.encoder = encoder  # sentence-transformers
        self.knowledge_base = []

    def stage1_gather(self, queries: list[str]):
        """阶段 1: 从互联网采集"""
        for query in queries:
            results = self.web_searcher.search(query)
            self.knowledge_base.extend(results)
        # 编码所有采集到的信息
        self.embeddings = self.encoder.encode(
            [r.text for r in self.knowledge_base]
        )

    def stage2_retrieve(self, section_query: str, top_k: int = 5):
        """阶段 2: 从知识库中语义检索"""
        query_embedding = self.encoder.encode(section_query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.knowledge_base[i] for i in top_indices]
```

---

## 六、成本与效率对比

| 维度 | autoresearch | AI-Scientist | RD-Agent | DeerFlow | STORM |
|------|-------------|-------------|---------|----------|-------|
| **LLM 调用** | 0 (Agent 是外部 LLM) | 100-200 次/论文 | 50-100 次/循环 | 10-50 次/任务 | 50-100 次/文章 |
| **典型成本** | $0 (仅 GPU 电费) | $5-15/论文 | $2-10/循环 | $0.50-5/任务 | $0.50-2/文章 |
| **耗时** | 8 小时 (~100 实验) | 12-24 小时/论文 | 1-4 小时/循环 | 5-30 分钟 | 10-30 分钟 |
| **主要瓶颈** | GPU 训练时间 | 实验执行 + LaTeX 编译 | 代码评估 + 反馈 | 子 Agent 执行 | Web 搜索延迟 |
| **并行收益** | N/A (单 GPU) | 多 idea 并行 | CoSTEER 方案并行 | 3x (子 Agent) | 3-5x (Persona) |

---

## 七、关键教训 (Key Takeaways)

### 1. 简单胜于复杂

autoresearch 用 630 行代码 + 一个 program.md 就实现了全自动研究。不需要复杂的框架，有时候一个清晰的循环 + 明确的评估指标就够了。

### 2. 约束驱动创新

限制 Agent 只能改一个文件、只给 5 分钟、只看一个指标 — 这些约束反而让 Agent 更聚焦、更有创造力。设计你的系统时，先想好约束条件。

### 3. 多视角 > 单一深度

STORM 的核心洞察：与其让一个 Agent 深入研究，不如让多个 Persona 从不同角度提问。信息的广度往往比深度更重要。

### 4. 中间件是正确的抽象

DeerFlow 的 12 个中间件覆盖了安全、限流、压缩、记忆、澄清等横切关注点。如果你的 Agent 越来越复杂，中间件是管理复杂度的正确方式。

### 5. 可审计性是刚需

三个项目都保存了完整的决策日志。这不仅是调试需要，更是信任建立的基础。你的 Auto Researcher 必须能回答"你为什么做了这个决定?"

### 6. 用对模型做对事

别用 GPT-4 做搜索 query 生成，也别用 GPT-3.5 做最终综合。不同阶段的任务复杂度不同，模型选择要匹配。

---

## 八、快速上手指南

```bash
# 1. autoresearch (最简单，需要 NVIDIA GPU)
git clone https://github.com/karpathy/autoresearch
cd autoresearch
uv run prepare.py          # 下载数据
# 然后用 Claude Code 打开，运行 program.md 中的指令

# 2. AI-Scientist (做原创研究，需要 GPU + API keys)
git clone https://github.com/SakanaAI/AI-Scientist
cd AI-Scientist
pip install -e .
# 配置 OPENAI_API_KEY + S2_API_KEY (Semantic Scholar)
python launch_scientist.py --model gpt-4o --experiment nanoGPT

# 3. RD-Agent (数据科学/量化，需要 API keys)
git clone https://github.com/microsoft/RD-Agent
cd RD-Agent
pip install -e .
# 配置 LLM API keys
rdagent fin_factor          # 量化因子挖掘
rdagent kaggle              # Kaggle 竞赛

# 4. DeerFlow (最通用，需要 API keys)
git clone https://github.com/bytedance/deer-flow
cd deer-flow
cp .env.example .env       # 配置 API keys
make install               # 安装依赖
make dev                   # 启动全栈 (前端 + 后端)

# 5. STORM (写综述，需要搜索 API)
git clone https://github.com/stanford-oval/storm
cd storm
pip install -e .
# 参考 examples/ 目录中的示例脚本
```

---

## 九、项目选择速查

| 如果你想... | 选择 | 原因 |
|------------|------|------|
| **AI 自动写论文** | AI-Scientist | 唯一产出被同行评审接收论文的系统 |
| 自动优化 ML 训练 | autoresearch | 630 行代码，overnight 跑 100 个实验 |
| 打 Kaggle / 量化交易 | RD-Agent | MLE-Bench 最强，R&D 双循环持续进化 |
| 自动写综述/报告 | STORM | 多视角 + 引用，维基百科级 |
| 建一个通用研究助手 | DeerFlow | 最灵活，中间件 + 沙箱 + 多通道 |
| 学习 Agent 设计模式 | autoresearch | 代码最少，模式最清晰 |
| 生产环境部署 | DeerFlow | 最完整的工程化 |
| 做原创科学发现 | AI-Scientist | 假设→实验→论文→审稿全闭环 |

---

*Generated on 2026-03-28 by analyzing karpathy/autoresearch, bytedance/deer-flow, stanford-oval/storm, SakanaAI/AI-Scientist, and microsoft/RD-Agent*
