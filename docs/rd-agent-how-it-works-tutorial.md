# RD-Agent 是怎么工作的？— 从原理到代码的完整教程

> RD-Agent = **R**esearch & **D**evelopment Agent
>
> 微软出品，一句话：**用 AI 模拟"研究员提假说 + 工程师写代码验证"的 R&D 循环，自动进化因子、模型和数据科学方案**

---

## 一、先看结果：RD-Agent 输入什么，输出什么

RD-Agent 支持多种场景，每种场景的输入输出不同：

### 场景 1：量化金融因子进化 (`rdagent fin_factor`)

```
输入: 无 (系统自动生成假说)
     或：一组金融报告 PDF

输出:
  log/
  ├── __session__/
  │   ├── 0/                        ← 第 0 轮循环
  │   │   ├── 0_direct_exp_gen/     ← 假说 + 实验设计
  │   │   ├── 1_coding/             ← 生成的因子代码 (.py)
  │   │   ├── 2_running/            ← Qlib 回测结果
  │   │   ├── 3_feedback/           ← IC/ICIR/收益率 反馈
  │   │   └── 4_record/             ← 记录到 Trace
  │   ├── 1/                        ← 第 1 轮循环
  │   └── ...
  └── factor_library/               ← 不断进化的因子库
```

### 场景 2：Kaggle / Data Science (`rdagent data_science`)

```
输入: --competition tabular-playground-series-dec-2021

输出:
  workspace/
  ├── load_data.py                  ← 数据加载代码
  ├── feature_engineering.py        ← 特征工程代码
  ├── model_*.py                    ← 模型代码 (可能多个)
  ├── ensemble.py                   ← 集成代码
  ├── workflow.py                   ← 训练/预测工作流
  └── submission.csv                ← 最终提交文件
```

### 场景 3：通用模型实现 (`rdagent general_model`)

```
输入: 一篇论文的 URL, 如 "https://arxiv.org/pdf/2210.09789"

输出: 从论文中提取模型结构并实现为可运行的 PyTorch 代码
```

**核心共性：所有场景都遵循同一个 R&D 循环，不断提出假说、编码实现、运行验证、反馈进化。**

---

## 二、核心理念：Research + Development 双循环

RD-Agent 的核心洞察是：**真实世界的科研和工程流程可以拆解为两个交替进行的循环。**

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Research 循环 (R)            Development 循环 (D)          │
│                                                             │
│   ┌─────────────┐             ┌─────────────────┐          │
│   │ 观察现状     │             │ 接收假说         │          │
│   │ (Trace历史)  │             │ (Hypothesis)     │          │
│   └──────┬──────┘             └────────┬────────┘          │
│          ↓                             ↓                    │
│   ┌─────────────┐             ┌─────────────────┐          │
│   │ 提出假说     │────────────→│ 生成实验代码      │          │
│   │ (HypothesisGen)           │ (CoSTEER)        │          │
│   └──────┬──────┘             └────────┬────────┘          │
│          │                             ↓                    │
│          │                    ┌─────────────────┐          │
│          │                    │ 运行实验          │          │
│          │                    │ (Runner)          │          │
│          │                    └────────┬────────┘          │
│          ↓                             ↓                    │
│   ┌─────────────┐             ┌─────────────────┐          │
│   │ 接收反馈     │←────────────│ 生成反馈         │          │
│   │ (更新 Trace) │             │ (Feedback)       │          │
│   └──────┬──────┘             └─────────────────┘          │
│          │                                                  │
│          └──── 下一轮循环 ────→                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**类比：**
- **R 循环** = 研究员/PI：看之前实验的结果，提出新的研究假说
- **D 循环** = 工程师/RA：拿到假说，写代码实现，跑实验，汇报结果

两者交替运行，形成一个闭环进化系统。

---

## 三、全局架构

### 3.1 核心抽象层

```
rdagent/core/                      ← 所有抽象基类
├── scenario.py                    ← Scenario: 场景描述 (金融/医疗/Kaggle...)
├── proposal.py                    ← Hypothesis, HypothesisGen, Trace, Feedback
├── experiment.py                  ← Task, Workspace, Experiment
├── developer.py                   ← Developer: 代码开发者
├── evaluation.py                  ← Evaluator, Feedback
├── evolving_framework.py          ← EvolvingStrategy, RAGStrategy
├── evolving_agent.py              ← EvoAgent, RAGEvoAgent
├── knowledge_base.py              ← KnowledgeBase
└── conf.py                        ← RDAgentSettings (全局配置)
```

### 3.2 完整流水线

```
                 ┌──────────────────────────────────────────────┐
                 │           RDLoop (主循环控制器)                │
                 │   rdagent/components/workflow/rd_loop.py      │
                 └──────────────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
   Step 1: direct_exp_gen      Step 2: coding             Step 3: running
   (提假说 + 设计实验)          (CoSTEER 写代码)           (执行实验)
        │                           │                           │
        ↓                           ↓                           ↓
  ┌───────────┐             ┌──────────────┐           ┌──────────────┐
  │HypothesisGen│            │  Developer   │           │   Runner     │
  │   ↓         │            │  (CoSTEER)   │           │  (Docker环境) │
  │Hypothesis   │            │              │           │              │
  │   ↓         │            │ EvoAgent     │           │ 执行代码      │
  │Hypothesis2  │            │   ↓          │           │ 收集结果      │
  │ Experiment  │            │ EvolvingStr  │           │              │
  └───────────┘             │   ↓          │           └──────┬───────┘
                             │ RAGStrategy  │                  │
                             └──────────────┘                  │
                                                               ↓
        ┌──────────────────────────────────────────────────────┤
        │                                                      │
   Step 4: feedback                                    Step 5: record
   (生成反馈)                                           (记录到 Trace)
        │                                                      │
        ↓                                                      ↓
  ┌───────────────┐                                   ┌──────────────┐
  │Experiment2    │                                   │  Trace       │
  │  Feedback     │                                   │  (DAG 结构)   │
  │               │                                   │  hist[]      │
  │ LLM 比较结果  │                                   │  dag_parent[]│
  │ 判定假说真假  │                                   │              │
  └───────────────┘                                   └──────────────┘
```

### 3.3 对应代码：RDLoop 的核心 5 步

```python
# rdagent/components/workflow/rd_loop.py
class RDLoop(LoopBase, metaclass=LoopMeta):

    # Step 1: 提出假说 + 设计实验
    async def direct_exp_gen(self, prev_out):
        hypo = self._propose()            # HypothesisGen.gen(trace)
        exp = self._exp_gen(hypo)          # Hypothesis2Experiment.convert(hypo, trace)
        return {"propose": hypo, "exp_gen": exp}

    # Step 2: 写代码 (CoSTEER 进化式编码)
    def coding(self, prev_out):
        exp = self.coder.develop(prev_out["direct_exp_gen"]["exp_gen"])
        return exp

    # Step 3: 运行实验 (Docker 环境)
    def running(self, prev_out):
        exp = self.runner.develop(prev_out["coding"])
        return exp

    # Step 4: 生成反馈 (LLM 分析结果)
    def feedback(self, prev_out):
        feedback = self.summarizer.generate_feedback(prev_out["running"], self.trace)
        return feedback

    # Step 5: 记录到 Trace (DAG 结构)
    def record(self, prev_out):
        self.trace.sync_dag_parent_and_hist(
            (exp, feedback), prev_out[self.LOOP_IDX_KEY]
        )
```

**LoopMeta 的巧妙设计：** `LoopMeta` 是一个元类，它自动检测类中所有非下划线开头的方法，按定义顺序注册为 `steps`。子类可以通过覆盖方法来定制某一步，也可以添加新方法来插入新步骤。

```python
# rdagent/utils/workflow/loop.py
class LoopMeta(type):
    def __new__(mcs, clsname, bases, attrs):
        steps = LoopMeta._get_steps(bases)
        for name, attr in attrs.items():
            if not name.startswith("_") and callable(attr):
                if name not in steps:
                    steps.append(name)
        attrs["steps"] = steps
        return super().__new__(mcs, clsname, bases, attrs)
```

---

## 四、支持的场景

RD-Agent 的场景系统是高度模块化的。每个场景提供不同的 `Scenario` 描述、`HypothesisGen`、`Developer`、`Runner` 和 `Feedback` 实现。

| 场景 | CLI 命令 | 功能描述 | 核心类 |
|------|---------|---------|--------|
| 量化因子进化 | `rdagent fin_factor` | 自动生成、验证、进化 alpha 因子 | `QlibFactorHypothesisGen` |
| 量化模型进化 | `rdagent fin_model` | 自动设计、训练、迭代预测模型 | `QlibModelHypothesisGen` |
| 因子+模型联合进化 | `rdagent fin_quant` | Factor 和 Model 交替优化 | `QlibQuantHypothesisGen` |
| 报告因子提取 | `rdagent fin_factor_report` | 从金融报告中提取因子并实现 | `FactorFromReportPropSetting` |
| 论文模型实现 | `rdagent general_model` | 从论文中提取模型并实现为代码 | `extract_models_and_implement` |
| Data Science/Kaggle | `rdagent data_science` | 自动特征工程+模型调优+集成 | `DataScienceRDLoop` |
| 医疗预测模型 | `rdagent data_science` | 医疗数据场景的模型进化 | `DataScienceScen` |
| LLM 微调 | `rdagent llm_finetune` | 自动化 LLM 微调实验 | `llm_finetune` |

### 4.1 量化金融场景 (Qlib 系列)

这是 RD-Agent 最成熟的场景。系统与微软的 [Qlib](https://github.com/microsoft/qlib) 量化投资平台深度集成。

```
┌─────────────────────────────────────────────────────────┐
│  fin_quant: Factor + Model 联合进化                      │
│                                                         │
│   Bandit/LLM 决策                                       │
│      ↓                                                  │
│  ┌────────────┐      ┌────────────┐                     │
│  │ action =   │      │ action =   │                     │
│  │ "factor"   │      │ "model"    │                     │
│  └─────┬──────┘      └─────┬──────┘                     │
│        ↓                    ↓                            │
│  QlibFactorHypothesisGen   QlibModelHypothesisGen       │
│        ↓                    ↓                            │
│  QlibFactorCoSTEER         QlibModelCoSTEER             │
│        ↓                    ↓                            │
│  QlibFactorRunner          QlibModelRunner               │
│        ↓                    ↓                            │
│  QlibFactor                QlibModel                     │
│  Experiment2Feedback       Experiment2Feedback            │
│        ↓                    ↓                            │
│        └────────┬───────────┘                            │
│                 ↓                                        │
│           QuantTrace (统一 Trace)                         │
│                                                         │
│  评估指标: IC, ICIR, Rank IC, Rank ICIR,                 │
│           年化收益率, 最大回撤, 信息比率                    │
└─────────────────────────────────────────────────────────┘
```

**Bandit 决策机制：** `fin_quant` 使用 Thompson Sampling (Linear Thompson Two-Arm) 来决定每轮是优化因子还是优化模型。

```python
# rdagent/scenarios/qlib/proposal/quant_proposal.py
class QlibQuantHypothesisGen(FactorAndModelHypothesisGen):
    def prepare_context(self, trace: Trace):
        if QUANT_PROP_SETTING.action_selection == "bandit":
            # 使用 Linear Thompson Sampling 选择 action
            metric = extract_metrics_from_experiment(trace.hist[-1][0])
            action = trace.controller.decide(metric)  # "factor" 或 "model"
        elif QUANT_PROP_SETTING.action_selection == "llm":
            # 用 LLM 判断下一步该做什么
            action = json.loads(resp).get("action", "factor")
        elif QUANT_PROP_SETTING.action_selection == "random":
            action = random.choice(["factor", "model"])
```

### 4.2 Data Science / Kaggle 场景

这是 RD-Agent 最复杂的场景，支持完整的数据科学流水线：

```
┌─────────────────────────────────────────────────────────┐
│  DataScienceRDLoop                                      │
│                                                         │
│  编码阶段分为 6 个组件，按依赖顺序执行:                    │
│                                                         │
│  ┌──────────────┐                                       │
│  │ DataLoader   │ ← 数据加载 + 预处理                    │
│  └──────┬───────┘                                       │
│         ↓                                               │
│  ┌──────────────┐                                       │
│  │ Feature      │ ← 特征工程                             │
│  └──────┬───────┘                                       │
│         ↓                                               │
│  ┌──────────────┐                                       │
│  │ Model        │ ← 模型 (可以有多个)                    │
│  └──────┬───────┘                                       │
│         ↓                                               │
│  ┌──────────────┐                                       │
│  │ Ensemble     │ ← 模型集成                             │
│  └──────┬───────┘                                       │
│         ↓                                               │
│  ┌──────────────┐                                       │
│  │ Workflow     │ ← 训练/预测流程编排                     │
│  └──────┬───────┘                                       │
│         ↓                                               │
│  ┌──────────────┐                                       │
│  │ Pipeline     │ ← 端到端 pipeline (可选)               │
│  └──────────────┘                                       │
│                                                         │
│  每个组件都有自己的 CoSTEER:                              │
│  DataLoaderCoSTEER, FeatureCoSTEER, ModelCoSTEER,        │
│  EnsembleCoSTEER, WorkflowCoSTEER, PipelineCoSTEER       │
└─────────────────────────────────────────────────────────┘
```

```python
# rdagent/scenarios/data_science/loop.py — coding 步骤
def coding(self, prev_out):
    exp = prev_out["direct_exp_gen"]
    for tasks in exp.pending_tasks_list:
        exp.sub_tasks = tasks
        if isinstance(exp.sub_tasks[0], DataLoaderTask):
            exp = self.data_loader_coder.develop(exp)
        elif isinstance(exp.sub_tasks[0], FeatureTask):
            exp = self.feature_coder.develop(exp)
        elif isinstance(exp.sub_tasks[0], ModelTask):
            exp = self.model_coder.develop(exp)
        elif isinstance(exp.sub_tasks[0], EnsembleTask):
            exp = self.ensemble_coder.develop(exp)
        elif isinstance(exp.sub_tasks[0], WorkflowTask):
            exp = self.workflow_coder.develop(exp)
        elif isinstance(exp.sub_tasks[0], PipelineTask):
            exp = self.pipeline_coder.develop(exp)
    return exp
```

---

## 五、Research 循环详解：假说是怎么生成的

### 5.1 假说的数据结构

```python
# rdagent/core/proposal.py
class Hypothesis:
    def __init__(self, hypothesis, reason, concise_reason,
                 concise_observation, concise_justification, concise_knowledge):
        self.hypothesis = hypothesis           # "动量因子的 5 日均线斜率可以捕捉短期趋势"
        self.reason = reason                   # 详细推理过程
        self.concise_reason = concise_reason   # 简洁理由
        self.concise_observation = concise_observation    # 从历史数据中观察到的现象
        self.concise_justification = concise_justification # 为什么这个假说有道理
        self.concise_knowledge = concise_knowledge         # 利用了什么先验知识
```

在量化场景中，`QlibQuantHypothesis` 还增加了 `action` 字段，指明是 "factor" 还是 "model" 方向的假说。

### 5.2 假说生成的流程

```
┌─────────────────────────────────────────────────────────────┐
│ LLMHypothesisGen.gen(trace, plan)                           │
│                                                             │
│  1. prepare_context(trace)     ← 子类实现，准备上下文         │
│     ├── 收集历史假说和反馈                                    │
│     ├── 获取 SOTA 实验结果                                   │
│     └── 附加 RAG 建议                                        │
│                                                             │
│  2. 构建 Prompt                                              │
│     ├── system_prompt:                                       │
│     │   - 场景描述 (scenario.get_scenario_all_desc())        │
│     │   - 输出格式要求                                       │
│     │   - 假说规范                                           │
│     │                                                       │
│     └── user_prompt:                                         │
│         - 历史假说和反馈 (hypothesis_and_feedback)            │
│         - 上一轮的假说和反馈 (last_hypothesis_and_feedback)   │
│         - SOTA 实验信息 (sota_hypothesis_and_feedback)        │
│         - RAG 知识补充                                       │
│                                                             │
│  3. LLM 调用 → JSON 响应                                    │
│                                                             │
│  4. convert_response(resp)    ← 子类实现，解析为 Hypothesis   │
└─────────────────────────────────────────────────────────────┘
```

**对应代码：**

```python
# rdagent/components/proposal/__init__.py
class LLMHypothesisGen(HypothesisGen):
    def gen(self, trace: Trace, plan: ExperimentPlan | None = None) -> Hypothesis:
        # 1. 准备上下文 (子类实现)
        context_dict, json_flag = self.prepare_context(trace)

        # 2. 构建 prompt
        system_prompt = T(".prompts:hypothesis_gen.system_prompt").r(
            targets=self.targets,
            scenario=self.scen.get_scenario_all_desc(...),
            hypothesis_output_format=context_dict["hypothesis_output_format"],
            hypothesis_specification=context_dict["hypothesis_specification"],
        )
        user_prompt = T(".prompts:hypothesis_gen.user_prompt").r(
            targets=self.targets,
            hypothesis_and_feedback=context_dict["hypothesis_and_feedback"],
            RAG=context_dict["RAG"],
        )

        # 3. LLM 调用
        resp = APIBackend().build_messages_and_create_chat_completion(
            user_prompt, system_prompt, json_mode=json_flag
        )

        # 4. 解析响应
        return self.convert_response(resp)
```

### 5.3 Trace：研究历史的 DAG 结构

`Trace` 不是简单的列表，而是一个有向无环图 (DAG)，支持多分支探索：

```python
# rdagent/core/proposal.py
class Trace:
    NodeType = tuple[Experiment, ExperimentFeedback]

    def __init__(self, scen, knowledge_base=None):
        self.scen = scen
        self.hist: list[Trace.NodeType] = []           # 实验+反馈的历史
        self.dag_parent: list[tuple[int, ...]] = []    # DAG 父节点索引
        self.knowledge_base = knowledge_base
        self.current_selection = self.SEL_LATEST_SOTA  # (-1,) 默认选最新 SOTA
```

```
DAG 示例 (多分支探索):

  实验0 (Factor: 5日动量)
    │  decision=True (SOTA)
    ├──→ 实验1 (Factor: 10日波动率)
    │      │  decision=False
    │      └──→ 实验3 (Factor: 波动率改进版)
    │             decision=True (新 SOTA)
    │
    └──→ 实验2 (Model: LSTM)
           │  decision=True
           └──→ 实验4 (Model: GRU)
                  decision=False
```

`get_sota_hypothesis_and_experiment()` 方法从 DAG 中找到当前最佳实验，供下一轮假说生成使用。

---

## 六、Development 循环详解：CoSTEER 如何生成代码

### 6.1 CoSTEER 架构

CoSTEER (Collaborative Strategy for Task Evolving and Execution with Retrieval) 是 RD-Agent 的核心代码生成引擎。它不是一次性生成代码，而是**多轮进化**。

```
┌─────────────────────────────────────────────────────────────┐
│  CoSTEER.develop(exp)                                       │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ RAGEvoAgent.multistep_evolve(evo, eva)                │  │
│  │                                                       │  │
│  │  for loop in range(max_loop):     ← 默认 10 轮        │  │
│  │                                                       │  │
│  │    1. RAG 查询知识库                                   │  │
│  │       queried_knowledge = rag.query(evo, trace)       │  │
│  │                                                       │  │
│  │    2. 进化策略生成代码                                  │  │
│  │       for evolved_evo in strategy.evolve_iter(...):    │  │
│  │           # LLM 为每个子任务生成/改进代码               │  │
│  │                                                       │  │
│  │    3. 迭代评估                                         │  │
│  │       step_feedback = eva_iter.send(evolved_evo)      │  │
│  │       # 执行代码，检查输出，评估质量                    │  │
│  │                                                       │  │
│  │    4. 记录进化步骤                                     │  │
│  │       evolving_trace.append(EvoStep(evo, kb, fb))     │  │
│  │                                                       │  │
│  │    5. 知识自进化 (可选)                                 │  │
│  │       rag.generate_knowledge(evolving_trace)           │  │
│  │       rag.dump_knowledge_base()                       │  │
│  │                                                       │  │
│  │    6. 检查是否所有任务完成                               │  │
│  │       if feedback.finished(): break                    │  │
│  │                                                       │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  输出: exp (带有 sub_workspace_list 的代码实现)               │
└─────────────────────────────────────────────────────────────┘
```

**对应代码：**

```python
# rdagent/core/evolving_agent.py
class RAGEvoAgent(EvoAgent):
    def multistep_evolve(self, evo, eva):
        for evo_loop_id in range(self.max_loop):
            # 1. RAG
            queried_knowledge = self.rag.query(evo, self.evolving_trace)

            # 2. Evolve: 进化策略可能分多步
            evo_iter = self.evolving_strategy.evolve_iter(
                evo=evo,
                evolving_trace=self.evolving_trace,
                queried_knowledge=queried_knowledge,
            )
            # 3. Evaluate: 迭代评估
            eva_iter = eva.evaluate_iter(...)
            next(eva_iter)
            for evolved_evo in evo_iter:
                step_feedback = eva_iter.send(evolved_evo)

            # 4. 记录
            self.evolving_trace.append(EvoStep(evolved_evo, queried_knowledge, overall_feedback))

            # 5. 知识自进化
            if self.knowledge_self_gen:
                self.rag.generate_knowledge(self.evolving_trace)
                self.rag.dump_knowledge_base()

            yield evo  # 返回控制权给调用者

            # 6. 检查完成
            if es.feedback.finished():
                break
```

### 6.2 Fallback 机制

CoSTEER 在多轮进化中维护一个 fallback 快照。如果后续进化导致代码变差，可以回退到之前最好的版本：

```python
# rdagent/components/coder/CoSTEER/__init__.py
class CoSTEER(Developer):
    def develop(self, exp):
        fallback_evo_exp = None
        fallback_evo_fb = None

        for evo_exp in self.evolve_agent.multistep_evolve(evo_exp, self.evaluator):
            evo_fb = self._get_last_fb()

            # 如果当前结果比 fallback 更好，更新 fallback
            if self.should_use_new_evo(fallback_evo_fb, evo_fb):
                fallback_evo_exp = deepcopy(evo_exp)
                fallback_evo_fb = deepcopy(evo_fb)
                fallback_evo_exp.create_ws_ckp()  # 创建 workspace 快照

        # 最终回退到最好的版本
        if fallback_evo_exp is not None:
            evo_exp = fallback_evo_exp
            evo_exp.recover_ws_ckp()  # 恢复 workspace 快照
```

### 6.3 CoSTEER 评估反馈

每个子任务的评估反馈包含三个维度：

```python
# rdagent/components/coder/CoSTEER/evaluators.py
@dataclass
class CoSTEERSingleFeedback(Feedback):
    execution: str           # 代码是否能执行
    return_checking: str     # 返回值是否符合预期 (形状/类型/约束)
    code: str                # 代码质量评审
    final_decision: bool     # 最终判定: 通过/不通过
```

```
评估流水线:

  代码 → [执行] → 成功? → [返回值检查] → 正确? → [代码审查] → 通过?
           ↓ 失败        ↓ 错误            ↓ 问题
     execution        return_checking      code
     feedback         feedback            feedback
           ↓               ↓                ↓
                final_decision = False
                (附带详细的失败原因)
```

---

## 七、Factor/Model 进化

### 7.1 因子进化流程

```
第 1 轮:
  假说: "5日收益率动量因子可以预测短期趋势"
  代码: def alpha001(df): return df['close'].pct_change(5)
  结果: IC=0.023, ICIR=0.31
  反馈: 假说部分验证，IC 正面但不显著

第 2 轮 (基于第 1 轮的反馈):
  假说: "改用成交量加权的动量因子可以提高信噪比"
  代码: def alpha002(df): return (df['close'] * df['volume']).pct_change(5) / df['volume'].rolling(5).mean()
  结果: IC=0.035, ICIR=0.48
  反馈: 显著改善! 设为新 SOTA

第 3 轮 (基于第 2 轮的 SOTA):
  假说: "波动率调整的动量因子可以降低噪声"
  代码: def alpha003(df): return alpha002(df) / df['close'].pct_change(5).rolling(20).std()
  结果: IC=0.041, ICIR=0.55
  反馈: 进一步改善! 设为新 SOTA

  ...
```

### 7.2 模型进化流程

```
第 1 轮:
  假说: "LSTM 可以捕捉时序特征中的长期依赖"
  代码: class Model(nn.Module): # 2层 LSTM + FC
  结果: IC=0.038, ARR=8.2%
  反馈: 模型性能合理，但训练不稳定

第 2 轮:
  假说: "GRU 比 LSTM 更适合较短的金融时序，且训练更稳定"
  代码: class Model(nn.Module): # 2层 GRU + Dropout + FC
  结果: IC=0.042, ARR=11.5%
  反馈: 显著改善! 设为新 SOTA

  ...
```

### 7.3 Factor + Model 联合进化

在 `fin_quant` 场景中，系统使用 Bandit 算法（Linear Thompson Sampling）来决定每轮应该改进因子还是改进模型：

```python
# rdagent/scenarios/qlib/proposal/bandit.py
class LinearThompsonTwoArm:
    def __init__(self, dim, prior_var=1.0, noise_var=1.0):
        self.dim = dim
        self.mean = {"factor": np.zeros(dim), "model": np.zeros(dim)}
        self.precision = {"factor": np.eye(dim)/prior_var, "model": np.eye(dim)/prior_var}

# 每轮循环:
# 1. 执行完实验后，提取指标向量 (IC, ICIR, ARR, MDD, Sharpe...)
# 2. Thompson Sampling 采样决定下一轮 action
# 3. 更新后验分布
```

```
联合进化轨迹示例:

  轮次  Action   假说                    IC      ARR
  ──── ──────── ─────────────────────── ─────── ────────
  1    factor   5日动量因子              0.023   5.1%
  2    factor   成交量加权动量           0.035   7.2%
  3    model    LSTM 时序模型            0.038   8.2%
  4    factor   波动率调整动量           0.041   9.5%
  5    model    GRU 替换 LSTM            0.044   11.5%
  6    factor   ML 特征因子              0.048   12.8%
  ...

  经过 ~20 轮迭代，通常可以达到:
  - 比基准因子库 ~2x 的年化收益率
  - 使用比基准少 70% 以上的因子数量
```

---

## 八、评估与反馈

### 8.1 两层反馈系统

RD-Agent 有两个层级的反馈：

```
┌─────────────────────────────────────────────────────────────┐
│ 层级 1: CoSTEER 内部反馈 (Development 阶段)                  │
│                                                             │
│ CoSTEERSingleFeedback:                                      │
│   - execution: 代码能否成功执行                               │
│   - return_checking: 输出格式/值/形状是否正确                  │
│   - code: 代码质量评审                                       │
│   - final_decision: bool                                    │
│                                                             │
│ 用途: 指导代码在 CoSTEER 内部的多轮进化                       │
│ 反馈频率: 每轮 CoSTEER 进化都会产生                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 层级 2: 假说级反馈 (Research 阶段)                            │
│                                                             │
│ HypothesisFeedback:                                         │
│   - decision: 假说是否被验证 (新实验是否比 SOTA 好)           │
│   - observations: 实验观察到了什么                             │
│   - hypothesis_evaluation: 假说的评估                        │
│   - new_hypothesis: 建议的下一步假说                          │
│   - reason: 详细推理                                         │
│   - acceptable: 整体是否可接受                                │
│                                                             │
│ 用途: 指导下一轮的假说生成                                    │
│ 反馈频率: 每完成一轮完整的 R&D 循环产生一次                    │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 量化场景的反馈生成

```python
# rdagent/scenarios/qlib/developer/feedback.py
class QlibFactorExperiment2Feedback(Experiment2Feedback):
    def generate_feedback(self, exp, trace):
        # 1. 提取当前实验和 SOTA 实验的指标
        current_result = exp.result
        sota_result = exp.based_experiments[-1].result

        # 2. 比较关键指标
        combined_result = process_results(current_result, sota_result)
        # 输出格式: "IC of Current Result is 0.041000, of SOTA Result is 0.035000; ..."

        # 3. LLM 分析比较结果，生成反馈
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=f"假说: {hypothesis_text}\n结果: {combined_result}",
            system_prompt=sys_prompt,
            json_mode=True,
        )
        # 4. 解析反馈
        feedback = HypothesisFeedback(
            decision=response["decision"],      # True: 假说被验证
            observations=response["observations"],
            hypothesis_evaluation=response["hypothesis_evaluation"],
            new_hypothesis=response["new_hypothesis"],
            reason=response["reason"],
        )
        return feedback
```

关键指标：

| 指标 | 含义 | 好的方向 |
|------|------|---------|
| IC | Information Coefficient, 预测值与真实值的相关性 | 越高越好 |
| ICIR | IC Information Ratio, IC 的稳定性 | 越高越好 |
| Rank IC | 排名版 IC | 越高越好 |
| ARR | Annualized Return Rate, 年化收益率 | 越高越好 |
| MDD | Max Drawdown, 最大回撤 | 越小越好 |
| Sharpe | 夏普比率 (ARR/MDD) | 越高越好 |

---

## 九、一个完整的 R&D 周期 Walkthrough

以 Kaggle 竞赛 `tabular-playground-series-dec-2021` 为例，完整走一遍。

### Step 0: 启动

```bash
rdagent data_science --competition tabular-playground-series-dec-2021
```

系统初始化：

```
DataScienceRDLoop.__init__():
  1. scen = KaggleScen("tabular-playground-series-dec-2021")
     - 下载竞赛数据
     - 解析竞赛描述
     - 确定评估指标 (如 RMSLE)
  2. exp_gen = DSProposalV2ExpGen(scen)        ← 假说生成器
  3. data_loader_coder = DataLoaderCoSTEER(scen)
  4. feature_coder = FeatureCoSTEER(scen)
  5. model_coder = ModelCoSTEER(scen)
  6. ensemble_coder = EnsembleCoSTEER(scen)
  7. workflow_coder = WorkflowCoSTEER(scen)
  8. runner = DSCoSTEERRunner(scen)
  9. summarizer = DSExperiment2Feedback(scen)
  10. trace = DSTrace(scen)
```

### Step 1: direct_exp_gen (假说 + 实验设计)

```
ExpGen.gen(trace):
  观察: "这是一个表格数据的回归问题，有 X 个特征..."
  假说: "使用 LightGBM 模型配合基本的数值特征处理应该能建立基线"
  实验设计:
    - pending_tasks: [DataLoaderTask, FeatureTask, ModelTask, WorkflowTask]
    - 每个 task 包含具体描述和约束
```

### Step 2: coding (CoSTEER 逐组件编码)

```
coding(exp):
  for task_group in exp.pending_tasks_list:
    if DataLoaderTask:
      DataLoaderCoSTEER.develop(exp)
      → 生成 load_data.py
      → CoSTEER 内部 10 轮进化，直到数据加载正确

    if FeatureTask:
      FeatureCoSTEER.develop(exp)
      → 生成 feature_engineering.py
      → 确保特征输出格式正确

    if ModelTask:
      ModelCoSTEER.develop(exp)
      → 生成 model_lgbm.py
      → 确保模型能训练和预测

    if WorkflowTask:
      WorkflowCoSTEER.develop(exp)
      → 生成 workflow.py
      → 编排整个训练/预测流程
```

### Step 3: running (Docker 中执行)

```
DSCoSTEERRunner.develop(exp):
  1. 准备 Docker 环境
  2. 将所有代码注入 workspace
  3. 执行 workflow.py
  4. 收集输出:
     - submission.csv
     - 训练日志
     - 验证集分数
```

### Step 4: feedback (LLM 分析结果)

```
DSExperiment2Feedback.generate_feedback(exp, trace):
  当前分数: RMSLE = 0.892
  SOTA 分数: None (第一轮)

  反馈:
    decision: True (基线建立成功)
    observations: "LightGBM 基线成功运行，RMSLE=0.892"
    new_hypothesis: "可以尝试特征交互和目标编码来改善"
```

### Step 5: record (记录到 Trace)

```
trace.sync_dag_parent_and_hist((exp, feedback), loop_idx=0)
  hist = [(exp_0, feedback_0)]
  dag_parent = [()]  # 根节点
```

### Step 6: 下一轮循环

```
第 2 轮:
  假说: "添加特征交互项和多项式特征可以捕捉非线性关系"
  → 只修改 FeatureTask 的代码
  → RMSLE 改善到 0.857
  → decision=True, 设为新 SOTA

第 3 轮:
  假说: "使用 XGBoost + LightGBM 集成可以进一步提升"
  → 添加 ModelTask (XGBoost) + 修改 EnsembleTask
  → RMSLE 改善到 0.831
  → decision=True, 设为新 SOTA

  ...持续迭代，分数不断改善
```

---

## 十、快速上手

### 10.1 安装

```bash
# 1. 创建 conda 环境
conda create -n rdagent python=3.10
conda activate rdagent

# 2. 安装 RD-Agent
pip install rdagent

# 3. 确保 Docker 已安装并可免 sudo 运行
docker run hello-world
```

### 10.2 配置 LLM

```bash
# 方式一: OpenAI
cat << EOF > .env
CHAT_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-xxxxxxxxxxxxx
EOF

# 方式二: Azure OpenAI
cat << EOF > .env
CHAT_MODEL=azure/gpt-4o
EMBEDDING_MODEL=azure/text-embedding-3-small
AZURE_API_KEY=xxxxxxxxxxxxx
AZURE_API_BASE=https://your-endpoint.openai.azure.com/
AZURE_API_VERSION=2024-02-01
EOF

# 方式三: DeepSeek (便宜)
cat << EOF > .env
CHAT_MODEL=deepseek/deepseek-chat
DEEPSEEK_API_KEY=xxxxxxxxxxxxx
EMBEDDING_MODEL=litellm_proxy/BAAI/bge-m3
LITELLM_PROXY_API_KEY=xxxxxxxxxxxxx
LITELLM_PROXY_API_BASE=https://api.siliconflow.cn/v1
EOF
```

### 10.3 健康检查

```bash
rdagent health_check
```

### 10.4 运行不同场景

```bash
# 量化因子进化
rdagent fin_factor

# 量化模型进化
rdagent fin_model

# 因子+模型联合进化 (推荐)
rdagent fin_quant

# 从金融报告提取因子
rdagent fin_factor_report --report-folder=reports/

# 从论文实现模型
rdagent general_model "https://arxiv.org/pdf/2210.09789"

# Kaggle 竞赛
dotenv set DS_LOCAL_DATA_PATH "$(pwd)/git_ignore_folder/ds_data"
dotenv set DS_CODER_ON_WHOLE_PIPELINE True
dotenv set DS_IF_USING_MLE_DATA True
dotenv set DS_SAMPLE_DATA_BY_LLM True
dotenv set DS_SCEN rdagent.scenarios.data_science.scen.KaggleScen
rdagent data_science --competition tabular-playground-series-dec-2021

# 医疗预测模型
wget https://github.com/SunsetWolf/rdagent_resource/releases/download/ds_data/arf-12-hours-prediction-task.zip
unzip arf-12-hours-prediction-task.zip -d ./git_ignore_folder/ds_data/
dotenv set DS_LOCAL_DATA_PATH "$(pwd)/git_ignore_folder/ds_data"
dotenv set DS_SCEN rdagent.scenarios.data_science.scen.DataScienceScen
rdagent data_science --competition arf-12-hours-prediction-task
```

### 10.5 查看运行结果

```bash
# Streamlit UI
rdagent ui --port 19899 --log-dir log/ --data-science

# Web UI (需先构建前端)
cd web && npm install && npm run build:flask && cd ..
rdagent server_ui --port 19899
```

### 10.6 配置系统 (BasePropSetting)

每个场景通过 `BasePropSetting` 子类配置所有组件：

```python
# rdagent/app/qlib_rd_loop/conf.py
class FactorBasePropSetting(BasePropSetting):
    scen: str = "rdagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
    hypothesis_gen: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesisGen"
    hypothesis2experiment: str = "rdagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment"
    coder: str = "rdagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    runner: str = "rdagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    summarizer: str = "rdagent.scenarios.qlib.developer.feedback.QlibFactorExperiment2Feedback"
    evolving_n: int = 10
```

所有类名都是字符串，通过 `import_class()` 动态加载。这意味着你可以通过**环境变量**替换任何组件，而不需要修改代码：

```bash
# 例如: 替换假说生成器
export QLIB_FACTOR_HYPOTHESIS_GEN="my_custom_module.MyHypothesisGen"
```

---

## 十一、Benchmark 成绩

### 11.1 MLE-Bench (机器学习工程 Benchmark)

[MLE-bench](https://github.com/openai/mle-bench) 使用 75 个 Kaggle 竞赛评估 AI agent 的 ML 工程能力。RD-Agent 目前是公开排行榜第一：

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MLE-Bench 成绩                               │
│                                                                     │
│  Agent                          Low(%)   Medium(%)  High(%)  All(%) │
│  ────────────────────────────── ──────── ────────── ──────── ────── │
│  R&D-Agent o3(R)+GPT-4.1(D)   51.5±6.9 19.3±5.5   26.7±0   30.2  │
│  R&D-Agent o1-preview         48.2±2.5  8.9±2.4   18.7±3.0  22.4  │
│  AIDE o1-preview              34.3±2.4  8.8±1.1   10.0±1.9  16.9  │
│                                                                     │
│  Low: <2h  |  Medium: 2-10h  |  High: >10h (人类工程师估计时间)       │
└─────────────────────────────────────────────────────────────────────┘
```

**关键设计洞察**：o3(R)+GPT-4.1(D) 版本使用不同 LLM 处理 R 和 D 阶段：
- **Research Agent (o3)**: 用推理更强的模型来提假说
- **Development Agent (GPT-4.1)**: 用更快更便宜的模型来写代码

这种分离式设计使得：
- 平均每轮循环时间减少
- 成本更低（代码生成不需要最贵的模型）
- 整体效果反而更好

### 11.2 量化金融 (R&D-Agent-Quant)

在真实股票市场的实验中（论文发表于 NeurIPS 2025）：

```
┌──────────────────────────────────────────────────────────────┐
│  R&D-Agent(Q) 实验结果 (成本 < $10)                          │
│                                                              │
│  vs 基准因子库:                                               │
│    年化收益率: ~2x 提升                                       │
│    因子数量:   减少 70% 以上                                   │
│                                                              │
│  vs SOTA 深度时序模型:                                        │
│    在更少的资源预算下超越最先进的深度时序模型                     │
│                                                              │
│  因子-模型交替优化:                                           │
│    在预测精度和策略鲁棒性之间实现优秀的 trade-off               │
└──────────────────────────────────────────────────────────────┘
```

---

## 十二、设计哲学

### 12.1 为什么 R+D 分离如此重要

```
❌ 单循环方案 (如简单的 Agent 编程):
   LLM → 写代码 → 运行 → 看结果 → 再写代码
   问题: 没有系统性的假说驱动，容易随机游走

✅ R+D 双循环方案 (RD-Agent):
   R: 基于历史反馈，系统性地提出新假说
   D: 基于假说，多轮进化地实现代码
   问题被分解: "想什么" 和 "怎么做" 解耦
```

**分离的好处：**

| 方面 | R 循环 | D 循环 |
|------|--------|--------|
| 核心能力 | 推理、创意、策略 | 编码、调试、优化 |
| 适合的 LLM | 推理型 (o3, o1) | 编码型 (GPT-4.1, Claude) |
| 迭代频率 | 每个 R&D 循环一次 | 每轮内部多次 (CoSTEER 10轮) |
| 失败代价 | 浪费一轮假说 | 代码可以重试/回退 |

### 12.2 CoSTEER 的进化式编码 vs 一次性生成

```
❌ 一次性生成:
   LLM → 代码 → 完
   问题: LLM 一次生成正确代码的概率不高，尤其是复杂任务

✅ CoSTEER 进化式:
   LLM → 代码 v1 → 评估 → 反馈 → LLM → 代码 v2 → 评估 → ... → 代码 vN
   优势:
   - 每轮有具体的错误反馈 (执行错误/返回值错误/代码质量)
   - RAG 知识库积累经验 (之前的成功/失败案例)
   - 自动 fallback 到最佳版本
```

### 12.3 DAG Trace vs 线性历史

```
❌ 线性历史:
   exp0 → exp1 → exp2 → exp3
   问题: 只能沿着一条路线走，不能回退到早期探索新方向

✅ DAG Trace:
   exp0 ──→ exp1 ──→ exp3
     │
     └───→ exp2 ──→ exp4
   优势:
   - 支持多分支探索
   - 可以从任意节点出发
   - 并行探索多条路线 (async)
   - MCTS 调度器可以智能选择探索方向
```

### 12.4 全组件可替换的插件式架构

```python
# 所有组件都是字符串路径，通过 import_class() 动态加载
class BasePropSetting:
    scen: str | None = None                # 场景类
    hypothesis_gen: str | None = None      # 假说生成器
    hypothesis2experiment: str | None = None # 假说转实验
    coder: str | None = None               # 编码器
    runner: str | None = None              # 运行器
    summarizer: str | None = None          # 反馈生成器
```

这种设计意味着：
- **用户可以通过环境变量替换任何组件**，不需要改代码
- **新场景只需要实现对应的接口**，就能复用整个框架
- **不同 LLM 可以用在不同阶段**，优化成本/性能平衡

### 12.5 Workspace 与 Docker 隔离

所有实验代码都在 Docker 容器中执行，确保：
- **安全性**：LLM 生成的代码不会影响主机
- **可重现性**：每次实验在相同环境中运行
- **隔离性**：不同实验互不干扰

```python
# rdagent/core/experiment.py
class FBWorkspace(Workspace):
    def execute(self, env: Env, entry: str) -> str:
        self.prepare()
        self.inject_files(**self.file_dict)
        return env.run(entry, str(self.workspace_path))
```

---

## 总结

RD-Agent 的核心创新在于将 AI 研发过程结构化为 **Research + Development 双循环**：

1. **Research**: LLM 扮演研究员，基于历史反馈提出新假说
2. **Development**: CoSTEER 进化式编码器，多轮迭代生成代码
3. **Evaluation**: 自动化评估，提供多维度反馈
4. **Evolution**: 知识库 + DAG Trace，持续积累经验

这种设计使得 RD-Agent 在 MLE-Bench 上取得了 30.2% 的最佳成绩，并在量化金融实验中以不到 $10 的成本实现了 2x 基准收益。

---

*Generated on 2026-03-28 from RD-Agent source code analysis*
