# AI-Scientist 是怎么工作的？ — 从原理到代码的完整教程

> AI-Scientist = **The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery**
>
> SakanaAI 出品，一句话：**让 LLM 自己提出研究想法、写代码跑实验、生成 LaTeX 论文、再自我评审打分 — 全自动完成一个完整的科研周期**

---

## 一、先看结果：AI-Scientist 输入什么，输出什么

```
输入: 一个实验模板 (如 nanoGPT) + 一个 LLM (如 Claude 3.5 Sonnet)

输出:
  results/nanoGPT/
  └── 20240815_120000_adaptive_block_size/
      ├── experiment.py               <- AI 修改后的实验代码
      ├── run_0/final_info.json       <- 基线结果
      ├── run_1/final_info.json       <- 实验结果 (run 1)
      ├── run_2/final_info.json       <- 实验结果 (run 2)
      ├── run_1.py, run_2.py          <- 每次 run 的代码快照
      ├── plot.py                     <- AI 修改后的画图代码
      ├── *.png                       <- 生成的图表
      ├── notes.txt                   <- 实验笔记
      ├── latex/template.tex          <- 完整 LaTeX 论文
      ├── adaptive_block_size.pdf     <- 编译后的 PDF 论文
      ├── review.txt                  <- AI 审稿意见 (JSON)
      ├── log.txt                     <- 完整运行日志
      └── *_aider.txt                 <- Aider 对话历史
```

**一个典型的输出论文长什么样？** 项目提供了 10 篇示例论文，比如：

| 论文标题 | 模板 | 说明 |
|---------|------|------|
| DualScale Diffusion: Adaptive Feature Balancing for Low-Dimensional Generative Models | 2D Diffusion | 提出双尺度去噪方法 |
| StyleFusion: Adaptive Multi-style Generation in Character-Level Language Models | NanoGPT | 多风格文本生成 |
| Grokking Through Compression: Unveiling Sudden Generalization via MDL | Grokking | 用最小描述长度解释 Grokking |
| Adaptive Learning Rates for Transformers via Q-Learning | NanoGPT | 用强化学习调学习率 |

每篇论文都是完整的学术格式：有摘要、引言、方法、实验、结论、参考文献，配有代码生成的图表。**整个过程不需要人参与。**

---

## 二、为什么它是"真正的AI科学家"

### 和 STORM / DeerFlow 的本质区别

```
                   STORM / DeerFlow              AI-Scientist
                   ───────────────              ─────────────
做的事:            搜索+综述已有知识             提出新想法+跑实验+写论文
类比:              图书管理员/综述作者           研究生/科研工作者
输入:              一个主题 (字符串)             一个代码模板 (可运行的实验)
核心动作:          搜索引擎 → 整理 → 写综述     生成idea → 改代码 → 跑GPU → 写论文
产出新知识?        否 (总结已有知识)             是 (通过实验发现新结果)
需要GPU?           否                           是 (要跑实验)
```

**关键区别**：STORM 从互联网*搜集*信息然后*整理*成文章，本质是一个更高级的搜索引擎。AI-Scientist 是*创造*新知识 — 它修改代码、在 GPU 上跑真实实验、分析结果、然后报告发现。这更接近真正的科研过程。

```
STORM/DeerFlow 的循环:
  主题 → 搜索 → 阅读 → 综述 → 文章

AI-Scientist 的循环:
  模板 → 想idea → 改代码 → 跑实验 → 分析结果 → 写论文 → 评审 → (改进)
```

---

## 三、全局流水线：五个阶段

```
阶段1           阶段2           阶段3          阶段4           阶段5
Idea生成        实验执行         论文写作        模拟评审        (可选)改进
(Generate)     (Experiment)    (Writeup)      (Review)       (Improve)

┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ LLM生成   │  │ Aider改  │  │ Aider填  │  │ LLM打分  │  │ 基于评审  │
│ 研究想法  │→ │ 实验代码  │→ │ LaTeX模板 │→ │ 写Review │→ │ 改进论文  │
│ +新颖性  │  │ +跑实验  │  │ +加引用  │  │ Accept/  │  │ 重新评审  │
│  检查     │  │ +画图    │  │ +编译PDF │  │ Reject   │  │          │
└──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
     ↓              ↓              ↓              ↓              ↓
  ideas.json    run_*/          template.tex   review.txt    *_improved.pdf
  (JSON列表)    final_info.json  → PDF        (JSON评审)    review_improved.txt
```

**入口文件是 `launch_scientist.py`，核心调用链如下：**

```python
# launch_scientist.py — 主流程 (简化版)

# 阶段 1: 生成 ideas + 新颖性检查
ideas = generate_ideas(base_dir, client, model, ...)
ideas = check_idea_novelty(ideas, base_dir, client, model, ...)

# 对每个被判定为 novel 的 idea:
for idea in novel_ideas:
    # 阶段 2: 跑实验
    success = perform_experiments(idea, folder_name, coder, baseline_results)

    # 阶段 3: 写论文
    perform_writeup(idea, folder_name, coder, client, client_model)

    # 阶段 4: AI 评审
    paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
    review = perform_review(paper_text, model="gpt-4o-2024-05-13", client=openai.OpenAI(), ...)

    # 阶段 5 (可选): 基于评审改进
    if improvement:
        perform_improvement(review, coder)
```

---

## 四、Idea 生成 — 让 LLM 当研究生想点子

### 4.1 输入：模板上下文

LLM 不是凭空想 idea。它能看到：
1. **任务描述** (`prompt.json` 中的 `task_description`)
2. **实验代码** (`experiment.py` 的全部源码)
3. **已有 ideas** (种子 ideas + 之前生成的 ideas)

```python
# ai_scientist/generate_ideas.py

idea_first_prompt = """{task_description}
<experiment.py>
{code}
</experiment.py>

Here are the ideas that you have already generated:
'''
{prev_ideas_string}
'''

Come up with the next impactful and creative idea for research experiments
and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
...
"""
```

**种子 Idea 示例 (nanoGPT 模板)：**

```json
[
  {
    "Name": "adaptive_block_size",
    "Title": "Adaptive Block Size: Dynamic Context Window Adjustment for Efficient Training",
    "Experiment": "Modify the model to dynamically adjust its block size during training...",
    "Interestingness": 6,
    "Feasibility": 4,
    "Novelty": 4
  },
  {
    "Name": "layerwise_learning_rates",
    "Title": "Layer-wise Learning Rate Adaptation: Optimizing Training Dynamics...",
    "Experiment": "Implement layer-wise learning rates, where each transformer layer...",
    "Interestingness": 4,
    "Feasibility": 6,
    "Novelty": 2
  }
]
```

### 4.2 迭代式自我反思 (Reflection)

不是一次生成就完事。每个 idea 都经过多轮反思打磨：

```
Round 1: LLM 生成初始 idea (含 Name, Title, Experiment, 评分)
    ↓
Round 2: LLM 回顾 idea, 评估质量/新颖性/可行性, 改进
    ↓
Round 3: LLM 继续改进, 如果满意则输出 "I am done"
    ↓
  ...最多 num_reflections 轮 (默认 5 轮)
```

```python
# generate_ideas() 核心循环
for _ in range(max_num_generations):  # 默认生成 50 个 ideas
    msg_history = []

    # 第 1 轮: 生成初始 idea
    text, msg_history = get_response_from_llm(
        idea_first_prompt.format(...),
        client=client, model=model,
        system_message=idea_system_prompt,
        msg_history=msg_history,
    )
    json_output = extract_json_between_markers(text)

    # 第 2~N 轮: 反思改进
    for j in range(num_reflections - 1):
        text, msg_history = get_response_from_llm(
            idea_reflection_prompt.format(
                current_round=j + 2,
                num_reflections=num_reflections
            ),
            client=client, model=model,
            system_message=idea_system_prompt,
            msg_history=msg_history,
        )
        json_output = extract_json_between_markers(text)
        if "I am done" in text:
            break  # idea 已经够好了

    idea_str_archive.append(json.dumps(json_output))
```

### 4.3 新颖性检查 — 用 Semantic Scholar 查重

生成 idea 后，不是直接跑实验，而是先检查这个 idea 是否已经有人做过了。

```
┌──────────────────────────────────────────────────────────────┐
│                    新颖性检查流程                              │
│                                                              │
│  idea: "Adaptive Block Size for Transformer Training"        │
│                                                              │
│  Round 1:                                                    │
│    LLM 思考: "我应该搜索 adaptive block size transformer"     │
│    → 调用 Semantic Scholar API, 返回 top 10 论文             │
│    → LLM 阅读摘要, 判断是否和自己的 idea 重叠               │
│                                                              │
│  Round 2:                                                    │
│    LLM 思考: "让我换个角度搜 dynamic context window training" │
│    → 再搜一次, 再看 10 篇                                    │
│                                                              │
│  Round 3:                                                    │
│    LLM: "没有找到显著重叠的工作"                               │
│    → "Decision made: novel."                                 │
│                                                              │
│  或者:                                                       │
│    LLM: "这篇 Smith et al. 2023 做了几乎一样的事"             │
│    → "Decision made: not novel."                             │
└──────────────────────────────────────────────────────────────┘
```

```python
# check_idea_novelty() — 核心逻辑
def check_idea_novelty(ideas, base_dir, client, model, max_num_iterations=10):
    for idx, idea in enumerate(ideas):
        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            text, msg_history = get_response_from_llm(
                novelty_prompt.format(
                    current_round=j + 1,
                    num_rounds=max_num_iterations,
                    idea=idea,
                    last_query_results=papers_str,
                ),
                client=client, model=model, ...
            )

            if "decision made: novel" in text.lower():
                novel = True
                break
            if "decision made: not novel" in text.lower():
                break

            # 解析搜索 query, 调 Semantic Scholar API
            json_output = extract_json_between_markers(text)
            query = json_output["Query"]
            papers = search_for_papers(query, result_limit=10)
            # 格式化论文列表给 LLM 看
            papers_str = format_papers(papers)

        idea["novel"] = novel
```

**搜索引擎支持两种后端：**

| 后端 | 需要 API Key | 备注 |
|------|-------------|------|
| Semantic Scholar | 推荐 (`S2_API_KEY`) | 默认, 学术搜索更精准 |
| OpenAlex | 不需要 (免费, 需设邮箱) | 备选方案, 实验性 |

**只有被标记为 `novel=True` 的 ideas 才会进入后续阶段。**

---

## 五、实验设计与执行 — AI 改代码、跑 GPU

### 5.1 核心工具：Aider

AI-Scientist 不是自己从头写代码，而是使用 **[Aider](https://aider.chat/)** — 一个 AI 编程助手 — 来修改现有代码。Aider 可以理解代码上下文，生成 diff 格式的修改。

```python
# launch_scientist.py 中创建 Aider coder 实例
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput

fnames = [exp_file, vis_file, notes]  # experiment.py, plot.py, notes.txt
io = InputOutput(yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt")
main_model = Model(model)  # 如 "claude-3-5-sonnet-20241022"

coder = Coder.create(
    main_model=main_model,
    fnames=fnames,          # Aider 能看到和修改的文件
    io=io,
    stream=False,
    use_git=False,          # 不用 git, 直接改文件
    edit_format="diff",     # 用 diff 格式 (搜索/替换块)
)
```

### 5.2 实验执行循环

```
┌─────────────────────────────────────────────────────────────────────┐
│                      实验执行流程                                    │
│                                                                     │
│  Step 1: 把 idea 和 baseline 结果告诉 Aider                        │
│    "你的目标是实现这个 idea: {title}                                 │
│     实验计划: {experiment}                                          │
│     你有最多 5 次运行机会. baseline 结果如下: {baseline_results}"     │
│                                                                     │
│  Step 2: Aider 修改 experiment.py (通过 SEARCH/REPLACE 块)          │
│                                                                     │
│  Step 3: 系统自动执行:                                               │
│    $ python experiment.py --out_dir=run_1                           │
│    (超时限制: 7200 秒 = 2 小时)                                      │
│                                                                     │
│  Step 4: 如果成功 → 把结果告诉 Aider, 问它要不要继续                 │
│          如果失败 → 把错误信息告诉 Aider, 让它修复 (最多 4 次重试)   │
│                                                                     │
│  Step 5: Aider 说 "ALL_COMPLETED" 或用完 5 次 run → 结束           │
│                                                                     │
│  Step 6: Aider 修改 plot.py 生成图表                                │
│  Step 7: Aider 更新 notes.txt 写实验笔记                            │
└─────────────────────────────────────────────────────────────────────┘
```

**对应代码：**

```python
# ai_scientist/perform_experiments.py

MAX_ITERS = 4   # 每次 run 最多重试 4 次
MAX_RUNS = 5    # 最多跑 5 次实验
MAX_STDERR_OUTPUT = 1500

def perform_experiments(idea, folder_name, coder, baseline_results) -> bool:
    current_iter = 0
    run = 1

    # 初始 prompt: 告诉 Aider 要实现什么
    next_prompt = coder_prompt.format(
        title=idea["Title"],
        idea=idea["Experiment"],
        max_runs=MAX_RUNS,
        baseline_results=baseline_results,
    )

    while run < MAX_RUNS + 1:
        if current_iter >= MAX_ITERS:
            break

        # Aider 修改代码
        coder_out = coder.run(next_prompt)
        if "ALL_COMPLETED" in coder_out:
            break

        # 运行实验
        return_code, next_prompt = run_experiment(folder_name, run)
        if return_code == 0:
            run += 1          # 成功 → 下一个 run
            current_iter = 0  # 重置重试计数
        current_iter += 1     # 失败 → 重试计数 +1

    # 画图阶段
    next_prompt = """Please modify plot.py to generate the most relevant plots...
    Only the runs in the `labels` dictionary will be plotted..."""
    while True:
        coder.run(next_prompt)
        return_code, next_prompt = run_plotting(folder_name)
        current_iter += 1
        if return_code == 0 or current_iter >= MAX_ITERS:
            break

    # 写实验笔记
    coder.run("Please modify notes.txt with a description of what each plot shows...")
    return True
```

### 5.3 实验运行的关键细节

```python
def run_experiment(folder_name, run_num, timeout=7200):
    cwd = osp.abspath(folder_name)

    # 保存当前代码快照 (便于追溯每次 run 的代码版本)
    shutil.copy(
        osp.join(folder_name, "experiment.py"),
        osp.join(folder_name, f"run_{run_num}.py"),
    )

    # 固定的命令格式, 不允许额外参数
    command = ["python", "experiment.py", f"--out_dir=run_{run_num}"]

    result = subprocess.run(command, cwd=cwd, stderr=subprocess.PIPE,
                           text=True, timeout=timeout)

    if result.returncode != 0:
        # 失败 → 删除不完整的结果目录
        shutil.rmtree(osp.join(cwd, f"run_{run_num}"), ignore_errors=True)
        # 截断错误信息 (最多 1500 字符), 反馈给 Aider
        stderr_output = result.stderr[-MAX_STDERR_OUTPUT:]
        next_prompt = f"Run failed with the following error {stderr_output}"
    else:
        # 成功 → 读取结果, 反馈给 Aider
        with open(osp.join(cwd, f"run_{run_num}", "final_info.json")) as f:
            results = json.load(f)
        next_prompt = f"Run {run_num} completed. Here are the results:\n{results}\n..."

    return result.returncode, next_prompt
```

**重要设计决策：**

- 实验代码每次 run 都保存快照 (`run_1.py`, `run_2.py`, ...)
- 超时 2 小时 — 防止实验代码死循环
- 失败时自动清理不完整的结果目录
- 错误信息截断到 1500 字符 — 避免超出 LLM 上下文窗口
- 固定命令格式 `--out_dir=run_i` — 标准化输出路径

---

## 六、论文写作 — 逐节填充 LaTeX 模板

### 6.1 写作流程总览

AI-Scientist 使用预置的 LaTeX 模板 (`template.tex`)，通过 Aider 逐节填充内容：

```
┌─────────────────────────────────────────────────────────────────┐
│                        论文写作流程                              │
│                                                                 │
│  Phase 1: 逐节填充 (7 个 section, 每个 2 轮)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Abstract │→│  Intro   │→│Background│→│  Method  │→ ...     │
│  │ (写+改)  │ │ (写+改)  │ │ (写+改)  │ │ (写+改)  │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│                                                                 │
│  Phase 2: Related Work + 自动加引用 (最多 20 轮)               │
│  ┌──────────────────────────────────────────────────┐          │
│  │ 草拟 Related Work → 搜索 Semantic Scholar        │          │
│  │ → 选最相关论文 → 加 bibtex → Aider 整合到正文    │          │
│  │ → 重复直到 "No more citations needed"            │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
│  Phase 3: 全文二次打磨 (8 个 section 再过一遍)                  │
│  ┌──────────────────────────────────────────────────┐          │
│  │ 重新考虑标题 → Abstract → Related Work → ...     │          │
│  │ 精简冗余、统一风格、修复错误                       │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
│  Phase 4: LaTeX 编译 + 错误修复                                 │
│  ┌──────────────────────────────────────────────────┐          │
│  │ chktex 检查 → Aider 修复 → pdflatex 编译 → PDF  │          │
│  └──────────────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 每个 Section 的写作指导

系统为每个 section 提供了详细的写作建议（`per_section_tips`）：

```python
# ai_scientist/perform_writeup.py

per_section_tips = {
    "Abstract": """
    - TL;DR of the paper
    - What are we trying to do and why is it relevant?
    - Why is this hard?
    - How do we solve it (i.e. our contribution!)
    - How do we verify that we solved it (e.g. Experiments and results)
    Please make sure the abstract reads smoothly and is well-motivated.
    This should be one continuous paragraph with no breaks.""",

    "Introduction": """
    - Longer version of the Abstract, i.e. of the entire paper
    - New trend: specifically list your contributions as bullet points
    - Extra space? Future work!""",

    "Related Work": """
    - Academic siblings of our work
    - Goal is to "Compare and contrast"
    - Note: Just describing what another paper is doing is not enough.""",

    "Method": """
    - What we do. Why we do it.
    - All described using the general Formalism introduced in the Problem Setting.""",

    "Results": """
    - Only includes results that have actually been run and saved in the logs.
    - Do not hallucinate results that don't exist.
    - If results exist: compares to baselines and includes statistics.""",
    # ... 其他 sections
}
```

### 6.3 逐节写作 + 两轮打磨

```python
# perform_writeup() — 逐节写作
def perform_writeup(idea, folder_name, coder, cite_client, cite_model, ...):
    # === Phase 1: 逐节填充 ===
    # 先写 Abstract (含 Title)
    coder.run(abstract_prompt)
    coder.run(refinement_prompt.format(section="Abstract"))  # 第 1 轮打磨

    # 依次写其他 sections
    for section in ["Introduction", "Background", "Method",
                    "Experimental Setup", "Results", "Conclusion"]:
        coder.run(section_prompt)           # 写
        coder.run(refinement_prompt)        # 打磨

    # === Phase 2: Related Work + 自动引用 ===
    coder.run(related_work_sketch_prompt)   # 先草拟结构

    for _ in range(num_cite_rounds):        # 最多 20 轮加引用
        draft = read_template_tex()
        prompt, done = get_citation_aider_prompt(cite_client, cite_model, draft, ...)
        if done:
            break
        if prompt is not None:
            # 自动把 bibtex 条目插入 references.bib
            bibtex_string = prompt.split('"""')[1]
            draft = draft.replace(r"\end{filecontents}", f"{bibtex_string}\\end{{filecontents}}")
            write_template_tex(draft)
            coder.run(prompt)  # Aider 整合引用到正文

    coder.run(refinement_prompt.format(section="Related Work"))

    # === Phase 3: 全文二次打磨 ===
    coder.run("Re-think the Title if necessary...")
    for section in ["Abstract", "Related Work", "Introduction", "Background",
                    "Method", "Experimental Setup", "Results", "Conclusion"]:
        coder.run(second_refinement_prompt.format(section=section, tips=...))

    # === Phase 4: LaTeX 编译 ===
    generate_latex(coder, folder_name, f"{folder_name}/{idea['Name']}.pdf")
```

### 6.4 自动引用系统

引用添加过程是一个两步对话循环：

```
┌──────────────────────────────────────────────────────────────────┐
│                   引用添加循环 (每轮)                              │
│                                                                  │
│  Step 1: LLM 阅读当前论文草稿                                    │
│    → 输出: "在 Method 第 2 段需要引用 attention mechanism 的论文"  │
│    → 输出: Query = "attention is all you need"                   │
│                                                                  │
│  Step 2: 调用 Semantic Scholar API, 返回 top 10                  │
│    → LLM 选择最相关的论文 (可多选)                                │
│    → 输出: Selected = [0, 3], Description = "在 xx 处加 \cite"    │
│                                                                  │
│  Step 3: 系统自动把选中论文的 bibtex 插入 references.bib          │
│    → Aider 根据 description 把 \cite{} 加到正文                  │
│                                                                  │
│  重复直到 LLM 说 "No more citations needed" (最多 20 轮)         │
└──────────────────────────────────────────────────────────────────┘
```

### 6.5 LaTeX 编译与错误修复

```python
def generate_latex(coder, folder_name, pdf_file, timeout=30, num_error_corrections=5):
    # 1. 检查所有 \cite{} 引用是否在 references.bib 中存在
    cites = re.findall(r"\\cite[a-z]*{([^}]*)}", tex_text)
    for cite in cites:
        if cite not in bib_text:
            coder.run(f"Reference {cite} not found in references.bib...")

    # 2. 检查所有 \includegraphics{} 引用的图片是否存在
    referenced_figs = re.findall(r"\\includegraphics.*?{(.*?)}", tex_text)
    all_figs = [f for f in os.listdir(folder) if f.endswith(".png")]
    for figure in referenced_figs:
        if figure not in all_figs:
            coder.run(f"The image {figure} not found in the directory...")

    # 3. 检查并修复重复的 figure 和 section header
    # 4. 用 chktex 检查 LaTeX 语法错误, 让 Aider 修复 (最多 5 轮)
    for i in range(num_error_corrections):
        check_output = os.popen(f"chktex {writeup_file} -q -n2 -n24 -n13 -n1").read()
        if check_output:
            coder.run(f"Please fix the following LaTeX errors:\n{check_output}")
        else:
            break

    # 5. 编译 LaTeX → PDF
    compile_latex(cwd, pdf_file)  # pdflatex → bibtex → pdflatex → pdflatex
```

---

## 七、模拟 Peer Review — AI 当审稿人

### 7.1 评审格式：NeurIPS 标准

AI-Scientist 使用 NeurIPS 会议的评审表格式，评审维度包括：

| 维度 | 评分范围 | 说明 |
|------|---------|------|
| Originality | 1-4 | 新颖性 |
| Quality | 1-4 | 技术质量 |
| Clarity | 1-4 | 表达清晰度 |
| Significance | 1-4 | 重要性 |
| Soundness | 1-4 | 技术可靠性 |
| Presentation | 1-4 | 展示质量 |
| Contribution | 1-4 | 贡献程度 |
| Overall | 1-10 | 总分 (1=Very Strong Reject, 10=Award quality) |
| Confidence | 1-5 | 评审信心 |
| Decision | Accept/Reject | 最终决定 |

此外还有文字性评审：Summary, Strengths, Weaknesses, Questions, Limitations。

### 7.2 评审流程：Ensemble + Reflection

```
┌──────────────────────────────────────────────────────────────────────┐
│                        评审流程                                       │
│                                                                      │
│  输入: PDF 论文 (通过 pymupdf4llm 转为 Markdown 文本)                │
│                                                                      │
│  Step 1: 加载 Few-shot 示例                                          │
│    → 真实的 ML 论文 + 真实的人类评审 (最多 3 对)                      │
│    → 让 LLM 学会评审的格式和标准                                      │
│                                                                      │
│  Step 2: Ensemble 评审 (默认 5 个独立评审)                            │
│    ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐          │
│    │Review 1│ │Review 2│ │Review 3│ │Review 4│ │Review 5│          │
│    │(t=0.75)│ │(t=0.75)│ │(t=0.75)│ │(t=0.75)│ │(t=0.75)│          │
│    └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘          │
│        └──────────┼──────────┼──────────┼──────────┘                 │
│                   ↓                                                  │
│         ┌─────────────────┐                                          │
│         │  Meta-Reviewer   │ ← 聚合 5 个评审意见                     │
│         │  (Area Chair)    │                                         │
│         └────────┬────────┘                                          │
│                  ↓                                                   │
│         数值分数: 取 5 个评审的平均值                                  │
│         文字评审: Meta-reviewer 综合                                  │
│                                                                      │
│  Step 3: Reflection (默认 5 轮)                                      │
│    → LLM 回顾自己的评审, 检查准确性和合理性                           │
│    → 如果有改进空间, 修正评审; 否则 "I am done"                       │
└──────────────────────────────────────────────────────────────────────┘
```

```python
# ai_scientist/perform_review.py

def perform_review(text, model, client,
                   num_reflections=5,        # 反思轮数
                   num_fs_examples=1,        # few-shot 示例数
                   num_reviews_ensemble=5,    # 集成评审数
                   temperature=0.1):         # 低温度 = 更稳定

    # 1. 构建 prompt: 评审表 + few-shot 示例 + 论文全文
    base_prompt = review_instruction_form + fs_prompt + paper_text

    # 2. 获取 N 个独立评审
    llm_review, msg_histories = get_batch_responses_from_llm(
        base_prompt, model=model, client=client,
        temperature=0.75,  # 高温度鼓励多样性
        n_responses=num_reviews_ensemble,
    )

    # 3. 解析每个评审
    parsed_reviews = [extract_json_between_markers(rev) for rev in llm_review]

    # 4. Meta-reviewer 聚合
    review = get_meta_review(model, client, temperature, parsed_reviews)

    # 5. 数值分数取平均
    for score in ["Originality", "Quality", "Clarity", "Significance",
                  "Soundness", "Presentation", "Contribution",
                  "Overall", "Confidence"]:
        scores = [r[score] for r in parsed_reviews if score in r]
        review[score] = int(round(np.mean(scores)))

    # 6. 反思改进
    for j in range(num_reflections - 1):
        text, msg_history = get_response_from_llm(
            reviewer_reflection_prompt, ...)
        review = extract_json_between_markers(text)
        if "I am done" in text:
            break

    return review
```

### 7.3 Few-shot 示例

系统自带 3 对真实论文-评审样本，用于让 LLM 学习评审标准：

```
ai_scientist/fewshot_examples/
├── 132_automated_relational.pdf   + .json (评审)
├── attention.pdf                  + .json (Attention is All You Need 的评审)
└── 2_carpe_diem.pdf               + .json (评审)
```

### 7.4 评审后改进 (可选)

如果启用 `--improvement` 标志，系统会基于评审意见改进论文：

```python
def perform_improvement(review, coder):
    improvement_prompt = '''The following review has been created for your paper:
    """
    {review}
    """
    Improve the text using the review.'''
    coder.run(improvement_prompt.format(review=json.dumps(review)))
```

改进后重新编译 PDF，再次评审，保存为 `review_improved.txt`。

---

## 八、实验模板系统

### 8.1 模板结构

每个模板是一个独立的目录，包含运行实验所需的全部文件：

```
templates/<experiment_name>/
├── experiment.py       <- 核心实验代码 (AI 会修改这个文件)
├── plot.py             <- 画图代码 (AI 会修改这个文件)
├── prompt.json         <- 任务描述 (system prompt + task_description)
├── seed_ideas.json     <- 种子 ideas (给 LLM 的示例)
├── latex/              <- LaTeX 模板
│   ├── template.tex    <- 论文模板 (含预置引用)
│   ├── iclr2024_conference.sty
│   ├── iclr2024_conference.bst
│   ├── natbib.sty
│   └── fancyhdr.sty
├── run_0/              <- 基线结果 (需要预先跑好)
│   └── final_info.json
└── (其他数据/依赖文件)
```

### 8.2 官方模板 (3 个)

| 模板 | 研究领域 | experiment.py 做什么 |
|------|---------|---------------------|
| **nanoGPT** | Transformer 语言模型 | 在字符级数据集 (shakespeare, enwik8, text8) 上训练小型 GPT |
| **nanoGPT_lite** | 同上 (轻量版) | 同上, 但更快完成 (适合测试) |
| **2d_diffusion** | 扩散生成模型 | 在 2D 数据集上训练 diffusion model, 评估 KL 散度 |
| **grokking** | 泛化/过拟合现象 | 在模运算任务上训练 Transformer, 研究 grokking 现象 |

### 8.3 社区贡献模板 (7 个)

| 模板 | 研究领域 | PR |
|------|---------|-----|
| seir | 传染病建模 | #137 |
| mobilenetV3 | 图像分类 | #141 |
| sketch_rnn | 手绘图生成 | #143 |
| MACE | 量子化学 AI | #157 |
| earthquake-prediction | 地震预测 | #167 |
| tensorf | 神经辐射场 | #175 |
| probes | LLM Steering/Probes | #215 |

### 8.4 如何创建自己的模板

**关键约束：**

1. `experiment.py` 必须接受 `--out_dir` 参数
2. 运行结果必须保存到 `{out_dir}/final_info.json`
3. `final_info.json` 格式: `{"metric_name": {"means": value, ...}}`
4. 必须先手动跑一次 `run_0` 作为基线

```bash
# 创建新模板的最小步骤
mkdir templates/my_experiment
cd templates/my_experiment

# 1. 写 experiment.py (接受 --out_dir 参数, 输出 final_info.json)
# 2. 写 plot.py (从 run_* 目录读取结果, 生成 .png 图)
# 3. 写 prompt.json
cat > prompt.json << 'EOF'
{
    "system": "You are an ambitious AI researcher...",
    "task_description": "You are given code that trains a model on ..."
}
EOF

# 4. 写 seed_ideas.json (至少 1-2 个示例 idea)
# 5. 复制 latex/ 目录, 修改 template.tex 中的预置引用
# 6. 跑基线
python experiment.py --out_dir run_0
python plot.py
```

---

## 九、一个完整的研究周期 Walkthrough

让我们完整走一遍，以 `nanoGPT_lite` 模板 + Claude 3.5 Sonnet 为例：

```
$ python launch_scientist.py \
    --model claude-3-5-sonnet-20241022 \
    --experiment nanoGPT_lite \
    --num-ideas 2
```

**Step 1: 初始化**

```
Using Anthropic API with model claude-3-5-sonnet-20241022.
Using GPUs: [0]
```

系统检查 LaTeX 依赖 (pdflatex, chktex)，创建 API client。

**Step 2: 生成 Ideas (阶段 1)**

```
读取 templates/nanoGPT_lite/prompt.json      → 获取任务描述
读取 templates/nanoGPT_lite/experiment.py     → 获取实验代码
读取 templates/nanoGPT_lite/seed_ideas.json   → 获取种子 ideas

Generating idea 1/2
  Iteration 1/5: LLM 生成初始 idea
    → {"Name": "attention_temperature", "Title": "...", ...}
  Iteration 2/5: LLM 反思, 改进实验设计
  Iteration 3/5: "I am done"

Generating idea 2/2
  Iteration 1/5: LLM 生成另一个 idea
  ...
```

**Step 3: 新颖性检查**

```
Checking novelty of idea 0: attention_temperature
  Round 1: Query = "attention temperature scaling transformer"
    → Semantic Scholar 返回 10 篇论文
    → LLM: "这些论文讨论了类似概念但角度不同"
  Round 2: Query = "learnable temperature self-attention"
    → 返回 10 篇论文
    → LLM: "Decision made: novel."

Checking novelty of idea 1: ...
```

**Step 4: 对每个 novel idea 执行完整流水线**

```
Processing idea: attention_temperature
  ┌─────────────────────────────────────────┐
  │ 创建项目目录:                            │
  │ results/nanoGPT_lite/                   │
  │   20240815_120000_attention_temperature/ │
  │ 复制模板文件到项目目录                    │
  └─────────────────────────────────────────┘

  *Starting Experiments*
    Aider 修改 experiment.py: 添加 temperature 参数到 attention
    $ python experiment.py --out_dir=run_1  → 成功
    Aider 分析结果, 决定做另一组实验
    $ python experiment.py --out_dir=run_2  → 成功
    Aider: "ALL_COMPLETED"
    Aider 修改 plot.py, 生成对比图
    Aider 更新 notes.txt

  *Starting Writeup*
    填写 Abstract + Title
    填写 Introduction, Background, Method, ...
    自动搜索并添加 12 个引用
    全文二次打磨
    编译 LaTeX → attention_temperature.pdf

  *Starting Review*
    加载 PDF, 转为文本
    5 个独立 LLM 评审 → Meta-review 聚合
    Overall: 4, Decision: Reject
    (写入 review.txt)

Completed idea: attention_temperature, Success: True
```

---

## 十、快速上手

### 10.1 安装

```bash
# 创建 conda 环境
conda create -n ai_scientist python=3.11
conda activate ai_scientist

# 安装 LaTeX (Ubuntu)
sudo apt-get install texlive-full

# 安装 Python 依赖
cd AI-Scientist
pip install -r requirements.txt
```

### 10.2 设置 API Keys

```bash
# 至少需要一个 LLM API Key
export OPENAI_API_KEY="sk-..."           # GPT-4o 系列
# 或
export ANTHROPIC_API_KEY="sk-ant-..."    # Claude 系列
# 或
export DEEPSEEK_API_KEY="sk-..."         # DeepSeek 系列

# 可选: 文献搜索
export S2_API_KEY="..."                  # Semantic Scholar (推荐)
```

### 10.3 准备模板

```bash
# 以 nanoGPT_lite 为例 (最快的模板)
# 1. 准备数据
python data/enwik8/prepare.py
python data/shakespeare_char/prepare.py
python data/text8/prepare.py

# 2. 跑基线
cd templates/nanoGPT_lite
python experiment.py --out_dir run_0
python plot.py
cd ../..
```

### 10.4 运行

```bash
# 最小示例: 2 个 ideas, 用 GPT-4o
python launch_scientist.py \
    --model gpt-4o-2024-05-13 \
    --experiment nanoGPT_lite \
    --num-ideas 2

# 用 Claude 3.5 Sonnet (推荐, 成功率最高)
python launch_scientist.py \
    --model claude-3-5-sonnet-20241022 \
    --experiment nanoGPT_lite \
    --num-ideas 2

# 多 GPU 并行
python launch_scientist.py \
    --model claude-3-5-sonnet-20241022 \
    --experiment nanoGPT_lite \
    --num-ideas 10 \
    --parallel 4 \
    --gpus 0,1,2,3

# 启用评审后改进
python launch_scientist.py \
    --model claude-3-5-sonnet-20241022 \
    --experiment 2d_diffusion \
    --num-ideas 5 \
    --improvement

# 跳过已有 ideas / 新颖性检查
python launch_scientist.py \
    --model gpt-4o-2024-05-13 \
    --experiment grokking \
    --num-ideas 5 \
    --skip-idea-generation \
    --skip-novelty-check
```

### 10.5 单独跑评审

```python
import openai
from ai_scientist.perform_review import load_paper, perform_review

client = openai.OpenAI()
paper_txt = load_paper("path/to/paper.pdf")
review = perform_review(
    paper_txt,
    model="gpt-4o-2024-05-13",
    client=client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)
print(f"Overall: {review['Overall']}/10")
print(f"Decision: {review['Decision']}")
print(f"Strengths: {review['Strengths']}")
print(f"Weaknesses: {review['Weaknesses']}")
```

### 10.6 支持的模型

| 提供商 | 模型 | 环境变量 |
|--------|------|---------|
| Anthropic | claude-3-5-sonnet-20240620, claude-3-5-sonnet-20241022 | `ANTHROPIC_API_KEY` |
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4.1, o1, o3-mini | `OPENAI_API_KEY` |
| DeepSeek | deepseek-chat, deepseek-coder, deepseek-reasoner | `DEEPSEEK_API_KEY` |
| Google | gemini-1.5-pro, gemini-2.0-flash, gemini-2.5-pro | `GEMINI_API_KEY` |
| OpenRouter | llama3.1-405b | `OPENROUTER_API_KEY` |
| AWS Bedrock | claude-3 系列 | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` |
| Vertex AI | claude-3 系列 | GCP 认证 |

---

## 十一、局限与思考

### 质量方面

- **成功率依赖模型**：Claude Sonnet 3.5 成功率最高，较弱的模型 (如 GPT-4o-mini) 经常失败
- **论文质量参差不齐**：大部分生成的论文得分在 3-5 分 (NeurIPS 标准), 相当于 Borderline Reject 到 Borderline Accept
- **实验范围受限**：只能在模板提供的代码框架内做修改，不能引入新数据集或全新的模型架构
- **结果可能不可靠**：AI 可能误解实验结果、产生错误的分析

### 安全方面

- **执行 LLM 生成的代码**：这是最大的风险。LLM 可能写出危险的代码（删文件、访问网络、无限循环等）
- **强烈建议容器化运行**：项目提供了 Docker 配置
- **限制网络访问**：防止 AI 生成的代码访问外部资源

### 伦理方面

- **学术诚信**：使用时必须明确标注 AI 生成（项目 License 有强制披露要求）
- **审稿偏差**：AI 审稿员存在正向偏差（GPT-4o 以外的模型尤其明显）
- **学术泡沫**：可能导致低质量论文泛滥

### 成本方面

- **每篇论文约 $15**（Claude Sonnet 3.5），DeepSeek 更便宜
- **需要 GPU 跑实验**：CPU 不可行（实验太慢）
- **Semantic Scholar API 有速率限制**：如果没有 API Key，新颖性检查可能很慢

### 适合的场景

- 快速探索一个研究方向的可行性
- 在已有实验框架上系统地搜索 idea 空间
- 学习 "AI 如何做研究" 的教育用途
- 作为科研辅助工具（人类负责最终判断）

### 不适合的场景

- 需要真正突破性创新的顶会论文
- 需要新数据集或复杂实验环境的研究
- 对安全性要求高的生产环境
- 替代人类科研工作者（至少目前还远远不够）

---

## 十二、关键设计决策

### 1. 为什么用 Aider 而不是让 LLM 直接生成代码？

```
直觉方案:  LLM 从头生成完整的 experiment.py
问题:      生成的代码和模板不兼容, 无法运行

AI-Scientist 方案:  用 Aider 做 diff 编辑 (SEARCH/REPLACE)
优点:
  - 保持代码结构不变, 只修改需要改的部分
  - LLM 看到完整上下文 (原始代码 + 实验笔记 + 画图代码)
  - 编辑粒度可控, 不会意外删掉关键代码
  - 利用 Aider 成熟的 diff 解析和应用机制
```

### 2. 为什么用固定的 `--out_dir=run_i` 格式？

```
设计选择:  所有实验必须用 python experiment.py --out_dir=run_i
原因:
  - 标准化: 系统知道去哪里找结果 (run_i/final_info.json)
  - 可追溯: 每次 run 的代码快照保存为 run_i.py
  - 防止 AI 添加额外参数导致不可预测的行为
  - 简化结果解析逻辑
```

### 3. 为什么评审用 GPT-4o 而不是和实验相同的模型？

```python
# launch_scientist.py 中硬编码了评审模型
review = perform_review(
    paper_text,
    model="gpt-4o-2024-05-13",    # 固定使用 GPT-4o
    client=openai.OpenAI(),        # 固定使用 OpenAI
    ...
)
```

原因：论文中发现 GPT-4o 的评审质量最接近人类审稿人。其他模型（包括 Claude）存在明显的正向偏差（倾向于给高分）。评审需要独立于写作模型，避免"自己审自己"的问题。

### 4. 为什么 idea 生成需要看到之前所有 idea？

```python
# 每次生成新 idea 时, 把所有已有 ideas 都放进 prompt
prev_ideas_string = "\n\n".join(idea_str_archive)
```

这是为了**避免重复**和**鼓励多样性**。LLM 看到已有的 ideas 后，被提示要想一些不同的方向。这类似于头脑风暴中"不要重复别人说过的"。

### 5. 为什么用 Semantic Scholar 而不是 Google Scholar？

```
Semantic Scholar:
  + 有免费的 REST API
  + 返回结构化数据 (标题, 作者, 摘要, 引用数, bibtex)
  + 可直接获取 bibtex 格式的引用 (用于 LaTeX)
  + 学术专用, 结果质量高

Google Scholar:
  - 没有官方 API
  - 容易被封 IP
  - 不直接提供 bibtex
```

### 6. 论文模板为什么用 ICLR 2024 格式？

```
templates/*/latex/template.tex 使用 iclr2024_conference.sty
原因:
  - 标准的 ML 会议格式, LLM 训练数据中见过大量类似论文
  - 有现成的 .sty 和 .bst 文件
  - 评审系统也是基于 NeurIPS/ICLR 标准
  - 用户可以替换为自己需要的会议格式
```

### 7. 为什么实验有 2 小时超时？

```python
def run_experiment(folder_name, run_num, timeout=7200):  # 7200 秒 = 2 小时
```

AI 生成的代码可能包含无限循环、内存泄漏、或计算量过大的实验。2 小时是一个经验值：足够跑完模板中的大部分合理实验，但能及时终止失控的代码。

### 8. 开放式 idea 生成 (Open-Ended) 模式

除了一次性生成 N 个 ideas，项目还提供了 `generate_next_idea()` 函数，支持**增量式**生成。之前完成的 idea 会带上评审分数，帮助 LLM 学习什么样的 idea 更容易被接受：

```python
# 开放式模式: 已完成的 idea 带有 "Score" 字段
# LLM 可以从之前的成功/失败中学习
idea_first_prompt += """
Completed ideas have an additional "Score" field which indicates
the assessment by an expert ML reviewer.
Scores of 0 indicate the idea failed during experimentation, writeup or reviewing.
"""
```

这为未来的**自主科研闭环**打下了基础：AI 不断提出 idea → 跑实验 → 被评审 → 从反馈中学习 → 提出更好的 idea。

---

*Generated on 2026-03-28 from AI-Scientist source code analysis (commit from SakanaAI/AI-Scientist)*
