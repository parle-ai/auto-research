# STORM 是怎么工作的？— 从原理到代码的完整教程

> STORM = **S**ynthesis of **T**opic **O**utlines through **R**etrieval and **M**ulti-perspective Question Asking
>
> 斯坦福出品，一句话：**用 AI 模拟"编辑部开会调研"的过程，自动写出带引用的维基百科级文章**

---

## 一、先看结果：STORM 输入什么，输出什么

```
输入: "量子计算" (一个主题，就一个字符串)

输出:
  output/量子计算/
  ├── storm_gen_article_polished.txt   ← 最终文章 (带引用的长文)
  ├── storm_gen_outline.txt            ← 文章大纲
  ├── conversation_log.json            ← 所有"调研对话"记录
  ├── raw_search_results.json          ← 所有搜索结果
  ├── url_to_info.json                 ← 引用来源汇总
  └── llm_call_history.jsonl           ← 每次 LLM 调用的日志
```

**整个过程不需要人参与。** 你给一个主题，它还你一篇 3000-5000 字的、每句话都有引用的文章。

---

## 二、全局流水线：四个阶段

```
阶段1              阶段2             阶段3              阶段4
知识策展            大纲生成           文章生成            文章润色
(Research)         (Outline)        (Write)            (Polish)

┌─────────┐      ┌─────────┐      ┌─────────┐      ┌──────────┐
│ 多Persona│  →   │ 生成大纲 │  →   │ 逐节写作 │  →   │ 写摘要   │
│ 多轮对话  │      │ (两版)   │      │ (并行)   │      │ 去重复   │
│ 搜索采集  │      │         │      │ 带引用   │      │ 整理引用  │
└─────────┘      └─────────┘      └─────────┘      └──────────┘
     ↓                ↓                ↓                ↓
 信息表             文章大纲          带引用草稿         最终文章
(InformationTable) (Outline)     (Draft Article)  (Polished Article)
```

每个阶段可以独立运行、跳过、或从中间恢复。下面逐个讲透。

---

## 三、阶段 1：知识策展 — 这是 STORM 最核心的创新

### 3.1 问题：如何让 AI 全面地调研一个主题？

最直觉的方案是让 LLM 直接搜索然后写文章。但问题是：

```
❌ 直觉方案: LLM → 搜索 "量子计算" → 拿到结果 → 写文章
   问题: 搜索方向单一，覆盖面窄，像一个外行随便搜了搜就开始写
```

STORM 的解法是模拟真实世界中**编辑部的工作方式**：

```
✅ STORM 方案:
   1. 先确定需要哪些"专家视角"来审视这个主题
   2. 每个视角的"编辑"各自去向"百科专家"提问
   3. 每次提问，专家都会去搜索引擎查资料后回答
   4. 最后把所有编辑采集到的资料合并到一起
```

### 3.2 第一步：生成 Persona（视角）

**Persona 不是随机编的，而是从相关的 Wikipedia 文章推导出来的。**

```
输入: "量子计算"
         ↓
  ┌──────────────────────────────────────────────────────┐
  │ Step A: 让 LLM 找相关的 Wikipedia 页面               │
  │                                                      │
  │  LLM: "跟量子计算相关的 Wikipedia 页面有:             │
  │   - Quantum mechanics                                │
  │   - Post-quantum cryptography                        │
  │   - Qubit                                            │
  │   - Shor's algorithm"                                │
  └──────────────────────┬───────────────────────────────┘
                         ↓
  ┌──────────────────────────────────────────────────────┐
  │ Step B: 提取这些页面的目录结构                         │
  │                                                      │
  │  Quantum mechanics 目录:                              │
  │    - Mathematical formulation                        │
  │    - Philosophical implications                      │
  │    - Applications                                    │
  │                                                      │
  │  Post-quantum cryptography 目录:                     │
  │    - Lattice-based                                   │
  │    - Hash-based                                      │
  │    - Standardization efforts                         │
  └──────────────────────┬───────────────────────────────┘
                         ↓
  ┌──────────────────────────────────────────────────────┐
  │ Step C: 基于目录结构，生成多样化的 Persona            │
  │                                                      │
  │  LLM: "要全面覆盖这个主题，需要以下视角的编辑:        │
  │   1. 量子物理学家 — 关注物理原理和量子比特            │
  │   2. 密码学研究者 — 关注安全影响和后量子密码          │
  │   3. 科技产业分析师 — 关注商业化和产业竞争"           │
  └──────────────────────────────────────────────────────┘
```

**对应代码：**

```python
# persona_generator.py
class CreateWriterWithPersona(dspy.Module):
    def forward(self, topic: str):
        # Step A: 找相关 Wikipedia 页面
        related_topics = self.find_related_topic(topic=topic).related_topics

        # Step B: 提取每个页面的目录
        examples = []
        for url in extract_urls(related_topics):
            title, toc = get_wiki_page_title_and_toc(url)
            examples.append(f"Title: {title}\nTable of Contents: {toc}")

        # Step C: 基于目录生成 Persona
        personas = self.gen_persona(
            topic=topic,
            examples="\n----------\n".join(examples)
        ).personas

        return personas
```

最终的 Persona 列表永远以一个 "Basic fact writer"（通用事实编辑）开头，再加上 LLM 生成的专业视角（默认 3 个）。

### 3.3 第二步：每个 Persona 和 Expert 进行多轮对话

这是最需要理解的部分。**不是 Agent 之间在自由聊天**，而是一个固定结构的循环：

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  WikiWriter                              TopicExpert            │
│  (一个 LLM 调用)                          (一个 LLM 调用 + 搜索) │
│                                                                 │
│  输入:                                   输入:                   │
│  - topic (主题)                          - topic (主题)          │
│  - persona (视角)                        - question (问题)       │
│  - history (对话历史)                                            │
│                                                                 │
│  输出:                                   内部流程:               │
│  - question (一个问题)                   1. 问题 → 搜索 queries  │
│                                          2. queries → Web Search │
│                                          3. snippets → 组织回答  │
│                                                                 │
│                                          输出:                   │
│                                          - answer (带引用的回答)  │
│                                          - queries (搜索词)      │
│                                          - search_results (原始) │
└─────────────────────────────────────────────────────────────────┘
```

**理解各部分的"灵魂"：**

- **WikiWriter 的灵魂 = Persona（决定问什么）** — Persona 不是一个独立的 Agent，只是传给 WikiWriter prompt 的一个字符串参数，改变它的提问方向
- **TopicExpert 的灵魂 = Retriever/搜索引擎（决定从哪找答案）** — TopicExpert 自己不"懂"任何东西，它被约束为只能基于搜索结果回答，搜不到就认怂。本质上是"搜索引擎的嘴替"

```
┌─────────────────────────────────────────────────────┐
│              ConvSimulator                           │
│                                                     │
│  WikiWriter                    TopicExpert          │
│  ┌──────────┐                 ┌──────────────┐     │
│  │          │   question      │              │     │
│  │  灵魂:   │ ─────────────→ │  灵魂:       │     │
│  │  Persona │                 │  Retriever   │     │
│  │  (字符串) │ ←───────────── │  (搜索引擎)   │     │
│  │          │    answer       │              │     │
│  └──────────┘                 └──────┬───────┘     │
│                                      │             │
│                                ┌─────┴─────┐       │
│                                │ Bing/You/ │       │
│                                │ DuckDuckGo│       │
│                                │ /Qdrant   │       │
│                                └───────────┘       │
└─────────────────────────────────────────────────────┘
```

**完整的对话流程（以"密码学研究者"Persona 为例）：**

```
╔══════════════════════════════════════════════════════════════════╗
║  Persona: "密码学研究者 — 关注安全影响和后量子密码"               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  第 1 轮:                                                        ║
║  ┌─────────────────────────────────────────────────────────────┐ ║
║  │ WikiWriter 思考:                                            │ ║
║  │   我是密码学研究者，topic 是量子计算                          │ ║
║  │   对话历史: 空                                               │ ║
║  │   → 我应该先问最核心的安全问题                                │ ║
║  │                                                             │ ║
║  │ WikiWriter 输出:                                            │ ║
║  │   "量子计算对现有公钥加密体系的威胁具体有多大？"               │ ║
║  └──────────────────────────┬──────────────────────────────────┘ ║
║                             ↓                                    ║
║  ┌─────────────────────────────────────────────────────────────┐ ║
║  │ TopicExpert 内部:                                           │ ║
║  │   1. 问题 → 生成搜索词:                                     │ ║
║  │      - "quantum computing threat public key cryptography"   │ ║
║  │      - "Shor algorithm RSA breaking"                        │ ║
║  │   2. 搜索词 → 调 Bing/You.com API → 拿到网页 snippets      │ ║
║  │   3. snippets → LLM 组织回答:                               │ ║
║  │      "Shor 算法可在多项式时间内分解大整数[1]，              │ ║
║  │       这意味着 RSA-2048 将在量子计算机成熟时被破解[2]..."     │ ║
║  └──────────────────────────┬──────────────────────────────────┘ ║
║                             ↓                                    ║
║  记录: DialogueTurn {                                            ║
║    question: "量子计算对现有公钥加密体系的威胁具体有多大？"        ║
║    answer: "Shor 算法可在多项式时间内...[1][2]"                   ║
║    queries: ["quantum computing threat...", "Shor algorithm..."] ║
║    search_results: [Information(url=..., snippets=[...])]        ║
║  }                                                               ║
║                                                                  ║
║  第 2 轮:                                                        ║
║  ┌─────────────────────────────────────────────────────────────┐ ║
║  │ WikiWriter 思考:                                            │ ║
║  │   我是密码学研究者，topic 是量子计算                          │ ║
║  │   对话历史: 上一轮聊了 Shor 算法和 RSA 威胁                  │ ║
║  │   → 那应对措施呢？问问后量子密码                              │ ║
║  │                                                             │ ║
║  │ WikiWriter 输出:                                            │ ║
║  │   "目前后量子密码学有哪些被广泛认可的方案？"                   │ ║
║  └──────────────────────────┬──────────────────────────────────┘ ║
║                             ↓                                    ║
║  ┌─────────────────────────────────────────────────────────────┐ ║
║  │ TopicExpert 内部:                                           │ ║
║  │   搜索 "post-quantum cryptography NIST standards"           │ ║
║  │   回答: "NIST 于 2024 年正式发布了三项标准[3]..."            │ ║
║  └──────────────────────────┬──────────────────────────────────┘ ║
║                             ↓                                    ║
║  第 3 轮:                                                        ║
║  WikiWriter: "这些方案在工业界的实际部署进展如何？"                ║
║  TopicExpert: (搜索) "Chrome 已实验性启用 ML-KEM [4]..."         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

**对应代码：**

```python
# knowledge_curation.py — ConvSimulator
class ConvSimulator(dspy.Module):
    def forward(self, topic, persona, ground_truth_url, callback_handler):
        dlg_history = []

        for _ in range(self.max_turn):  # 默认 3 轮
            # ---- WikiWriter: 基于 persona + 历史 生成问题 ----
            question = self.wiki_writer(
                topic=topic,
                persona=persona,           # ← persona 在这里注入
                dialogue_turns=dlg_history  # ← 能看到之前聊了什么
            ).question

            # 对话结束信号
            if question.startswith("Thank you so much for your help!"):
                break

            # ---- TopicExpert: 搜索 + 回答 ----
            expert_output = self.topic_expert(
                topic=topic,
                question=question,
                ground_truth_url=ground_truth_url  # 排除这个 URL
            )

            # ---- 记录这一轮 ----
            dlg_turn = DialogueTurn(
                agent_utterance=expert_output.answer,
                user_utterance=question,
                search_queries=expert_output.queries,
                search_results=expert_output.searched_results,
            )
            dlg_history.append(dlg_turn)

        return dlg_history
```

**关键细节：**
- `persona` 只是 prompt 中的一个字符串参数，不是独立进程
- `dlg_history` 让 WikiWriter 能追问（不重复问、能深入）
- TopicExpert **每次都真实搜索**，不是凭 LLM 记忆回答
- `ground_truth_url` 用于评估时排除答案来源，正常使用可忽略

### 3.4 第三步：并行跑多个 Persona，然后合并

```
              ┌──────────────────┐
              │ 主题: "量子计算"  │
              └────────┬─────────┘
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │ Persona 1  │ │ Persona 2  │ │ Persona 3  │
   │ 基础事实    │ │ 密码学     │ │ 产业分析    │    ← 并行 (ThreadPool)
   │            │ │            │ │            │
   │ 3轮对话    │ │ 3轮对话    │ │ 3轮对话    │
   │ ~6次搜索   │ │ ~6次搜索   │ │ ~6次搜索   │
   └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
         │              │              │
         └──────────────┼──────────────┘
                        ↓
              ┌──────────────────┐
              │    按 URL 合并    │
              │    去重 snippets  │
              └────────┬─────────┘
                       ↓
              ┌──────────────────┐
              │ InformationTable │
              │ ~18 次搜索结果   │
              │ ~50+ snippets    │
              │ ~30+ 独立 URL    │
              └──────────────────┘
```

**合并逻辑非常简单：**

```python
# storm_dataclass.py
def construct_url_to_info(conversations):
    url_to_info = {}

    for persona, conv in conversations:     # 遍历每个 Persona 的对话
        for turn in conv:                   # 遍历每轮对话
            for info in turn.search_results:  # 遍历每个搜索结果
                if info.url in url_to_info:
                    # 同一个 URL → 合并 snippets
                    url_to_info[info.url].snippets.extend(info.snippets)
                else:
                    # 新 URL → 直接加入
                    url_to_info[info.url] = info

    # 去重
    for url in url_to_info:
        url_to_info[url].snippets = list(set(url_to_info[url].snippets))

    return url_to_info
```

**为什么按 URL 合并？** 因为不同 Persona 可能搜到同一篇文章的不同段落。比如物理学家和密码学家都搜到了同一篇 Nature 论文，但关注的段落不同，合并后这篇论文的信息就更完整了。

---

## 四、阶段 2：大纲生成

有了 InformationTable 之后，STORM 生成大纲（用两种方式，取更好的那个）：

```
方式 A: 直接大纲 (Direct Outline)
  输入: topic
  LLM 仅凭自身知识生成大纲
  → 覆盖面依赖 LLM 训练数据

方式 B: 基于对话的大纲 (Refined Outline)        ← STORM 采用这个
  输入: topic + 所有对话历史
  LLM 根据实际采集到的信息生成大纲
  → 覆盖面反映了真实搜索结果
```

```python
# outline_generation.py
class WriteOutline(dspy.Module):
    def forward(self, topic, conv):
        # 方式 A
        draft = self.write_page_outline(topic=topic).outline

        # 方式 B — 基于实际调研内容
        refined = self.write_page_outline_from_conv(
            topic=topic,
            conv=conv     # ← 所有 Persona 的对话记录
        ).outline

        return draft, refined
```

输出示例：
```markdown
# 量子计算
## 基本原理
### 量子比特
### 量子纠缠
### 量子门
## 发展历史
### 早期理论
### 实验突破
## 主要技术路线
### 超导量子计算
### 离子阱
### 光量子
## 应用前景
### 密码学影响
### 药物发现
### 金融优化
## 挑战与局限
```

---

## 五、阶段 3：文章生成 — 逐节并行写作 + 二阶段检索

这里有一个巧妙的设计：**写文章时不再去网上搜索，而是从阶段 1 采集到的 InformationTable 中做语义检索。**

```
                    ┌──────────────────────────┐
                    │    InformationTable       │
                    │  (阶段1采集的所有信息)      │
                    │                          │
                    │  url_1: [snippet, ...]   │
                    │  url_2: [snippet, ...]   │
                    │  url_3: [snippet, ...]   │
                    │         ...              │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────┴─────────────┐
                    │  Sentence-Transformer    │
                    │  把所有 snippets 编码为    │
                    │  向量 (一次性)             │
                    └────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          ↓                      ↓                      ↓
  ┌───────────────┐    ┌───────────────┐     ┌───────────────┐
  │ "基本原理"     │    │ "发展历史"     │     │ "应用前景"     │
  │               │    │               │     │               │
  │ 语义检索 top5  │    │ 语义检索 top5  │     │ 语义检索 top5  │    ← 并行
  │ → LLM 写此节  │    │ → LLM 写此节  │     │ → LLM 写此节  │
  │ → 带引用 [1]  │    │ → 带引用 [3]  │     │ → 带引用 [5]  │
  └───────────────┘    └───────────────┘     └───────────────┘
```

**为什么不直接再搜索一次？**
1. **一致性**：所有引用都来自已采集的信息，不会出现"引用了但没在知识库里"的情况
2. **效率**：不需要再调搜索 API
3. **质量**：语义检索比关键词搜索更精准地匹配段落需要的信息

**对应代码：**

```python
# article_generation.py
class StormArticleGenerationModule:
    def generate_section(self, topic, section_name, information_table):
        # 1. 用 section_name 做语义检索
        relevant_info = information_table.retrieve_information(
            queries=[section_name],
            search_top_k=self.search_top_k
        )

        # 2. 格式化检索到的信息
        info_text = format_info_with_citations(relevant_info)

        # 3. LLM 基于信息写这一节
        section_content = self.write_section(
            topic=topic,
            section=section_name,
            info=info_text
        ).output

        return section_content

    def generate_article(self, topic, outline, information_table):
        # 所有章节并行写作
        with ThreadPoolExecutor(max_workers=self.max_thread_num) as executor:
            for section in outline.sections:
                if section.name not in ["Introduction", "Conclusion"]:
                    executor.submit(
                        self.generate_section,
                        topic, section.name, information_table
                    )
```

---

## 六、阶段 4：文章润色

```python
# article_polish.py — 两步完成
class PolishPageModule(dspy.Module):
    def forward(self, topic, article):
        # Step 1: 写 Lead Section (摘要/引言)
        # 遵循 Wikipedia 规范: 概述全文核心内容
        lead = self.write_lead_section(topic=topic, article=article).output

        # Step 2: 去除重复内容 (可选)
        # 不同 Persona 可能导致不同节有重复信息
        polished = self.polish_page(topic=topic, article=article).output

        return lead + polished
```

---

## 七、数据流全景图

把四个阶段串起来看完整的数据流：

```
"量子计算"
     │
     ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段 1: 知识策展                                             │
│                                                             │
│  "量子计算" → 找相关 Wikipedia → 提取目录 → 生成 Persona     │
│                                                             │
│  Persona 1 ──┐                                              │
│  Persona 2 ──┤→ 各自 3 轮对话 (并行) → 按 URL 合并          │
│  Persona 3 ──┘                                              │
│                                                             │
│  输出: InformationTable { url → [snippets] }                │
│        conversation_log (所有对话记录)                        │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段 2: 大纲生成                                             │
│                                                             │
│  conversation_log → LLM → 层级大纲                          │
│  # 量子计算                                                 │
│  ## 基本原理                                                │
│  ### 量子比特                                                │
│  ...                                                        │
│                                                             │
│  输出: StormArticle (仅有大纲骨架)                            │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段 3: 文章生成                                             │
│                                                             │
│  对大纲中每一节:                                              │
│    section_name → 语义检索 InformationTable → top5 snippets  │
│    → LLM 基于 snippets 写这一节 (带 [1][2] 引用)            │
│                                                             │
│  所有节并行写作                                               │
│                                                             │
│  输出: StormArticle (填充了内容 + 引用)                       │
└──────────────────────────┬──────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ 阶段 4: 润色                                                 │
│                                                             │
│  写 Lead Section (全文摘要)                                   │
│  去除跨节重复内容                                             │
│  整理引用编号                                                 │
│                                                             │
│  输出: storm_gen_article_polished.txt (最终成品)              │
└─────────────────────────────────────────────────────────────┘
```

---

## 八、LLM 调用策略：不同阶段用不同模型

STORM 允许为每个阶段配置不同的 LLM，原因是**不同阶段对智能的需求不同**：

```python
lm_configs = STORMWikiLMConfigs()

# 阶段 1 的对话模拟 — 用便宜快速的模型
# 原因: 这里只是生成问题和基于 snippet 组织回答，不需要最强推理
lm_configs.set_conv_simulator_lm(LM(model='gpt-4o-mini', max_tokens=500))
lm_configs.set_question_asker_lm(LM(model='gpt-4o-mini', max_tokens=500))

# 阶段 2 的大纲生成 — 用中等模型
# 原因: 需要理解对话全局结构，但不需要生成长文
lm_configs.set_outline_gen_lm(LM(model='gpt-4o', max_tokens=400))

# 阶段 3 的文章写作 — 用最强模型
# 原因: 这里需要把 snippets 综合成流畅、准确、有引用的长文
lm_configs.set_article_gen_lm(LM(model='gpt-4o', max_tokens=3000))

# 阶段 4 的润色 — 用最强模型
lm_configs.set_article_polish_lm(LM(model='gpt-4o', max_tokens=4000))
```

**成本分布（典型一篇文章）：**

| 阶段 | LLM 调用次数 | 模型 | 占总成本 |
|------|------------|------|---------|
| 知识策展 | ~30-40 次 | mini | ~15% |
| 大纲生成 | 2-3 次 | 4o | ~5% |
| 文章生成 | 5-10 次 | 4o | ~50% |
| 润色 | 1-2 次 | 4o | ~30% |
| **总计** | ~40-55 次 | 混合 | **$0.50-2.00** |

---

## 九、快速上手

### 9.1 安装

```bash
cd storm
pip install -e .
```

### 9.2 最小运行示例

```python
from knowledge_storm import STORMWikiRunnerArguments, STORMWikiRunner, STORMWikiLMConfigs
from knowledge_storm.lm import LitellmModel
from knowledge_storm.rm import BingSearch  # 或 YouRM, DuckDuckGoSearchRM 等

# 1. 配置 LLM
lm_configs = STORMWikiLMConfigs()
fast_lm = LitellmModel(model='gpt-4o-mini', max_tokens=500)
strong_lm = LitellmModel(model='gpt-4o', max_tokens=3000)

lm_configs.set_conv_simulator_lm(fast_lm)
lm_configs.set_question_asker_lm(fast_lm)
lm_configs.set_outline_gen_lm(strong_lm)
lm_configs.set_article_gen_lm(strong_lm)
lm_configs.set_article_polish_lm(strong_lm)

# 2. 配置搜索引擎
rm = BingSearch(bing_search_api_key="YOUR_KEY", k=3)

# 3. 配置运行参数
args = STORMWikiRunnerArguments(
    output_dir='./output',
    max_conv_turn=3,       # 每个 Persona 对话 3 轮
    max_perspective=3,     # 3 个 Persona 视角
    search_top_k=3,        # 每次搜索取 top 3
)

# 4. 创建并运行
runner = STORMWikiRunner(args, lm_configs, rm)
runner.run(
    topic="量子计算",
    do_research=True,           # 阶段 1
    do_generate_outline=True,   # 阶段 2
    do_generate_article=True,   # 阶段 3
    do_polish_article=True,     # 阶段 4
)
runner.post_run()  # 保存日志和成本统计

# 5. 查看结果
# 输出在 ./output/量子计算/ 目录下
```

### 9.3 支持的搜索引擎

| 搜索引擎 | 类名 | 需要 API Key |
|---------|------|-------------|
| Bing | `BingSearch` | 是 |
| You.com | `YouRM` | 是 |
| DuckDuckGo | `DuckDuckGoSearchRM` | 否 (免费) |
| Brave | `BraveRM` | 是 |
| Tavily | `TavilySearchRM` | 是 |
| Serper | `SerperRM` | 是 |
| 自定义文档 | `VectorRM` (Qdrant) | 否 |

### 9.4 支持的 LLM（通过 LiteLLM）

OpenAI, Anthropic Claude, Google Gemini, Together.ai, Ollama (本地), Azure OpenAI, 以及 LiteLLM 支持的 100+ 其他模型。

---

## 十、STORM 的局限和适用场景

### 适合的场景
- 写某个主题的综述/概览（Wikipedia 风格）
- 需要多角度覆盖的调研报告
- 需要每句话都有引用来源的严谨写作

### 不适合的场景
- 需要跑实验的 ML 研究（用 autoresearch）
- 需要实时交互的问答（用 DeerFlow）
- 写观点性/创意性内容（STORM 是事实导向的）
- 特别窄的专业领域（搜索引擎可能找不到足够信息）

### 主要局限
- **质量依赖搜索结果**：如果搜索引擎返回的信息质量差，文章也会差
- **成本不低**：40-55 次 LLM 调用，每篇 $0.5-2
- **无法验证事实**：它引用了来源，但不会交叉验证来源的正确性
- **英文效果最好**：搜索和 LLM 对英文主题的支持最成熟

---

*Generated on 2026-03-28 from storm source code analysis*
