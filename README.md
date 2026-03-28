# Auto Research

Parle AI 团队对 Auto Researcher 领域的探索与分析。

我们深入研究了当前最热门的 5 个自动化研究项目，提取了共性架构模式，并为每个项目编写了详细的 how-it-works tutorial。

## 分析的项目

| 项目 | Stars | 定位 | 原创性 |
|------|-------|------|--------|
| [karpathy/autoresearch](https://github.com/karpathy/autoresearch) | 59k | 单 GPU 自主 ML 实验 | 有 (新实验结果) |
| [bytedance/deer-flow](https://github.com/bytedance/deer-flow) | 51k | 通用超级 Agent 框架 | 无 |
| [stanford-oval/storm](https://github.com/stanford-oval/storm) | 28k | 多 Persona 知识策展 + 长文生成 | 无 |
| [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist) | 12.8k | 全自动科学家：假设→实验→论文→审稿 | 最高 (论文被 ICLR 接收) |
| [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) | 12.1k | R&D 双循环：假设→代码→评估→进化 | 有 (代码持续进化) |

## 文档

- **[docs/README.md](docs/README.md)** — 五大项目对比分析 + 六大共性架构模式
- [docs/autoresearch-how-it-works-tutorial.md](docs/autoresearch-how-it-works-tutorial.md) — autoresearch 详解
- [docs/deerflow-how-it-works-tutorial.md](docs/deerflow-how-it-works-tutorial.md) — DeerFlow 详解
- [docs/storm-how-it-works-tutorial.md](docs/storm-how-it-works-tutorial.md) — STORM 详解
- [docs/ai-scientist-how-it-works-tutorial.md](docs/ai-scientist-how-it-works-tutorial.md) — AI-Scientist 详解
- [docs/rd-agent-how-it-works-tutorial.md](docs/rd-agent-how-it-works-tutorial.md) — RD-Agent 详解

## 六大共性模式

| # | 模式 | 一句话 |
|---|------|--------|
| 1 | **循环驱动** | 所有系统本质都是 Hypothesize→Execute→Evaluate→Decide 循环 |
| 2 | **多阶段流水线** | 规划→采集→执行→综合，阶段间解耦 |
| 3 | **异构模型分配** | 便宜模型做检索，贵模型做综合 |
| 4 | **激进并行化** | 多视角/多子任务并行，缩短 3-5x 时间 |
| 5 | **可审计状态追踪** | Git/日志/中间产物，每步可回溯 |
| 6 | **约束即创造力** | 限制越明确，Agent 行为越聚焦 |

## 下一步

这是团队对 Auto Researcher 领域的第一步探索。后续计划：
- 实际运行各项目，记录真实体验和结果
- 基于共性模式，构建适合我们场景的 Auto Researcher
- 持续跟踪这个快速发展的领域

## 本地补充

`ai-scientist/` 因体积过大 (~197MB) 未包含在 repo 中，需要单独 clone：

```bash
git clone --depth 1 https://github.com/SakanaAI/AI-Scientist.git ai-scientist
```

## License

MIT
