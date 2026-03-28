# DeerFlow 是怎么工作的？ — 从架构到源码的完整教程

> DeerFlow = **D**eep **E**xploration and **E**fficient **R**esearch **Flow**
>
> 字节跳动出品，一句话：**一个开源的"超级 Agent 底座"，能编排子 Agent、记忆、沙箱，通过可扩展的 Skill 做几乎任何事**

---

## 一、先看结果：DeerFlow 输入什么，输出什么

```
输入: 任何自然语言指令
  例如: "比较 AWS、Azure 和 GCP 三家云服务商的定价策略"
  例如: "帮我写一个 Python 爬虫，抓取 Hacker News 首页"
  例如: "分析这份 PDF 报告的核心结论"  (支持文件上传)

输出:
  - 对话式回复 (带引用、带 Markdown 格式)
  - 生成的文件 (代码、报告、图表 → /mnt/user-data/outputs/)
  - 长期记忆 (跨会话记住你的偏好和背景)
  - 子 Agent 并行执行的中间结果 (实时 SSE 推送)
```

**核心卖点：**
- 不只是一个 Chatbot，而是一个能**分解任务、并行执行、操作文件系统、执行代码**的 Agent
- 12 个中间件层层过滤，保证安全性和稳定性
- 支持 MCP 协议，一键接入数百个外部工具
- 长期记忆系统，越用越懂你

---

## 二、整体架构：四层服务

```
                           用户
                            │
                     ┌──────┴──────┐
                     │  Nginx:2026  │  ← 统一入口
                     └──────┬──────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
    ┌─────────┴──────┐  ┌──┴──────────┐  ┌┴──────────────┐
    │ Frontend:3000  │  │ Gateway:8001│  │ LangGraph:2024 │
    │ (Next.js)      │  │ (FastAPI)   │  │ (Agent 运行时) │
    └────────────────┘  └─────────────┘  └───────┬────────┘
                                                  │
                              ┌────────────────────┼────────────────────┐
                              │                    │                    │
                        ┌─────┴─────┐       ┌─────┴──────┐     ┌──────┴──────┐
                        │  Sandbox  │       │   Memory   │     │ MCP Servers │
                        │ (本地/Docker)│     │ (memory.json)│    │ (stdio/SSE) │
                        └───────────┘       └────────────┘     └─────────────┘
```

**Nginx 路由规则：**

| 请求路径 | 转发目标 | 说明 |
|---------|---------|------|
| `/api/langgraph/*` | LangGraph:2024 | Agent 对话、Thread 管理 |
| `/api/*` (其它) | Gateway:8001 | 模型列表、MCP配置、技能管理、记忆、文件上传 |
| `/` (非API) | Frontend:3000 | Web 界面 |

**四层各自负责什么？**

| 层 | 代码位置 | 职责 |
|----|---------|------|
| **LangGraph Server** | `backend/packages/harness/deerflow/` | Agent 构建、中间件链、工具执行、沙箱、记忆 |
| **Gateway API** | `backend/app/gateway/` | REST API：模型切换、MCP 配置、技能启停、文件上传、Thread 清理 |
| **Frontend** | `frontend/` | Next.js Web UI，SSE 流式显示 |
| **IM Channels** | `backend/app/channels/` | 飞书/Slack/Telegram 集成 |

**架构约束（Harness / App 分层）：**

```
deerflow.*  (harness 层，可发布为独立包)
    ↑ 允许被引用
app.*       (应用层，不发布)
    ✗ 禁止反向引用 deerflow → app
```

这个约束由 `tests/test_harness_boundary.py` 在 CI 中强制执行。

---

## 三、Lead Agent 详解：如何构建"主 Agent"

Lead Agent 是整个系统的核心。每次用户发消息，都会经过 `make_lead_agent()` 创建（或复用）一个 Agent 实例。

### 3.1 入口：make_lead_agent()

```python
# agents/lead_agent/agent.py

def make_lead_agent(config: RunnableConfig):
    cfg = config.get("configurable", {})

    # ---- 1. 解析运行时参数 ----
    thinking_enabled = cfg.get("thinking_enabled", True)
    reasoning_effort = cfg.get("reasoning_effort", None)
    requested_model_name = cfg.get("model_name") or cfg.get("model")
    is_plan_mode = cfg.get("is_plan_mode", False)
    subagent_enabled = cfg.get("subagent_enabled", False)
    max_concurrent_subagents = cfg.get("max_concurrent_subagents", 3)
    agent_name = cfg.get("agent_name")

    # ---- 2. 模型解析链：请求指定 → Agent配置 → 全局默认 ----
    agent_config = load_agent_config(agent_name)
    agent_model_name = agent_config.model if agent_config and agent_config.model else _resolve_model_name()
    model_name = requested_model_name or agent_model_name

    # ---- 3. 组装四要素：model + tools + middleware + system_prompt ----
    return create_agent(
        model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled),
        tools=get_available_tools(model_name=model_name, subagent_enabled=subagent_enabled),
        middleware=_build_middlewares(config, model_name=model_name, agent_name=agent_name),
        system_prompt=apply_prompt_template(subagent_enabled=subagent_enabled, ...),
        state_schema=ThreadState,
    )
```

### 3.2 模型解析链

模型名怎么确定？有三层 fallback：

```
优先级:
  1. 请求中指定的 model_name (用户在前端选的)
  2. Agent 配置文件中的 model (自定义 Agent 的默认模型)
  3. config.yaml 中的第一个 model (全局默认)

如果最终选中的模型不在 config.yaml 中 → 回退到全局默认
如果启用了 thinking 但模型不支持 → 自动降级为非 thinking 模式
```

### 3.3 模型工厂：create_chat_model()

```python
# models/factory.py

def create_chat_model(name: str | None = None, thinking_enabled: bool = False, **kwargs):
    config = get_app_config()
    model_config = config.get_model_config(name)

    # 通过反射加载 LLM 类
    model_class = resolve_class(model_config.use, BaseChatModel)
    # 例如: "langchain_openai:ChatOpenAI" → ChatOpenAI 类

    # 处理 thinking 模式
    if thinking_enabled and model_config.supports_thinking:
        # 合并 when_thinking_enabled 中的额外参数
        model_settings.update(effective_wte)

    model_instance = model_class(**kwargs, **model_settings)

    # 可选：附加 LangSmith tracing
    if is_tracing_enabled():
        tracer = LangChainTracer(project_name=tracing_config.project)
        model_instance.callbacks = [tracer]

    return model_instance
```

**`use` 字段的魔法：** `config.yaml` 中的 `use: langchain_openai:ChatOpenAI` 会通过反射系统（`deerflow.reflection`）动态 import 并实例化。所有以 `$` 开头的值会解析为环境变量。

### 3.4 系统 Prompt 构建

系统 Prompt 不是一个固定字符串，而是**动态拼装**的：

```
apply_prompt_template()
    │
    ├── <role>        ← Agent 名称
    ├── <soul>        ← SOUL.md (自定义人格，可选)
    ├── <memory>      ← 从 memory.json 注入的长期记忆
    ├── <thinking_style>  ← 思考风格指导
    ├── <clarification_system>  ← 澄清机制
    ├── <skill_system>    ← 已启用技能的列表和路径
    ├── <available-deferred-tools>  ← 延迟加载工具的名称列表
    ├── <subagent_system> ← 子 Agent 编排指令（如果启用）
    ├── <working_directory> ← 沙箱路径映射
    ├── <citations>   ← 引用格式规范
    ├── <critical_reminders> ← 关键提醒
    └── <current_date>    ← 当前日期
```

### 3.5 ThreadState：Agent 的状态容器

```python
# agents/thread_state.py

class ThreadState(AgentState):
    sandbox: SandboxState | None        # 沙箱 ID
    thread_data: ThreadDataState | None  # 线程目录路径
    title: str | None                   # 自动生成的对话标题
    artifacts: list[str]                # 生成的文件列表 (去重)
    todos: list | None                  # 计划模式的任务列表
    uploaded_files: list[dict] | None   # 用户上传的文件
    viewed_images: dict[str, ViewedImageData]  # 已查看的图片
```

每个 Thread 都有自己的 ThreadState，线程隔离。

---

## 四、中间件链：12 个横切关注点

这是 DeerFlow 最精妙的设计之一。**不用 DAG，不用 workflow，而是用中间件链来实现所有横切逻辑。**

```
用户消息
  │
  ↓
┌──────────────────────────────────────────────────────────────────┐
│                      中间件链 (严格顺序)                          │
│                                                                  │
│  1. ThreadDataMiddleware     ← 创建线程目录                       │
│  2. UploadsMiddleware        ← 注入上传文件列表                   │
│  3. SandboxMiddleware        ← 获取沙箱环境                       │
│  4. DanglingToolCallMiddleware ← 修补缺失的 ToolMessage          │
│  5. GuardrailMiddleware      ← 工具调用鉴权 (可选)                │
│  6. ToolErrorHandlingMiddleware ← 工具异常转 ToolMessage          │
│  7. SummarizationMiddleware  ← 超长对话自动摘要 (可选)            │
│  8. TodoMiddleware           ← 计划模式任务跟踪 (可选)            │
│  9. TokenUsageMiddleware     ← Token 用量追踪 (可选)              │
│  10. TitleMiddleware         ← 首次对话后自动生成标题              │
│  11. MemoryMiddleware        ← 对话入队异步更新记忆               │
│  12. ViewImageMiddleware     ← 注入图片 base64 (仅视觉模型)       │
│  13. DeferredToolFilterMiddleware ← 隐藏延迟工具 schema (可选)    │
│  14. SubagentLimitMiddleware ← 截断超限的 task 调用 (可选)        │
│  15. LoopDetectionMiddleware ← 检测重复工具调用循环                │
│  16. ClarificationMiddleware ← 拦截澄清请求,中断执行 (必须最后)   │
│                                                                  │
│  注: 实际数量根据配置动态增减, 6-16 是一般情况                     │
└──────────────────────────────────────────────────────────────────┘
  │
  ↓
LLM 调用 → 工具调用 → LLM 调用 → ... → 最终回复
```

### 4.1 ThreadDataMiddleware — 线程隔离的基础

```python
# middlewares/thread_data_middleware.py

class ThreadDataMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        thread_id = context.get("thread_id")
        # 计算（不创建）三个目录的路径：
        return {
            "thread_data": {
                "workspace_path": f"{base}threads/{thread_id}/user-data/workspace",
                "uploads_path":   f"{base}threads/{thread_id}/user-data/uploads",
                "outputs_path":   f"{base}threads/{thread_id}/user-data/outputs",
            }
        }
```

**为什么放第一个？** 因为后续所有中间件（UploadsMiddleware、SandboxMiddleware）都需要 `thread_data` 来知道文件放在哪。

**虚拟路径映射：**
```
Agent 看到的路径          →  宿主机实际路径
/mnt/user-data/workspace  →  backend/.deer-flow/threads/{thread_id}/user-data/workspace
/mnt/user-data/uploads    →  backend/.deer-flow/threads/{thread_id}/user-data/uploads
/mnt/user-data/outputs    →  backend/.deer-flow/threads/{thread_id}/user-data/outputs
/mnt/skills               →  deer-flow/skills/
```

### 4.2 SandboxMiddleware — 沙箱生命周期

```python
# sandbox/middleware.py

class SandboxMiddleware(AgentMiddleware):
    def before_agent(self, state, runtime):
        if self._lazy_init:
            return None  # 延迟到第一次工具调用才获取沙箱

    def after_agent(self, state, runtime):
        sandbox_id = state.get("sandbox", {}).get("sandbox_id")
        if sandbox_id:
            get_sandbox_provider().release(sandbox_id)  # 释放沙箱
```

### 4.3 DanglingToolCallMiddleware — 修补中断的工具调用

当用户中断对话时，可能会出现 AIMessage 有 tool_calls 但没有对应的 ToolMessage。这个中间件会注入占位 ToolMessage，防止 LLM 看到不完整的对话历史而困惑。

### 4.4 GuardrailMiddleware — 工具调用鉴权

可选的安全门，在工具执行前检查权限。支持三种 Provider：

| Provider | 说明 |
|----------|------|
| `AllowlistProvider` | 内置，零依赖，基于黑白名单 |
| OAP Provider | 遵循 Open Agent Passport 标准 |
| Custom Provider | 自定义 `evaluate()` 方法 |

### 4.5 ToolErrorHandlingMiddleware — 异常不崩溃

```python
# middlewares/tool_error_handling_middleware.py

class ToolErrorHandlingMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except GraphBubbleUp:
            raise  # LangGraph 控制流信号要透传
        except Exception as exc:
            # 把异常转成 ToolMessage，让 Agent 继续运行
            return ToolMessage(
                content=f"Error: Tool '{tool_name}' failed: {detail}. "
                        "Continue with available context.",
                status="error",
            )
```

**设计哲学：** 单个工具失败不应该导致整个对话崩溃。Agent 可以看到错误信息，选择重试或换个方案。

### 4.6 SummarizationMiddleware — 长对话不丢上下文

当对话超过 token 阈值（默认 15564 tokens），自动摘要旧消息，保留最近 10 条。

### 4.7 TitleMiddleware — 自动生成标题

```python
# middlewares/title_middleware.py

class TitleMiddleware(AgentMiddleware):
    def after_model(self, state, runtime):
        if self._should_generate_title(state):
            # 条件：第一次完整交换（1 user + 1 assistant）后
            model = create_chat_model(name=config.model_name, thinking_enabled=False)
            response = model.invoke(title_prompt)
            return {"title": title}
```

### 4.8 MemoryMiddleware — 异步记忆更新

```python
# middlewares/memory_middleware.py

class MemoryMiddleware(AgentMiddleware):
    def after_agent(self, state, runtime):
        # 1. 过滤消息：只保留 user + final AI response
        filtered = _filter_messages_for_memory(state["messages"])

        # 2. 入队（不阻塞当前请求）
        queue = get_memory_queue()
        queue.add(thread_id=thread_id, messages=filtered)
        # 队列有 30 秒 debounce，自动批处理
```

### 4.9 LoopDetectionMiddleware — P0 安全机制

```python
# middlewares/loop_detection_middleware.py

# 滑动窗口检测重复工具调用
# 同一组工具调用出现 3 次 → 注入警告
# 出现 5 次 → 强制剥离 tool_calls，逼 Agent 输出文本回答
```

### 4.10 ClarificationMiddleware — "先问再做"

```python
# middlewares/clarification_middleware.py

class ClarificationMiddleware(AgentMiddleware):
    def wrap_tool_call(self, request, handler):
        if request.tool_call.get("name") != "ask_clarification":
            return handler(request)

        # 拦截 ask_clarification 调用，中断执行
        return Command(
            update={"messages": [formatted_tool_message]},
            goto=END,  # 终止当前轮次，等用户回复
        )
```

**为什么放最后？** 因为它需要拦截 `ask_clarification` 工具调用，而这个调用只有在模型已经决策之后才会出现。

### 4.11 SubagentLimitMiddleware — 硬性并发限制

```python
# middlewares/subagent_limit_middleware.py

class SubagentLimitMiddleware(AgentMiddleware):
    def after_model(self, state, runtime):
        # 模型生成了 N 个 task() 调用
        # 只保留前 max_concurrent 个，多余的直接截断
        # max_concurrent 限定在 [2, 4] 范围
```

### 中间件顺序总结

| # | 中间件 | 时机 | 核心作用 |
|---|--------|------|---------|
| 1 | ThreadData | before_agent | 创建线程目录 |
| 2 | Uploads | before_agent | 注入上传文件列表 |
| 3 | Sandbox | before/after_agent | 沙箱生命周期 |
| 4 | DanglingToolCall | before_model | 修补缺失 ToolMessage |
| 5 | Guardrail | wrap_tool_call | 工具调用鉴权 |
| 6 | ToolErrorHandling | wrap_tool_call | 异常转 ToolMessage |
| 7 | Summarization | before_model | 超长对话摘要 |
| 8 | Todo | before_model | 计划模式任务跟踪 |
| 9 | TokenUsage | after_model | Token 用量追踪 |
| 10 | Title | after_model | 自动生成标题 |
| 11 | Memory | after_agent | 异步更新记忆 |
| 12 | ViewImage | before_model | 注入图片 base64 |
| 13 | DeferredToolFilter | before_model | 隐藏延迟工具 schema |
| 14 | SubagentLimit | after_model | 截断超限 task 调用 |
| 15 | LoopDetection | after_model | 检测重复工具调用 |
| 16 | Clarification | wrap_tool_call | 拦截澄清请求 |

---

## 五、子 Agent 并行系统

### 5.1 设计原理

当用户问一个复杂问题（比如"比较 5 个云服务商"），Lead Agent 不会自己逐一搜索，而是：

```
用户: "比较 AWS, Azure, GCP, 阿里云, Oracle Cloud"
       │
       ↓
Lead Agent 思考:
  "这个任务可以分解为 5 个独立子任务"
  "并发限制是 3，所以分两批"
       │
       ↓
第一批 (并行):              第二批 (并行):
┌─────────┐ ┌─────────┐ ┌──────┐    ┌─────────┐ ┌──────────┐
│task: AWS│ │task:Azure│ │task: │    │task:阿里│ │task:Oracle│
│         │ │         │ │ GCP  │    │  云     │ │          │
└────┬────┘ └────┬────┘ └──┬───┘    └────┬────┘ └─────┬────┘
     │           │          │             │            │
     ↓           ↓          ↓             ↓            ↓
 搜索+分析    搜索+分析   搜索+分析    搜索+分析    搜索+分析
     │           │          │             │            │
     └───────────┼──────────┘             └────────────┘
                 ↓                              ↓
           第一批结果                       第二批结果
                 │                              │
                 └──────────┬───────────────────┘
                            ↓
                    Lead Agent 综合分析
                            ↓
                    最终比较报告 (带引用)
```

### 5.2 SubagentExecutor：双线程池架构

```python
# subagents/executor.py

# 调度线程池（3 workers）— 管理任务生命周期
_scheduler_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-scheduler-")

# 执行线程池（3 workers）— 实际运行 Agent
_execution_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-exec-")
```

**为什么要两个线程池？**

```
_scheduler_pool:
  ┌────────────┐
  │ run_task() │ ← 管理超时、状态更新
  │            │
  │  submit → ─┼─→ _execution_pool:
  │            │    ┌────────────────┐
  │  wait ← ──┼─── │ executor.execute()│ ← 真正跑 Agent
  │            │    │ (asyncio.run)  │
  │  update ── │    └────────────────┘
  │  result    │
  └────────────┘
```

调度池负责超时控制（默认 15 分钟），执行池负责实际 Agent 运行。分开是为了避免超时逻辑和 Agent 执行互相阻塞。

### 5.3 内置子 Agent 类型

```python
# subagents/builtins/general_purpose.py

GENERAL_PURPOSE_CONFIG = SubagentConfig(
    name="general-purpose",
    system_prompt="...",       # 自治工作，不问用户
    tools=None,                # 继承父 Agent 所有工具
    disallowed_tools=["task", "ask_clarification", "present_files"],  # 防止嵌套
    model="inherit",           # 使用父 Agent 的模型
    max_turns=50,              # 最多 50 轮工具调用
    timeout_seconds=900,       # 15 分钟超时
)
```

| 子 Agent | 特点 | 典型用途 |
|---------|------|---------|
| `general-purpose` | 所有工具（除 task/clarification），50 轮 | 复杂多步任务、网络调研、代码分析 |
| `bash` | 仅命令执行工具 | git 操作、构建、部署 |

### 5.4 task_tool：Lead Agent 和子 Agent 的桥梁

```python
# tools/builtins/task_tool.py

@tool("task")
async def task_tool(runtime, description, prompt, subagent_type, tool_call_id, max_turns=None):
    # 1. 获取子 Agent 配置
    config = get_subagent_config(subagent_type)

    # 2. 从父 Agent 继承上下文
    executor = SubagentExecutor(
        config=config,
        tools=get_available_tools(subagent_enabled=False),  # 子 Agent 不能再分身
        parent_model=parent_model,
        sandbox_state=sandbox_state,   # 共享沙箱
        thread_data=thread_data,       # 共享文件目录
        thread_id=thread_id,
    )

    # 3. 异步执行 + 轮询
    task_id = executor.execute_async(prompt)

    writer = get_stream_writer()
    writer({"type": "task_started", "task_id": task_id})

    while True:
        result = get_background_task_result(task_id)
        if result.status == SubagentStatus.COMPLETED:
            writer({"type": "task_completed", ...})
            return f"Task Succeeded. Result: {result.result}"
        await asyncio.sleep(5)
```

**关键设计：**
- 子 Agent 的 `subagent_enabled=False`，防止无限嵌套
- 共享 sandbox 和 thread_data，子 Agent 可以读写父 Agent 的文件
- 通过 SSE `stream_writer` 实时推送 `task_started`/`task_running`/`task_completed` 事件
- 自动清理完成的后台任务，防止内存泄漏

---

## 六、工具系统

### 6.1 工具组装过程

```python
# tools/tools.py

def get_available_tools(groups, include_mcp, model_name, subagent_enabled):
    tools = []

    # ---- Layer 1: 配置定义的工具 ----
    # 从 config.yaml 的 tools[] 加载
    tools += [resolve_variable(tool.use) for tool in config.tools]

    # ---- Layer 2: 内置工具 ----
    builtin = [present_file_tool, ask_clarification_tool]
    if model_supports_vision:
        builtin.append(view_image_tool)
    if subagent_enabled:
        builtin.append(task_tool)

    # ---- Layer 3: MCP 工具 ----
    mcp_tools = get_cached_mcp_tools()  # 延迟加载 + mtime 缓存

    # ---- Layer 4: ACP Agent 工具 ----
    acp_tools = [build_invoke_acp_agent_tool(acp_agents)]

    return tools + builtin + mcp_tools + acp_tools
```

### 6.2 默认工具清单

| 工具 | 组 | 说明 |
|------|---|------|
| `web_search` | web | DuckDuckGo 搜索（默认），也支持 Tavily |
| `web_fetch` | web | Jina AI Reader 抓取网页 |
| `image_search` | web | DuckDuckGo 图片搜索 |
| `bash` | bash | 沙箱内执行命令 |
| `ls` | file:read | 目录列表（树形，最深 2 层） |
| `read_file` | file:read | 读取文件（支持行号范围） |
| `write_file` | file:write | 写入/追加文件 |
| `str_replace` | file:write | 字符串替换（精确/全局） |
| `present_files` | 内置 | 将文件呈现给用户 |
| `ask_clarification` | 内置 | 向用户提问 |
| `view_image` | 内置 | 读取图片为 base64（仅视觉模型） |
| `task` | 内置 | 委托给子 Agent（仅启用子 Agent 时） |
| `tool_search` | 内置 | 延迟工具发现（仅启用 tool_search 时） |

### 6.3 Tool Search：延迟加载大量工具

当接入多个 MCP Server、工具总数达到数十甚至上百个时，把所有工具的 schema 都塞进 LLM context 是浪费且有害的。Tool Search 解决了这个问题：

```
                           启动时
                             │
               ┌─────────────┴─────────────┐
               ↓                           ↓
        工具少 (< 10)               工具多 (10+)
               │                           │
        全部加载到 context          只在 prompt 列出名字
               │                    + 注册到 DeferredToolRegistry
               │                           │
               │                    Agent 看到:
               │                    <available-deferred-tools>
               │                    github__create_issue
               │                    github__list_repos
               │                    postgres__query
               │                    ...
               │                    </available-deferred-tools>
               │                           │
               │                    Agent 需要用某个工具时:
               │                    tool_search("select:github__create_issue")
               │                           │
               │                    返回完整 schema → Agent 可以调用了
```

```python
# tools/builtins/tool_search.py

class DeferredToolRegistry:
    def search(self, query: str) -> list[BaseTool]:
        # "select:name1,name2" → 精确匹配
        # "+keyword rest"     → 名字必须包含 keyword
        # "keyword"           → 正则搜索 name + description
```

### 6.4 MCP 集成

MCP (Model Context Protocol) 让 DeerFlow 可以接入任何实现了 MCP 协议的外部服务。

```json
// extensions_config.json
{
  "mcpServers": {
    "github": {
      "enabled": true,
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "$GITHUB_TOKEN" }
    },
    "postgres": {
      "enabled": false,
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    }
  }
}
```

**支持的传输方式：**

| 传输 | 说明 |
|------|------|
| `stdio` | 启动子进程，通过 stdin/stdout 通信 |
| `sse` | Server-Sent Events 连接 |
| `http` | HTTP 端点 |

**缓存机制：**

```python
# mcp/cache.py

def get_cached_mcp_tools():
    # 1. 检查 extensions_config.json 的 mtime
    if _is_cache_stale():
        reset_mcp_tools_cache()  # 重新加载

    # 2. 首次调用时 lazy init
    if not _cache_initialized:
        asyncio.run(initialize_mcp_tools())

    return _mcp_tools_cache
```

**mtime 对齐：** Gateway API（进程 A）修改了 `extensions_config.json`，LangGraph Server（进程 B）通过检测文件 mtime 变化自动重载。无需重启服务。

### 6.5 同步/异步兼容

MCP 工具天然是异步的（`coroutine`），但 DeerFlow 的 subagent 执行链是同步的。解决方案：

```python
# mcp/tools.py

def _make_sync_tool_wrapper(coro, tool_name):
    def sync_wrapper(*args, **kwargs):
        loop = asyncio.get_running_loop()
        if loop is not None and loop.is_running():
            # 在全局线程池中运行，避免嵌套 event loop
            future = _SYNC_TOOL_EXECUTOR.submit(asyncio.run, coro(*args, **kwargs))
            return future.result()
        else:
            return asyncio.run(coro(*args, **kwargs))
    return sync_wrapper
```

---

## 七、记忆系统

### 7.1 整体流程

```
用户对话
  │
  ↓
MemoryMiddleware
  │ 过滤消息: 只保留 user + final AI response
  │ 去掉 tool messages 和中间 AI 消息
  │ 去掉 <uploaded_files> 标签 (文件上传是临时的)
  ↓
MemoryQueue (30秒 debounce)
  │ 同一 thread 的多次更新合并
  │ 后台线程处理
  ↓
MemoryUpdater
  │ 发送当前记忆 + 新对话给 LLM
  │ LLM 返回结构化更新
  ↓
memory.json (原子写入: tmp + rename)
  │
  ↓
下次对话时, 通过 <memory> 标签注入 system prompt
```

### 7.2 记忆数据结构

```json
// memory.json
{
  "version": "1.0",
  "lastUpdated": "2026-03-28T10:00:00Z",
  "user": {
    "workContext": {
      "summary": "DeerFlow 核心贡献者，负责 Agent 架构...",
      "updatedAt": "..."
    },
    "personalContext": {
      "summary": "中英双语，偏好简洁代码风格...",
      "updatedAt": "..."
    },
    "topOfMind": {
      "summary": "正在优化子 Agent 并发性能，同时调研 MCP 生态...",
      "updatedAt": "..."
    }
  },
  "history": {
    "recentMonths": { "summary": "...", "updatedAt": "..." },
    "earlierContext": { "summary": "...", "updatedAt": "..." },
    "longTermBackground": { "summary": "...", "updatedAt": "..." }
  },
  "facts": [
    {
      "id": "fact_a1b2c3d4",
      "content": "偏好使用 Python 3.12+ 类型注解",
      "category": "preference",
      "confidence": 0.9,
      "createdAt": "2026-03-15T...",
      "source": "thread_abc123"
    }
  ]
}
```

### 7.3 Fact 提取与去重

```python
# agents/memory/updater.py

class MemoryUpdater:
    def _apply_updates(self, current_memory, update_data, thread_id):
        # ---- 去除旧 fact ----
        facts_to_remove = set(update_data.get("factsToRemove", []))
        current_memory["facts"] = [f for f in facts if f["id"] not in facts_to_remove]

        # ---- 添加新 fact (去重) ----
        existing_keys = {fact["content"].strip() for fact in current_memory["facts"]}
        for fact in update_data.get("newFacts", []):
            if fact["confidence"] >= config.fact_confidence_threshold:  # 默认 0.7
                key = fact["content"].strip()
                if key not in existing_keys:
                    current_memory["facts"].append({
                        "id": f"fact_{uuid4().hex[:8]}",
                        "content": key,
                        "category": fact["category"],
                        "confidence": fact["confidence"],
                        "createdAt": now,
                        "source": thread_id,
                    })
                    existing_keys.add(key)

        # ---- 超过 100 条 → 按 confidence 排序截断 ----
        if len(current_memory["facts"]) > config.max_facts:
            current_memory["facts"] = sorted(
                current_memory["facts"],
                key=lambda f: f.get("confidence", 0),
                reverse=True
            )[:config.max_facts]
```

### 7.4 记忆注入

```python
# agents/memory/prompt.py

def format_memory_for_injection(memory_data, max_tokens=2000):
    sections = []

    # User Context (简短)
    sections.append("User Context:\n- Work: ...\n- Personal: ...\n- Current Focus: ...")

    # History (中等长度)
    sections.append("History:\n- Recent: ...\n- Earlier: ...")

    # Facts (按 confidence 降序，逐条添加直到 token 预算耗尽)
    ranked_facts = sorted(facts, key=lambda f: f["confidence"], reverse=True)
    for fact in ranked_facts:
        line = f"- [{category} | {confidence:.2f}] {content}"
        if running_tokens + line_tokens <= max_tokens:
            fact_lines.append(line)
    sections.append("Facts:\n" + "\n".join(fact_lines))

    return "\n\n".join(sections)
```

### 7.5 防止文件上传污染记忆

文件上传是临时的，不应该进入长期记忆。DeerFlow 用两道防线：

1. **MemoryMiddleware** 阶段：strip 掉 `<uploaded_files>` 标签
2. **MemoryUpdater** 阶段：用正则清除所有提及文件上传的句子

```python
# agents/memory/updater.py

_UPLOAD_SENTENCE_RE = re.compile(
    r"[^.!?]*\b(?:upload(?:ed|ing)?.*?file|/mnt/user-data/uploads/)[^.!?]*[.!?]?\s*",
    re.IGNORECASE,
)

def _strip_upload_mentions_from_memory(memory_data):
    # 从所有 summary 和 facts 中清除上传相关内容
```

---

## 八、沙箱执行

### 8.1 抽象接口

```python
# sandbox/sandbox.py

class Sandbox(ABC):
    def execute_command(self, command: str) -> str: ...
    def read_file(self, path: str) -> str: ...
    def write_file(self, path: str, content: str, append: bool = False): ...
    def list_dir(self, path: str, max_depth=2) -> list[str]: ...

# sandbox/sandbox_provider.py

class SandboxProvider(ABC):
    def acquire(self, thread_id: str) -> str: ...  # 获取沙箱，返回 ID
    def get(self, sandbox_id: str) -> Sandbox: ...  # 通过 ID 获取沙箱
    def release(self, sandbox_id: str): ...         # 释放沙箱
```

### 8.2 三种沙箱实现

| 沙箱 | 配置 | 隔离级别 | 适用场景 |
|------|------|---------|---------|
| **LocalSandboxProvider** | 默认 | 无（直接在宿主机执行） | 开发/单用户 |
| **AioSandboxProvider** (Docker) | `deerflow.community.aio_sandbox:AioSandboxProvider` | 容器级别 | 多用户/需要隔离 |
| **AioSandboxProvider** (Provisioner) | 加上 `provisioner_url` | Pod 级别 (k3s) | 生产环境 |

**本地沙箱的安全机制：**

```python
# sandbox/tools.py

@tool("bash")
def bash_tool(runtime, description, command):
    if is_local_sandbox(runtime):
        # 1. 验证命令中的绝对路径
        validate_local_bash_command_paths(command, thread_data)
        # 只允许 /mnt/user-data/*, /mnt/skills/*, /bin/*, /usr/bin/* 等

        # 2. 替换虚拟路径为实际路径
        command = replace_virtual_paths_in_command(command, thread_data)

        # 3. 执行后遮掩宿主机路径
        output = sandbox.execute_command(command)
        return mask_local_paths_in_output(output, thread_data)
```

**路径遮掩示例：**
```
实际输出: File saved to /Users/danny/Developer/deer-flow/backend/.deer-flow/threads/abc/user-data/outputs/report.md
Agent 看到: File saved to /mnt/user-data/outputs/report.md
```

### 8.3 沙箱工具详解

| 工具 | 读/写 | 特殊行为 |
|------|-------|---------|
| `bash` | 读写 | 路径验证 → 虚拟路径替换 → 执行 → 路径遮掩 |
| `ls` | 只读 | 树形输出，最深 2 层 |
| `read_file` | 只读 | 支持 `start_line`/`end_line` 行号范围 |
| `write_file` | 写 | 支持 append 模式，自动创建目录 |
| `str_replace` | 写 | 默认只替换第一个匹配，`replace_all=True` 全局替换 |

---

## 九、技能系统

### 9.1 SKILL.md 格式

```markdown
---
name: deep-research
description: Deep research skill for comprehensive topic analysis
license: MIT
---

# Deep Research Skill

## Overview
This skill guides the agent through a systematic research workflow...

## Workflow
1. Read `references/search_strategy.md` for search optimization
2. Conduct multi-perspective research using web_search
3. Synthesize findings into a structured report
...
```

### 9.2 技能加载流程

```
skills/
├── public/              ← 公开技能 (提交到 Git)
│   ├── deep-research/
│   │   ├── SKILL.md     ← 入口文件 (YAML frontmatter + 指令)
│   │   └── references/  ← 附加资源 (按需加载)
│   └── bootstrap/
│       └── SKILL.md
└── custom/              ← 自定义技能 (gitignored)
    └── my-skill/
        └── SKILL.md
```

```python
# skills/parser.py

def parse_skill_file(skill_file: Path, category: str):
    content = skill_file.read_text()
    # 提取 YAML front matter
    front_matter_match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    # 解析 name, description, license
    return Skill(name=name, description=description, ...)
```

### 9.3 Progressive Loading 模式

技能不是一次性全部加载到 context 的，而是**渐进式加载**：

```
System Prompt 中只列出名字和描述:
  <skill_system>
    <available_skills>
      <skill>
        <name>deep-research</name>
        <description>Deep research for comprehensive analysis</description>
        <location>/mnt/skills/public/deep-research/SKILL.md</location>
      </skill>
    </available_skills>
  </skill_system>

Agent 匹配到需要时:
  → read_file("/mnt/skills/public/deep-research/SKILL.md")
  → 按照 SKILL.md 中的指令执行
  → 按需 read_file() 加载 references/ 下的额外资源
```

### 9.4 运行时启停

通过 Gateway API 或 `extensions_config.json`：

```json
// extensions_config.json
{
  "skills": {
    "deep-research": { "enabled": true },
    "bootstrap": { "enabled": false }
  }
}
```

```
PUT /api/skills/deep-research
{ "enabled": false }
```

### 9.5 技能安装

```
POST /api/skills/install
Content-Type: multipart/form-data
file: my-skill.skill  (ZIP 格式)
```

解压到 `skills/custom/` 目录。

---

## 十、配置系统

DeerFlow 有三层配置文件：

### 10.1 config.yaml — 主配置

```yaml
# 模型配置
models:
  - name: gpt-4
    display_name: GPT-4
    use: langchain_openai:ChatOpenAI   # 反射加载类
    model: gpt-4
    api_key: $OPENAI_API_KEY           # 环境变量解析
    max_tokens: 4096
    supports_thinking: false
    supports_vision: true

# 工具配置
tools:
  - name: web_search
    group: web
    use: deerflow.community.ddg_search.tools:web_search_tool

# 沙箱配置
sandbox:
  use: deerflow.sandbox.local:LocalSandboxProvider

# 记忆配置
memory:
  enabled: true
  debounce_seconds: 30
  max_facts: 100
  fact_confidence_threshold: 0.7
  max_injection_tokens: 2000

# 摘要配置
summarization:
  enabled: true
  trigger:
    - type: tokens
      value: 15564
  keep:
    type: messages
    value: 10

# 标题生成
title:
  enabled: true
  max_words: 6
  max_chars: 60
```

**配置版本控制：** `config_version` 字段用于检测配置文件过时。运行 `make config-upgrade` 自动合并新字段。

**配置查找优先级：**
1. 显式 `config_path` 参数
2. `DEER_FLOW_CONFIG_PATH` 环境变量
3. 当前目录 (`backend/`) 的 `config.yaml`
4. 父目录（项目根目录）的 `config.yaml` -- 推荐

**配置热加载：** `get_app_config()` 缓存配置，但会检测文件 mtime 变化自动重载。无需重启。

### 10.2 extensions_config.json — MCP 和技能

```json
{
  "mcpServers": {
    "github": {
      "enabled": true,
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "$GITHUB_TOKEN" }
    }
  },
  "skills": {
    "deep-research": { "enabled": true }
  }
}
```

可通过 Gateway API 运行时修改。

### 10.3 .env — 密钥管理

config.yaml 中以 `$` 开头的值会从环境变量解析。推荐用 `.env` 文件管理：

```bash
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=ghp_...
TAVILY_API_KEY=tvly-...
```

### 10.4 运行时覆盖

前端/API 可以通过 `config.configurable` 在运行时覆盖：

| 参数 | 说明 |
|------|------|
| `model_name` | 切换模型 |
| `thinking_enabled` | 开关思考模式 |
| `reasoning_effort` | 思考深度（low/medium/high） |
| `is_plan_mode` | 开关计划模式 |
| `subagent_enabled` | 开关子 Agent |
| `max_concurrent_subagents` | 最大并发子 Agent 数 |

---

## 十一、一个完整的研究任务 Walkthrough

用户问："比较 5 个云服务商的定价策略"。以下是系统内部完整的执行过程：

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 1: 请求进入                                                          │
│                                                                          │
│ POST /api/langgraph/threads/{thread_id}/runs/stream                     │
│ Body: { "input": { "messages": [{"role":"user","content":"比较 5 个..."}] }, │
│        "config": { "configurable": {                                     │
│          "model_name": "gpt-4",                                          │
│          "thinking_enabled": true,                                       │
│          "subagent_enabled": true,                                       │
│          "max_concurrent_subagents": 3                                   │
│        }}}                                                               │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 2: make_lead_agent()                                                │
│                                                                          │
│ ● 解析模型: gpt-4 (config.yaml 中找到)                                    │
│ ● 创建模型: create_chat_model("gpt-4", thinking_enabled=True)            │
│ ● 加载工具: get_available_tools(subagent_enabled=True)                    │
│   → [web_search, web_fetch, bash, ls, read_file, write_file,            │
│      str_replace, present_files, ask_clarification, task, ...]          │
│ ● 构建中间件链: 16 个中间件 (含 SubagentLimitMiddleware)                  │
│ ● 生成系统 Prompt: 包含 <subagent_system> 和 <memory> 标签               │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 3: 中间件链 - before_agent                                          │
│                                                                          │
│ ThreadDataMiddleware:                                                    │
│   → thread_data = {workspace_path, uploads_path, outputs_path}          │
│                                                                          │
│ SandboxMiddleware:                                                       │
│   → lazy_init=True, 暂不获取沙箱                                         │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 4: LLM 第一轮思考                                                   │
│                                                                          │
│ GPT-4 (thinking): "这个任务可以分解为 5 个独立子任务。                      │
│  并发限制是 3，所以分两批：                                                │
│  Batch 1: AWS, Azure, GCP                                                │
│  Batch 2: 阿里云, Oracle Cloud"                                          │
│                                                                          │
│ 输出 3 个 tool_calls:                                                    │
│   task(description="AWS 定价分析", prompt="...", subagent_type="general") │
│   task(description="Azure 定价分析", prompt="...", subagent_type="general")│
│   task(description="GCP 定价分析", prompt="...", subagent_type="general") │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 5: 中间件链 - after_model                                           │
│                                                                          │
│ SubagentLimitMiddleware:                                                  │
│   → 3 个 task 调用 ≤ max_concurrent(3)，通过                             │
│                                                                          │
│ LoopDetectionMiddleware:                                                  │
│   → 第一次出现，无重复，通过                                               │
│                                                                          │
│ TitleMiddleware:                                                          │
│   → 第一次交换，生成标题: "云服务商定价策略比较"                             │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 6: 3 个 task 工具并行执行                                            │
│                                                                          │
│ ┌──────────────────────────────────────────────────────────┐              │
│ │ task_tool("AWS 定价分析", ...)                            │              │
│ │  → SubagentExecutor.execute_async(prompt)                │              │
│ │  → _scheduler_pool → _execution_pool                    │              │
│ │  → 创建 general-purpose Agent (不含 task 工具)           │              │
│ │  → Agent 循环: web_search → web_fetch → 分析 → 总结     │              │
│ │  → SSE: task_started → task_running → task_completed    │              │
│ │  → 结果: "AWS 采用按需/预留/节省计划三种定价..."         │              │
│ └──────────────────────────────────────────────────────────┘              │
│                                                                          │
│ (Azure 和 GCP 同时并行，类似过程)                                         │
│                                                                          │
│ 前端实时收到 SSE 事件:                                                    │
│   {type: "task_started", task_id: "tc_001"}                              │
│   {type: "task_running", task_id: "tc_001", message: {...}}              │
│   {type: "task_completed", task_id: "tc_001", result: "..."}             │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 7: LLM 第二轮 — 处理第一批结果 + 启动第二批                          │
│                                                                          │
│ GPT-4 看到 3 个 ToolMessage (AWS/Azure/GCP 的结果)                       │
│                                                                          │
│ 输出 2 个 tool_calls:                                                    │
│   task(description="阿里云定价分析", ...)                                 │
│   task(description="Oracle Cloud 定价分析", ...)                         │
│                                                                          │
│ (第二批并行执行，同样的流程)                                               │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 8: LLM 第三轮 — 综合分析                                            │
│                                                                          │
│ GPT-4 看到全部 5 个结果，生成综合比较报告:                                 │
│                                                                          │
│ "## 五大云服务商定价策略比较                                               │
│                                                                          │
│ ### 1. AWS                                                               │
│ AWS 采用三层定价... [citation:AWS Pricing](https://aws.amazon.com/...)    │
│                                                                          │
│ ### 2. Azure                                                             │
│ ...                                                                      │
│                                                                          │
│ ## 对比总结                                                               │
│ | 维度 | AWS | Azure | GCP | 阿里云 | Oracle |                           │
│ |------|-----|-------|-----|--------|--------|                            │
│ | 按需定价 | ... | ... | ... | ... | ... |                               │
│                                                                          │
│ ## Sources                                                               │
│ - [AWS Pricing](https://...) - 官方定价页面                               │
│ ..."                                                                     │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 9: 中间件链 - after_agent                                           │
│                                                                          │
│ MemoryMiddleware:                                                        │
│   → 过滤消息 (只保留 user + final AI)                                    │
│   → 入队: get_memory_queue().add(thread_id, filtered_messages)           │
│   → 30 秒后后台线程: LLM 提取 facts, 更新 memory.json                   │
│                                                                          │
│ SandboxMiddleware:                                                        │
│   → release(sandbox_id)                                                  │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 ↓
┌────────────────────────────────────────────────────────────────────────────┐
│ Step 10: 记忆更新 (30秒后异步)                                           │
│                                                                          │
│ MemoryUpdater.update_memory():                                           │
│   LLM 分析对话, 提取:                                                    │
│   - topOfMind: "正在进行云服务商对比分析"                                  │
│   - newFacts: [                                                          │
│       {content: "关注云服务商定价策略", category: "goal", confidence: 0.8}│
│     ]                                                                    │
│   → 原子写入 memory.json                                                 │
└────────────────────────────────────────────────────────────────────────────┘
```

**时间线总结：**

| 阶段 | 耗时 | 并行度 |
|------|------|--------|
| Agent 构建 | <100ms | 1 |
| LLM 第一轮思考 | 2-5s | 1 |
| Batch 1 (3 子 Agent) | 15-30s | 3 并行 |
| LLM 第二轮 | 1-2s | 1 |
| Batch 2 (2 子 Agent) | 15-30s | 2 并行 |
| LLM 综合分析 | 5-10s | 1 |
| **总计** | **~40-80s** | |

如果不用子 Agent，Lead Agent 需要串行搜索 5 次、串行分析 5 次，总计可能 3-5 分钟。

---

## 十二、快速上手

### 12.1 系统要求

- Python 3.12+
- Node.js 22+
- uv (Python 包管理)
- pnpm (Node.js 包管理)

### 12.2 安装

```bash
# 克隆仓库
git clone https://github.com/bytedance/deer-flow.git
cd deer-flow

# 检查环境
make check

# 安装依赖
make install

# 生成配置文件
make config
# → 创建 config.yaml 和 extensions_config.json
```

### 12.3 配置模型

编辑 `config.yaml`，至少配置一个模型：

```yaml
models:
  - name: gpt-4
    display_name: GPT-4
    use: langchain_openai:ChatOpenAI
    model: gpt-4
    api_key: $OPENAI_API_KEY
    max_tokens: 4096
    supports_vision: true
```

设置环境变量：
```bash
export OPENAI_API_KEY=sk-...
```

### 12.4 启动

```bash
# 一键启动所有服务（开发模式，带热重载）
make dev
```

打开 `http://localhost:2026` 即可使用。

### 12.5 Docker 部署

```bash
# 生产模式
make up

# 停止
make down
```

### 12.6 嵌入式 Python 客户端

不需要启动任何服务，直接在 Python 中使用：

```python
from deerflow.client import DeerFlowClient

client = DeerFlowClient()

# 同步对话
response = client.chat("帮我搜索最新的 AI 论文")
print(response)

# 流式对话
for event in client.stream("分析这个代码库的架构"):
    if event.type == "messages-tuple":
        print(event.data)  # 实时输出
```

---

## 十三、设计哲学

### 13.1 中间件而非 DAG

**为什么不用 LangGraph 的 DAG 定义复杂工作流？**

```
传统 DAG 方案:
  ┌──────┐     ┌──────┐     ┌──────┐
  │ Node1│ ──→ │ Node2│ ──→ │ Node3│
  └──────┘     └──────┘     └──────┘
  问题: 每个横切关注点 (记忆、标题、安全) 都要侵入每个节点

DeerFlow 方案:
  中间件链 → 通用 Agent 循环 (model → tool → model → ...)
  横切关注点在中间件中统一处理，Agent 逻辑保持简洁
```

**优势：**
- 添加新功能只需写一个中间件，不影响现有代码
- 中间件顺序有严格语义，易于推理
- 测试简单：每个中间件独立测试

### 13.2 Harness / App 分离

```
deerflow-harness (可发布的 pip 包)
├── agents/       ← Agent 构建
├── tools/        ← 工具系统
├── sandbox/      ← 沙箱执行
├── mcp/          ← MCP 集成
├── models/       ← 模型工厂
├── skills/       ← 技能系统
├── memory/       ← 记忆系统
├── config/       ← 配置系统
└── client.py     ← 嵌入式客户端

app (不发布, 应用层)
├── gateway/      ← REST API
└── channels/     ← IM 集成
```

**为什么要分？** 因为 harness 可以独立使用（通过 `DeerFlowClient`），不需要 FastAPI 或 Next.js。这让 DeerFlow 可以嵌入到任何 Python 应用中。

### 13.3 线程隔离

每个对话 Thread 有：
- 独立的文件系统目录
- 独立的 ThreadState
- 独立的沙箱环境（Docker 模式下）
- 独立的记忆更新队列

用户 A 的文件和用户 B 完全隔离。

### 13.4 懒加载优先

几乎所有资源都是懒加载的：
- 沙箱：第一次工具调用才 acquire
- MCP 工具：第一次需要才初始化
- 记忆：第一次读取才加载文件
- 线程目录：第一次写入才创建

这样冷启动快，资源消耗低。

### 13.5 反射系统 — 一切皆可扩展

`config.yaml` 中的 `use` 字段通过反射系统动态加载类或变量：

```yaml
# 加载类
use: langchain_openai:ChatOpenAI
# → from langchain_openai import ChatOpenAI

# 加载变量
use: deerflow.community.tavily.tools:web_search_tool
# → from deerflow.community.tavily.tools import web_search_tool
```

**扩展方式：** 只要实现对应的接口（`BaseChatModel`、`BaseTool`、`SandboxProvider`），就可以在 `config.yaml` 中引用。

### 13.6 配置即代码

所有运行时行为都通过配置控制：
- 模型切换：改 `config.yaml` 或运行时 `model_name` 参数
- 工具增减：改 `config.yaml` 的 `tools[]`
- MCP 服务：改 `extensions_config.json`（支持 API 修改）
- 技能启停：改 `extensions_config.json`（支持 API 修改）
- 记忆开关：改 `config.yaml` 的 `memory.enabled`
- 沙箱切换：改 `config.yaml` 的 `sandbox.use`

不需要改代码，不需要重新部署。

---

*Generated on 2026-03-28 from DeerFlow 2.0 source code analysis*
