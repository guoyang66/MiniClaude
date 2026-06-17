# MiniClaude

不是要一比一复刻 Claude Code 的所有产品能力，而是把 Claude Code 这类 AI 编程 Agent 最核心的运行机制拆出来：

* 用户输入一个目标，Agent 能自己规划下一步
* 模型不是只回答文本，而是能主动发起工具调用
* 工具调用不是直接裸跑，而是有参数校验和权限审批
* 执行过程不是黑盒，而是通过事件流实时展示到 TUI
* 每一次 run 都能留下 events、trace、session 记录，方便复盘和排查
* 多轮会话不是简单拼接历史，而是有 thread、notes、context 分层记忆
* 上下文快爆了，不是粗暴截断，而是有水位检测和 compact 压缩
* 复杂任务可以交给子 Agent，外部工具可以通过 MCP 接进来

我们要做的是一个真正能跑任务、能调工具、能看过程、能管权限、能续上下文、能扩展生态的本地 Agent 运行时。

你学完之后，再看 Claude Code、Codex、Cursor 这些 AI 编程工具，就不会只停留在“它好像很智能”。

你能看懂它背后那条工程主线：

**用户目标 → Agent Loop → 模型思考 → 工具调用 → 结果回填 → 事件展示 → 会话续航。**

一次完整的 Agent 使用体验：

* 克隆项目和切换阶段分支
* 配置 `.env`
* 让 Agent 写一个一个任务
* 配置 Skill 和 MCP
* 在 TUI 里看到工具调用、事件流、权限审批和上下文水位

### MiniClaude 长什么样？

MiniClaude 的最终形态是这样的：

用户不是直接和一个脚本对话，而是通过 `mini` CLI 或 `mini-tui` 连接到常驻的 `mini-core` 守护进程。

真正执行任务的是 Core daemon。

CLI 和 TUI 只是客户端。

这意味着：

* TUI 崩了，Agent 任务不一定要跟着死
* 后续可以同时接 CLI、TUI、Web 前端
* 所有任务过程都能通过事件流订阅
* 所有命令、响应、事件都要通过类型化协议通信
* Agent 的工具调用、会话记忆、权限审批、上下文压缩，都在同一条运行链路里完成


### 项目架构图

![](docs/images/20260610114820_MiniClaude架构图-分层版.png)

MiniClaude 的核心不是一个 prompt，而是一套完整的本地 Agent 运行链路：

```latex
用户目标
  → CLI / TUI
  → JSON-RPC over NDJSON
  → mini-core daemon
  → AgentRunner
  → AgentLoop
  → LLM Provider
  → ToolRegistry
  → PermissionManager
  → EventBus
  → Session Store
  → TUI 实时渲染 / events.jsonl 持久化 / trace 回放
```

MiniClaude 是把 Claude Code 这类 AI 编程 Agent 背后的核心机制，用一个 mini 版工程完整跑通：它不是单进程脚本，而是 `mini-core` daemon + CLI/TUI 多客户端架构；

不是一次性调大模型，而是 ReAct AgentLoop，支持模型思考、工具调用、结果回填和多步执行；

不是让模型说执行就执行，而是把工具调用放进 `ToolRegistry` 和 `PermissionManager`，先做参数校验、权限审批、失败分类，再把 tool result 返回给模型；

不是只展示最终答案，而是通过 `EventBus`、events、trace 和 TUI，把 token 流、工具调用、审批、上下文水位都实时展示并可回放；

不是简单拼接聊天历史，而是用 session、thread、notes、context 和 compact 做上下文治理；

最后还支持 Skills、Subagents、MCP，把工作流、子 Agent 和外部工具统一接进同一套运行链路。

也就是说，这个项目真正能讲的不是“我接了一个大模型接口”，而是“我实现了一个本地 Agent 运行时”。






