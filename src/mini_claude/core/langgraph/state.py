from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# LangGraph Agent 状态 — 包含 interrupt、压缩、子Agent 全部状态
class AgentState(TypedDict):
    # ── 消息与对话
    messages: Annotated[list[BaseMessage], add_messages]

    # ── 任务信息
    goal: str
    step: int
    max_steps: int
    status: str  # running | waiting_permission | success | failed
    reason: str | None
    result: str | None
    run_id: str
    session_id: str | None

    # ── 系统提示
    system_prompt_override: str | None
    global_context: str
    project_context: str
    session_notes: str
    tool_whitelist: list[str] | None

    # ── interrupt 权限（LangGraph 原生）
    pending_permission: dict[str, Any] | None
    permission_response: str | None  # allow_once | always_allow | deny_once | always_deny

    # ── 上下文压缩
    context_pct: float
    compaction_threshold: float | None
    original_token_count: int
    summary_token_count: int

    # ── 子 Agent (Send 并行)
    subagent_tasks: list[dict[str, Any]]
    subagent_results: dict[str, str]


# 从参数构建初始状态
def build_initial_state(
    goal: str,
    run_id: str,
    max_steps: int = 20,
    session_id: str | None = None,
    system_prompt_override: str | None = None,
    global_context: str = "",
    project_context: str = "",
    session_notes: str = "",
    tool_whitelist: list[str] | None = None,
    compaction_threshold: float | None = None,
) -> AgentState:
    return AgentState(
        messages=[],
        goal=goal,
        step=0,
        max_steps=max_steps,
        status="running",
        reason=None,
        result=None,
        run_id=run_id,
        session_id=session_id,
        system_prompt_override=system_prompt_override,
        global_context=global_context,
        project_context=project_context,
        session_notes=session_notes,
        tool_whitelist=tool_whitelist,
        pending_permission=None,
        permission_response=None,
        context_pct=0.0,
        compaction_threshold=compaction_threshold,
        original_token_count=0,
        summary_token_count=0,
        subagent_tasks=[],
        subagent_results={},
    )
