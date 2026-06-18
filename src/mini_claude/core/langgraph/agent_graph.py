from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from mini_claude.core.langgraph.state import AgentState
from mini_claude.core.langgraph.tools_adapter import build_default_tools

logger = logging.getLogger(__name__)


# ── 系统提示词 ──
def _build_system_prompt(state: AgentState) -> str:
    parts = [state["system_prompt_override"] or (
        "You are MiniClaude, an AI assistant for software engineering. "
        "Work step by step. Use tools to achieve the goal."
    )]
    if state["global_context"]:
        parts.append(f"\n## Global Context\n{state['global_context']}")
    if state["project_context"]:
        parts.append(f"\n## Project Context\n{state['project_context']}")
    if state["session_notes"]:
        parts.append(f"\n## Session Notes\n{state['session_notes']}")
    parts.append(
        f"\n## Current Task\nGoal: {state['goal']}\n"
        f"Step {state['step']}/{state['max_steps']}\n"
        "When complete, explain your result clearly."
    )
    return "\n".join(parts)


# ── agent 节点：LLM 调用 ──
async def _call_model(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    model: BaseChatModel = config["configurable"]["model"]  # type: ignore[index]
    tools: list[StructuredTool] = config["configurable"]["tools"]  # type: ignore[index]

    system = _build_system_prompt(state)
    msgs: list = [SystemMessage(content=system)]
    if state["messages"]:
        msgs.extend(list(state["messages"]))
    if not state["messages"]:
        msgs.append(HumanMessage(content=state["goal"]))

    step = state["step"] + 1
    t0 = time.monotonic()
    bound = model.bind_tools(tools) if tools else model

    try:
        response = await bound.ainvoke(msgs)
    except Exception as exc:
        logger.error("LLM error step=%d: %s", step, exc)
        return {"step": step, "status": "failed", "reason": f"LLM error: {exc}"}

    elapsed = int((time.monotonic() - t0) * 1000)
    usage = getattr(response, "usage_metadata", None) or {}
    input_tk = usage.get("input_tokens", 0)
    output_tk = usage.get("output_tokens", 0)
    ctx_max = getattr(model, "max_tokens", 200000) or 200000
    pct = (input_tk / ctx_max) * 100 if ctx_max else 0.0

    logger.debug("step=%d model=%s in=%d out=%d %dms ctx=%.1f%%",
                 step, getattr(model, "model", "?"), input_tk, output_tk, elapsed, pct)

    return {
        "messages": [response],
        "step": step,
        "context_pct": pct,
        "original_token_count": input_tk,
    }


# ── 权限审查：LangGraph interrupt ──
async def _check_permission(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    msgs = state["messages"]
    if not msgs:
        return {}
    last = msgs[-1]
    if not (hasattr(last, "tool_calls") and last.tool_calls):
        return {}

    for tc in last.tool_calls:
        name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
        args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
        tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")

        if name in ("read_file", "list_dir", "note_save",
                    "task_create", "task_list", "task_get", "task_update"):
            continue

        decision = interrupt({
            "type": "permission_request",
            "tool_name": name,
            "tool_call_id": tc_id,
            "args": args,
            "message": f"Approve '{name}'?",
        })
        logger.info("permission %s → %s", name, decision)

        if decision in ("deny_once", "always_deny"):
            return {
                "messages": [
                    ToolMessage(
                        content=f"Permission denied for '{name}'.",
                        tool_call_id=tc_id,
                    )
                ],
                "permission_response": decision,
            }
    return {}


# ── 上下文压缩 ──
async def _compact_context(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    model: BaseChatModel = config["configurable"]["model"]  # type: ignore[index]
    threshold = state.get("compaction_threshold") or 80.0
    if state["context_pct"] < threshold:
        return {}

    logger.info("compacting ctx=%.1f%% (threshold %.1f%%)", state["context_pct"], threshold)
    msgs = list(state["messages"])
    if not msgs:
        return {}

    prompt = HumanMessage(content=(
        "Summarize the above conversation. Preserve: "
        "1) original goal 2) key decisions 3) files touched "
        "4) errors and resolutions. Under 1000 words."
    ))
    try:
        resp = await model.ainvoke(msgs + [prompt])
        summary = str(resp.content)
        return {
            "messages": [
                SystemMessage(content=f"[Summary]\n{summary}"),
                HumanMessage(content="Understood. Continue."),
            ],
            "summary_token_count": len(summary) // 4,
            "context_pct": 0.0,
        }
    except Exception as exc:
        logger.warning("compact failed: %s", exc)
        return {}


# ── 路由函数 ──
def _route_after_agent(state: AgentState) -> Literal["permission", "compact", "tools", "__end__"]:
    if state["status"] != "running" or state["step"] >= state["max_steps"]:
        return "__end__"
    msgs = state["messages"]
    if not msgs:
        return "__end__"
    last = msgs[-1]
    has_tools = hasattr(last, "tool_calls") and last.tool_calls
    if not has_tools:
        return "__end__"
    for tc in last.tool_calls:
        name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
        if name in ("bash", "write_file"):
            return "permission"
    threshold = state.get("compaction_threshold") or 80.0
    if state["context_pct"] >= threshold:
        return "compact"
    return "tools"


def _route_after_permission(state: AgentState) -> Literal["tools", "compact", "__end__"]:
    if state["permission_response"] in ("deny_once", "always_deny"):
        return "__end__"
    if state["context_pct"] >= (state.get("compaction_threshold") or 80.0):
        return "compact"
    return "tools"


def _route_after_compact(state: AgentState) -> Literal["tools", "__end__"]:
    if state["status"] != "running":
        return "__end__"
    return "tools"


# ── 构建图 ──
def build_agent_graph(model: BaseChatModel, tools: list[StructuredTool]) -> StateGraph:
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", _call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("permission", _check_permission)
    workflow.add_node("compact", _compact_context)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", _route_after_agent, {
        "permission": "permission", "compact": "compact",
        "tools": "tools", "__end__": END,
    })
    workflow.add_conditional_edges("permission", _route_after_permission, {
        "tools": "tools", "compact": "compact", "__end__": END,
    })
    workflow.add_conditional_edges("compact", _route_after_compact, {
        "tools": "tools", "__end__": END,
    })
    workflow.add_edge("tools", "agent")
    return workflow


def compile_agent(
    model: BaseChatModel,
    tools: list[StructuredTool] | None = None,
    tool_whitelist: list[str] | None = None,
    checkpointer: Any = None,
) -> Any:
    t = tools or build_default_tools(tool_whitelist=tool_whitelist)
    return build_agent_graph(model, t).compile(checkpointer=checkpointer)


async def run_agent_stream(
    compiled_graph: Any, state: AgentState,
    model: BaseChatModel, tools: list[StructuredTool],
    config_extra: dict[str, Any] | None = None,
) -> Any:
    cfg: RunnableConfig = {
        "configurable": {"model": model, "tools": tools, **(config_extra or {})}
    }
    async for evt in compiled_graph.astream_events(state, cfg, version="v2"):
        yield evt
