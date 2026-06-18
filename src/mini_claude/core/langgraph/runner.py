from __future__ import annotations

import asyncio
import datetime
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from mini_claude.core.config import MiniConfig
from mini_claude.core.events.writer import EventWriter
from mini_claude.core.langgraph.agent_graph import compile_agent, run_agent_stream
from mini_claude.core.langgraph.llm_adapter import create_chat_model
from mini_claude.core.langgraph.state import build_initial_state
from mini_claude.core.langgraph.tools_adapter import build_default_tools
from mini_claude.core.memory.loader import load_context_file
from mini_claude.core.runs import ensure_run_dir, new_run_id
from mini_claude.core.tools.builtin.note_save import NoteSaveTool
from mini_claude.core.langgraph.tools_adapter import kama_tool_to_langchain

logger = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    status: str
    result: str
    reason: str | None


@dataclass
class RunStreamEvent:
    """astream_events 的统一事件包装，供 TUI/CLI 消费"""
    # event_type: llm_token | tool_start | tool_end | step_start |
    #             step_end | run_finished | permission_request
    event_type: str
    data: dict[str, Any]


def _now() -> str:
    return datetime.datetime.now(UTC).isoformat()


class LangGraphRunner:
    # 基于 LangGraph 的 Agent 运行器，使用 astream_events 流式输出

    def __init__(
        self,
        config: MiniConfig,
        *,
        runs_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._runs_dir = runs_dir or Path("runs")

    # 运行 agent 并流式产出事件（astream_events）
    async def run_stream(
        self,
        goal: str,
        *,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
        session_id: str | None = None,
        session_notes: str = "",
        extra_messages: list | None = None,
        permission_handler: Any = None,  # 权限响应回调: async func(event) -> str
    ) -> AsyncIterator[RunStreamEvent]:
        run_id = new_run_id()
        ensure_run_dir(run_id)

        # 加载上下文
        global_ctx = load_context_file(Path("~/.mini/context.md").expanduser())
        project_ctx = load_context_file(Path(".mini/context.md"))

        # 创建事件持久化
        events_path = Path(self._runs_dir) / run_id / "events.jsonl"
        event_writer = EventWriter(events_path)
        await event_writer.__aenter__()

        try:
            # 创建 LLM 和工具
            model = create_chat_model(self._config.llm)
            tools = build_default_tools(tool_whitelist=tool_whitelist)

            if session_id:
                tools.append(kama_tool_to_langchain(NoteSaveTool()))

            # 初始状态
            state = build_initial_state(
                goal=goal,
                run_id=run_id,
                max_steps=self._config.agent.max_steps,
                session_id=session_id,
                system_prompt_override=system_prompt_override,
                global_context=global_ctx,
                project_context=project_ctx,
                session_notes=session_notes,
                tool_whitelist=tool_whitelist,
                compaction_threshold=(
                    self._config.compaction.auto_threshold if self._config.compaction.auto_threshold > 0 else None
                ),
            )

            if extra_messages:
                state["messages"] = list(extra_messages)

            # 编译图
            compiled = compile_agent(model, tools)

            # 流式运行并产出事件
            async for langgraph_event in run_agent_stream(compiled, state, model, tools):
                event = self._convert_event(langgraph_event)

                # 处理 interrupt（权限请求）
                if self._is_permission_interrupt(langgraph_event):
                    if permission_handler:
                        yield event
                        try:
                            decision = await permission_handler(event)
                            compiled.update_state(
                                {"configurable": {"model": model, "tools": tools}},
                                {"permission_response": decision},
                            )
                        except Exception:
                            compiled.update_state(
                                {"configurable": {"model": model, "tools": tools}},
                                {"permission_response": "deny_once"},
                            )
                    else:
                        yield event
                        yield RunStreamEvent(
                            event_type="run_finished",
                            data={"run_id": run_id, "status": "failed", "reason": "permission_timeout"},
                        )
                        return

                yield event

            # 判定最终结果
            last_event = event if 'event' in dir() else None
            if last_event and last_event.event_type == "llm_token":
                pass  # already streamed

        except asyncio.CancelledError:
            yield RunStreamEvent(
                event_type="run_finished",
                data={"run_id": run_id, "status": "failed", "reason": "cancelled"},
            )
        except Exception as exc:
            logger.exception("agent run failed")
            yield RunStreamEvent(
                event_type="run_finished",
                data={"run_id": run_id, "status": "failed", "reason": str(exc)},
            )
        finally:
            await event_writer.__aexit__(None, None, None)

    # 运行 agent（非流式），返回 RunOutcome
    async def run(self, **kwargs: Any) -> RunOutcome:
        final_text = ""
        final_status = "success"
        final_reason = None

        async for evt in self.run_stream(**kwargs):
            if evt.event_type == "llm_token":
                final_text += evt.data.get("token", "")
            elif evt.event_type == "run_finished":
                final_status = evt.data.get("status", "success")
                final_reason = evt.data.get("reason")

        return RunOutcome(status=final_status, result=final_text, reason=final_reason)

    # 将 LangGraph 事件转换为统一的 RunStreamEvent
    def _convert_event(self, raw: dict[str, Any]) -> RunStreamEvent:
        event_name = raw.get("event", "")
        data = raw.get("data", {})

        # on_chat_model_stream → llm_token
        if event_name == "on_chat_model_stream":
            chunk = data.get("chunk", {})
            if hasattr(chunk, "content") and chunk.content and not hasattr(chunk, "tool_calls"):
                return RunStreamEvent(
                    event_type="llm_token",
                    data={"token": str(chunk.content) if isinstance(chunk.content, str) else str(list(chunk.content))},
                )
            return RunStreamEvent(event_type="unknown", data={})

        # on_tool_start
        if event_name == "on_tool_start":
            return RunStreamEvent(
                event_type="tool_start",
                data={
                    "tool_name": data.get("name", ""),
                    "tool_input": data.get("input", {}),
                },
            )

        # on_tool_end
        if event_name == "on_tool_end":
            output = data.get("output", "")
            return RunStreamEvent(
                event_type="tool_end",
                data={
                    "tool_name": data.get("name", ""),
                    "output": str(output) if output else "",
                },
            )

        # on_chain_start (agent node)
        if event_name == "on_chain_start" and data.get("name") == "agent":
            step_input = raw.get("input", {})
            return RunStreamEvent(
                event_type="step_start",
                data={"step": step_input.get("step", 0) + 1},
            )

        # on_chain_end (agent node)
        if event_name == "on_chain_end" and data.get("name") == "agent":
            return RunStreamEvent(event_type="step_end", data={})

        # __interrupt__ → permission
        if self._is_permission_interrupt(raw):
            interrupt_data = self._extract_interrupt(raw)
            return RunStreamEvent(
                event_type="permission_request",
                data=interrupt_data,
            )

        return RunStreamEvent(event_type="unknown", data={})

    def _is_permission_interrupt(self, raw: dict[str, Any]) -> bool:
        event_name = raw.get("event", "")
        if event_name == "on_chain_interrupt":
            return True
        # alternative check
        if event_name == "on_chain_end":
            interrupt_data = raw.get("data", {})
            if isinstance(interrupt_data, dict):
                output = interrupt_data.get("output", {})
                if isinstance(output, dict) and output.get("__interrupt__"):
                    return True
        return False

    def _extract_interrupt(self, raw: dict[str, Any]) -> dict[str, Any]:
        event_name = raw.get("event", "")
        if event_name == "on_chain_interrupt":
            val = raw.get("data", {})
            if isinstance(val, dict):
                return val
            return {"value": val}
        if event_name == "on_chain_end":
            output = raw.get("data", {}).get("output", {})
            if isinstance(output, dict):
                interrupts = output.get("__interrupt__", [])
                if interrupts:
                    return interrupts[0].get("value", {})
        return {}
