from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from mini_claude.core.config import MiniConfig
from mini_claude.core.langgraph.runner import LangGraphRunner, RunStreamEvent

logger = logging.getLogger(__name__)


class TuiAgentAdapter:
    """TUI 使用的 Agent 适配器：优先 daemon IPC，否则本地 LangGraph astream_events"""

    def __init__(self, config: MiniConfig) -> None:
        self._config = config
        self._runner = LangGraphRunner(config)

    # 尝试 daemon 连接；如果失败则本地运行
    async def try_connect_daemon(self) -> Any | None:
        """返回 SocketClient 实例，或 None（daemon 不可用）"""
        from mini_claude.core.transport.socket_client import SocketClient
        client = SocketClient(self._config.host, self._config.port)
        try:
            await client.connect()
            return client
        except (ConnectionRefusedError, OSError):
            await client.close()
            return None

    # 本地 LangGraph 流式运行
    def stream_local(
        self, goal: str, **kwargs: Any
    ) -> AsyncIterator[RunStreamEvent]:
        return self._runner.run_stream(goal=goal, **kwargs)

    # 将 daemon IPC 事件字典转为 RunStreamEvent
    @staticmethod
    def ipc_to_stream_event(event: dict[str, Any]) -> RunStreamEvent:
        event_type_map = {
            "run.started": "run_start",
            "run.finished": "run_finished",
            "step.started": "step_start",
            "step.finished": "step_end",
            "llm.token": "llm_token",
            "tool.call_started": "tool_start",
            "tool.call_finished": "tool_end",
            "tool.call_failed": "tool_end",
            "permission.requested": "permission_request",
            "llm.usage": "llm_usage",
            "context.compacted": "compact",
        }
        mapped = event_type_map.get(event.get("type", ""), event.get("type", ""))
        return RunStreamEvent(event_type=mapped, data=event)

    # 本地流式运行并处理中断（权限审批）
    async def run_with_interrupt(
        self,
        goal: str,
        *,
        interrupt_handler: Any,
        **kwargs: Any,
    ) -> AsyncIterator[RunStreamEvent]:
        """流式运行，遇到 interrupt 时调用 interrupt_handler 获取决策"""
        async for evt in self._runner.run_stream(
            goal=goal,
            permission_handler=interrupt_handler,
            **kwargs,
        ):
            yield evt
