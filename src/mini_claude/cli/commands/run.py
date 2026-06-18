from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

from mini_claude.core.config import MiniConfig
from mini_claude.core.langgraph.runner import LangGraphRunner


class StdoutPrinter:
    """astream_events 输出格式化器，直接消费 RunStreamEvent"""

    def __init__(self) -> None:
        self._inline = False
        self._run_start: float = 0.0
        self._step = 0

    def _ensure_newline(self) -> None:
        if self._inline:
            print()
            self._inline = False

    async def handle(self, event_type: str | dict[str, Any], data: dict[str, Any] | None = None) -> int | None:
        """处理单个事件。兼容两种调用方式：
         - handle(dict) — daemon IPC 路径和测试
         - handle(event_type, data) — LangGraph astream_events 路径
        返回 exit_code 或 None"""
        if isinstance(event_type, dict):
            event = event_type
            event_type = event.get("type", "")
            data = event

        if data is None:
            data = {}

        type_aliases: dict[str, str] = {
            "run.started": "run_start",
            "run.finished": "run_finished",
            "step.started": "step_start",
            "step.finished": "step_end",
            "llm.token": "llm_token",
            "tool.call_started": "tool_start",
            "tool.call_finished": "tool_end",
            "tool.call_failed": "tool_end",
            "llm.usage": "llm_usage",
            "permission.requested": "permission_request",
        }
        event_type = type_aliases.get(event_type, event_type)

        if event_type == "run_start":
            self._run_start = time.monotonic()
            print(f"[run] {data.get('run_id', '')} goal={data.get('goal', '')}")

        elif event_type == "step_start":
            self._ensure_newline()
            self._step = data.get("step", self._step)
            print(f"[step {self._step}] plan...")

        elif event_type == "step_end":
            print(f"[step {self._step}] done")

        elif event_type == "llm_token":
            token = data.get("token", "")
            if token:
                print(token, end="", flush=True)
                self._inline = True

        elif event_type == "tool_start":
            self._ensure_newline()
            params = data.get("tool_input") or data.get("params", {})
            params_str = json.dumps(params, ensure_ascii=False)
            print(f"[tool] {data.get('tool_name', '')} {params_str}")

        elif event_type == "tool_end":
            print(f"[tool] {data.get('tool_name', '')} OK")

        elif event_type == "permission_request":
            self._ensure_newline()
            tool_name = data.get("tool_name", "unknown")
            args = data.get("args", {})
            print(f"[perm] Tool '{tool_name}' needs approval: {json.dumps(args, ensure_ascii=False)}")
            print("[perm] [y]Allow [a]Always allow [n]Deny [d]Deny always")

        elif event_type == "run_finished":
            self._ensure_newline()
            elapsed = time.monotonic() - self._run_start
            status = data.get("status", "?")
            steps = data.get("steps", self._step)
            reason = data.get("reason", "")
            print(f"[run] {status}  {steps} steps  {elapsed:.1f}s")
            if reason:
                print(f"[run] reason: {reason}")
            return 0 if status == "success" else 1

        return None


# 异步核心：优先 daemon，否则本地 LangGraph 运行
async def _run_async(goal: str, config: MiniConfig) -> int:
    from mini_claude.core.transport.socket_client import IpcError, SocketClient

    client = SocketClient(config.host, config.port)
    try:
        await client.connect()
    except (ConnectionRefusedError, OSError):
        await client.close()
        return await _run_langgraph_stream(goal, config)

    printer = StdoutPrinter()
    finished = asyncio.Event()
    exit_code = 0

    async def on_event(event: dict[str, Any]) -> None:
        nonlocal exit_code
        rc = await printer.handle(event)
        if event.get("type") == "run.finished":
            if event.get("status") != "success":
                exit_code = 1
            finished.set()
        if rc is not None:
            exit_code = rc

    client.on_event(on_event)
    loop_task = asyncio.create_task(client.run_event_loop())

    try:
        await client.send_command("event.subscribe", {
            "topics": ["run.*", "step.*", "tool.*", "llm.token", "llm.usage"],
            "scope": "global",
        })
        await client.send_command("agent.run", {"goal": goal})
    except IpcError as e:
        print(f"error: {e}", file=sys.stderr)
        loop_task.cancel()
        await client.close()
        return 1

    await finished.wait()
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass

    await client.close()
    return exit_code


# 本地 LangGraph 流式运行（使用 astream_events）
async def _run_langgraph_stream(goal: str, config: MiniConfig) -> int:
    print(f"[langgraph] running locally model={config.llm.default_model} goal={goal[:80]}...")

    printer = StdoutPrinter()
    printer._run_start = time.monotonic()
    runner = LangGraphRunner(config)
    exit_code = 0

    try:
        async for evt in runner.run_stream(goal=goal):
            rc = printer.handle(evt.event_type, evt.data)
            if rc is not None:
                exit_code = rc
    except KeyboardInterrupt:
        print("\n[langgraph] cancelled")
        return 130

    return exit_code


# 执行 mini run --goal "..." 命令
def cmd_run(goal: str, config: MiniConfig) -> None:
    try:
        exit_code = asyncio.run(_run_async(goal, config))
    except KeyboardInterrupt:
        sys.exit(130)
    sys.exit(exit_code)
