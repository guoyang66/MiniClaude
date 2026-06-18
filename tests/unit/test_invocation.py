from __future__ import annotations

import asyncio

from pydantic import BaseModel

from mini_claude.core.llm.types import ToolCallBlock
from mini_claude.core.tools.base import BaseTool, ToolResult
from mini_claude.core.tools.invocation import invoke_tool
from mini_claude.core.tools.registry import ToolRegistry


class _EchoParams(BaseModel):
    msg: str


class _EchoTool(BaseTool):
    name = "echo"
    description = "Echoes the msg param"
    input_schema: dict[str, object] = {
        "type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"],
    }
    params_model = _EchoParams

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(content=str(params["msg"]))


class _SlowTool(BaseTool):
    name = "slow"
    description = "Sleeps forever"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        await asyncio.sleep(60)
        return ToolResult(content="done")


class _BrokenTool(BaseTool):
    name = "broken"
    description = "Always raises"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        raise RuntimeError("boom")


def _call(name: str, inp: dict[str, object] | None = None, uid: str = "t1") -> ToolCallBlock:
    return ToolCallBlock(id=uid, name=name, input=inp or {})


# 功能：验证正常调用时返回工具内容
async def test_success_returns_content() -> None:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    result = await invoke_tool(registry, _call("echo", {"msg": "hi"}))
    assert not result.is_error
    assert result.content == "hi"


# 功能：验证未知工具返回运行时错误
async def test_unknown_tool_returns_runtime_error() -> None:
    result = await invoke_tool(ToolRegistry(), _call("nonexistent"))
    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "unknown tool" in result.content


# 功能：验证参数缺失返回 schema 错误
async def test_missing_required_param_gives_schema_error() -> None:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    result = await invoke_tool(registry, _call("echo", {}))
    assert result.is_error
    assert result.error_type == "schema_error"


# 功能：验证超时返回 timeout 错误
async def test_timeout_gives_timeout_error() -> None:
    registry = ToolRegistry()
    registry.register(_SlowTool())
    result = await invoke_tool(registry, _call("slow"), timeout=0.05)
    assert result.is_error
    assert result.error_type == "timeout"


# 功能：验证运行时异常返回 runtime_error
async def test_runtime_exception_gives_runtime_error() -> None:
    registry = ToolRegistry()
    registry.register(_BrokenTool())
    result = await invoke_tool(registry, _call("broken"))
    assert result.is_error
    assert result.error_type == "runtime_error"
    assert "boom" in result.content
