from __future__ import annotations

import pytest

import mini_claude.core.tools.invocation as inv_mod
from mini_claude.core.llm.types import ToolCallBlock
from mini_claude.core.tools.base import BaseTool, ToolResult
from mini_claude.core.tools.errors import RateLimitedError
from mini_claude.core.tools.invocation import invoke_tool
from mini_claude.core.tools.registry import ToolRegistry


class _FailNTimes(BaseTool):
    name = "fail_n"
    description = "Fails n times then succeeds"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    def __init__(self, n: int, *, error_type: str = "runtime_error") -> None:
        self._remaining = n
        self._error_type = error_type

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        if self._remaining > 0:
            self._remaining -= 1
            return ToolResult(content="transient error", is_error=True, error_type=self._error_type)
        return ToolResult(content="ok")


class _RateLimitedNTimes(BaseTool):
    name = "rate_n"
    description = "Rate-limits n times then succeeds"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    def __init__(self, n: int) -> None:
        self._remaining = n

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        if self._remaining > 0:
            self._remaining -= 1
            raise RateLimitedError("429 Too Many Requests")
        return ToolResult(content="ok")


class _AlwaysFails(BaseTool):
    name = "always_fail"
    description = "Always fails"
    input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

    def __init__(self, error_type: str = "runtime_error") -> None:
        self._error_type = error_type

    async def invoke(self, params: dict[str, object]) -> ToolResult:
        return ToolResult(content="permanent error", is_error=True, error_type=self._error_type)


def _call(name: str) -> ToolCallBlock:
    return ToolCallBlock(id="t1", name=name, input={})


async def _run(tool: BaseTool, *, monkeypatch: pytest.MonkeyPatch) -> ToolResult:
    monkeypatch.setattr(inv_mod, "_RETRY_BASE_S", 0.0)
    registry = ToolRegistry()
    registry.register(tool)
    return await invoke_tool(registry, _call(tool.name))


# ── 重试逻辑测试（invoke_tool 精简版，无 EventBus 依赖）──

# 功能：runtime_error 重试后最终成功
async def test_runtime_error_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    result = await _run(_FailNTimes(1), monkeypatch=monkeypatch)
    assert not result.is_error
    assert result.content == "ok"


# 功能：rate_limited 异常触发重试，成功后返回正常
async def test_rate_limited_retries_and_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    result = await _run(_RateLimitedNTimes(1), monkeypatch=monkeypatch)
    assert not result.is_error
    assert result.content == "ok"


# 功能：runtime_error 超过 2 次重试后最终失败
async def test_runtime_error_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    result = await _run(_AlwaysFails("runtime_error"), monkeypatch=monkeypatch)
    assert result.is_error
    assert result.error_type == "runtime_error"


# 功能：rate_limited 耗尽重试后最终失败
async def test_rate_limited_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    result = await _run(_RateLimitedNTimes(10), monkeypatch=monkeypatch)
    assert result.is_error
    assert result.error_type == "rate_limited"


# 功能：schema_error 不触发重试，直接失败
async def test_schema_error_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    result = await _run(_AlwaysFails("schema_error"), monkeypatch=monkeypatch)
    assert result.is_error
    assert result.error_type == "schema_error"


# 功能：timeout 不触发重试，直接失败
async def test_timeout_no_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    class _SlowTool(BaseTool):
        name = "slow"
        description = "sleeps"
        input_schema: dict[str, object] = {"type": "object", "properties": {}, "required": []}

        async def invoke(self, params: dict[str, object]) -> ToolResult:
            await asyncio.sleep(60)
            return ToolResult(content="done")

    monkeypatch.setattr(inv_mod, "_RETRY_BASE_S", 0.0)
    registry = ToolRegistry()
    registry.register(_SlowTool())
    result = await invoke_tool(registry, _call("slow"), timeout=0.05)
    assert result.is_error
    assert result.error_type == "timeout"
