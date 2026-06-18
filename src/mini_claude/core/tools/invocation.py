from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import ValidationError

from mini_claude.core.llm.types import ToolCallBlock
from mini_claude.core.tools.base import ToolResult
from mini_claude.core.tools.errors import RateLimitedError
from mini_claude.core.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT: float = 120.0
_MAX_RETRIES: int = 2
_RETRY_BASE_S: float = 2.0
_RETRYABLE: frozenset[str] = frozenset({"runtime_error", "rate_limited"})


# 简化版工具调用：校验参数 → 执行（含超时+重试），不涉及权限和事件发布
# 权限审查和事件推送由 LangGraph agent_graph 的 permission 节点和 astream_events 负责
async def invoke_tool(
    registry: ToolRegistry,
    tool_call: ToolCallBlock,
    *,
    timeout: float = _DEFAULT_TIMEOUT,
) -> ToolResult:
    tool = registry.get(tool_call.name)
    if tool is None:
        return ToolResult(
            content=f"unknown tool: {tool_call.name}",
            is_error=True, error_type="runtime_error",
        )

    if tool.params_model is not None:
        try:
            tool.params_model.model_validate(dict(tool_call.input))
        except ValidationError as exc:
            return ToolResult(
                content=str(exc), is_error=True, error_type="schema_error",
            )

    for attempt in range(1, _MAX_RETRIES + 2):
        error_class: str | None = None
        error_message: str | None = None

        try:
            result = await asyncio.wait_for(
                tool.invoke(dict(tool_call.input)), timeout=timeout,
            )
            if result.is_error:
                error_class = result.error_type or "runtime_error"
                error_message = result.content
            else:
                return result
        except RateLimitedError as exc:
            error_class = "rate_limited"
            error_message = str(exc)
        except TimeoutError:
            return ToolResult(
                content=f"tool timed out after {timeout}s",
                is_error=True, error_type="timeout",
            )
        except Exception as exc:
            error_class = "runtime_error"
            error_message = str(exc)

        if error_class in _RETRYABLE and attempt <= _MAX_RETRIES:
            delay = _RETRY_BASE_S * (2 ** (attempt - 1))
            logger.debug("tool %s retry %d/%d after %.1fs: %s",
                         tool_call.name, attempt, _MAX_RETRIES, delay, error_message)
            await asyncio.sleep(delay)
            continue

        return ToolResult(
            content=error_message or "unknown error",
            is_error=True, error_type=error_class or "runtime_error",
        )

    return ToolResult(content="internal error", is_error=True, error_type="runtime_error")
