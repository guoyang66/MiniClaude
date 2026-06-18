from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.tools import StructuredTool

from mini_claude.core.tools.base import BaseTool as KamaBaseTool
from mini_claude.core.tools.builtin.bash import BashTool
from mini_claude.core.tools.builtin.list_dir import ListDirTool
from mini_claude.core.tools.builtin.read_file import ReadFileTool
from mini_claude.core.tools.builtin.write_file import WriteFileTool

logger = logging.getLogger(__name__)


# 将 MiniClaude BaseTool 转换为 LangChain StructuredTool
def kama_tool_to_langchain(kama_tool: KamaBaseTool) -> StructuredTool:
    params_model = kama_tool.params_model

    # 为每个工具创建同步包装函数
    async def _invoke(**kwargs: Any) -> str:
        params = {k: v for k, v in kwargs.items()}
        result = await kama_tool.invoke(params)
        if result.is_error:
            return f"[{result.error_type or 'error'}] {result.content}"
        return result.content

    # 同步包装（LangChain 工具调用使用 run_in_executor）
    def _sync_invoke(**kwargs: Any) -> str:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_invoke(**kwargs))
        return loop.run_until_complete(_invoke(**kwargs))

    return StructuredTool.from_function(
        func=_sync_invoke,
        coroutine=_invoke,
        name=kama_tool.name,
        description=kama_tool.description,
        args_schema=params_model,
    )


# 创建默认的内置工具集（转为 LangChain 兼容格式）
def build_default_tools(
    tool_whitelist: list[str] | None = None,
    extra_tools: list[StructuredTool] | None = None,
) -> list[StructuredTool]:
    all_tools: list[KamaBaseTool] = [
        BashTool(),
        ReadFileTool(),
        WriteFileTool(),
        ListDirTool(),
    ]

    result: list[StructuredTool] = []
    for tool in all_tools:
        if tool_whitelist is None or tool.name in tool_whitelist:
            result.append(kama_tool_to_langchain(tool))

    if extra_tools:
        result.extend(extra_tools)

    return result
