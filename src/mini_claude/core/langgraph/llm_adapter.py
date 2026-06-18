from __future__ import annotations

import logging
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from mini_claude.core.config import LlmConfig

logger = logging.getLogger(__name__)


# 创建 LangChain 兼容的 LLM 实例，支持 DeepSeek 等兼容 API
def create_chat_model(config: LlmConfig) -> BaseChatModel:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")

    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set, LLM calls will fail")

    kwargs: dict[str, object] = {
        "model": config.default_model,
        "api_key": api_key or "",
        "temperature": 0.0,
        "max_tokens": 4096,
    }

    if base_url:
        kwargs["base_url"] = base_url

    return ChatAnthropic(**kwargs)  # type: ignore[arg-type]


# 创建流式调用 LLM，返回消息迭代器
async def stream_chat(
    model: BaseChatModel,
    messages: list,
    tools: list | None = None,
    system: str | None = None,
) -> list:
    kwargs: dict[str, object] = {
        "input": messages,
        "streaming": True,
    }
    if tools:
        kwargs["tools"] = tools
    if system:
        kwargs["system"] = system

    # 使用 ainvoke 获取完整响应（兼容模式）
    result = await model.ainvoke(**kwargs)  # type: ignore[arg-type]
    return [result]
