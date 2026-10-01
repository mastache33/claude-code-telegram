"""Typing indicator is sent into the same topic the user is reading."""

import pytest

from src.bot.orchestrator import MessageOrchestrator


class _Chat:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def send_action(self, action: str, message_thread_id: int | None = None) -> None:
        self.calls.append((action, message_thread_id))


@pytest.mark.asyncio
async def test_typing_uses_the_topic() -> None:
    chat = _Chat()
    await MessageOrchestrator._show_typing(chat, 243543)
    assert chat.calls == [("typing", 243543)]


@pytest.mark.asyncio
async def test_typing_without_topic() -> None:
    chat = _Chat()
    await MessageOrchestrator._show_typing(chat, None)
    assert chat.calls == [("typing", None)]
