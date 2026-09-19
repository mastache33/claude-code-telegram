"""Tests for fork features: AskUserQuestion answers, /model, pending text input."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

from src.bot.utils import pending_input
from src.claude.run_context import current_model
from src.claude.sdk_integration import _make_can_use_tool_callback

QUESTION_INPUT = {
    "questions": [
        {
            "question": "Which color?",
            "header": "Color",
            "multiSelect": False,
            "options": [{"label": "Red"}, {"label": "Green"}],
        }
    ]
}


async def test_ask_user_question_answers_flow_into_updated_input():
    async def answer(tool_input):
        return {"Which color?": "Green"}

    cb = _make_can_use_tool_callback(
        security_validator=None,
        working_directory=Path("/tmp"),
        approved_directory=Path("/tmp"),
        question_callback=answer,
    )
    result = await cb("AskUserQuestion", QUESTION_INPUT, MagicMock())
    assert isinstance(result, PermissionResultAllow)
    assert result.updated_input["answers"] == {"Which color?": "Green"}
    assert result.updated_input["questions"] == QUESTION_INPUT["questions"]


async def test_unanswered_question_is_denied():
    async def no_answer(tool_input):
        return None

    cb = _make_can_use_tool_callback(
        security_validator=None,
        working_directory=Path("/tmp"),
        approved_directory=Path("/tmp"),
        question_callback=no_answer,
    )
    result = await cb("AskUserQuestion", QUESTION_INPUT, MagicMock())
    assert isinstance(result, PermissionResultDeny)


async def test_other_tools_allowed_without_validator():
    cb = _make_can_use_tool_callback(
        security_validator=None,
        working_directory=Path("/tmp"),
        approved_directory=Path("/tmp"),
        question_callback=AsyncMock(),
    )
    result = await cb("Bash", {"command": "ls"}, MagicMock())
    assert isinstance(result, PermissionResultAllow)


async def test_pending_input_resolves_once():
    future = asyncio.get_running_loop().create_future()
    pending_input.waiting_text[42] = future
    assert pending_input.is_waiting(42)
    assert pending_input.resolve(42, "my answer")
    assert await future == "my answer"
    assert not pending_input.is_waiting(42)
    assert not pending_input.resolve(42, "again")


@pytest.mark.parametrize(
    "arg,expected",
    [("sonnet", "sonnet"), ("OPUS", "opus"), ("claude-sonnet-5", "claude-sonnet-5")],
)
async def test_model_command_sets_user_model(arg, expected):
    from src.bot.orchestrator import MessageOrchestrator

    settings = MagicMock()
    settings.claude_model = None
    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch.settings = settings
    update = MagicMock()
    update.message.text = f"/model {arg}"
    update.message.reply_text = AsyncMock()
    context = MagicMock()
    context.user_data = {}
    await orch.agentic_model(update, context)
    assert context.user_data["model"] == expected


async def test_model_command_rejects_unknown():
    from src.bot.orchestrator import MessageOrchestrator

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch.settings = MagicMock(claude_model=None)
    update = MagicMock()
    update.message.text = "/model gpt-4o"
    update.message.reply_text = AsyncMock()
    context = MagicMock()
    context.user_data = {}
    await orch.agentic_model(update, context)
    assert "model" not in context.user_data


def test_current_model_default_is_none():
    assert current_model.get() is None


@pytest.mark.parametrize(
    "mode,from_voice,should_speak",
    [("auto", True, True), ("auto", False, False), ("on", False, True), ("off", True, False)],
)
async def test_voice_reply_modes(monkeypatch, mode, from_voice, should_speak):
    import sys
    import types

    from src.bot.orchestrator import MessageOrchestrator

    spoken = []

    class FakeSpeech:
        async def create(self, **kwargs):
            spoken.append(kwargs["input"])
            return MagicMock(content=b"ogg")

    fake_openai = types.SimpleNamespace(
        AsyncOpenAI=lambda api_key: types.SimpleNamespace(audio=types.SimpleNamespace(speech=FakeSpeech()))
    )
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch.settings = MagicMock()
    update = MagicMock()
    update.message.reply_voice = AsyncMock()
    context = MagicMock()
    context.user_data = {"voice_reply": mode}
    await orch._maybe_send_voice_reply(update, context, "**Привет**, всё готово", from_voice=from_voice)
    assert bool(spoken) is should_speak
    if should_speak:
        assert spoken[0] == "Привет, всё готово"
        update.message.reply_voice.assert_awaited_once()


def test_environment_profile_does_not_override_explicit_env(monkeypatch):
    from src.config.loader import _apply_environment_overrides
    from src.config.settings import Settings

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:x")
    monkeypatch.setenv("TELEGRAM_BOT_USERNAME", "b")
    monkeypatch.setenv("APPROVED_DIRECTORY", "/tmp")
    monkeypatch.setenv("CLAUDE_MAX_COST_PER_USER", "1000000")
    settings = _apply_environment_overrides(Settings(), "production")
    assert settings.claude_max_cost_per_user == 1000000
    assert settings.rate_limit_requests == 5  # not set explicitly -> profile default applies


@pytest.mark.parametrize("thread_id", [None, 777])
async def test_unmapped_topic_falls_back_to_default_project(tmp_path, thread_id):
    from src.bot.orchestrator import MessageOrchestrator

    general = MagicMock(slug="general", absolute_path=tmp_path)
    general.name = "Общее"
    manager = MagicMock()
    manager.resolve_project = AsyncMock(return_value=None)
    manager.registry.get_by_slug = MagicMock(return_value=general)

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch.settings = MagicMock(project_threads_mode="private", project_threads_default_slug="general")
    orch._extract_message_thread_id = MagicMock(return_value=thread_id)
    update = MagicMock()
    update.effective_chat.id = 1
    update.effective_chat.type = "private"
    context = MagicMock()
    context.bot_data = {"project_threads_manager": manager}
    context.user_data = {}

    assert await orch._apply_thread_routing_context(update, context) is True
    assert context.user_data["current_directory"] == tmp_path
    assert context.user_data["_thread_context"]["state_key"] == f"1:{thread_id or 'main'}"


async def test_unmapped_topic_rejected_without_default():
    from src.bot.orchestrator import MessageOrchestrator

    manager = MagicMock()
    manager.resolve_project = AsyncMock(return_value=None)
    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch.settings = MagicMock(project_threads_mode="private", project_threads_default_slug=None)
    orch._extract_message_thread_id = MagicMock(return_value=5)
    orch._reject_for_thread_mode = AsyncMock()
    update = MagicMock()
    update.effective_chat.type = "private"
    context = MagicMock()
    context.bot_data = {"project_threads_manager": manager}
    context.user_data = {}
    assert await orch._apply_thread_routing_context(update, context) is False
