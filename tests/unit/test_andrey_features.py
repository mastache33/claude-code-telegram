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


async def test_speak_to_user_tool_call_is_collected_and_spoken():
    from src.bot.orchestrator import MessageOrchestrator
    from src.claude.sdk_integration import StreamUpdate

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch.settings = MagicMock(openai_api_key=None)
    collected: list[str] = []
    cb = orch._make_stream_callback(
        verbose_level=0, progress_msg=MagicMock(), tool_log=[], start_time=0.0, mcp_voice=collected,
    )
    await cb(StreamUpdate(type="tool_calls", tool_calls=[
        {"name": "mcp__telegram__speak_to_user", "input": {"text": "Готово, отчёт собран"}}]))
    assert collected == ["Готово, отчёт собран"]

    spoken = []
    orch._speak = AsyncMock(side_effect=lambda u, t: spoken.append(t))
    context = MagicMock()
    context.user_data = {"voice_reply": "off"}
    await orch._maybe_send_voice_reply(MagicMock(), context, "текст ответа", from_voice=False, requested=collected)
    assert spoken == ["Готово, отчёт собран"]


async def test_location_message_builds_prompt_for_claude():
    from src.bot.orchestrator import MessageOrchestrator

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch._handle_agentic_media_message = AsyncMock()
    update = MagicMock()
    update.message.location.latitude = 57.15222
    update.message.location.longitude = 65.52722
    update.message.location.horizontal_accuracy = 12.0
    update.message.caption = "куда доехать на велике?"
    update.message.reply_text = AsyncMock()
    await orch.agentic_location(update, MagicMock())
    prompt = orch._handle_agentic_media_message.await_args.kwargs["prompt"]
    assert "57.15222" in prompt and "65.52722" in prompt and "велике" in prompt


async def test_checklist_toggle_and_report():
    from src.bot.orchestrator import Checklist, MessageOrchestrator

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch._handle_agentic_media_message = AsyncMock()
    state = Checklist(title="Выкат", items=["тесты", "деплой"], done=[False, False], user_id=7)
    orch._checklists = {55: state}

    update = MagicMock()
    update.effective_user.id = 7
    update.callback_query.data = "chk:55:0"
    update.callback_query.message.message_id = 55
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    await orch._handle_checklist_callback(update, MagicMock())
    assert state.done == [True, False]
    assert "✅" in orch._checklist_text(state) and "1/2" in orch._checklist_text(state)

    update.callback_query.data = "chk:55:report"
    update.callback_query.message.reply_text = AsyncMock()
    await orch._handle_checklist_callback(update, MagicMock())
    prompt = orch._handle_agentic_media_message.await_args.kwargs["prompt"]
    assert "сделано 1 из 2" in prompt and "тесты" in prompt and "деплой" in prompt


async def test_checklist_tool_call_collected():
    from src.bot.orchestrator import MessageOrchestrator
    from src.claude.sdk_integration import StreamUpdate

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    collected: list = []
    cb = orch._make_stream_callback(
        verbose_level=0, progress_msg=MagicMock(), tool_log=[], start_time=0.0, mcp_checklists=collected,
    )
    await cb(StreamUpdate(type="tool_calls", tool_calls=[
        {"name": "mcp__telegram__send_checklist_to_user",
         "input": {"title": "Выкат", "items": ["тесты", " ", "деплой"]}}]))
    assert collected == [("Выкат", ["тесты", "деплой"])]


async def test_pinned_status_creates_then_edits(tmp_path):
    from src.bot.orchestrator import MessageOrchestrator

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch.settings = MagicMock(enable_pinned_status=True, claude_model=None, approved_directory=tmp_path)
    orch._status_messages = {}
    orch._extract_message_thread_id = MagicMock(return_value=None)

    update = MagicMock()
    update.effective_chat.id = 10
    context = MagicMock()
    context.user_data = {"current_directory": tmp_path / "PHT", "model": "sonnet"}
    context.bot.send_message = AsyncMock(return_value=MagicMock(message_id=99))
    context.bot.pin_chat_message = AsyncMock()
    context.bot.edit_message_text = AsyncMock()

    await orch._update_status(update, context, "⏳ работаю", "почини логи")
    assert orch._status_messages == {"10:main": 99}
    context.bot.pin_chat_message.assert_awaited_once()
    sent_text = context.bot.send_message.await_args.kwargs["text"]
    assert "PHT" in sent_text and "sonnet" in sent_text and "почини логи" in sent_text

    await orch._update_status(update, context, "✅ готов")
    context.bot.edit_message_text.assert_awaited_once()
    assert context.bot.send_message.await_count == 1


async def test_pinned_status_off_by_default():
    from src.bot.orchestrator import MessageOrchestrator

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch.settings = MagicMock(enable_pinned_status=False)
    context = MagicMock()
    context.bot.send_message = AsyncMock()
    await orch._update_status(MagicMock(), context, "⏳ работаю")
    context.bot.send_message.assert_not_awaited()


async def test_webapp_data_becomes_claude_prompt():
    from src.bot.orchestrator import MessageOrchestrator

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch._handle_agentic_media_message = AsyncMock()
    update = MagicMock()
    update.message.web_app_data.data = '{"action":"logs","label":"Логи PHT","prompt":"покажи логи pht"}'
    update.message.reply_text = AsyncMock()
    await orch.agentic_webapp_data(update, MagicMock())
    assert orch._handle_agentic_media_message.await_args.kwargs["prompt"] == "покажи логи pht"
    assert "Логи PHT" in update.message.reply_text.await_args.args[0]


async def test_panel_without_url_explains():
    from src.bot.orchestrator import MessageOrchestrator

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch.settings = MagicMock(webapp_url=None)
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    await orch.agentic_panel(update, MagicMock())
    assert "не настроена" in update.message.reply_text.await_args.args[0]


async def test_configure_bot_applies_settings(tmp_path):
    from src.bot.orchestrator import MessageOrchestrator

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch.settings = MagicMock(approved_directory=tmp_path)
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    context = MagicMock()
    context.bot_data = {}
    context.user_data = {"model": "opus", "claude_session_id": "abc"}

    await orch._apply_bot_config(update, context, [
        {"model": "sonnet", "voice": "on", "verbosity": 0, "new_session": True},
    ])
    assert context.user_data["model"] == "sonnet"
    assert context.user_data["voice_reply"] == "on"
    assert context.user_data["verbose_level"] == 0
    assert context.user_data["force_new_session"] is True
    assert context.user_data["claude_session_id"] is None
    text = update.message.reply_text.await_args.args[0]
    assert "sonnet" in text and "новая сессия" in text


async def test_configure_bot_switches_project(tmp_path):
    from src.bot.orchestrator import MessageOrchestrator

    (tmp_path / "Developer" / "tea-app").mkdir(parents=True)
    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch.settings = MagicMock(approved_directory=tmp_path)
    update = MagicMock()
    update.message.reply_text = AsyncMock()
    context = MagicMock()
    context.bot_data = {}
    context.user_data = {}

    await orch._apply_bot_config(update, context, [{"project": "tea"}])
    assert context.user_data["current_directory"] == tmp_path / "Developer" / "tea-app"


async def test_configure_bot_tool_call_collected():
    from src.bot.orchestrator import MessageOrchestrator
    from src.claude.sdk_integration import StreamUpdate

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    collected: list = []
    cb = orch._make_stream_callback(
        verbose_level=0, progress_msg=MagicMock(), tool_log=[], start_time=0.0, mcp_config=collected,
    )
    await cb(StreamUpdate(type="tool_calls", tool_calls=[
        {"name": "mcp__telegram__configure_bot", "input": {"model": "sonnet"}}]))
    assert collected == [{"model": "sonnet"}]


async def test_create_topic_tool_creates_and_maps():
    from src.bot.orchestrator import MessageOrchestrator

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    manager = MagicMock()
    manager.repository.upsert_mapping = AsyncMock()
    update = MagicMock()
    update.effective_chat.id = 5
    update.message.reply_text = AsyncMock()
    context = MagicMock()
    context.bot_data = {"project_threads_manager": manager}
    context.bot.create_forum_topic = AsyncMock(return_value=MagicMock(message_thread_id=321))

    await orch._create_topics(update, context, [{"name": "Ремонт кухни"}, {"name": "Gavan", "project": "gavan"}])
    assert context.bot.create_forum_topic.await_count == 2
    manager.repository.upsert_mapping.assert_awaited_once()
    assert manager.repository.upsert_mapping.await_args.kwargs["project_slug"] == "gavan"
    assert "Ремонт кухни" in update.message.reply_text.await_args_list[0].args[0]


async def test_elevenlabs_provider_used_when_selected(monkeypatch):
    import sys
    import types

    from src.bot.orchestrator import MessageOrchestrator

    calls = {}

    class FakeResponse:
        content = b"ogg"

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, params=None, json=None):
            calls.update(url=url, headers=headers, json=json)
            return FakeResponse()

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(AsyncClient=FakeClient))

    orch = MessageOrchestrator.__new__(MessageOrchestrator)
    orch.settings = MagicMock(
        tts_provider="elevenlabs",
        elevenlabs_api_key=MagicMock(get_secret_value=lambda: "k"),
        elevenlabs_voice_id="lena123",
        elevenlabs_model="eleven_v3",
    )
    update = MagicMock()
    update.message.reply_voice = AsyncMock()
    await orch._speak_elevenlabs(update, "Привет, это Лена")
    assert "lena123" in calls["url"] and calls["json"]["text"] == "Привет, это Лена"
    update.message.reply_voice.assert_awaited_once()
