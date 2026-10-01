"""Claude usage-limit switches the same chat to Cursor."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from src.claude.cursor_backend import bare_tool_name, claude_mcp_to_cursor
from src.claude.exceptions import ClaudeProcessError
from src.claude.facade import ClaudeIntegration
from src.claude.limits import is_claude_limit
from src.claude.sdk_integration import ClaudeResponse
from src.claude.session import SessionManager
from src.config.settings import Settings

from .conftest import InMemorySessionStorage


def _settings(tmp_path: Path, **overrides) -> Settings:
    data = dict(
        telegram_bot_token="test:token",
        telegram_bot_username="testbot",
        approved_directory=tmp_path,
        cursor_api_key=SecretStr("cursor-test"),
        database_url=f"sqlite:///{tmp_path}/bot.db",
    )
    data.update(overrides)
    return Settings(**data)


def _response(
    content: str = "ok", session_id: str = "claude-session"
) -> ClaudeResponse:
    return ClaudeResponse(
        content=content,
        session_id=session_id,
        cost=0.0,
        duration_ms=10,
        num_turns=1,
    )


def _cursor_response(content: str = "ответ курсора") -> ClaudeResponse:
    return ClaudeResponse(
        content=content,
        session_id="agent-cursor",
        cost=0.1,
        duration_ms=20,
        num_turns=1,
    )


@pytest.fixture
def stack(tmp_path):
    config = _settings(tmp_path)
    storage = InMemorySessionStorage()
    sessions = SessionManager(config, storage)
    cursor = MagicMock()
    cursor.execute_command = AsyncMock(return_value=_cursor_response())
    cursor.forget = MagicMock()
    facade = ClaudeIntegration(
        config=config,
        sdk_manager=MagicMock(),
        session_manager=sessions,
        cursor_manager=cursor,
    )
    facade.sdk_manager.execute_command = AsyncMock(return_value=_response())
    return facade, cursor


def test_limit_phrases():
    assert is_claude_limit("You've hit your weekly limit · resets Oct 8")
    assert is_claude_limit("usage limit reached")
    assert not is_claude_limit("починил лимит в коде")
    assert not is_claude_limit("rate limit of the bot is 10")


def test_mcp_and_tool_names():
    converted = claude_mcp_to_cursor(
        {
            "telegram": {
                "command": "python",
                "args": ["-m", "src.mcp.telegram_server"],
                "env": {"A": 1},
            },
            "web": {"url": "https://example.test/mcp", "headers": {"X": "y"}},
        }
    )
    assert converted["telegram"]["command"] == "python"
    assert converted["telegram"]["env"] == {"A": "1"}
    assert converted["web"]["url"] == "https://example.test/mcp"
    assert bare_tool_name("mcp__telegram__send_file_to_user") == "send_file_to_user"


async def test_limit_text_switches_to_cursor(stack):
    facade, cursor = stack
    project = Path("/work/pht")
    facade.sdk_manager.execute_command.return_value = _response(
        "You've hit your weekly limit"
    )

    response = await facade.run_command(
        prompt="поправь бота",
        working_directory=project,
        user_id=7,
    )

    cursor.execute_command.assert_awaited_once()
    assert response.content.startswith("Claude упёрся в лимит")
    assert "ответ курсора" in response.content
    assert response.session_id != "agent-cursor"


async def test_limit_error_switches_to_cursor(stack):
    facade, cursor = stack
    facade.sdk_manager.execute_command.side_effect = ClaudeProcessError(
        "Claude process error: usage limit reached"
    )

    response = await facade.run_command(
        prompt="продолжай",
        working_directory=Path("/work/pht"),
        user_id=7,
    )

    assert "ответ курсора" in response.content
    cursor.execute_command.assert_awaited()


async def test_other_errors_stay_on_claude(stack):
    facade, cursor = stack
    facade.sdk_manager.execute_command.side_effect = ClaudeProcessError("disk full")

    with pytest.raises(ClaudeProcessError, match="disk full"):
        await facade.run_command(
            prompt="продолжай",
            working_directory=Path("/work/pht"),
            user_id=7,
        )

    cursor.execute_command.assert_not_awaited()


async def test_followup_skips_claude_while_limit_holds(stack):
    facade, cursor = stack
    project = Path("/work/pht")
    facade.sdk_manager.execute_command.return_value = _response(
        "You've hit your weekly limit"
    )
    await facade.run_command(prompt="раз", working_directory=project, user_id=7)

    facade.sdk_manager.execute_command.reset_mock()
    cursor.execute_command.reset_mock()
    cursor.execute_command.return_value = _cursor_response("второй")

    response = await facade.run_command(
        prompt="два", working_directory=project, user_id=7
    )

    facade.sdk_manager.execute_command.assert_not_awaited()
    cursor.execute_command.assert_awaited_once()
    assert response.content == "второй"


async def test_without_key_the_limit_message_is_kept(tmp_path):
    config = _settings(tmp_path, cursor_api_key=None)
    sessions = SessionManager(config, InMemorySessionStorage())
    facade = ClaudeIntegration(
        config=config,
        sdk_manager=MagicMock(),
        session_manager=sessions,
    )
    facade.sdk_manager.execute_command = AsyncMock(
        return_value=_response("You've hit your weekly limit")
    )

    response = await facade.run_command(
        prompt="раз",
        working_directory=Path("/work/pht"),
        user_id=7,
    )

    assert response.content == "You've hit your weekly limit"


async def test_agent_ids_survive_a_restart(tmp_path):
    from src.claude.cursor_backend import CursorSDKManager

    config = _settings(tmp_path)
    first = CursorSDKManager(config)
    first._agents[(7, "/work/pht")] = "agent-kept"
    first._save_agents()

    second = CursorSDKManager(config)
    assert second._agents[(7, "/work/pht")] == "agent-kept"
