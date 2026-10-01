"""Cursor agent used when Claude's usage limit is spent.

Same machine, same project folder, same Telegram tools (files, voice,
checklists). Conversation state is a Cursor agent id, separate from the
Claude session.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from ..config.settings import Settings
from .exceptions import ClaudeProcessError, ClaudeTimeoutError
from .sdk_integration import ClaudeResponse, StreamUpdate

logger = structlog.get_logger()

_TELEGRAM_TOOLS = (
    "send_file_to_user, send_image_to_user, speak_to_user, "
    "send_checklist_to_user, configure_bot, create_topic"
)
_QUIET_REPLY = (
    "В ответе Андрею только результат. "
    "Не описывай команды, инструменты, очереди и ход работы. "
    "Ссылку на товар присылай только если страница открылась и товар можно купить. "
    "Если товара нет или он закончился — ищи другой."
)


def _bridge_dropped(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        mark in text
        for mark in (
            "readerror",
            "connecterror",
            "remoteprotocol",
            "bridge request failed",
            "connection reset",
            "broken pipe",
            "server disconnected",
        )
    )


def bare_tool_name(name: str) -> str:
    """Cursor may prefix MCP tools. The bot matches the bare tool name."""
    cleaned = name or ""
    for separator in ("__", ".", "/"):
        if separator in cleaned:
            cleaned = cleaned.split(separator)[-1]
    return cleaned


def unwrap_cursor_tool(name: str, raw: Any) -> Tuple[str, Dict[str, Any]]:
    """Turn Cursor's ``mcp`` wrapper into the tool the Telegram bot intercepts.

    A speak request arrives as name ``mcp`` with
    ``{toolName: speak_to_user, args: {text: ...}}``. The bot only delivers
    voice, files and settings when it sees the inner name and arguments.
    """
    payload = raw if isinstance(raw, dict) else {}
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            loaded = None
        if isinstance(loaded, dict):
            payload = loaded
    inner = payload.get("toolName") or payload.get("tool_name")
    if name in {"mcp", "CallDynamicTool"} and isinstance(inner, str) and inner.strip():
        args = payload.get("args")
        if args is None:
            args = payload.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        return inner.strip(), args
    return bare_tool_name(name), payload


def claude_mcp_to_cursor(servers: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert a Claude ``mcpServers`` map into Cursor SDK server dicts."""
    converted: Dict[str, Dict[str, Any]] = {}
    for name, raw in servers.items():
        if not isinstance(raw, dict):
            continue
        command = raw.get("command")
        url = raw.get("url")
        if command:
            env = raw.get("env") or {}
            converted[name] = {
                "command": str(command),
                "args": [str(arg) for arg in (raw.get("args") or [])],
                "env": {str(key): str(value) for key, value in env.items()},
            }
            if raw.get("cwd"):
                converted[name]["cwd"] = str(raw["cwd"])
        elif url:
            converted[name] = {
                "url": str(url),
                "type": str(raw.get("type") or "http"),
                "headers": {
                    str(key): str(value)
                    for key, value in (raw.get("headers") or {}).items()
                },
            }
    return converted


class CursorSDKManager:
    """Run a local Cursor agent and shape the result like a Claude response."""

    def __init__(self, config: Settings):
        self.config = config
        self._client: Any = None
        self._lock = asyncio.Lock()
        self._agents: Dict[Tuple[int, str], str] = {}
        self._load_agents()

    def forget(self, user_id: int, working_directory: Path) -> None:
        """Drop the remembered agent so the next message starts a new chat."""
        self._agents.pop(self._key(user_id, working_directory), None)
        self._save_agents()

    async def _drop_client(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception:
                logger.debug("Cursor client close failed")

    async def shutdown(self) -> None:
        self._agents.clear()
        await self._drop_client()

    async def execute_command(
        self,
        prompt: str,
        working_directory: Path,
        user_id: int,
        stream_callback: Optional[Callable[[StreamUpdate], Any]] = None,
        interrupt_event: Optional[asyncio.Event] = None,
        images: Optional[List[Dict[str, str]]] = None,
    ) -> ClaudeResponse:
        api_key = self.config.cursor_api_key_str
        if not api_key:
            raise ClaudeProcessError(
                "Cursor не настроен: в окружении бота нет CURSOR_API_KEY."
            )

        if interrupt_event is not None and interrupt_event.is_set():
            return ClaudeResponse(
                content="",
                session_id="",
                cost=0.0,
                duration_ms=0,
                num_turns=0,
                interrupted=True,
            )

        start = asyncio.get_event_loop().time()
        key = self._key(user_id, working_directory)
        mcp_servers = self._mcp_servers()
        agent: Any = None
        result: Any = None
        tools_used: List[Dict[str, Any]] = []
        num_turns = 0
        cost = 0.0

        for attempt in (1, 2):
            agent = None
            try:
                client = await self._get_client()
                agent, created = await self._open_agent(
                    client, key, working_directory, api_key, mcp_servers
                )
                text = self._opening_prompt(prompt, working_directory) if created else prompt
                text = _QUIET_REPLY + "\n\n" + text
                run = await agent.send(
                    self._message(text, images), {"mcp_servers": mcp_servers}
                )
                tools_used = []
                seen_calls: set[str] = set()
                num_turns = 0

                async def _consume() -> Any:
                    nonlocal num_turns
                    async for event in run.messages():
                        if getattr(event, "type", "") == "assistant":
                            num_turns += 1
                            await self._emit_assistant(event, stream_callback)
                        elif getattr(event, "type", "") == "tool_call":
                            await self._emit_tool(
                                event, stream_callback, tools_used, seen_calls
                            )
                    return await run.wait()

                consume = asyncio.create_task(_consume())
                watcher: Optional[asyncio.Task[None]] = None
                if interrupt_event is not None:

                    async def _cancel_on_interrupt() -> None:
                        await interrupt_event.wait()
                        await run.cancel()

                    watcher = asyncio.create_task(_cancel_on_interrupt())

                try:
                    result = await asyncio.wait_for(
                        consume, timeout=self.config.claude_timeout_seconds
                    )
                except asyncio.TimeoutError:
                    consume.cancel()
                    try:
                        await consume
                    except (asyncio.CancelledError, Exception):
                        pass
                    try:
                        await run.cancel()
                    except Exception:
                        logger.debug("Cursor cancel after timeout failed")
                    raise ClaudeTimeoutError(
                        f"Cursor timed out after {self.config.claude_timeout_seconds}s"
                    )
                finally:
                    if watcher is not None:
                        watcher.cancel()

                cost = await self._cost(agent)
                break
            except (ClaudeProcessError, ClaudeTimeoutError):
                raise
            except Exception as exc:
                self._agents.pop(key, None)
                self._save_agents()
                if attempt == 1 and _bridge_dropped(exc):
                    logger.warning(
                        "Cursor bridge dropped, retrying once",
                        error=str(exc),
                        user_id=user_id,
                    )
                    await self._drop_client()
                    continue
                logger.error("Cursor agent failed", error=str(exc), user_id=user_id)
                raise ClaudeProcessError(f"Cursor не ответил: {exc}") from exc
            finally:
                if agent is not None:
                    try:
                        await agent.close()
                    except Exception:
                        logger.debug("Cursor agent close failed")

        content = (getattr(result, "result", "") or "").strip()
        if not content and tools_used:
            content = "Готово."

        status = str(getattr(result, "status", "") or "")
        interrupted = status == "cancelled"
        if interrupt_event is not None and interrupt_event.is_set():
            interrupted = True

        duration_ms = int((asyncio.get_event_loop().time() - start) * 1000)
        agent_id = self._agents.get(key, "")
        return ClaudeResponse(
            content=content,
            session_id=agent_id,
            cost=cost,
            duration_ms=duration_ms,
            num_turns=num_turns,
            is_error=status == "error",
            error_type=status if status == "error" else None,
            tools_used=tools_used,
            interrupted=interrupted,
        )

    async def _open_agent(
        self,
        client: Any,
        key: Tuple[int, str],
        working_directory: Path,
        api_key: str,
        mcp_servers: Dict[str, Dict[str, Any]],
    ) -> Tuple[Any, bool]:
        options = self._agent_options(working_directory, api_key, mcp_servers)
        existing = self._agents.get(key)
        if existing:
            try:
                return await client.agents.resume(existing, options), False
            except Exception as exc:
                logger.warning(
                    "Cursor resume failed, starting a new agent",
                    agent_id=existing,
                    error=str(exc),
                )
                self._agents.pop(key, None)
                self._save_agents()
        agent = await client.agents.create(options)
        self._agents[key] = agent.agent_id
        self._save_agents()
        return agent, True

    def _agent_options(
        self,
        working_directory: Path,
        api_key: str,
        mcp_servers: Dict[str, Dict[str, Any]],
    ) -> Any:
        from cursor_sdk import AgentOptions, LocalAgentOptions

        return AgentOptions(
            model=self.config.cursor_model,
            api_key=api_key,
            local=LocalAgentOptions(
                cwd=str(working_directory),
                setting_sources=["project"],
                auto_review=True,
            ),
            mcp_servers=mcp_servers,
        )

    async def _get_client(self) -> Any:
        async with self._lock:
            if self._client is None:
                from cursor_sdk import AsyncClient

                self._client = await AsyncClient.launch_bridge(
                    workspace=str(self.config.approved_directory)
                )
            return self._client

    def _mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        path = self.config.mcp_config_path
        if not self.config.enable_mcp or not path:
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load MCP config for Cursor", error=str(exc))
            return {}
        servers = data.get("mcpServers", data if isinstance(data, dict) else {})
        if not isinstance(servers, dict):
            return {}
        return claude_mcp_to_cursor(servers)

    def _opening_prompt(self, prompt: str, working_directory: Path) -> str:
        notes = [
            "Ты продолжаешь работу Андрея в Telegram.",
            f"Все файлы и команды — внутри {working_directory}.",
            "Отвечай по-русски, коротко, только результатом.",
            f"Файлы, картинки, голос, чек-листы и темы отдавай инструментами: "
            f"{_TELEGRAM_TOOLS}. В тексте ответа об этом не пиши.",
        ]
        claude_md = working_directory / "CLAUDE.md"
        if claude_md.is_file():
            try:
                notes.append(claude_md.read_text(encoding="utf-8")[:20000])
            except OSError:
                pass
        return "\n\n".join(notes) + "\n\n" + prompt

    def _message(self, prompt: str, images: Optional[List[Dict[str, str]]]) -> Any:
        if not images:
            return prompt
        from cursor_sdk import SDKImage, UserMessage

        attached = [
            SDKImage.from_data(image["data"], image.get("media_type") or "image/png")
            for image in images
            if image.get("data")
        ]
        return UserMessage(text=prompt, images=attached or None)

    async def _emit_assistant(
        self, event: Any, stream_callback: Optional[Callable[[StreamUpdate], Any]]
    ) -> None:
        if stream_callback is None:
            return
        content = getattr(getattr(event, "message", None), "content", ()) or ()
        parts = [text for block in content if (text := getattr(block, "text", None))]
        if parts:
            await _maybe_await(
                stream_callback(
                    StreamUpdate(type="assistant", content="\n".join(parts))
                )
            )

    async def _emit_tool(
        self,
        event: Any,
        stream_callback: Optional[Callable[[StreamUpdate], Any]],
        tools_used: List[Dict[str, Any]],
        seen_calls: set[str],
    ) -> None:
        call_id = str(getattr(event, "call_id", "") or "")
        status = str(getattr(event, "status", "") or "")
        args = getattr(event, "args", None)
        ready = isinstance(args, dict) and bool(args)
        if status == "running" and not ready:
            return
        if status != "running" and (not call_id or call_id in seen_calls):
            return
        if call_id:
            seen_calls.add(call_id)
        name, tool_input = unwrap_cursor_tool(
            str(getattr(event, "name", "") or ""), args
        )
        tools_used.append(
            {
                "name": name,
                "timestamp": asyncio.get_event_loop().time(),
                "input": tool_input,
            }
        )
        if stream_callback is None:
            return
        await _maybe_await(
            stream_callback(
                StreamUpdate(
                    type="assistant",
                    tool_calls=[{"name": name, "input": tool_input, "id": call_id}],
                )
            )
        )

    async def _cost(self, agent: Any) -> float:
        try:
            usage = await agent.get_usage()
        except Exception:
            return 0.0
        cost = getattr(usage, "cost", None)
        charged = getattr(cost, "charged_cents", None)
        if charged is None:
            return 0.0
        return float(charged) / 100.0

    def _state_path(self) -> Path:
        url = self.config.database_url
        if url.startswith("sqlite:///"):
            database = Path(url.removeprefix("sqlite:///"))
            if not database.is_absolute():
                database = Path.cwd() / database
            return database.with_name("cursor-agents.json")
        return Path.cwd() / "cursor-agents.json"

    def _load_agents(self) -> None:
        path = self._state_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, list):
            return
        for item in raw:
            if not isinstance(item, dict):
                continue
            user_id = item.get("user_id")
            folder = item.get("path")
            agent_id = item.get("agent_id")
            if isinstance(user_id, int) and folder and agent_id:
                self._agents[(user_id, str(folder))] = str(agent_id)

    def _save_agents(self) -> None:
        path = self._state_path()
        payload = [
            {"user_id": user_id, "path": folder, "agent_id": agent_id}
            for (user_id, folder), agent_id in self._agents.items()
        ]
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to save Cursor agents", error=str(exc))

    @staticmethod
    def _key(user_id: int, working_directory: Path) -> Tuple[int, str]:
        return (user_id, str(working_directory))


async def _maybe_await(value: Any) -> None:
    if asyncio.iscoroutine(value):
        await value
