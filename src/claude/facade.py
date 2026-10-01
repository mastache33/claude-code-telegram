"""High-level Claude Code integration facade.

Provides simple interface for bot handlers.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import structlog

from ..config.settings import Settings
from .cursor_backend import CursorSDKManager
from .exceptions import ClaudeProcessError
from .limits import is_claude_limit
from .sdk_integration import ClaudeResponse, ClaudeSDKManager, StreamUpdate
from .session import SessionManager

logger = structlog.get_logger()


class ClaudeIntegration:
    """Main integration point for Claude Code."""

    def __init__(
        self,
        config: Settings,
        sdk_manager: Optional[ClaudeSDKManager] = None,
        session_manager: Optional[SessionManager] = None,
        cursor_manager: Optional[CursorSDKManager] = None,
    ):
        """Initialize Claude integration facade."""
        self.config = config
        self.sdk_manager = sdk_manager or ClaudeSDKManager(config)
        self.session_manager = session_manager
        self._cursor_manager = cursor_manager
        self._cursor_until: Dict[Tuple[int, str], datetime] = {}

    async def run_command(
        self,
        prompt: str,
        working_directory: Path,
        user_id: int,
        session_id: Optional[str] = None,
        on_stream: Optional[Callable[[StreamUpdate], None]] = None,
        force_new: bool = False,
        interrupt_event: Optional["asyncio.Event"] = None,
        images: Optional[List[Dict[str, str]]] = None,
        approval_callback: Optional[
            Callable[[str, Dict[str, Any]], Awaitable[bool]]
        ] = None,
    ) -> ClaudeResponse:
        """Run Claude Code command with full integration."""
        logger.info(
            "Running Claude command",
            user_id=user_id,
            working_directory=str(working_directory),
            session_id=session_id,
            prompt_length=len(prompt),
            force_new=force_new,
        )

        # If no session_id provided, try to find an existing session for this
        # user+directory combination (auto-resume).
        # Skip auto-resume when force_new is set (e.g. after /new command).
        if not session_id and not force_new:
            existing_session = await self._find_resumable_session(
                user_id, working_directory
            )
            if existing_session:
                session_id = existing_session.session_id
                logger.info(
                    "Auto-resuming existing session for project",
                    session_id=session_id,
                    project_path=str(working_directory),
                    user_id=user_id,
                )

        # Get or create session
        session = await self.session_manager.get_or_create_session(
            user_id, working_directory, session_id
        )

        # Execute command
        cursor_key = (user_id, str(working_directory))
        if force_new and self._cursor_manager is not None:
            self._cursor_manager.forget(user_id, working_directory)

        try:
            # Continue session if we have an existing session with a real ID
            is_new = getattr(session, "is_new_session", False)
            should_continue = not is_new and bool(session.session_id)

            # For new sessions, don't pass session_id to Claude Code
            claude_session_id = session.session_id if should_continue else None

            if self._cursor_is_active(cursor_key):
                response = await self._run_cursor(
                    prompt=prompt,
                    working_directory=working_directory,
                    user_id=user_id,
                    session=session,
                    on_stream=on_stream,
                    interrupt_event=interrupt_event,
                    images=images,
                    announce=False,
                )
            else:
                try:
                    response = await self._execute(
                        prompt=prompt,
                        working_directory=working_directory,
                        session_id=claude_session_id,
                        continue_session=should_continue,
                        stream_callback=on_stream,
                        interrupt_event=interrupt_event,
                        images=images,
                        approval_callback=approval_callback,
                    )
                except Exception as resume_error:
                    # A spent quota will fail the fresh retry too.
                    if self._cursor_ready() and is_claude_limit(str(resume_error)):
                        raise
                    # If resume failed (e.g., session expired/missing on Claude's side),
                    # retry as a fresh session.  The CLI returns a generic exit-code-1
                    # when the session is gone, so we catch *any* error during resume.
                    if should_continue:
                        logger.warning(
                            "Session resume failed, starting fresh session",
                            failed_session_id=claude_session_id,
                            error=str(resume_error),
                        )
                        # Clean up the stale session
                        await self.session_manager.remove_session(session.session_id)

                        # Create a fresh session and retry
                        session = await self.session_manager.get_or_create_session(
                            user_id, working_directory
                        )
                        response = await self._execute(
                            prompt=prompt,
                            working_directory=working_directory,
                            session_id=None,
                            continue_session=False,
                            stream_callback=on_stream,
                            interrupt_event=interrupt_event,
                            images=images,
                            approval_callback=approval_callback,
                        )
                    else:
                        raise

                if (
                    self._cursor_ready()
                    and not response.interrupted
                    and is_claude_limit(response.content or "")
                ):
                    response = await self._run_cursor(
                        prompt=prompt,
                        working_directory=working_directory,
                        user_id=user_id,
                        session=session,
                        on_stream=on_stream,
                        interrupt_event=interrupt_event,
                        images=images,
                        announce=True,
                    )

            # Update session (assigns real session_id for new sessions)
            await self.session_manager.update_session(session, response)

            # Ensure response has the session's final ID
            response.session_id = session.session_id

            if not response.session_id:
                logger.warning(
                    "No session_id after execution; session cannot be resumed",
                    user_id=user_id,
                )

            logger.info(
                "Claude command completed",
                session_id=response.session_id,
                cost=response.cost,
                duration_ms=response.duration_ms,
                num_turns=response.num_turns,
                is_error=response.is_error,
            )

            return response

        except Exception as e:
            if self._cursor_ready() and is_claude_limit(str(e)):
                logger.warning(
                    "Claude limit reached, switching to Cursor",
                    error=str(e),
                    user_id=user_id,
                )
                try:
                    response = await self._run_cursor(
                        prompt=prompt,
                        working_directory=working_directory,
                        user_id=user_id,
                        session=session,
                        on_stream=on_stream,
                        interrupt_event=interrupt_event,
                        images=images,
                        announce=True,
                    )
                except Exception as cursor_error:
                    logger.error(
                        "Cursor fallback failed",
                        error=str(cursor_error),
                        user_id=user_id,
                    )
                    raise ClaudeProcessError(
                        f"Claude упёрся в лимит, Cursor тоже не ответил: {cursor_error}"
                    ) from cursor_error
                await self.session_manager.update_session(session, response)
                response.session_id = session.session_id
                return response

            logger.error(
                "Claude command failed",
                error=str(e),
                user_id=user_id,
                session_id=session.session_id,
            )
            raise

    async def _execute(
        self,
        prompt: str,
        working_directory: Path,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        stream_callback: Optional[Callable] = None,
        interrupt_event: Optional[asyncio.Event] = None,
        images: Optional[List[Dict[str, str]]] = None,
        approval_callback: Optional[
            Callable[[str, Dict[str, Any]], Awaitable[bool]]
        ] = None,
    ) -> ClaudeResponse:
        """Execute command via SDK."""
        return await self.sdk_manager.execute_command(
            prompt=prompt,
            working_directory=working_directory,
            session_id=session_id,
            continue_session=continue_session,
            stream_callback=stream_callback,
            interrupt_event=interrupt_event,
            images=images,
            approval_callback=approval_callback,
        )

    async def _find_resumable_session(
        self,
        user_id: int,
        working_directory: Path,
    ) -> Optional["ClaudeSession"]:  # noqa: F821
        """Find the most recent resumable session for a user in a directory.

        Returns the session if one exists that is non-expired and has a real
        (non-temporary) session ID from Claude. Returns None otherwise.
        """

        sessions = await self.session_manager._get_user_sessions(user_id)

        matching_sessions = [
            s
            for s in sessions
            if s.project_path == working_directory
            and bool(s.session_id)
            and not self.session_manager._is_session_expired(s)
        ]

        if not matching_sessions:
            return None

        return max(matching_sessions, key=lambda s: s.last_used)

    async def continue_session(
        self,
        user_id: int,
        working_directory: Path,
        prompt: Optional[str] = None,
        on_stream: Optional[Callable[[StreamUpdate], None]] = None,
    ) -> Optional[ClaudeResponse]:
        """Continue the most recent session."""
        logger.info(
            "Continuing session",
            user_id=user_id,
            working_directory=str(working_directory),
            has_prompt=bool(prompt),
        )

        # Get user's sessions
        sessions = await self.session_manager._get_user_sessions(user_id)

        # Find most recent session in this directory (exclude sessions without IDs)
        matching_sessions = [
            s
            for s in sessions
            if s.project_path == working_directory and bool(s.session_id)
        ]

        if not matching_sessions:
            logger.info("No matching sessions found", user_id=user_id)
            return None

        # Get most recent
        latest_session = max(matching_sessions, key=lambda s: s.last_used)

        # Continue session with default prompt if none provided
        # Claude CLI requires a prompt, so we use a placeholder
        return await self.run_command(
            prompt=prompt or "Please continue where we left off",
            working_directory=working_directory,
            user_id=user_id,
            session_id=latest_session.session_id,
            on_stream=on_stream,
        )

    async def get_session_info(
        self, session_id: str, user_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get session information (scoped to requesting user)."""
        return await self.session_manager.get_session_info(session_id, user_id)

    async def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all sessions for a user."""
        sessions = await self.session_manager._get_user_sessions(user_id)
        return [
            {
                "session_id": s.session_id,
                "project_path": str(s.project_path),
                "created_at": s.created_at.isoformat(),
                "last_used": s.last_used.isoformat(),
                "total_cost": s.total_cost,
                "message_count": s.message_count,
                "tools_used": s.tools_used,
                "expired": self.session_manager._is_session_expired(s),
            }
            for s in sessions
        ]

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        return await self.session_manager.cleanup_expired_sessions()

    async def get_user_summary(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive user summary."""
        session_summary = await self.session_manager.get_user_session_summary(user_id)

        return {
            "user_id": user_id,
            **session_summary,
        }

    def _cursor_ready(self) -> bool:
        return self.config.cursor_fallback_ready

    def _cursor_is_active(self, key: Tuple[int, str]) -> bool:
        if not self._cursor_ready():
            return False
        until = self._cursor_until.get(key)
        return until is not None and datetime.now(UTC) < until

    def _cursor(self) -> CursorSDKManager:
        if self._cursor_manager is None:
            self._cursor_manager = CursorSDKManager(self.config)
        return self._cursor_manager

    async def _run_cursor(
        self,
        prompt: str,
        working_directory: Path,
        user_id: int,
        session: "ClaudeSession",  # noqa: F821
        on_stream: Optional[Callable[[StreamUpdate], None]],
        interrupt_event: Optional[asyncio.Event],
        images: Optional[List[Dict[str, str]]],
        announce: bool,
    ) -> ClaudeResponse:
        """Run the prompt on Cursor and keep the Claude session id intact."""
        if not self._cursor_ready():
            raise ClaudeProcessError(
                "Claude упёрся в лимит. Чтобы бот переключился на Cursor, "
                "положи CURSOR_API_KEY в окружение бота."
            )

        key = (user_id, str(working_directory))
        self._cursor_until[key] = datetime.now(UTC) + timedelta(
            hours=self.config.cursor_fallback_hours
        )
        if announce and on_stream is not None:
            await _maybe_stream(
                on_stream,
                StreamUpdate(
                    type="assistant",
                    content="Claude упёрся в лимит, подключаю Cursor.",
                ),
            )

        kept_session_id = ""
        if session.session_id and not getattr(session, "is_new_session", False):
            kept_session_id = session.session_id

        response = await self._cursor().execute_command(
            prompt=prompt,
            working_directory=working_directory,
            user_id=user_id,
            stream_callback=on_stream,
            interrupt_event=interrupt_event,
            images=images,
        )
        if announce and response.content:
            response.content = (
                "Claude упёрся в лимит — дальше этот чат ведёт Cursor.\n\n"
                + response.content
            )
        response.session_id = kept_session_id
        logger.info(
            "Cursor fallback completed",
            user_id=user_id,
            working_directory=str(working_directory),
            announced=announce,
        )
        return response

    async def shutdown(self) -> None:
        """Shutdown integration and cleanup resources."""
        logger.info("Shutting down Claude integration")

        await self.cleanup_expired_sessions()
        if self._cursor_manager is not None:
            await self._cursor_manager.shutdown()

        logger.info("Claude integration shutdown complete")


async def _maybe_stream(
    callback: Callable[[StreamUpdate], Any], update: StreamUpdate
) -> None:
    result = callback(update)
    if asyncio.iscoroutine(result):
        await result
