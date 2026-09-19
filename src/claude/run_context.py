"""Per-request context shared between the Telegram layer and the SDK layer.

The orchestrator binds these ContextVars while it handles an update; the SDK
integration reads them when it builds ``ClaudeAgentOptions``. asyncio tasks
inherit the context, so the values follow the request without threading new
arguments through every facade call.
"""

from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Dict, Optional

# Answers keyed by question text, or None if the user did not answer.
QuestionCallback = Callable[[Dict[str, Any]], Awaitable[Optional[Dict[str, str]]]]

current_model: ContextVar[Optional[str]] = ContextVar("current_model", default=None)
current_question_callback: ContextVar[Optional[QuestionCallback]] = ContextVar(
    "current_question_callback", default=None
)
