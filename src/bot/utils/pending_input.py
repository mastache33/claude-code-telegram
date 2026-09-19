"""Free-text answers the bot is waiting for, keyed by Telegram user id.

When Claude asks a question and the user taps "own answer", the next text
message from that user resolves the future instead of starting a new Claude
run. The update processor consults this registry so such a message bypasses
the sequential queue (the handler that asked is still holding it).
"""

import asyncio
from typing import Dict

waiting_text: Dict[int, "asyncio.Future[str]"] = {}


def is_waiting(user_id: int) -> bool:
    future = waiting_text.get(user_id)
    return future is not None and not future.done()


def resolve(user_id: int, text: str) -> bool:
    future = waiting_text.pop(user_id, None)
    if future is None or future.done():
        return False
    future.set_result(text)
    return True
