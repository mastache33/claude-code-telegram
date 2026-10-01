"""Detect Claude usage-limit failures that should switch the bot to Cursor."""

_MARKERS = (
    "you've hit your weekly limit",
    "you've hit your limit",
    "you have hit your weekly limit",
    "you have hit your limit",
    "usage limit reached",
    "usage limit",
    "out of extra usage",
    "extra usage required",
    "credit balance is too low",
)


def is_claude_limit(text: str) -> bool:
    """True when Claude (not the user) is reporting that its quota is spent."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _MARKERS)
