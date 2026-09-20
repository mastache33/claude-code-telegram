"""MCP server exposing Telegram-specific tools to Claude.

Runs as a stdio transport server. The ``send_file_to_user`` tool validates
file existence and size, then returns a success string; ``send_image_to_user``
is kept as a deprecated image-only alias. Actual Telegram delivery is handled
by the bot's stream callback which intercepts the tool call and applies full
security checks (approved directory, secrets blocklist).
"""

from pathlib import Path

from mcp.server.fastmcp import FastMCP

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"}

# Telegram Bot API document upload limit.
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024

mcp = FastMCP("telegram")


@mcp.tool()
async def send_file_to_user(file_path: str, caption: str = "") -> str:
    """Send a file of any type to the Telegram user as a document.

    Preferred tool for delivering files (PDF, zip, csv, logs, images, etc.)
    back to the user. Full security validation (approved directory, secrets
    blocklist) happens on the bot side; this tool only performs basic syntax
    checks so it can run without access to the bot's runtime configuration.

    Args:
        file_path: Absolute path to the file.
        caption: Optional caption to display with the file.

    Returns:
        Confirmation string when the file is queued for delivery.
    """
    path = Path(file_path)

    if not path.is_absolute():
        return f"Error: path must be absolute, got '{file_path}'"

    if not path.is_file():
        return f"Error: file not found: {file_path}"

    size = path.stat().st_size
    if size == 0:
        return f"Error: file is empty: {file_path}"
    if size > MAX_FILE_SIZE_BYTES:
        return (
            f"Error: file too large ({size} bytes). "
            f"Telegram Bot API limit is {MAX_FILE_SIZE_BYTES} bytes (50 MB)."
        )

    return f"File queued for delivery: {path.name}"


@mcp.tool()
async def configure_bot(
    model: str = "",
    voice: str = "",
    verbosity: int = -1,
    new_session: bool = False,
    project: str = "",
    show_panel: bool = False,
) -> str:
    """Change the Telegram bot's own settings for this chat, as the user asked in words.

    Use whenever the user asks for something the bot owns rather than you:
    "включи соннет", "перейди на opus", "отвечай голосом", "поменьше деталей",
    "начни заново", "переключись на PHT". Do not tell the user to type /model
    or /voice — call this tool instead and confirm in one short line.

    Args:
        model: "opus" | "sonnet" | "haiku" | "default" (or a full claude-* id).
        voice: "auto" (voice answer to voice messages) | "on" | "off".
        verbosity: 0 quiet, 1 normal, 2 detailed; -1 leaves it unchanged.
        new_session: True to drop the conversation context and start fresh.
        project: project slug or folder name to switch the workspace to.
        show_panel: True to show the 🛠 Панель button (Mini App) in the chat.

    Returns:
        Confirmation string listing what will be applied.
    """
    changes = []
    if model:
        allowed = {"opus", "sonnet", "haiku", "fable", "default", "reset"}
        if model.lower() not in allowed and not model.lower().startswith("claude-"):
            return f"Error: unknown model '{model}'. Use opus, sonnet, haiku or default."
        changes.append(f"model={model.lower()}")
    if voice:
        if voice.lower() not in {"auto", "on", "off"}:
            return f"Error: voice must be auto, on or off, got '{voice}'"
        changes.append(f"voice={voice.lower()}")
    if verbosity != -1:
        if verbosity not in (0, 1, 2):
            return f"Error: verbosity must be 0, 1 or 2, got {verbosity}"
        changes.append(f"verbosity={verbosity}")
    if new_session:
        changes.append("new session")
    if project:
        changes.append(f"project={project}")
    if show_panel:
        changes.append("show panel")
    if not changes:
        return "Error: nothing to change — pass model, voice, verbosity, new_session or project."
    return "Bot settings queued: " + ", ".join(changes)


@mcp.tool()
async def send_checklist_to_user(title: str, items: list[str]) -> str:
    """Send a tappable checklist to the Telegram user.

    Use for multi-step work the user will do (or verify) themselves: release
    steps, manual QA, shopping, packing. The user taps items to tick them off
    and can report the state back to you with one button.

    Args:
        title: Short checklist title, e.g. "Выкат TeaMate".
        items: 1-20 short steps, each under 80 characters.

    Returns:
        Confirmation string when the checklist is queued.
    """
    clean = [str(i).strip() for i in items if str(i).strip()]
    if not clean:
        return "Error: items are empty"
    if len(clean) > 20:
        return f"Error: too many items ({len(clean)}). Keep it under 20."
    too_long = [i for i in clean if len(i) > 80]
    if too_long:
        return f"Error: item too long ({len(too_long[0])} chars): {too_long[0][:40]}..."
    return f"Checklist queued: {title.strip() or 'Чек-лист'} ({len(clean)} items)"


@mcp.tool()
async def speak_to_user(text: str) -> str:
    """Speak a short message to the Telegram user as a voice message.

    Use when the user asks for a spoken/voice answer, or when the reply is
    better heard than read. Keep it under ~1200 characters of plain prose:
    no markdown, no code, no long lists. The bot synthesises the speech and
    delivers it after the text reply.

    Args:
        text: Plain text to read aloud, in the user's language.

    Returns:
        Confirmation string when the voice message is queued.
    """
    clean = text.strip()
    if not clean:
        return "Error: text is empty"
    if len(clean) > 1500:
        return f"Error: text too long ({len(clean)} chars). Keep it under 1500."
    return f"Voice message queued ({len(clean)} chars)"


@mcp.tool()
async def send_image_to_user(file_path: str, caption: str = "") -> str:
    """DEPRECATED: use ``send_file_to_user`` instead.

    Kept for backward compatibility with existing prompts and MCP configs.
    Accepts only image extensions; ``send_file_to_user`` accepts any file type
    and is the preferred tool.

    Args:
        file_path: Absolute path to the image file.
        caption: Optional caption to display with the image.

    Returns:
        Confirmation string when the image is queued for delivery.
    """
    path = Path(file_path)

    if not path.is_absolute():
        return f"Error: path must be absolute, got '{file_path}'"

    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return (
            f"Error: unsupported image extension '{path.suffix}'. "
            f"Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )

    if not path.is_file():
        return f"Error: file not found: {file_path}"

    return f"Image queued for delivery: {path.name}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
