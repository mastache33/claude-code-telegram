"""One-shot reminders for Andrey. Stored in the bot database, fired in Telegram.

Times without a timezone are Tyumen (UTC+5). The agent writes a row through the
MCP tool; this process sends the message when the time comes.
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

TYUMEN = ZoneInfo("Asia/Yekaterinburg")
_MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    chat_id INTEGER NOT NULL,
    message_thread_id INTEGER,
    text TEXT NOT NULL,
    fire_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    sent_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(status, fire_at);
"""

_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_CLOCK = re.compile(r"(?:в\s+)?(\d{1,2})[:.](\d{2})")
_REL = re.compile(
    r"через\s+(\d+)\s*(минут\w*|мин\b|час\w*|день|дня|дней)",
    re.IGNORECASE,
)


def db_path() -> Path:
    override = os.environ.get("REMINDERS_DB", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".claude-telegram" / "bot.db"


def context_path() -> Path:
    override = os.environ.get("REMINDER_CONTEXT", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".claude-telegram" / "reminder-context.json"


def write_chat_context(update: object) -> None:
    """Remember which chat a later remind_user call should answer in."""
    user = getattr(update, "effective_user", None)
    chat = getattr(update, "effective_chat", None)
    if user is None or chat is None:
        return
    message = getattr(update, "effective_message", None)
    thread = getattr(message, "message_thread_id", None) if message else None
    path = context_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "user_id": int(user.id),
                    "chat_id": int(chat.id),
                    "message_thread_id": int(thread) if thread else None,
                }
            ),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError):
        return


def _connect(path: Optional[Path] = None) -> sqlite3.Connection:
    target = path or db_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def format_local(moment: datetime) -> str:
    local = moment.astimezone(TYUMEN)
    return f"{local.day} {_MONTHS[local.month - 1]} {local:%H:%M}"


def parse_when(text: str, now: Optional[datetime] = None) -> datetime:
    """Return an aware UTC datetime. Naive values are read as Tyumen time."""
    raw = (text or "").strip()
    if not raw:
        raise ValueError("пустое время")
    current = now.astimezone(TYUMEN) if now else datetime.now(TYUMEN)

    iso = raw.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        parsed = None
    if parsed is not None and (_DATE.search(raw) or "T" in raw or "+" in raw):
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=TYUMEN)
        return parsed.astimezone(timezone.utc).replace(microsecond=0)

    lower = raw.lower().replace("ё", "е")
    relative = _REL.search(lower)
    if relative:
        amount = int(relative.group(1))
        unit = relative.group(2)
        if amount <= 0:
            raise ValueError("срок должен быть больше нуля")
        if unit.startswith("мин"):
            delta = timedelta(minutes=amount)
        elif unit.startswith("час"):
            delta = timedelta(hours=amount)
        else:
            delta = timedelta(days=amount)
        return (current + delta).astimezone(timezone.utc).replace(microsecond=0)

    day = None
    if "послезавтра" in lower:
        day = (current + timedelta(days=2)).date()
    elif "завтра" in lower:
        day = (current + timedelta(days=1)).date()
    elif "сегодня" in lower:
        day = current.date()
    date_match = _DATE.search(raw)
    if date_match:
        day = datetime(
            int(date_match.group(1)),
            int(date_match.group(2)),
            int(date_match.group(3)),
        ).date()

    clock = _CLOCK.search(lower)
    if clock is None or int(clock.group(1)) > 23 or int(clock.group(2)) > 59:
        raise ValueError(
            "нужно время вида 2026-10-02T15:43:00+05:00, "
            "«завтра 15:43» или «через 30 минут»"
        )
    hour, minute = int(clock.group(1)), int(clock.group(2))
    if day is None:
        candidate = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if candidate <= current:
            candidate += timedelta(days=1)
    else:
        candidate = datetime(day.year, day.month, day.day, hour, minute, tzinfo=TYUMEN)
    return candidate.astimezone(timezone.utc).replace(microsecond=0)


def _read_context() -> dict:
    path = context_path()
    if not path.is_file():
        raise ValueError("не вижу, в какой чат писать напоминание")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("chat_id"):
        raise ValueError("не вижу, в какой чат писать напоминание")
    return data


def add_reminder(when: str, text: str, now: Optional[datetime] = None) -> str:
    body = (text or "").strip()
    if not body:
        return "Error: пустой текст напоминания"
    if len(body) > 400:
        return "Error: текст длиннее 400 символов"
    current = now.astimezone(TYUMEN) if now else datetime.now(TYUMEN)
    try:
        fire = parse_when(when, now=current)
        ctx = _read_context()
    except (ValueError, json.JSONDecodeError, OSError) as exc:
        return f"Error: {exc}"
    if fire < current.astimezone(timezone.utc) - timedelta(seconds=90):
        return (
            f"Error: это время уже прошло ({format_local(fire)}). "
            "Передай будущее время, например 2026-10-02T15:43:00+05:00."
        )
    if fire > current.astimezone(timezone.utc) + timedelta(days=366 * 3):
        return "Error: так далеко вперёд не ставлю, максимум три года."

    fire_iso = fire.isoformat()
    conn = _connect()
    try:
        existing = conn.execute(
            """
            SELECT id, fire_at FROM reminders
            WHERE status = 'pending' AND lower(text) = lower(?)
            """,
            (body,),
        ).fetchall()
        for row in existing:
            previous = datetime.fromisoformat(row["fire_at"])
            if abs(previous - fire) < timedelta(minutes=1):
                return (
                    f"Уже стоит на {format_local(fire)}: {body} (номер {row['id']})"
                )
        created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        thread = ctx.get("message_thread_id")
        cur = conn.execute(
            """
            INSERT INTO reminders
                (user_id, chat_id, message_thread_id, text, fire_at, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending')
            """,
            (
                int(ctx.get("user_id") or 0),
                int(ctx["chat_id"]),
                int(thread) if thread else None,
                body,
                fire_iso,
                created,
            ),
        )
        conn.commit()
        return f"Напоминание поставлено на {format_local(fire)}: {body} (номер {cur.lastrowid})"
    finally:
        conn.close()


def list_reminders() -> str:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, text, fire_at FROM reminders
            WHERE status = 'pending'
            ORDER BY fire_at
            LIMIT 20
            """
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return "Пока ничего не записано."
    lines = []
    for row in rows:
        when = format_local(datetime.fromisoformat(row["fire_at"]))
        lines.append(f"{row['id']}. {when} — {row['text']}")
    return "Записано:\n" + "\n".join(lines)


def cancel_reminder(what: str) -> str:
    query = (what or "").strip()
    if not query:
        return "Error: скажи номер или текст напоминания"
    conn = _connect()
    try:
        if query.lstrip("#").isdigit():
            rows = conn.execute(
                "SELECT id, text, fire_at FROM reminders WHERE status = 'pending' AND id = ?",
                (int(query.lstrip("#")),),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, text, fire_at FROM reminders
                WHERE status = 'pending' AND lower(text) LIKE '%' || lower(?) || '%'
                """,
                (query,),
            ).fetchall()
        if not rows:
            return "Не нашёл такое напоминание."
        if len(rows) > 5:
            return "Слишком много совпадений. Назови номер из списка."
        ids = [row["id"] for row in rows]
        conn.execute(
            f"UPDATE reminders SET status = 'cancelled' WHERE id IN ({','.join('?' * len(ids))})",
            ids,
        )
        conn.commit()
        cancelled = [
            f"{row['id']}. {format_local(datetime.fromisoformat(row['fire_at']))} — {row['text']}"
            for row in rows
        ]
        return "Отменил:\n" + "\n".join(cancelled)
    finally:
        conn.close()


def claim_due(now: Optional[datetime] = None) -> list[dict]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(microsecond=0)
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT id, chat_id, message_thread_id, text, fire_at, attempts
            FROM reminders
            WHERE status = 'pending' AND fire_at <= ?
            ORDER BY fire_at
            """,
            (current.isoformat(),),
        ).fetchall()
        if not rows:
            return []
        ids = [row["id"] for row in rows]
        marks = ",".join("?" * len(ids))
        conn.execute(
            f"UPDATE reminders SET status = 'sending' WHERE status = 'pending' AND id IN ({marks})",
            ids,
        )
        conn.commit()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def mark_sent(reminder_id: int) -> None:
    conn = _connect()
    try:
        sent = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        conn.execute(
            "UPDATE reminders SET status = 'sent', sent_at = ? WHERE id = ?",
            (sent, reminder_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_retry(reminder_id: int, drop_thread: bool = False) -> None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT attempts FROM reminders WHERE id = ?", (reminder_id,)
        ).fetchone()
        attempts = int(row["attempts"]) + 1 if row else 1
        status = "failed" if attempts >= 5 else "pending"
        if drop_thread:
            conn.execute(
                """
                UPDATE reminders
                SET status = ?, attempts = ?, message_thread_id = NULL
                WHERE id = ?
                """,
                (status, attempts, reminder_id),
            )
        else:
            conn.execute(
                "UPDATE reminders SET status = ?, attempts = ? WHERE id = ?",
                (status, attempts, reminder_id),
            )
        conn.commit()
    finally:
        conn.close()


def notification_text(row: dict, now: Optional[datetime] = None) -> str:
    fire = datetime.fromisoformat(row["fire_at"])
    current = now or datetime.now(timezone.utc)
    if current - fire > timedelta(minutes=2):
        return f"Напоминание, должно было в {format_local(fire)}: {row['text']}"
    return f"Напоминание: {row['text']}"


async def reminder_loop(bot: object, interval: float = 15) -> None:
    """Send due reminders until the task is cancelled."""
    import asyncio

    import structlog

    logger = structlog.get_logger()
    while True:
        try:
            for row in claim_due():
                kwargs = {
                    "chat_id": row["chat_id"],
                    "text": notification_text(row),
                }
                if row.get("message_thread_id"):
                    kwargs["message_thread_id"] = row["message_thread_id"]
                try:
                    await bot.send_message(**kwargs)  # type: ignore[attr-defined]
                    mark_sent(int(row["id"]))
                except Exception as exc:
                    dropped = "thread" in str(exc).lower()
                    mark_retry(int(row["id"]), drop_thread=dropped)
                    logger.warning("Reminder send failed", id=row["id"], error=str(exc)[:200])
        except Exception as exc:
            logger.warning("Reminder check failed", error=str(exc)[:200])
        await asyncio.sleep(interval)
