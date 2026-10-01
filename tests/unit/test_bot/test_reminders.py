"""Reminders: Tyumen time parsing, save, list, cancel, due."""

import json
from datetime import datetime, timedelta, timezone

from src.bot.reminders import (
    TYUMEN,
    add_reminder,
    cancel_reminder,
    claim_due,
    list_reminders,
    notification_text,
    parse_when,
)


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, 1, hour, minute, tzinfo=TYUMEN)


def test_parse_iso_and_relative() -> None:
    now = _at(15, 0)
    iso = parse_when("2026-10-01T15:43:00+05:00", now=now)
    assert iso.astimezone(TYUMEN).hour == 15
    assert iso.astimezone(TYUMEN).minute == 43
    later = parse_when("через 30 минут", now=now)
    assert later.astimezone(TYUMEN) == _at(15, 30)
    tomorrow = parse_when("завтра 9:05", now=now)
    assert tomorrow.astimezone(TYUMEN) == datetime(2026, 10, 2, 9, 5, tzinfo=TYUMEN)


def test_clock_without_day_rolls_forward() -> None:
    assert parse_when("в 18:30", now=_at(15, 0)).astimezone(TYUMEN) == _at(18, 30)
    rolled = parse_when("в 14:00", now=_at(15, 0)).astimezone(TYUMEN)
    assert rolled == datetime(2026, 10, 2, 14, 0, tzinfo=TYUMEN)


def test_save_list_cancel_and_fire(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REMINDERS_DB", str(tmp_path / "bot.db"))
    monkeypatch.setenv("REMINDER_CONTEXT", str(tmp_path / "ctx.json"))
    (tmp_path / "ctx.json").write_text(
        json.dumps({"user_id": 7, "chat_id": 7, "message_thread_id": 243543}),
        encoding="utf-8",
    )
    now = _at(15, 0)
    saved = add_reminder("сегодня 15:43", "встреча в чайной", now=now)
    assert "15:43" in saved
    assert "чайной" in saved
    assert "встреча в чайной" in list_reminders()
    again = add_reminder("2026-10-01T15:43:00+05:00", "встреча в чайной", now=now)
    assert again.startswith("Уже стоит")

    missed = add_reminder("сегодня 14:00", "уже прошло", now=now)
    assert missed.startswith("Error")

    due = claim_due(_at(15, 43))
    assert len(due) == 1
    assert due[0]["chat_id"] == 7
    assert due[0]["message_thread_id"] == 243543
    assert notification_text(due[0], _at(15, 43)) == "Напоминание: встреча в чайной"
    late = notification_text(due[0], _at(15, 43) + timedelta(minutes=10))
    assert late.startswith("Напоминание, должно было")
    assert claim_due(_at(16, 0)) == []

    kept = add_reminder("завтра 10:00", "забрать заказ", now=now)
    assert "10:00" in kept
    cancelled = cancel_reminder("заказ")
    assert "забрать заказ" in cancelled
    assert "Пока ничего" in list_reminders()


def test_bad_time_is_an_error() -> None:
    try:
        parse_when("когда-нибудь", now=_at(12, 0))
    except ValueError as exc:
        assert "2026-10-02" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_notification_uses_utc_storage() -> None:
    fire = _at(15, 43).astimezone(timezone.utc).replace(microsecond=0)
    text = notification_text(
        {"fire_at": fire.isoformat(), "text": "чай"},
        now=_at(15, 43),
    )
    assert text == "Напоминание: чай"
