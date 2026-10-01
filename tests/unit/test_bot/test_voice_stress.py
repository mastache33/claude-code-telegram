import sqlite3

from src.bot.voice_stress import stress_russian


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE a (w TEXT PRIMARY KEY, s TEXT)")
    conn.executemany(
        "INSERT INTO a VALUES (?, ?)",
        [("встреча", "встр+еча"), ("чайной", "ч+айной")],
    )
    return conn


def test_marks_known_words_and_keeps_the_rest():
    text = stress_russian("Встреча в чайной завтра", db=_db())
    assert text == "Встр+еча в ч+айной завтра"


def test_leaves_text_alone_without_a_dictionary():
    assert stress_russian("Встреча") == "Встреча"
