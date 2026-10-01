"""Put a stress mark before the stressed vowel for the Russian voice model.

The dictionary is optional. Without ~/.claude-telegram/voice/accents.db the text
is spoken as it was written. Homographs are left unmarked: one spelling has two
stresses, and guessing the wrong one is worse than leaving it.
"""

import re
import sqlite3
from pathlib import Path
from typing import Optional

_WORD = re.compile(r"[А-Яа-яЁё]+")
_DB: Optional[sqlite3.Connection] = None
_LOOKED = False


def _connect() -> Optional[sqlite3.Connection]:
    global _DB, _LOOKED
    if _LOOKED:
        return _DB
    _LOOKED = True
    path = Path.home() / ".claude-telegram" / "voice" / "accents.db"
    if not path.is_file():
        return None
    _DB = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    return _DB


def stress_russian(text: str, db: Optional[sqlite3.Connection] = None) -> str:
    conn = db if db is not None else _connect()
    if conn is None or not text:
        return text

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        if "+" in word:
            return word
        key = word.lower().replace("ё", "е")
        row = conn.execute("SELECT s FROM a WHERE w = ?", (key,)).fetchone()
        if not row:
            return word
        marked = str(row[0])
        if word[:1].isupper() and marked:
            marked = marked[:1].upper() + marked[1:]
        return marked

    return _WORD.sub(repl, text)
