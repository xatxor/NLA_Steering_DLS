"""Доступ к Hugging Face Hub.

Полагаться на то, что `HF_TOKEN` в окружении подхватится сам, оказалось нельзя:
переменная выставлена, а Hub всё равно предупреждает про анонимные запросы.
Поэтому токен всюду передаётся в `from_pretrained(token=...)` явно, а `check()`
показывает, работает ли аутентификация на самом деле, а не только выставлена ли
переменная. Разница существенна: анонимные запросы упираются в лимиты как раз
на скачивании многогигабайтных весов.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

_loaded = False


def token() -> str | None:
    """Токен из окружения или .env. None, если не задан."""
    global _loaded
    if not _loaded:
        load_dotenv()
        _loaded = True
    value = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return value.strip() or None if value else None


def check() -> dict:
    """Проверка, что токен реально принимается Hub. Сам токен не возвращается."""
    tok = token()
    if not tok:
        return {"present": False, "authenticated": False, "user": None}
    try:
        from huggingface_hub import whoami

        return {"present": True, "authenticated": True, "user": whoami(tok).get("name")}
    except Exception as exc:
        return {"present": True, "authenticated": False, "error": repr(exc)}
