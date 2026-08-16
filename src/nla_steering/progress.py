"""Пульс долгих задач в отдельном файле.

Лог задачи воркер держит открытым на запись всё время её выполнения, и Google
Drive такие файлы синхронизирует рывками: локально лог может часами выглядеть
застывшим на заголовке, хотя задача идёт нормально. Отличить это от настоящего
зависания по логу невозможно, и на этом уже дважды было потеряно время.

Здесь на каждый вызов файл открывается, пишется и **закрывается**. Закрытый
файл Drive синхронизирует сразу, поэтому пульс виден локально почти в реальном
времени.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .paths import artifacts_dir


def beat(name: str, message: str) -> None:
    path = artifacts_dir("progress") / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now(timezone.utc):%H:%M:%S}"
    path.write_text(f"{stamp}  {message}\n", "utf-8")


def read(name: str) -> str | None:
    path = artifacts_dir("progress") / f"{name}.txt"
    return path.read_text("utf-8").strip() if path.exists() else None
