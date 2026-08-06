"""Разрешение путей: один и тот же код работает локально (Windows + Drive for
Desktop) и в Colab.

Асимметрия, из-за которой этот модуль вообще нужен: локально корень Drive
называется `G:\\Мой диск` (русская локаль), а Colab монтирует тот же самый диск
как `/content/drive/MyDrive`. Ничто, кроме этого модуля, не должно знать о
существовании такой разницы.
"""

from __future__ import annotations

import os
from pathlib import Path

WORKSPACE = "nla_steering"

_DRIVE_CANDIDATES = [
    Path("/content/drive/MyDrive"),  # Colab
    Path(r"G:\Мой диск"),  # Windows, русская локаль
    Path(r"G:\My Drive"),  # Windows, английская локаль
]


def in_colab() -> bool:
    return Path("/content").is_dir()


def drive_root() -> Path:
    """Корень Google Drive. Переопределяется через NLA_DRIVE_ROOT."""
    override = os.environ.get("NLA_DRIVE_ROOT")
    if override:
        return Path(override)
    for candidate in _DRIVE_CANDIDATES:
        if candidate.is_dir():
            return candidate
    raise RuntimeError(
        "Не найден корень Google Drive. Установи Drive for Desktop или задай "
        "NLA_DRIVE_ROOT. Проверенные пути: "
        + ", ".join(str(c) for c in _DRIVE_CANDIDATES)
    )


def workspace() -> Path:
    """Общая папка проекта на Drive — мост между Colab и локальной машиной."""
    return drive_root() / WORKSPACE


def repo_root() -> Path:
    """Корень гит-репозитория (там, где лежит pyproject.toml)."""
    for parent in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Не найден pyproject.toml выше по дереву от paths.py")


def artifacts_dir(*parts: str) -> Path:
    """Тяжёлые дампы: активации, вербализации. Живут на Drive, не в git."""
    path = workspace() / "artifacts"
    for part in parts:
        path = path / part
    return path


def results_dir(*parts: str) -> Path:
    """Мелкие агрегаты (json/csv/png), которые коммитятся в git."""
    path = repo_root() / "results"
    for part in parts:
        path = path / part
    return path


def jobs_dir(stage: str = "") -> Path:
    """Очередь задач: queue / running / done / failed / logs."""
    path = workspace() / "jobs"
    return path / stage if stage else path


def ensure_layout() -> None:
    """Создаёт папки на Drive. Идемпотентно, безопасно вызывать где угодно."""
    for stage in ("queue", "running", "done", "failed", "logs"):
        jobs_dir(stage).mkdir(parents=True, exist_ok=True)
    artifacts_dir().mkdir(parents=True, exist_ok=True)
    results_dir().mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_layout()
    print(f"colab:     {in_colab()}")
    print(f"drive:     {drive_root()}")
    print(f"workspace: {workspace()}")
    print(f"repo:      {repo_root()}")
