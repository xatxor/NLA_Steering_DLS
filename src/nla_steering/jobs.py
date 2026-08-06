"""Очередь задач между локальной машиной и Colab.

Job-файл кладётся в `<Drive>/nla_steering/jobs/queue/`, воркер в Colab
забирает самый старый, выполняет и перекладывает в `done/` или `failed/`.
Лог пишется в `logs/<id>.log` по ходу выполнения, так что за прогрессом можно
следить локально, не дожидаясь конца.
"""

from __future__ import annotations

import json
import shlex
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .paths import ensure_layout, jobs_dir


@dataclass
class Job:
    id: str
    script: str
    args: list[str] = field(default_factory=list)
    note: str = ""
    created_at: str = ""

    def command(self) -> str:
        return f"python {shlex.quote(self.script)} " + " ".join(
            shlex.quote(a) for a in self.args
        )


def new_id(slug: str) -> str:
    return f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{slug}"


def submit(slug: str, script: str, args: list[str] | None = None, note: str = "") -> Job:
    """Поставить задачу в очередь. Возвращает Job с присвоенным id."""
    ensure_layout()
    job = Job(
        id=new_id(slug),
        script=script,
        args=args or [],
        note=note,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    path = jobs_dir("queue") / f"{job.id}.json"
    path.write_text(json.dumps(asdict(job), indent=2, ensure_ascii=False), "utf-8")
    return job


def _load(path: Path) -> Job:
    return Job(**json.loads(path.read_text("utf-8")))


def claim() -> tuple[Job, Path] | None:
    """Забрать самую старую задачу из очереди, переложив её в running/.

    Перекладывание файла — это и есть блокировка: воркер один, гонок нет.
    """
    pending = sorted(jobs_dir("queue").glob("*.json"))
    if not pending:
        return None
    src = pending[0]
    dst = jobs_dir("running") / src.name
    src.replace(dst)
    return _load(dst), dst


def finish(path: Path, ok: bool) -> None:
    path.replace(jobs_dir("done" if ok else "failed") / path.name)


def log_path(job_id: str) -> Path:
    return jobs_dir("logs") / f"{job_id}.log"


def read_log(job_id: str, timeout: float = 90.0, poll: float = 3.0) -> str:
    """Прочитать лог, дождавшись его появления.

    Google Drive синхронизирует файлы независимо друг от друга, поэтому статус
    задачи регулярно доезжает раньше лога: `wait()` уже вернул 'done', а файла
    лога локально ещё нет. Ждём его отдельно, иначе на каждой задаче ловим
    FileNotFoundError на ровном месте.
    """
    path = log_path(job_id)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return path.read_text("utf-8")
        time.sleep(poll)
    raise TimeoutError(f"лог {job_id} не появился за {timeout} с (задержка Drive?)")


def status() -> dict[str, list[str]]:
    """Снимок очереди — чтобы проверить, что происходит."""
    ensure_layout()
    return {
        stage: sorted(p.stem for p in jobs_dir(stage).glob("*.json"))
        for stage in ("queue", "running", "done", "failed")
    }


def wait(job_id: str, timeout: float = 1800.0, poll: float = 10.0) -> str:
    """Дождаться завершения задачи. Возвращает 'done' или 'failed'."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for stage in ("done", "failed"):
            if (jobs_dir(stage) / f"{job_id}.json").exists():
                return stage
        time.sleep(poll)
    raise TimeoutError(f"{job_id} не завершилась за {timeout} с")


if __name__ == "__main__":
    for stage, ids in status().items():
        print(f"{stage:8} {len(ids):3}  {', '.join(ids[-3:])}")
