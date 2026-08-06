"""Смоук-тест GPU-стороны: проверяет связку целиком и протокол инъекции.

Весов не качает (только токенизаторы, сотни килобайт), выполняется за минуту.
Проверяет ровно то, на чём держится весь дальнейший пайплайн:

  * очередь задач доезжает до Colab, а лог и артефакт — обратно на Drive;
  * какая карта досталась и сколько на ней памяти (от этого зависит, влезут ли
    база и NLA одновременно или нужен двухфазный прогон);
  * есть ли **нативная** поддержка bf16 (на Turing её нет, только эмуляция,
    поэтому считаем в fp16 — ради скорости, а не потому что bf16 «не работает»);
  * **позиция инъекции находится однозначно**: символ U+320E токенизируется в
    один токен 149705, окружённый соседями 29 и 522 из nla_meta.yaml. Если это
    не так, весь план подстановки активации хуком неверен.

Запуск:
    python scripts/01_smoke_gpu.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering import hf  # noqa: E402
from nla_steering.paths import artifacts_dir, drive_root, in_colab, workspace  # noqa: E402


def gpu_report() -> dict:
    try:
        import torch
    except ImportError:
        return {"error": "torch не установлен"}

    report = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        return report

    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    report.update(
        {
            "gpu": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_gb": round(props.total_memory / 1024**3, 2),
            "free_gb": round(free / 1024**3, 2),
        }
    )

    # is_bf16_supported() по умолчанию учитывает программную эмуляцию и на
    # Turing возвращает True — то есть bf16 «работает», но без тензорных ядер.
    # Нас интересует именно нативная поддержка (Ampere, 8.0+), потому что от
    # неё зависит скорость. Аргумент including_emulation появился не во всех
    # версиях torch, поэтому вызов защищён.
    report["bf16_any"] = torch.cuda.is_bf16_supported()
    try:
        report["bf16_native"] = torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        report["bf16_native"] = (props.major, props.minor) >= (8, 0)
    return report


def injection_report(cfg: dict) -> dict:
    """Проверка позиции инъекции по токенизатору — весов не требует."""
    from transformers import AutoTokenizer

    nla = cfg["nla"]
    inj = nla["injection"]
    # токен передаём явно: на переменную окружения Hub полагается ненадёжно
    tok = AutoTokenizer.from_pretrained(nla["verbalizer"], token=hf.token())

    prompt = nla["prompts"]["av"].format(injection_char=inj["char"])
    ids = tok(prompt, add_special_tokens=False)["input_ids"]

    positions = [i for i, t in enumerate(ids) if t == inj["token_id"]]
    report = {
        "prompt_tokens": len(ids),
        "expected_token_id": inj["token_id"],
        "found_positions": positions,
        "unique": len(positions) == 1,
    }

    # Символ обязан быть ровно одним токеном, иначе подстановка одного вектора
    # вместо одного эмбеддинга некорректна по построению.
    report["char_is_single_token"] = (
        tok(inj["char"], add_special_tokens=False)["input_ids"] == [inj["token_id"]]
    )

    if positions:
        p = positions[0]
        left, right = (ids[p - 1] if p else None), (ids[p + 1] if p + 1 < len(ids) else None)
        report.update(
            {
                "left_id": left,
                "right_id": right,
                "left_matches_meta": left == inj["left_neighbor_id"],
                "right_matches_meta": right == inj["right_neighbor_id"],
                "left_text": tok.decode([left]) if left is not None else None,
                "right_text": tok.decode([right]) if right is not None else None,
                "context": tok.decode(ids[max(0, p - 4) : p + 5]),
            }
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text("utf-8"))

    report = {
        "when": datetime.now(timezone.utc).isoformat(),
        "in_colab": in_colab(),
        "python": platform.python_version(),
        "drive_root": str(drive_root()),
        "workspace": str(workspace()),
        # Не наличие переменной, а реальный ответ Hub: сам токен не печатаем.
        "hf": hf.check(),
        "gpu": gpu_report(),
    }
    if not report["hf"]["authenticated"]:
        print("! аутентификация в HF Hub не работает — скачивание весов упрётся"
              " в лимиты анонимных запросов\n")

    print("=== окружение ===")
    for key in ("in_colab", "python", "drive_root", "workspace"):
        print(f"  {key:17} {report[key]}")
    print(f"  {'hf':17} {report['hf']}")

    print("\n=== gpu ===")
    for key, value in report["gpu"].items():
        print(f"  {key:16} {value}")

    print("\n=== протокол инъекции ===")
    try:
        report["injection"] = injection_report(cfg)
        for key, value in report["injection"].items():
            print(f"  {key:22} {value!r}")
        checks = ("unique", "char_is_single_token", "left_matches_meta", "right_matches_meta")
        ok = all(report["injection"].get(c) for c in checks)
        report["injection_ok"] = ok
        print(f"\n  ИТОГ: {'протокол подтверждён' if ok else 'ПРОТОКОЛ НЕ СОШЁЛСЯ'}")
    except Exception as exc:
        report["injection"] = {"error": repr(exc)}
        report["injection_ok"] = False
        print(f"  ОШИБКА: {exc!r}")

    out = artifacts_dir("smoke") / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nОтчёт на Drive: {out}")

    # Аутентификация тоже входит в вердикт: следующий шаг качает ~6 ГБ весов,
    # и анонимные лимиты там перестают быть косметической проблемой.
    ok = bool(report.get("injection_ok")) and report["hf"]["authenticated"]
    print(f"\nСМОУК-ТЕСТ: {'пройден' if ok else 'НЕ ПРОЙДЕН'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
