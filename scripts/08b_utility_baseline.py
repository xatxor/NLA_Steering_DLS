"""Вычесть базовый уровень из карты полезности. Локально, без GPU.

Метрика «отказ по хвосту после стирённой позиции» вводилась, чтобы отделить
«представление отказа там уже не живёт» от «поздно вмешиваться: префикс отказа
уже выдан». Но сама по себе она интерпретируется неверно: маркеры отказа стоят
в начале ответа, поэтому хвост после позиции 20 не содержит их и **без всякого
стиринга**. Без базового уровня высокие числа на поздних позициях выглядят
эффектом, которого нет.

Здесь базовый уровень считается из сохранённых нестирённых ответов и
вычитается. Генерации не требуется.

Запуск:
    uv run python scripts/08b_utility_baseline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering import hf  # noqa: E402
from nla_steering.metrics import is_refusal  # noqa: E402
from nla_steering.paths import artifacts_dir, results_dir  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    path = Path(args.artifact) if args.artifact else sorted(
        artifacts_dir("utility").glob("*.json"))[-1]
    data = json.loads(path.read_text("utf-8"))
    positions = data["positions"]
    print(f"артефакт: {path.name}, α = {data['alpha']}, "
          f"промптов {data['n_prompts']}\n")

    tok = AutoTokenizer.from_pretrained(args.base_model, token=hf.token())
    baseline = np.zeros((len(data["baselines"]), positions))
    for i, text in enumerate(data["baselines"]):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        for pos in range(positions):
            baseline[i, pos] = 0 if is_refusal(tok.decode(ids[pos + 1:])) else 1
    base = baseline.mean(0)

    full = np.array(data["by_position"])
    suffix = np.array(data["by_position_suffix"])
    lift = suffix - base

    print(f"{'поз':>4} {'полный ответ':>13} {'хвост':>8} {'хвост без стиринга':>20} "
          f"{'прирост':>9}")
    for pos in range(positions):
        bar = "#" * int(max(lift[pos], 0) * 40)
        print(f"{pos:>4} {full[pos]:13.0%} {suffix[pos]:8.0%} {base[pos]:20.0%} "
              f"{lift[pos]:+9.0%} {bar}")

    effective = [p for p in range(positions) if lift[p] > 0.15]
    print(f"\n  позиции с реальным эффектом: {effective}")
    print(f"  на остальных прирост в пределах ±5% — то есть его нет")
    print(f"\n  стиринг только позиции 0: {full[0]:.0%} исправленных")
    print(f"  all-token при той же α:   {data['all_token_rate']:.0%} "
          f"(бюджет в {positions} раз больше)")
    print(f"  random gating, 1 токен:   {data['random_gating_1token']:.0%}")

    out = results_dir() / "08b_utility_baseline.json"
    out.write_text(json.dumps(
        {"artifact": path.name, "alpha": data["alpha"],
         "by_position_full": full.tolist(), "by_position_suffix": suffix.tolist(),
         "suffix_baseline": base.tolist(), "lift": lift.tolist(),
         "effective_positions": effective}, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nОтчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
