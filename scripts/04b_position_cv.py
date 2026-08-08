"""Выбор позиции извлечения вектора по кросс-валидации. Локально, без GPU.

Зачем отдельный шаг. В `04_steering_vector.py` позиция выбиралась по одному
val-сплиту из 60 примеров, и победила `-3` с AUROC 0.9665 против 0.9431 у `-1`.
На такой выборке разница в 0.02 — шум, а решение влияет на всё дальнейшее.
Вдобавок `-3` это токен `<|im_start|>`, служебный и одинаковый во всех промптах:
его `|v|` вчетверо меньше остальных (4.37 против 22.5), то есть направление
собрано из тонкой вариации, наведённой контекстом через attention. Такое может
не пережить перехода к генерации.

Поэтому позиция выбирается 5-fold кросс-валидацией на JBB, а **XSTest не
участвует в выборе** — он должен остаться честно held-out для этапа 8.

Кроме AUROC меряется устойчивость направления: косинус между векторами,
посчитанными на разных фолдах. Позиция, дающая высокий AUROC, но разное
направление от фолда к фолду, для стиринга не годится.

Читает артефакт `04_steering_vector.py` с Drive.

Запуск:
    uv run python scripts/04b_position_cv.py
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering.paths import artifacts_dir, results_dir  # noqa: E402

#: расшифровка позиций в хвосте промпта Qwen2.5 (см. tokenizer)
POSITION_TOKENS = {-1: "\\n", -2: "assistant", -3: "<|im_start|>", -5: "<|im_end|>"}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def latest_artifact() -> Path:
    files = sorted(artifacts_dir("vector").glob("*.npz"))
    if not files:
        raise SystemExit("нет артефактов в artifacts/vector — сначала запусти 04")
    return files[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.artifact) if args.artifact else latest_artifact()
    data = np.load(path)
    acts, labels = data["jbb_acts"], data["jbb_labels"]
    xs_acts, xs_labels = data["xstest_acts"], data["xstest_labels"]
    positions = data["positions"].tolist()
    print(f"артефакт: {path.name}")
    print(f"JBB {acts.shape}, XSTest {xs_acts.shape}\n")

    rng = np.random.default_rng(args.seed)
    folds = np.array_split(rng.permutation(len(labels)), args.folds)

    print(f"{'поз':>4} {'токен':<14} {'|v|':>8} {'AUROC cv':>18} {'устойчивость':>13} "
          f"{'XSTest':>8}")
    report = {}
    for j, pos in enumerate(positions):
        a = acts[:, j]
        aucs, directions = [], []
        for k in range(args.folds):
            test = folds[k]
            train = np.concatenate([folds[m] for m in range(args.folds) if m != k])
            v = a[train][labels[train] == 1].mean(0) - a[train][labels[train] == 0].mean(0)
            unit = v / np.linalg.norm(v)
            aucs.append(auroc(a[test] @ unit, labels[test]))
            directions.append(unit)

        # Устойчивость направления: средний косинус между векторами фолдов.
        stability = float(np.mean([d1 @ d2 for d1, d2 in combinations(directions, 2)]))

        full = a[labels == 1].mean(0) - a[labels == 0].mean(0)
        norm = float(np.linalg.norm(full))
        unit_full = full / norm
        xs_auc = auroc(xs_acts[:, j] @ unit_full, xs_labels)

        report[str(pos)] = {"token": POSITION_TOKENS.get(pos, "?"), "norm": norm,
                            "auroc_cv_mean": float(np.mean(aucs)),
                            "auroc_cv_std": float(np.std(aucs)),
                            "direction_stability": stability,
                            "auroc_xstest": xs_auc}
        print(f"{pos:>4} {POSITION_TOKENS.get(pos, '?'):<14} {norm:8.3f} "
              f"{np.mean(aucs):9.4f} ± {np.std(aucs):.4f} {stability:13.4f} {xs_auc:8.4f}")

    best = max(report, key=lambda p: report[p]["auroc_cv_mean"])
    print(f"\n  ВЫБРАНА ПОЗИЦИЯ {best} ({report[best]['token']})")
    print(f"  AUROC cv {report[best]['auroc_cv_mean']:.4f} ± {report[best]['auroc_cv_std']:.4f}, "
          f"устойчивость {report[best]['direction_stability']:.4f}")
    print("  XSTest показан справочно и в выборе не участвовал — он held-out.")

    out = results_dir() / "04b_position_cv.json"
    out.write_text(json.dumps({"artifact": path.name, "folds": args.folds,
                               "best_position": int(best), "by_position": report},
                              indent=2, ensure_ascii=False), "utf-8")
    print(f"\nОтчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
