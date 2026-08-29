"""Значимость эффекта стиринга на TruthfulQA. Локально, без GPU.

Свип по α на новом векторе дал «лучшую» точку с выигрышем 5.8–6.2 п.п., но
кривая по α немонотонна: база лучше большинства ненулевых α, а значения скачут.
Это ровно тот случай, когда без теста значимости утверждать нечего.

Тесты **парные**: каждая α измерялась на одних и тех же вопросах, поэтому
сравнение по парам на порядок мощнее сравнения средних.

* непрерывная шкала — критерий Уилкоксона по разностям `score(α) − score(0)`
  (без предположения о нормальности, разности MC2 тяжелохвостые);
* бинарная шкала (ошибка / не ошибка) — критерий Макнемара по вопросам, где
  исход при α и при нуле различается.

Отдельно считается поправка на множественность: α перебиралось семь раз, и
«лучшая» точка выбрана постфактум. Без поправки это классический способ найти
эффект там, где его нет.

Запуск:
    uv run python scripts/18_truth_significance.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering.paths import artifacts_dir, results_dir  # noqa: E402


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Точный критерий Макнемара для парных бинарных исходов."""
    from scipy.stats import binomtest

    only_a = int(np.sum(a & ~b))   # ошибка только при α
    only_b = int(np.sum(~a & b))   # ошибка только в базе
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    return only_a, only_b, float(binomtest(only_a, n, 0.5).pvalue)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=None)
    args = parser.parse_args()

    from scipy.stats import wilcoxon

    files = sorted(artifacts_dir("truth_inform").glob("*_per_question.npz"))
    if not files:
        raise SystemExit("нет артефакта с поединичными значениями — перезапусти 16")
    path = Path(args.artifact) if args.artifact else files[-1]
    data = np.load(path)
    alphas = data["alphas"].tolist()
    mc2_all, mc2_com = data["mc2_all"], data["mc2_committal"]
    zero = alphas.index(0.0)
    n = mc2_all.shape[1]
    print(f"артефакт: {path.name} | вопросов {n} | α: {alphas}\n")

    print("=== парные тесты против α = 0 ===")
    print(f"{'α':>6} {'Δ score (содерж.)':>18} {'Уилкоксон p':>13} "
          f"{'Δ ошибок':>10} {'Макнемар p':>12}")
    report = {}
    for i, alpha in enumerate(alphas):
        if alpha == 0.0:
            continue
        delta = mc2_com[i] - mc2_com[zero]
        w_p = float(wilcoxon(delta).pvalue) if np.any(delta != 0) else 1.0
        err_a = mc2_com[i] < 0
        err_0 = mc2_com[zero] < 0
        only_a, only_0, m_p = mcnemar(err_a, err_0)
        d_err = float(err_a.mean() - err_0.mean())
        report[str(alpha)] = {"delta_score": float(delta.mean()),
                              "wilcoxon_p": w_p, "delta_error": d_err,
                              "mcnemar_p": m_p, "discordant": only_a + only_0}
        print(f"{alpha:6.1f} {delta.mean():18.4f} {w_p:13.4f} "
              f"{d_err:+9.1%} {m_p:12.4f}")

    tested = len(report)
    best = min(report, key=lambda a: report[a]["delta_error"])
    b = report[best]
    bonferroni = min(1.0, b["mcnemar_p"] * tested)
    print(f"\n=== лучшая точка: α = {best} ===")
    print(f"  сдвиг ошибок: {b['delta_error']:+.1%}, Макнемар p = {b['mcnemar_p']:.4f}")
    print(f"  проверено α: {tested}, поправка Бонферрони -> p = {bonferroni:.4f}")
    verdict = bonferroni < 0.05
    print(f"\n  эффект {'ЗНАЧИМ' if verdict else 'НЕ ЗНАЧИМ'} после поправки "
          f"на множественность")
    if not verdict:
        print("  -> \"лучшая α\" выбрана постфактум из семи, и различие с базой")
        print("     объясняется перебором, а не управлением концептом")

    # Насколько вообще упорядочены значения по α — при настоящем управлении
    # ожидалась бы монотонность, как у вектора из azaria-mitchell (ρ = -0.96)
    means = [float(mc2_com[i].mean()) for i in range(len(alphas))]
    rho = float(np.corrcoef(np.argsort(np.argsort(alphas)),
                            np.argsort(np.argsort(means)))[0, 1])
    print(f"\n  монотонность score по α: ρ = {rho:+.3f}")

    out = results_dir() / "18_truth_significance.json"
    out.write_text(json.dumps({"artifact": path.name, "n": int(n),
                               "by_alpha": report, "best_alpha": best,
                               "bonferroni_p": bonferroni, "significant": verdict,
                               "monotonicity": rho}, indent=2, ensure_ascii=False),
                   "utf-8")
    print(f"\nОтчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
