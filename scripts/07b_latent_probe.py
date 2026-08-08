"""Есть ли информация об отказе в латенте NLA? Локально, без GPU.

`07_gating_signals.py` дал nla_latent AUROC 0.61 против 0.92 у проекции на `v`.
Вывод «NLA проигрывает» из этого делать рано: сигнал там читался как косинус до
**усреднённой** опоры, а этап 4 уже показал, что усреднение стирает различие.
Слабым может быть ридаут, а не сам латент.

Здесь вопрос ставится прямо: обучаем линейный классификатор поверх латентов и
сравниваем с таким же классификатором поверх сырых активаций слоя 24. Если
латент даёт сопоставимое качество — информация в нём есть, и надо менять способ
считывания. Если заметно хуже — NLA действительно теряет нужное.

**Кросс-валидация по промптам, а не по токенам.** Метка постоянна внутри
ответа, а активации соседних токенов сильно скоррелированы: разбиение по
токенам протащило бы один и тот же ответ в train и в test и завысило бы всё.

Запуск:
    uv run python scripts/07b_latent_probe.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering.paths import artifacts_dir, results_dir  # noqa: E402


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def grouped_cv(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
               folds: int, seed: int) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    chunks = np.array_split(rng.permutation(unique), folds)

    scores = []
    for k in range(folds):
        test_groups = set(chunks[k].tolist())
        test = np.array([g in test_groups for g in groups])
        scaler = StandardScaler().fit(features[~test])
        model = LogisticRegression(max_iter=3000, C=0.01)
        model.fit(scaler.transform(features[~test]), labels[~test])
        scores.append(auroc(model.decision_function(scaler.transform(features[test])),
                            labels[test]))
    return float(np.mean(scores)), float(np.std(scores))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sig_path = sorted(artifacts_dir("signals").glob("*.npz"))[-1]
    gen_path = sorted(artifacts_dir("generation").glob("*.npz"))[-1]
    sig, gen = np.load(sig_path), np.load(gen_path)
    if "token_latents" not in sig:
        raise SystemExit(f"{sig_path.name} без латентов — перезапусти 07")

    labels, meta = sig["labels"], sig["meta"]
    groups = meta[:, 0]  # индекс промпта
    latents, acts = sig["token_latents"], gen["acts"]
    print(f"сигналы: {sig_path.name}\nгенерация: {gen_path.name}")
    print(f"токенов {len(labels)}, промптов {len(np.unique(groups))}, "
          f"из отказов {labels.mean():.1%}\n")

    print(f"=== AUROC, {args.folds}-fold CV с разбиением по промптам ===")
    results = {}

    for name, features in (("активация слоя 24", acts), ("латент av", latents)):
        mean, std = grouped_cv(features, labels, groups, args.folds, args.seed)
        results[name] = {"auroc_mean": mean, "auroc_std": std}
        print(f"  probe / {name:20} {mean:.4f} ± {std:.4f}")

    # Проекция обучения не требует, но считаем её на тех же данных для сравнения.
    proj = auroc(sig["signal_cosine"], labels)
    latent_cos = auroc(sig["signal_nla_latent"], labels)
    results["cosine к v (без обучения)"] = {"auroc": proj}
    results["косинус к опорному латенту"] = {"auroc": latent_cos}
    print(f"  cosine к v (без обучения)      {proj:.4f}")
    print(f"  косинус к опорному латенту     {latent_cos:.4f}")

    # Временной профиль: на каких шагах генерации сигнал сильнее всего.
    print("\n=== профиль по шагу генерации (средний cosine к v) ===")
    steps = meta[:, 1]
    profile = {}
    for lo, hi in ((0, 1), (1, 4), (4, 8), (8, 16), (16, 32), (32, 48)):
        mask = (steps >= lo) & (steps < hi)
        if not mask.any():
            continue
        ref = float(sig["signal_cosine"][mask & (labels == 1)].mean())
        non = float(sig["signal_cosine"][mask & (labels == 0)].mean())
        profile[f"{lo}-{hi}"] = {"refusal": ref, "compliant": non, "gap": ref - non}
        print(f"  шаги {lo:>2}–{hi:<2}  отказ {ref:+.4f}  согласие {non:+.4f}  "
              f"разрыв {ref - non:+.4f}")

    out = results_dir() / "07b_latent_probe.json"
    out.write_text(json.dumps({"signals": sig_path.name, "generation": gen_path.name,
                               "folds": args.folds, "auroc": results,
                               "step_profile": profile}, indent=2, ensure_ascii=False),
                   "utf-8")
    print(f"\nОтчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
