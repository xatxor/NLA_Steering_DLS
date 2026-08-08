"""Этап 4: как NLA описывает центроиды и сам steering-вектор.

Это развилка проекта, а не рядовая проверка. В постановке она сформулирована
прямо: steering-вектор — это **разность активаций, а не активация**, и для NLA
он может оказаться out-of-distribution. Если так, весь замысел «гейтить по
близости к описанию концепта» надо строить иначе.

Что уже известно из этапа 3 и делает вопрос острым: косинус между центроидами
классов **0.9988**, а норма вектора 22.4 против типичной нормы активации ~95.
То есть различие между «вредным» и «безопасным» — тонкая добавка поверх
огромной общей компоненты, и предсказание такое: **NLA опишет оба центроида
практически одинаково**. Проверяем.

Условия прогона:
  * два центроида по отдельности;
  * сам вектор, как есть и отмасштабированный до типичной нормы активации
    (норма 22.4 сама по себе делает его OOD, и это надо развести с направлением);
  * безопасный центроид плюс α·v для нескольких α — пункт 5 постановки, «как
    NLA описывает разницу до и после добавления concept vector»;
  * отдельные активации, не усреднённые, как точка отсчёта.

Количественная мера OOD — reconstruction score: гоняем `av → ar` и меряем
косинус с исходным вектором. Для настоящих активаций этап 3 дал 0.77 при 0.52
у «чужих» пар. Если для `v` выйдет около случайного уровня, это и есть OOD,
измеренный, а не додуманный.

Запуск:
    python scripts/05_describe_vector.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering import hf  # noqa: E402
from nla_steering.nla import Reconstructor, Verbalizer, cosine  # noqa: E402
from nla_steering.paths import artifacts_dir  # noqa: E402

ALPHAS = [1.0, 2.0, 4.0]


def build_conditions(cfg: dict, path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Собрать все векторы-кандидаты из сырых активаций этапа 3."""
    data = np.load(path)
    positions = data["positions"].tolist()
    j = positions.index(cfg["steering"]["extraction_position"])
    acts, labels = data["jbb_acts"][:, j], data["jbb_labels"]

    harmful, harmless = acts[labels == 1], acts[labels == 0]
    c_harmful, c_harmless = harmful.mean(0), harmless.mean(0)
    v = c_harmful - c_harmless
    typical = float(np.linalg.norm(acts, axis=1).mean())

    conditions = {
        "centroid_harmful": c_harmful,
        "centroid_harmless": c_harmless,
        "vector_raw": v,
        # норма вектора вчетверо меньше типичной активации; масштабируем, чтобы
        # отделить эффект направления от эффекта величины
        "vector_scaled": v / np.linalg.norm(v) * typical,
    }
    for alpha in ALPHAS:
        conditions[f"harmless+{alpha:g}v"] = c_harmless + alpha * v
    for i in range(2):
        conditions[f"sample_harmful_{i}"] = harmful[i]
        conditions[f"sample_harmless_{i}"] = harmless[i]

    meta = {
        "position": cfg["steering"]["extraction_position"],
        "typical_activation_norm": typical,
        "vector_norm": float(np.linalg.norm(v)),
        "centroid_cosine": float(cosine(c_harmful, c_harmless)[0]),
        "norms": {k: float(np.linalg.norm(x)) for k, x in conditions.items()},
    }
    return conditions, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--artifact", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=70)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text("utf-8"))
    token = hf.token()

    path = Path(args.artifact) if args.artifact else sorted(
        artifacts_dir("vector").glob("*.npz"))[-1]
    conditions, meta = build_conditions(cfg, path)
    names = list(conditions)
    matrix = np.stack([conditions[n] for n in names])

    print(f"=== источник: {path.name}, позиция {meta['position']} ===")
    print(f"  типичная норма активации : {meta['typical_activation_norm']:.2f}")
    print(f"  норма вектора            : {meta['vector_norm']:.2f}")
    print(f"  косинус центроидов       : {meta['centroid_cosine']:.4f}")

    with Verbalizer(cfg, token) as av:
        explanations = av.verbalize(matrix, max_new_tokens=args.max_new_tokens)
    with Reconstructor(cfg, token) as ar:
        recon = ar.reconstruct(explanations)

    scores = cosine(matrix, recon)

    print("\n=== объяснения и reconstruction score ===")
    print("(ориентир из этапа 03: свои пары 0.77, чужие 0.52)")
    results = {}
    for name, text, score in zip(names, explanations, scores):
        results[name] = {"explanation": text, "reconstruction_cosine": float(score),
                         "norm": meta["norms"][name]}
        print(f"\n--- {name}   |v|={meta['norms'][name]:.1f}   recon={score:.4f}")
        print(f"    {text[:340]}")

    # Прямая проверка предсказания: одинаково ли NLA описывает два центроида.
    pair = cosine(recon[names.index("centroid_harmful")],
                  recon[names.index("centroid_harmless")])[0]
    print(f"\n=== косинус реконструкций двух центроидов: {pair:.4f} ===")

    out = artifacts_dir("describe") / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "results": results,
                               "centroid_reconstruction_cosine": float(pair)},
                              indent=2, ensure_ascii=False), "utf-8")
    print(f"\nОтчёт на Drive: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
