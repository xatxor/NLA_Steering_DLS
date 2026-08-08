"""Этап 5, часть 2: четыре гейтинг-сигнала на токенах генерации.

Считает и сравнивает кандидатов в гейт:

* **cosine** — проекция активации на `v`. Это CAST, прямой конкурент, и планка
  здесь высокая: на промптах он даёт AUROC 0.95.
* **probe** — логистическая регрессия по активациям.
* **nla_latent** — внутренняя сводка `av` на последней позиции промпта, один
  forward на токен. Ради этого варианта проект и затевался: текстовая
  вербализация стоит целой генерации на каждый токен, латент — одного прохода.
* (текстовый сигнал считается отдельно, `08_text_signal.py`: он на два порядка
  дороже и его приходится считать на подвыборке.)

`v` и probe калибруются на активациях **промптов** (этап 3), а применяются к
активациям **генерации** — это и есть реалистичный режим, и заодно честная
проверка переноса между распределениями.

Опора для NLA-сигнала строится из отдельных активаций с высокой проекцией на
`v`, а не из центроида: этап 4 показал, что усреднение стирает различие.

Оценка: AUROC против метки «этот токен из ответа-отказа». Это ещё не карта
полезности стиринга (она на этапе 6), но ближайший честный прокси к вопросу
«думает ли модель об отказе на этом токене».

Отдельно считается корреляция сигналов между собой. Если `nla_latent` совпадает
с проекцией, значит NLA ничего не добавляет, и это тоже результат.

Запуск:
    python scripts/07_gating_signals.py --config configs/base.yaml
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
from nla_steering.nla import Verbalizer, cosine  # noqa: E402
from nla_steering.paths import artifacts_dir, results_dir  # noqa: E402


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))


def latest(sub: str, pattern: str = "*.npz") -> Path:
    files = sorted(artifacts_dir(sub).glob(pattern))
    if not files:
        raise SystemExit(f"нет артефактов в artifacts/{sub}")
    return files[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--generation", default=None)
    parser.add_argument("--vector", default=None)
    parser.add_argument("--reference-k", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text("utf-8"))
    token = hf.token()

    gen_path = Path(args.generation) if args.generation else latest("generation")
    vec_path = Path(args.vector) if args.vector else latest("vector")
    gen, vec = np.load(gen_path), np.load(vec_path)
    print(f"=== дамп {gen_path.name}, вектор {vec_path.name} ===")

    acts, meta, refusals = gen["acts"], gen["meta"], gen["refusals"]
    labels = refusals[meta[:, 0]].astype(int)  # токен из ответа-отказа
    print(f"  токенов {len(acts)}, из ответов-отказов {labels.mean():.1%}")

    # --- опора и вектор из активаций промптов (этап 3) ---
    j = vec["positions"].tolist().index(cfg["steering"]["extraction_position"])
    prompt_acts, prompt_labels = vec["jbb_acts"][:, j], vec["jbb_labels"]
    v = (prompt_acts[prompt_labels == 1].mean(0) - prompt_acts[prompt_labels == 0].mean(0))
    unit = v / np.linalg.norm(v)

    signals = {}
    signals["cosine"] = cosine(acts, np.broadcast_to(unit, acts.shape))
    signals["projection"] = acts @ unit

    from sklearn.linear_model import LogisticRegression

    probe = LogisticRegression(max_iter=2000, C=1.0)
    probe.fit(prompt_acts, prompt_labels)
    signals["probe"] = probe.decision_function(acts)
    print(f"  probe обучен на {len(prompt_acts)} промптах, "
          f"train acc {probe.score(prompt_acts, prompt_labels):.3f}")

    # Опора: активации промптов с самой высокой проекцией на v — хвост
    # распределения, не центроид (этап 4: усреднение стирает различие).
    top = np.argsort(prompt_acts @ unit)[-args.reference_k:]
    print(f"  опора: {args.reference_k} активаций с наибольшей проекцией "
          f"(из них вредных {int(prompt_labels[top].sum())})")

    with Verbalizer(cfg, token) as av:
        reference = av.latents(prompt_acts[top], batch_size=args.batch_size).mean(0)
        print("  опорный латент посчитан, считаю латенты токенов...")
        token_latents = av.latents(acts, batch_size=args.batch_size)
    signals["nla_latent"] = cosine(token_latents, np.broadcast_to(reference, token_latents.shape))

    print("\n=== AUROC против метки «токен из ответа-отказа» ===")
    report = {}
    for name, score in signals.items():
        value = auroc(score, labels)
        report[name] = {"auroc": value}
        print(f"  {name:12} {value:.4f}")

    print("\n=== корреляция сигналов (Spearman) ===")
    names = list(signals)
    print(f"{'':12}" + "".join(f"{n:>12}" for n in names))
    correlations = {}
    for a in names:
        row = [spearman(signals[a], signals[b]) for b in names]
        correlations[a] = dict(zip(names, row))
        print(f"{a:12}" + "".join(f"{x:12.3f}" for x in row))

    print("\n  Если nla_latent почти совпадает с projection, NLA ничего не"
          " добавляет — это тоже результат.")

    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out_dir = artifacts_dir("signals")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Латенты сохраняем целиком: косинус до усреднённой опоры — слабый ридаут,
    # и вопрос «есть ли информация в латенте» надо уметь проверять отдельно,
    # не повторяя прогон через av.
    np.savez_compressed(out_dir / f"{stamp}.npz", labels=labels, meta=meta,
                        reference_latent=reference, vector=unit,
                        token_latents=token_latents.astype(np.float32),
                        prompt_acts=prompt_acts, prompt_labels=prompt_labels,
                        **{f"signal_{k}": v for k, v in signals.items()})

    summary = {"generation": gen_path.name, "vector": vec_path.name,
               "n_tokens": int(len(acts)), "refusal_token_frac": float(labels.mean()),
               "auroc": report, "spearman": correlations}
    payload = json.dumps(summary, indent=2, ensure_ascii=False)
    # results_dir() в Colab живёт в эфемерном клоне репозитория и пропадёт
    # вместе с сессией — дублируем сводку на Drive рядом с артефактом.
    (out_dir / f"{stamp}.json").write_text(payload, "utf-8")
    results_dir().mkdir(parents=True, exist_ok=True)
    (results_dir() / "07_gating_signals.json").write_text(payload, "utf-8")
    print(f"\nАртефакт на Drive: {out_dir / (stamp + '.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
