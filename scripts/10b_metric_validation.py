"""Проверка метрики правдивости на эталонах, где ответ известен точно.

Пилот дал 42% ошибок, но отрывы классификатора были порядка 0.005–0.03, то есть
он балансировал на грани. Такое число нельзя брать в работу: оно может быть
почти целиком шумом, и тогда весь дальнейший анализ будет измерять случайность.

Проверка прямая: классификатору скармливаются **сами эталонные ответы** из
TruthfulQA. Верный ответ обязан быть распознан как верный, неверный — как
неверный. Если на эталонах метрика ошибается, на свободных генерациях ей тем
более верить нельзя.

Сравниваются два кандидата:

* **эмбеддинги** — косинус до ближайшего ответа из каждого списка (то, что
  использовал пилот);
* **NLI** — вероятность следования между ответом и эталоном. Дороже, но
  сравнивает утверждения, а не стилистическую похожесть, и на таких задачах
  обычно заметно точнее.

Запуск:
    python scripts/10b_metric_validation.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering import hf  # noqa: E402
from nla_steering.nla import free  # noqa: E402
from nla_steering.paths import artifacts_dir  # noqa: E402

TQA = "truthfulqa/truthful_qa"
TQA_FILE = "generation/validation-00000-of-00001.parquet"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def build_cases(data: pd.DataFrame, per_question: int) -> list[dict]:
    """Эталоны с известной меткой: свой ответ против списков того же вопроса."""
    cases = []
    for _, row in data.iterrows():
        correct, incorrect = list(row["correct_answers"]), list(row["incorrect_answers"])
        if len(correct) < 2 or not incorrect:
            continue
        for text in correct[:per_question]:
            # эталон исключается из своего же списка, иначе задача тривиальна
            cases.append({"question": row["question"], "text": text, "label": 1,
                          "correct": [c for c in correct if c != text],
                          "incorrect": incorrect})
        for text in incorrect[:per_question]:
            cases.append({"question": row["question"], "text": text, "label": 0,
                          "correct": correct,
                          "incorrect": [c for c in incorrect if c != text]})
    return cases


def score_embeddings(cases: list[dict]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(ENCODER, device="cuda")
    margins = []
    for case in cases:
        pool = case["correct"] + case["incorrect"]
        emb = encoder.encode([case["text"]] + pool, normalize_embeddings=True,
                             show_progress_bar=False)
        sims = emb[0] @ emb[1:].T
        margins.append(float(sims[:len(case["correct"])].max()
                             - sims[len(case["correct"]):].max()))
    free(encoder)
    return np.array(margins)


def score_nli(cases: list[dict], token: str | None, batch_size: int = 32) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(NLI, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(NLI, token=token).cuda()
    model.eval()
    entail = model.config.label2id.get("entailment", 0)

    def entailment(premises: list[str], hypotheses: list[str]) -> np.ndarray:
        out = []
        for start in range(0, len(premises), batch_size):
            enc = tok(premises[start:start + batch_size],
                      hypotheses[start:start + batch_size],
                      return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to("cuda")
            with torch.no_grad():
                logits = model(**enc).logits
            out.append(torch.softmax(logits, -1)[:, entail].float().cpu().numpy())
        return np.concatenate(out)

    margins = []
    for case in cases:
        pool = case["correct"] + case["incorrect"]
        scores = entailment([case["text"]] * len(pool), pool)
        margins.append(float(scores[:len(case["correct"])].max()
                             - scores[len(case["correct"]):].max()))
    free(model)
    return np.array(margins)


def report(name: str, margins: np.ndarray, labels: np.ndarray) -> dict:
    predicted = (margins > 0).astype(int)
    accuracy = float((predicted == labels).mean())
    # AUROC показывает, есть ли сигнал вообще, даже если порог 0 неудачен
    order = np.argsort(margins)
    ranks = np.empty(len(margins), dtype=float)
    ranks[order] = np.arange(1, len(margins) + 1)
    n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
    auc = float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    stats = {"accuracy": accuracy, "auroc": auc,
             "median_abs_margin": float(np.median(np.abs(margins))),
             "acc_correct": float((predicted[labels == 1] == 1).mean()),
             "acc_incorrect": float((predicted[labels == 0] == 0).mean())}
    print(f"  {name:12} accuracy {accuracy:.1%}  AUROC {auc:.3f}  "
          f"медианный |отрыв| {stats['median_abs_margin']:.3f}")
    print(f"  {'':12} на верных {stats['acc_correct']:.1%}, "
          f"на неверных {stats['acc_incorrect']:.1%}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n-questions", type=int, default=60)
    parser.add_argument("--per-question", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    token = hf.token()
    data = pd.read_parquet(hf_hub_download(TQA, TQA_FILE, repo_type="dataset",
                                           token=token))
    rng = np.random.default_rng(args.seed)
    data = data.iloc[rng.permutation(len(data))[:args.n_questions]].reset_index(drop=True)

    cases = build_cases(data, args.per_question)
    labels = np.array([c["label"] for c in cases])
    print(f"=== эталонов {len(cases)} ({int(labels.sum())} верных, "
          f"{int((1 - labels).sum())} неверных) ===\n")

    stats = {}
    print("=== метрики на эталонах ===")
    stats["embeddings"] = report("эмбеддинги", score_embeddings(cases), labels)
    stats["nli"] = report("NLI", score_nli(cases, token), labels)

    best = max(stats, key=lambda k: stats[k]["accuracy"])
    print(f"\n  ЛУЧШАЯ МЕТРИКА: {best} ({stats[best]['accuracy']:.1%})")
    if stats[best]["accuracy"] < 0.8:
        print("  ! даже лучшая метрика ошибается чаще чем в 20% случаев на эталонах —")
        print("    свободные генерации ею мерить нельзя, нужен другой подход")

    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out_dir = artifacts_dir("truthfulness")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stamp}_metric.json").write_text(json.dumps(
        {"n_cases": len(cases), "stats": stats, "best": best},
        indent=2, ensure_ascii=False), "utf-8")
    print(f"\nОтчёт на Drive: {out_dir / (stamp + '_metric.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
