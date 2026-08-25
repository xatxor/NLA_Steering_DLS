"""Вектор правдивости и проверка его переноса на TruthfulQA.

Первый шаг работы с truthfulness. Отдельный скрипт, потому что это ранняя точка
отсечения: если направление не выделяется или не переносится, всю остальную
цепочку строить бессмысленно.

**Контрастные пары** — `notrichardren/azaria-mitchell`: 17 тысяч утверждений с
меткой истинности из 12 доменов (города, столицы, компании, элементы, факты и
отрицания). Выборка стратифицируется по доменам, иначе вектор выучит тематику
самого крупного из них (`cities`, 44% датасета), а не истинность.

**Два формата извлечения, и это не формальность.** Вектор будет применяться на
токенах, которые модель порождает сама, поэтому извлекать его из голых
предложений рискованно — распределение активаций другое.

* `bare` — просто текст утверждения, последний токен. Так делают в литературе.
* `assistant` — утверждение как продолжение ответа ассистента, то есть ровно та
  позиция, в которой вектор потом будет применяться.

**Перенос проверяется на настоящей задаче, а не на самих утверждениях.** Для
817 вопросов TruthfulQA берётся активация последнего токена промпта, проекция
на вектор сравнивается с меткой «модель предпочитает неверный ответ» (MC2,
`score(0) < 0` из эксперимента 10d). AUROC на утверждениях может быть высоким
при нулевом переносе — важен именно второй показатель.

Запуск:
    python scripts/14_truth_vector.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering import hf  # noqa: E402
from nla_steering.nla import free, load_model  # noqa: E402
from nla_steering.paths import artifacts_dir, results_dir  # noqa: E402
from nla_steering.progress import beat  # noqa: E402
from nla_steering.scoring import auroc  # noqa: E402

CLAIMS = "notrichardren/azaria-mitchell"
CLAIMS_FILE = "data/combined-00000-of-00001-823614a3dac99c50.parquet"
TQA = "truthfulqa/truthful_qa"
TQA_FILE = "generation/validation-00000-of-00001.parquet"
TQA_SYSTEM = "Answer the question directly and concisely."
#: нейтральная затравка для формата assistant — важно, чтобы она не намекала
#: на истинность или ложность продолжения
ASSISTANT_PROMPT = "Tell me a fact."


def stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Поровну из каждого домена и поровну по меткам."""
    rng = np.random.default_rng(seed)
    per_domain = max(2, n // (df["dataset"].nunique() * 2))
    parts = []
    for _, group in df.groupby("dataset"):
        for label in (0, 1):
            sub = group[group.label == label]
            take = min(per_domain, len(sub))
            if take:
                parts.append(sub.iloc[rng.permutation(len(sub))[:take]])
    out = pd.concat(parts, ignore_index=True)
    return out.iloc[rng.permutation(len(out))].reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n-claims", type=int, default=1200)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    cfg = yaml.safe_load(Path(args.config).read_text("utf-8"))
    token = hf.token()
    index = cfg["nla"]["hidden_state_index"]

    beat("truth_vector", "старт")
    claims = pd.read_parquet(hf_hub_download(CLAIMS, CLAIMS_FILE, repo_type="dataset",
                                             token=token))
    claims = stratified(claims, args.n_claims, args.seed)
    tqa = pd.read_parquet(hf_hub_download(TQA, TQA_FILE, repo_type="dataset",
                                          token=token))
    print(f"=== утверждений {len(claims)} из {claims['dataset'].nunique()} доменов, "
          f"истинных {int(claims.label.sum())} ===", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], token=token)
    beat("truth_vector", "гружу веса")
    model = load_model(cfg["base_model"], token)
    model.eval()
    print("веса загружены", flush=True)

    assistant_prefix = tok.apply_chat_template(
        [{"role": "user", "content": ASSISTANT_PROMPT}],
        add_generation_prompt=True, tokenize=False)

    def activation(text: str, fmt: str) -> np.ndarray:
        """Активация слоя 24 на последнем токене утверждения."""
        full = text if fmt == "bare" else assistant_prefix + text
        ids = tok(full, return_tensors="pt", add_special_tokens=(fmt == "bare"))
        ids = {k: v.to(model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        return out.hidden_states[index][0, -1].float().cpu().numpy()

    acts = {}
    for fmt in ("bare", "assistant"):
        rows = []
        for i, text in enumerate(claims["claim"]):
            rows.append(activation(text, fmt))
            if (i + 1) % 200 == 0:
                beat("truth_vector", f"{fmt}: {i + 1}/{len(claims)}")
        acts[fmt] = np.stack(rows)
        print(f"  {fmt}: снято {len(rows)}", flush=True)

    labels = claims["label"].to_numpy().astype(int)

    # Активации промптов TruthfulQA — на них проверяется перенос
    beat("truth_vector", "активации TruthfulQA")
    tqa_acts = []
    for question in tqa["question"]:
        enc = tok.apply_chat_template(
            [{"role": "system", "content": TQA_SYSTEM},
             {"role": "user", "content": question}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        tqa_acts.append(out.hidden_states[index][0, -1].float().cpu().numpy())
    tqa_acts = np.stack(tqa_acts)
    free(model)

    # Метка ошибки из эксперимента 10d: score(0) < 0, то есть модель уже по
    # одному вопросу предпочитает неверный ответ
    tqa_npz = np.load(sorted(artifacts_dir("truthfulness").glob("*_ll.npz"))[-1])
    tqa_errors = tqa_npz["errors"].astype(int)
    if len(tqa_errors) != len(tqa_acts):
        raise SystemExit(f"метки 10d ({len(tqa_errors)}) не совпадают с вопросами "
                         f"({len(tqa_acts)}) — перезапусти 10c на всех 817")

    print(f"\n=== качество направления, {args.folds}-fold CV по доменам ===")
    print(f"{'формат':12} {'AUROC на утвержд.':>18} {'устойчивость':>13} "
          f"{'AUROC на TQA':>14}")
    report, vectors = {}, {}
    rng = np.random.default_rng(args.seed)
    domains = claims["dataset"].to_numpy()
    unique_domains = np.unique(domains)

    for fmt, matrix in acts.items():
        # Разбиение по ДОМЕНАМ, а не по утверждениям: иначе «Yato is a city in
        # Japan» и «Kyoto is a city in Japan» попадут в train и test сразу,
        # и AUROC покажет запоминание темы, а не обобщение истинности.
        chunks = np.array_split(rng.permutation(unique_domains), args.folds)
        aucs, directions = [], []
        for k in range(args.folds):
            test = np.isin(domains, chunks[k])
            if test.all() or not test.any():
                continue
            train = ~test
            v = matrix[train][labels[train] == 1].mean(0) - \
                matrix[train][labels[train] == 0].mean(0)
            unit = v / np.linalg.norm(v)
            aucs.append(auroc(matrix[test] @ unit, labels[test]))
            directions.append(unit)

        full = matrix[labels == 1].mean(0) - matrix[labels == 0].mean(0)
        unit_full = full / np.linalg.norm(full)
        vectors[fmt] = unit_full
        stability = float(np.mean([a @ b for a, b in combinations(directions, 2)]))
        # знак: вектор указывает в сторону ИСТИНЫ, значит у вопросов, где модель
        # ошибается, проекция должна быть ниже — метку инвертируем
        tqa_auc = auroc(-(tqa_acts @ unit_full), tqa_errors)
        report[fmt] = {"auroc_claims": float(np.mean(aucs)),
                       "auroc_claims_std": float(np.std(aucs)),
                       "direction_stability": stability,
                       "auroc_tqa_transfer": tqa_auc,
                       "norm": float(np.linalg.norm(full))}
        print(f"{fmt:12} {np.mean(aucs):11.4f} ± {np.std(aucs):.3f} "
              f"{stability:13.4f} {tqa_auc:14.4f}")

    best = max(report, key=lambda f: report[f]["auroc_tqa_transfer"])
    print(f"\n  выбран формат: {best} (перенос AUROC {report[best]['auroc_tqa_transfer']:.4f})")
    print(f"  косинус между векторами двух форматов: "
          f"{float(vectors['bare'] @ vectors['assistant']):.4f}")
    print(f"\n  доля вопросов TQA, где модель ошибается: {tqa_errors.mean():.1%}")
    print("  напоминание: 0.5 на переносе означает, что направление на этой"
          " задаче бесполезно, каким бы ни был AUROC на утверждениях")

    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out_dir = artifacts_dir("truth_vector")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{stamp}.npz",
                        vector_bare=vectors["bare"],
                        vector_assistant=vectors["assistant"],
                        best_format=np.array(best),
                        claim_acts_assistant=acts["assistant"],
                        claim_labels=labels, claim_domains=domains,
                        tqa_acts=tqa_acts, tqa_errors=tqa_errors)
    results_dir().mkdir(parents=True, exist_ok=True)
    (results_dir() / "14_truth_vector.json").write_text(json.dumps(
        {"n_claims": len(claims), "by_format": report, "best": best,
         "cosine_between_formats": float(vectors["bare"] @ vectors["assistant"]),
         "tqa_error_rate": float(tqa_errors.mean())}, indent=2, ensure_ascii=False),
        "utf-8")
    print(f"\nАртефакт на Drive: {out_dir / (stamp + '.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
