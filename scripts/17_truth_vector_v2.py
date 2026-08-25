"""Плох ли вектор, или дело в самом концепте? Три проверки разом.

Вектор из `azaria-mitchell` даёт AUROC 0.876 на утверждениях (кросс-валидация по
доменам, устойчивость 0.985) и монотонный causal-эффект на MC2 — но в обратную
сторону. Гипотеза была, что он кодирует «уверенное фактическое утверждение», а
не истинность. Здесь она проверяется, и заодно проверяются два подозрения о том,
что вектор просто неудачный.

**Проверка 1 — интерпретация, напрямую.** Берём собственные ответы TruthfulQA
как продолжения ассистента и проецируем на текущий вектор. Если **неверные**
ответы (уверенные заблуждения) проецируются выше верных (часто уклончивых), то
«направление уверенного утверждения» — измеренный факт, а не догадка.

**Проверка 2 — источник контраста.** Строим направление из самой TruthfulQA:
верные ответы против неверных, разделение **по вопросам**, а не по ответам
(иначе варианты одного вопроса попадут в train и test сразу). Если такое
направление разделяет тест заметно лучше, значит дело было в разрыве
распределений, а не в концепте.

**Проверка 3 — слой.** Мы всё время работали на слое 24, потому что там живёт
NLA, но для truthfulness оптимальный слой никто не искал. Снимаем несколько
слоёв за один проход и смотрим, конкурентен ли 24-й. Если он заметно хуже
лучшего, это ограничение работы, которое надо назвать вслух.

Запуск:
    python scripts/17_truth_vector_v2.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
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

TQA = "truthfulqa/truthful_qa"
TQA_FILE = "generation/validation-00000-of-00001.parquet"
SYSTEM = "Answer the question directly and concisely."
LAYERS = [12, 18, 24, 30, 35]

NONCOMMITTAL = [
    "i have no comment", "no comment", "unknown", "it is unknown", "i don't know",
    "there is no", "nothing happens", "it depends", "not necessarily", "nothing",
    "no", "none", "nowhere", "nobody", "never", "it is not", "there are no",
]


def is_noncommittal(answer: str) -> bool:
    s = answer.lower().strip().rstrip(".")
    return any(s == m or s.startswith(m + " ") or s.startswith(m + ",")
               for m in NONCOMMITTAL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--n-questions", type=int, default=500)
    parser.add_argument("--per-side", type=int, default=2)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    cfg = yaml.safe_load(Path(args.config).read_text("utf-8"))
    token = hf.token()

    beat("truth_v2", "старт")
    old = np.load(sorted(artifacts_dir("truth_vector").glob("*.npz"))[-1],
                  allow_pickle=True)
    old_vec = old[f"vector_{str(old['best_format'])}"]

    data = pd.read_parquet(hf_hub_download(TQA, TQA_FILE, repo_type="dataset",
                                           token=token))
    rng = np.random.default_rng(args.seed)
    data = data.iloc[rng.permutation(len(data))[:args.n_questions]].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], token=token)
    beat("truth_v2", "гружу веса")
    model = load_model(cfg["base_model"], token)
    model.eval()
    print(f"=== вопросов {len(data)}, слои {LAYERS} ===", flush=True)

    # Активация последнего токена ответа, поданного как продолжение ассистента —
    # ровно та позиция, в которой вектор потом применяется при генерации.
    acts = {layer: [] for layer in LAYERS}
    labels, qids, noncom = [], [], []
    for qi, row in data.iterrows():
        enc = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": row["question"]}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        prompt_ids = enc["input_ids"][0].tolist()

        picks = ([(a, 1) for a in list(row["correct_answers"])[:args.per_side]] +
                 [(a, 0) for a in list(row["incorrect_answers"])[:args.per_side]])
        for answer, label in picks:
            ans_ids = tok(" " + answer, add_special_tokens=False)["input_ids"]
            if not ans_ids:
                continue
            ids = torch.tensor([prompt_ids + ans_ids], device=model.device)
            with torch.no_grad():
                out = model(input_ids=ids, output_hidden_states=True)
            for layer in LAYERS:
                acts[layer].append(out.hidden_states[layer + 1][0, -1].float().cpu().numpy())
            labels.append(label)
            qids.append(qi)
            noncom.append(int(is_noncommittal(answer)))
        if (qi + 1) % 50 == 0:
            beat("truth_v2", f"{qi + 1}/{len(data)}")
            print(f"  {qi + 1}/{len(data)}", flush=True)

    free(model)
    acts = {k: np.stack(v) for k, v in acts.items()}
    labels = np.array(labels)
    qids = np.array(qids)
    noncom = np.array(noncom)
    print(f"\n  собрано ответов: {len(labels)} "
          f"(верных {int(labels.sum())}, уклончивых среди верных "
          f"{int(noncom[labels == 1].sum())})", flush=True)

    # ---------- проверка 1: интерпретация старого вектора ----------
    print("\n=== 1. куда смотрит вектор из azaria-mitchell (слой 24) ===")
    proj = acts[24] @ old_vec
    mean_correct = float(proj[labels == 1].mean())
    mean_incorrect = float(proj[labels == 0].mean())
    auc_old = auroc(proj, labels)
    print(f"  проекция верных ответов:   {mean_correct:+.4f}")
    print(f"  проекция неверных ответов: {mean_incorrect:+.4f}")
    print(f"  AUROC (верные выше):       {auc_old:.4f}")
    committal = labels[(labels == 1) | (labels == 0)]
    only_committal = noncom == 0
    auc_committal = auroc(proj[only_committal], labels[only_committal])
    print(f"  AUROC только по содержательным ответам: {auc_committal:.4f}")
    print(f"  проекция уклончивых верных: "
          f"{float(proj[(labels == 1) & (noncom == 1)].mean()):+.4f}")
    print(f"  проекция содержательных верных: "
          f"{float(proj[(labels == 1) & (noncom == 0)].mean()):+.4f}")
    if mean_incorrect > mean_correct:
        print("  -> неверные проецируются ВЫШЕ: гипотеза «уверенное утверждение»"
              " подтверждается")

    # ---------- проверка 2 и 3: свой вектор, по слоям ----------
    unique_q = np.unique(qids)
    shuffled = rng.permutation(unique_q)
    n_train = int(len(shuffled) * args.train_frac)
    train_q, test_q = set(shuffled[:n_train]), set(shuffled[n_train:])
    train = np.array([q in train_q for q in qids])
    test = ~train
    print(f"\n=== 2–3. своё направление из TruthfulQA, сплит по вопросам "
          f"({len(train_q)}/{len(test_q)}) ===")
    print(f"{'слой':>6} {'AUROC тест':>12} {'AUROC только содерж.':>22} "
          f"{'|v|':>9}")

    report, vectors = {}, {}
    for layer in LAYERS:
        matrix = acts[layer]
        v = matrix[train & (labels == 1)].mean(0) - matrix[train & (labels == 0)].mean(0)
        unit = v / np.linalg.norm(v)
        vectors[layer] = unit
        auc = auroc(matrix[test] @ unit, labels[test])
        mask = test & only_committal
        auc_com = auroc(matrix[mask] @ unit, labels[mask])
        report[layer] = {"auroc_test": auc, "auroc_test_committal": auc_com,
                         "norm": float(np.linalg.norm(v))}
        print(f"{layer:6} {auc:12.4f} {auc_com:22.4f} {np.linalg.norm(v):9.3f}")

    best_layer = max(LAYERS, key=lambda l: report[l]["auroc_test"])
    print(f"\n  лучший слой: {best_layer} (AUROC {report[best_layer]['auroc_test']:.4f})")
    print(f"  слой 24 (там живёт NLA): {report[24]['auroc_test']:.4f}")
    print(f"  косинус нового вектора слоя 24 со старым: "
          f"{float(vectors[24] @ old_vec):+.4f}")

    print("\n  Важно: AUROC здесь — качество РАЗДЕЛЕНИЯ ответов, а не гарантия")
    print("  causal-эффекта. Проверять стирингом отдельно (как в 15/16).")

    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out_dir = artifacts_dir("truth_vector_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{stamp}.npz",
                        **{f"vector_layer{l}": vectors[l] for l in LAYERS},
                        acts24=acts[24], labels=labels, qids=qids, noncommittal=noncom,
                        best_layer=np.array(best_layer))
    payload = {"n_answers": int(len(labels)),
               "old_vector": {"mean_proj_correct": mean_correct,
                              "mean_proj_incorrect": mean_incorrect,
                              "auroc": auc_old, "auroc_committal": auc_committal},
               "by_layer": {str(k): v for k, v in report.items()},
               "best_layer": int(best_layer),
               "cosine_new_vs_old_layer24": float(vectors[24] @ old_vec)}
    (out_dir / f"{stamp}.json").write_text(json.dumps(payload, indent=2,
                                                      ensure_ascii=False), "utf-8")
    results_dir().mkdir(parents=True, exist_ok=True)
    (results_dir() / "17_truth_vector_v2.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nАртефакт на Drive: {out_dir / (stamp + '.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
