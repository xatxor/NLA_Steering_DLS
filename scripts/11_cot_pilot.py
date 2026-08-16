"""Пилот CoT на GSM8K: пригодно ли рассуждение для проверки гипотезы.

Дважды подряд концепт оказывался разрешим до начала генерации: у refusal весь
запрос лежит в контексте, поэтому решение принимается на нулевом токене, у
truthfulness так в 57% случаев. В цепочке рассуждений это невозможно по
построению — модель порождает промежуточные результаты, которых у неё не было
в начале, и «верно ли рассуждение» физически не определено на нулевом токене.

Заодно решается главная беда проекта — измерение. Ответ GSM8K это число, точное
совпадение, и меряется то, что модель реально написала, а не прокси. После
эмбеддингового классификатора с его 58.6% это решающий аргумент.

Четыре критерия:

  * **частота ошибок** — есть ли что чинить (нужно ≥30%);
  * **длина трейсов** — есть ли где выбирать токены (медиана ≥80);
  * **локализуема ли ошибка** — где по трейсу обваливается предпочтение верного
    ответа. Считается точно: на каждом префиксе сравнивается правдоподобие
    верного числа и того, которое модель выдала. Если обвал всегда в начале, мы
    снова в вырожденном случае;
  * **частота самопроверки** — как часто модель сама пишет «wait», «actually»
    и подобное. От этого зависит, удастся ли построить вектор backtracking на
    естественном контрасте или его придётся выбивать промптом (и тогда он будет
    про стиль промпта, а не про поведение).

Запуск:
    python scripts/11_cot_pilot.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering import hf  # noqa: E402
from nla_steering.nla import free, load_model  # noqa: E402
from nla_steering.paths import artifacts_dir  # noqa: E402
from nla_steering.scoring import mean_logprobs  # noqa: E402

GSM8K = "openai/gsm8k"
GSM8K_FILE = "main/test-00000-of-00001.parquet"
SYSTEM = ("Solve the problem step by step. "
          "Finish with a final line of the form 'The answer is <number>'.")
PROBE = "\nThe answer is"

BACKTRACK_MARKERS = [
    "wait", "actually", "hmm", "let me re", "recheck", "re-check", "reconsider",
    "hold on", "on second thought", "but wait", "correction", "i made a mistake",
    "that's wrong", "that is wrong", "let me verify", "let me check", "oops",
]

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    """Последнее число ответа: сначала после 'answer is', иначе просто последнее."""
    tail = text.lower().split("answer is")
    source = tail[-1] if len(tail) > 1 else text
    found = NUMBER.findall(source)
    if not found:
        found = NUMBER.findall(text)
    if not found:
        return None
    return found[-1].replace(",", "").rstrip(".")


def same_number(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a == b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    cfg = yaml.safe_load(Path(args.config).read_text("utf-8"))
    token = hf.token()

    data = pd.read_parquet(hf_hub_download(GSM8K, GSM8K_FILE, repo_type="dataset",
                                           token=token))
    rng = np.random.default_rng(args.seed)
    data = data.iloc[rng.permutation(len(data))[:args.n_problems]].reset_index(drop=True)
    gold = [extract_answer(a.split("####")[-1]) for a in data["answer"]]

    tok = AutoTokenizer.from_pretrained(cfg["base_model"], token=token)
    model = load_model(cfg["base_model"], token)
    model.eval()
    print(f"=== GSM8K: {len(data)} задач, до {args.max_new_tokens} токенов ===")

    traces, predictions, lengths = [], [], []
    for idx, row in data.iterrows():
        enc = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": row["question"]}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True)
        prompt_ids = enc["input_ids"][0].tolist()
        ids = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        gen = out[0, len(prompt_ids):].tolist()
        text = tok.decode(gen, skip_special_tokens=True)
        traces.append({"prompt_ids": prompt_ids, "gen": gen, "text": text})
        predictions.append(extract_answer(text))
        lengths.append(len(gen))
        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(data)}")

    correct = np.array([same_number(p, g) for p, g in zip(predictions, gold)])
    lengths = np.array(lengths)
    # Трейс, упёршийся в лимит, не дописан: парсер берёт последнее число из
    # незаконченного рассуждения и выдаёт его за ответ. В первом прогоне так
    # было у 40% задач, и мнимая частота ошибок 37% оказалась почти целиком
    # обрезкой — на дописанных модель ошибалась в 14%. Правильность считаем
    # только по завершённым.
    finished = lengths < args.max_new_tokens

    print(f"\n=== завершённость ===")
    print(f"  дописано до конца: {finished.mean():.0%} "
          f"({int(finished.sum())} из {len(finished)}), "
          f"обрезано лимитом: {int((~finished).sum())}")
    if not finished.any():
        print("  все трейсы обрезаны — поднимай --max-new-tokens")
        return 0

    print(f"\n=== правильность (только завершённые) ===")
    print(f"  верных ответов: {correct[finished].mean():.0%}  "
          f"ОШИБОК: {1 - correct[finished].mean():.0%} "
          f"({int((~correct[finished]).sum())} из {int(finished.sum())})")
    if (~finished).any():
        print(f"  на обрезанных: {correct[~finished].mean():.0%} — эти числа "
              f"бессмысленны, парсер читает недописанное рассуждение")
    print(f"\n=== длина трейсов (токены) ===")
    print(f"  медиана {np.median(lengths):.0f}, "
          f"квартили {np.percentile(lengths, 25):.0f}/{np.percentile(lengths, 75):.0f}, "
          f"max {lengths.max()}")

    print(f"\n=== самопроверка в трейсах ===")
    has_marker = np.array([any(m in t["text"].lower() for m in BACKTRACK_MARKERS)
                           for t in traces])
    print(f"  трейсов с маркерами самопроверки: {has_marker.mean():.0%} "
          f"({int(has_marker.sum())} из {len(traces)})")
    print(f"    среди верных {has_marker[correct].mean():.0%}, "
          f"среди неверных {has_marker[~correct].mean():.0%}")

    # Траектория считается для всех неверных (там и было бы вмешательство) и для
    # равного числа верных как контроль.
    wrong_idx = np.where(~correct & finished)[0]
    right_idx = np.where(correct & finished)[0][:len(wrong_idx)]
    probe_ids = tok(PROBE, add_special_tokens=False)["input_ids"]

    print(f"\n=== обвал предпочтения верного ответа ===")
    collapse, trajectories = {}, {}
    for group, indices in (("неверные", wrong_idx), ("верные", right_idx)):
        points = []
        for i in indices:
            trace, gold_i, pred_i = traces[i], gold[i], predictions[i]
            if pred_i is None or same_number(pred_i, gold_i):
                pred_i = "0" if not same_number("0", gold_i) else "1"
            good = tok(" " + str(gold_i), add_special_tokens=False)["input_ids"]
            bad = tok(" " + str(pred_i), add_special_tokens=False)["input_ids"]

            steps = list(range(0, len(trace["gen"]), args.stride))
            scores = []
            for step in steps:
                context = trace["prompt_ids"] + trace["gen"][:step] + probe_ids
                lp = mean_logprobs(model, tok, context, [good, bad])
                scores.append(float(lp[0] - lp[1]))
            trajectories[int(i)] = {"steps": steps, "scores": scores}

            # обвал — последний шаг, после которого верный ответ уже не
            # предпочитается ни разу
            point = 0
            for k in range(len(steps)):
                if all(s <= 0 for s in scores[k:]):
                    point = steps[k]
                    break
            else:
                point = steps[-1]
            points.append(point / max(len(trace["gen"]), 1))
        collapse[group] = np.array(points)
        arr = collapse[group]
        if arr.size:
            print(f"  {group:9} доля трейса до обвала: медиана {np.median(arr):.2f}, "
                  f"std {arr.std():.2f}, в первых 10% — {float((arr <= 0.1).mean()):.0%}")

    free(model)

    wrong_collapse = collapse["неверные"]
    ok_errors = float(1 - correct[finished].mean()) >= 0.30
    ok_length = float(np.median(lengths[finished])) >= 80
    ok_finished = float(finished.mean()) >= 0.80
    ok_spread = bool(wrong_collapse.size and wrong_collapse.std() > 0.15
                     and float((wrong_collapse <= 0.1).mean()) < 0.5)
    verdict = ok_errors and ok_length and ok_spread and ok_finished

    print(f"\n=== вердикт ===")
    print(f"  трейсы дописываются (>=80%):          {ok_finished}")
    print(f"  ошибается достаточно часто (>=30%):   {ok_errors}")
    print(f"  трейсы достаточно длинные (>=80):     {ok_length}")
    print(f"  обвал не сосредоточен в начале:       {ok_spread}")
    print(f"\n  ПОСТАНОВКА {'ГОДИТСЯ' if verdict else 'ПОД ВОПРОСОМ'}")
    print(f"\n  для вектора backtracking: естественных примеров "
          f"{has_marker.mean():.0%} — "
          f"{'хватит на контраст' if has_marker.mean() > 0.15 else 'мало, понадобится элицитация промптом'}")

    print("\n=== пример неверного трейса ===")
    if wrong_idx.size:
        i = int(wrong_idx[0])
        print(f"  В: {data.iloc[i]['question'][:200]}")
        print(f"  эталон {gold[i]}, выдано {predictions[i]}")
        print(f"  О: {traces[i]['text'][:400]}")

    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out_dir = artifacts_dir("cot")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stamp}.json").write_text(json.dumps(
        {"n": len(data), "accuracy_finished": float(correct[finished].mean()),
         "finished_rate": float(finished.mean()),
         "accuracy_all": float(correct.mean()),
         "median_length": float(np.median(lengths)),
         "backtrack_rate": float(has_marker.mean()),
         "collapse_wrong": wrong_collapse.tolist(),
         "collapse_right": collapse["верные"].tolist(),
         "verdict": verdict, "trajectories": trajectories,
         "questions": data["question"].tolist(),
         "traces": [t["text"] for t in traces],
         "gold": gold, "predictions": predictions},
        indent=2, ensure_ascii=False), "utf-8")
    print(f"\nОтчёт на Drive: {out_dir / (stamp + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
