"""Этап 8: сквозное сравнение режимов гейтинга на held-out XSTest.

Всё предыдущее меряло сигналы по отдельности. Здесь стиринг реально
применяется, и режимы сравниваются по тому, ради чего проект и затевался:
меньше ли over-refusal на безопасных запросах при том же уровне отказов на
опасных.

**XSTest ни разу не участвовал ни в подборе вектора, ни в выборе позиции
извлечения, ни в калибровке порогов** — он держался held-out с этапа 3 именно
ради этой таблицы.

Режимы:

* `none` — без стиринга, точка отсчёта;
* `all` — на всех токенах, классический режим из литературы;
* `random` — случайные `k` токенов. **Обязательный контроль**: гейтинг стирит
  меньше, поэтому любое улучшение при неравном бюджете может быть артефактом;
* `first_k` — первые `k` токенов, позиционная эвристика. Этап 6 показал, что
  для refusal она и есть оптимум, так что это не пустой бейзлайн, а фаворит;
* `cosine` — топ-`k` по проекции на `v`. Это CAST, прямой конкурент;
* `probe` — топ-`k` по линейному классификатору;
* `nla_latent` — топ-`k` по близости латента `av` к опорному.

**Все гейты получают ровно `k` токенов на ответ.** Равный бюджет вмешательства —
условие, без которого сравнение невалидно.

Гейт работает **офлайн**: счёт для токена `t` берётся с нестирённой активации
на `t`. Это каузально корректно (решение принимается до вмешательства) и снимает
вопрос переноса порогов на сдвинутое распределение, но означает, что режим
не полностью онлайновый. Для inference это и есть рекомендуемый постановкой
порядок: сначала офлайн-анализ.

Побочный ущерб меряется средним логарифмом правдоподобия сгенерированного
ответа **под нестирённой моделью**: если стиринг ломает беглость, это видно.

Запуск:
    python scripts/12_gating_comparison.py --config configs/base.yaml
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
from nla_steering.metrics import is_refusal  # noqa: E402
from nla_steering.nla import Verbalizer, cosine, free, load_model  # noqa: E402
from nla_steering.paths import artifacts_dir, results_dir  # noqa: E402
from nla_steering.progress import beat  # noqa: E402
from nla_steering.steering import Steerer  # noqa: E402

XSTEST = "natolambert/xstest-v2-copy"
XSTEST_FILE = "data/prompts-00000-of-00001.parquet"
MODES = ["all", "random", "first_k", "cosine", "probe", "nla_latent"]


def load_xstest(token: str | None, n_per_class: int, seed: int) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    df = pd.read_parquet(hf_hub_download(XSTEST, XSTEST_FILE, repo_type="dataset",
                                         token=token))
    # типы с префиксом contrast_ — действительно опасные, остальные безопасны,
    # но сформулированы так, что выглядят опасно
    df = df.assign(unsafe=df["type"].str.startswith("contrast_").astype(int))
    rng = np.random.default_rng(seed)
    parts = []
    for label in (0, 1):
        sub = df[df.unsafe == label]
        take = min(n_per_class, len(sub))
        parts.append(sub.iloc[rng.permutation(len(sub))[:take]])
    return pd.concat(parts, ignore_index=True)[["prompt", "unsafe"]]


def generate(model, tok, prompt, max_new_tokens, steerer=None, targets=None):
    import torch

    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = {k: v.to(model.device) for k, v in enc.items()}
    if steerer is not None:
        steerer.reset(targets)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    prompt_len = ids["input_ids"].shape[1]
    gen = out[0, prompt_len:]
    return tok.decode(gen, skip_special_tokens=True), out[0], prompt_len


def naturalness(model, sequence, prompt_len) -> float:
    """Средний logP ответа под моделью без стиринга — прокси беглости."""
    import torch

    with torch.no_grad():
        logits = model(input_ids=sequence.unsqueeze(0)).logits.float()
    logprobs = torch.log_softmax(logits, -1)[0]
    target = sequence[prompt_len:]
    if target.numel() == 0:
        return float("nan")
    picked = logprobs[prompt_len - 1:prompt_len - 1 + target.numel()]
    return float(picked.gather(-1, target.unsqueeze(-1)).squeeze(-1).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--vector", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--n-per-class", type=int, default=150)
    parser.add_argument("--budget", type=int, default=2, help="k токенов на ответ")
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--reference-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    cfg = yaml.safe_load(Path(args.config).read_text("utf-8"))
    token = hf.token()
    layer, index = cfg["layer"], cfg["nla"]["hidden_state_index"]
    rng = np.random.default_rng(args.seed)

    # Отметки до каждой потенциально долгой операции: первый прогон 37 минут
    # молчал, и было не понять, висит ли задача или умерла сессия. Чтение
    # артефакта идёт через FUSE-монтирование Drive и может быть медленным.
    print("старт", flush=True)
    beat("compare", "старт")
    vec_path = Path(args.vector) if args.vector else sorted(
        artifacts_dir("vector").glob("*.npz"))[-1]
    print(f"читаю вектор: {vec_path.name}", flush=True)
    vec = np.load(vec_path)
    print("вектор прочитан", flush=True)
    j = vec["positions"].tolist().index(cfg["steering"]["extraction_position"])
    prompt_acts, prompt_labels = vec["jbb_acts"][:, j], vec["jbb_labels"]
    v = prompt_acts[prompt_labels == 1].mean(0) - prompt_acts[prompt_labels == 0].mean(0)
    unit = v / np.linalg.norm(v)

    print("качаю XSTest", flush=True)
    beat("compare", "качаю XSTest")
    data = load_xstest(token, args.n_per_class, args.seed)
    print(f"=== XSTest: {len(data)} промптов "
          f"(безопасных {int((data.unsafe == 0).sum())}, "
          f"опасных {int((data.unsafe == 1).sum())}) ===")
    print(f"=== бюджет {args.budget} токенов на ответ, α = {args.alpha} ===")

    # Индикаторы скачивания весов рисуются возвратом каретки без перевода
    # строки, а воркер читает вывод построчно — такой прогресс в лог не попадает
    # и выглядит как зависание. Выключаем их и печатаем состояние сами.
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"gpu: {torch.cuda.get_device_name(0)}, свободно "
          f"{free_mem / 1024**3:.2f} из {total_mem / 1024**3:.2f} ГБ", flush=True)
    print("гружу токенизатор", flush=True)
    beat("compare", "гружу токенизатор")
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], token=token)
    print("токенизатор готов, гружу веса (может качаться ~6 ГБ)", flush=True)
    model = load_model(cfg["base_model"], token)
    model.eval()
    print("веса загружены", flush=True)
    beat("compare", "веса загружены")

    # --- фаза 1: нестирённая генерация + активации для гейтов ---
    print("\n=== фаза 1: базовая генерация и активации ===")
    baseline_text, baseline_nat, token_acts, lengths = [], [], [], []
    for i, row in data.iterrows():
        enc = tok.apply_chat_template([{"role": "user", "content": row["prompt"]}],
                                      add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        ids = {k: v_.to(model.device) for k, v_ in enc.items()}
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.eos_token_id,
                                 output_hidden_states=True, return_dict_in_generate=True)
        prompt_len = ids["input_ids"].shape[1]
        gen = out.sequences[0, prompt_len:]
        baseline_text.append(tok.decode(gen, skip_special_tokens=True))
        baseline_nat.append(naturalness(model, out.sequences[0], prompt_len))
        acts = []
        for step, states in enumerate(out.hidden_states):
            if step >= gen.numel():
                break
            acts.append(states[index][0, -1].float().cpu().numpy())
        token_acts.append(np.stack(acts) if acts else np.zeros((0, v.shape[0]), np.float32))
        lengths.append(len(acts))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(data)}", flush=True)
            beat("compare", f"фаза 1: {i + 1}/{len(data)}")

    from sklearn.linear_model import LogisticRegression

    probe = LogisticRegression(max_iter=2000, C=0.01).fit(prompt_acts, prompt_labels)
    scores = {
        "cosine": [a @ unit if len(a) else np.zeros(0) for a in token_acts],
        "probe": [probe.decision_function(a) if len(a) else np.zeros(0) for a in token_acts],
    }
    free(model)

    # --- фаза 2: латенты av для гейта nla_latent ---
    print("\n=== фаза 2: латенты av ===")
    top = np.argsort(prompt_acts @ unit)[-args.reference_k:]
    flat = np.concatenate([a for a in token_acts if len(a)])
    with Verbalizer(cfg, token) as av:
        reference = av.latents(prompt_acts[top]).mean(0)
        flat_latents = av.latents(flat)
    flat_scores = cosine(flat_latents, np.broadcast_to(reference, flat_latents.shape))
    nla_scores, cursor = [], 0
    for n in lengths:
        nla_scores.append(flat_scores[cursor:cursor + n])
        cursor += n
    scores["nla_latent"] = nla_scores

    # --- фаза 3: стиринг во всех режимах при равном бюджете ---
    print("\n=== фаза 3: генерация со стирингом ===")
    model = load_model(cfg["base_model"], token)
    model.eval()
    vector_t = torch.as_tensor(-args.alpha * v, dtype=torch.float32, device=model.device)

    def targets_for(mode: str, i: int) -> set[int] | None:
        n = lengths[i]
        k = min(args.budget, n)
        if n == 0:
            return set()
        if mode == "all":
            return None
        if mode == "first_k":
            return set(range(k))
        if mode == "random":
            return set(rng.choice(n, size=k, replace=False).tolist())
        return set(np.argsort(scores[mode][i])[-k:].tolist())

    results = {"none": {"text": baseline_text, "nat": baseline_nat}}
    with Steerer(model, layer, vector_t) as steerer:
        for mode in MODES:
            texts, nats = [], []
            for i, row in data.iterrows():
                text, seq, plen = generate(model, tok, row["prompt"],
                                           args.max_new_tokens, steerer,
                                           targets_for(mode, i))
                texts.append(text)
                nats.append((seq, plen))
            results[mode] = {"text": texts, "seq": nats}
            print(f"  {mode} готов")
            beat("compare", f"фаза 3: {mode} готов")

    # Беглость считается уже после выхода из Steerer: хук снят, значит logP
    # ответа берётся под нестирённой моделью — иначе стиринг оценивал бы сам себя.
    for mode in MODES:
        pairs = results[mode].pop("seq")
        results[mode]["nat"] = [naturalness(model, s, p) for s, p in pairs]
    free(model)

    # --- сводка ---
    safe = data.unsafe.to_numpy() == 0
    unsafe = ~safe
    print(f"\n{'режим':12} {'over-refusal':>13} {'отказ на опасных':>18} "
          f"{'разрыв':>8} {'беглость':>10}")
    table = {}
    for mode, payload in results.items():
        refusals = np.array([is_refusal(t) for t in payload["text"]])
        over = float(refusals[safe].mean())
        keep = float(refusals[unsafe].mean())
        nat = float(np.nanmean(payload["nat"]))
        table[mode] = {"over_refusal": over, "unsafe_refusal": keep,
                       "gap": keep - over, "naturalness": nat}
        print(f"{mode:12} {over:12.1%} {keep:17.1%} {keep - over:8.1%} {nat:10.3f}")

    base = table["none"]
    print(f"\n  цель: over-refusal ниже {base['over_refusal']:.1%} "
          f"при отказе на опасных около {base['unsafe_refusal']:.1%}")
    gated = {m: t for m, t in table.items() if m in ("cosine", "probe", "nla_latent")}
    if gated:
        best = min(gated, key=lambda m: gated[m]["over_refusal"])
        print(f"  лучший содержательный гейт: {best} "
              f"({gated[best]['over_refusal']:.1%} over-refusal)")
        print(f"  позиционная эвристика first_k: "
              f"{table['first_k']['over_refusal']:.1%}")
        print(f"  random при том же бюджете:     "
              f"{table['random']['over_refusal']:.1%}")

    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out_dir = artifacts_dir("comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"table": table, "budget": args.budget, "alpha": args.alpha,
               "n_safe": int(safe.sum()), "n_unsafe": int(unsafe.sum()),
               "vector": vec_path.name,
               "texts": {m: r["text"][:5] for m, r in results.items()}}
    (out_dir / f"{stamp}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
    results_dir().mkdir(parents=True, exist_ok=True)
    (results_dir() / "12_gating_comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), "utf-8")
    print(f"\nАртефакт на Drive: {out_dir / (stamp + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
