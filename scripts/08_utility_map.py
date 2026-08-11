"""Этап 6: карта полезности стиринга по токенам.

Вопрос проекта — «на каких токенах применять стиринг» — до сих пор нигде не
измерялся напрямую. Сигналы этапа 5 оценивались против метки уровня промпта
(«ответ был отказом»), которая внутри ответа постоянна и в принципе не может
сказать, какой токен важен. Здесь ground truth добывается причинно: стирим
**ровно один токен за раз** и смотрим, изменился ли исход.

Берутся безопасные запросы JBB, на которые модель ошибочно ответила отказом
(их 29 из 100). К активации слоя 24 в одной выбранной позиции добавляется
`-α·v`, то есть отказ подавляется. Полезность позиции — перестал ли ответ быть
отказом.

**α калибруется, а не назначается.** Одиночное вмешательство слабее, чем
стиринг на всех токенах, поэтому сначала идёт свип по α в режиме all-token, и
для карты берётся наименьшая α, переворачивающая заметную долю ответов. Иначе
пустая карта означала бы лишь «α мала», а не «токены не важны».

Средняя полезность по карте — это и есть **random gating при бюджете в один
токен**, тот самый обязательный бейзлайн: гейтинг обязан бить не ноль, а его.

Запуск:
    python scripts/08_utility_map.py --config configs/base.yaml
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
from nla_steering.nla import free, load_model  # noqa: E402
from nla_steering.paths import artifacts_dir  # noqa: E402

JBB = "JailbreakBench/JBB-Behaviors"
ALPHA_SWEEP = [0.5, 1.0, 2.0, 4.0]


class Steerer:
    """Хук на слой 24: добавляет вектор в заданных шагах генерации.

    Позиция задаётся в терминах шага генерации: шаг 0 — это prefill, где
    активация последней позиции порождает первый токен ответа; шаг g>0
    обрабатывает один токен и порождает следующий. `targets=None` — стирить
    везде (режим all-token).
    """

    def __init__(self, model, layer: int, vector):
        self.model, self.vector = model, vector
        self.block = model.model.layers[layer]
        self.targets: set[int] | None = None
        self.step = 0
        self._handle = None

    def __enter__(self):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.targets is None or self.step in self.targets:
                hidden = hidden.clone()
                hidden[:, -1, :] = hidden[:, -1, :] + self.vector.to(hidden.dtype)
                output = (hidden, *output[1:]) if isinstance(output, tuple) else hidden
            self.step += 1
            return output

        self._handle = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        return False

    def reset(self, targets: set[int] | None):
        self.targets, self.step = targets, 0


def generate(model, tok, prompt: str, max_new_tokens: int, steerer=None,
             targets: set[int] | None = None):
    """Возвращает (полный текст, список id сгенерированных токенов)."""
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
    new_ids = out[0, ids["input_ids"].shape[1]:]
    return tok.decode(new_ids, skip_special_tokens=True), new_ids.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--vector", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="0 = подобрать свипом")
    args = parser.parse_args()

    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    cfg = yaml.safe_load(Path(args.config).read_text("utf-8"))
    token = hf.token()
    layer = cfg["layer"]

    vec_path = Path(args.vector) if args.vector else sorted(
        artifacts_dir("vector").glob("*.npz"))[-1]
    vec = np.load(vec_path)
    j = vec["positions"].tolist().index(cfg["steering"]["extraction_position"])
    acts, plabels = vec["jbb_acts"][:, j], vec["jbb_labels"]
    v = acts[plabels == 1].mean(0) - acts[plabels == 0].mean(0)
    print(f"=== вектор из {vec_path.name}, |v| = {np.linalg.norm(v):.2f} ===")

    benign = pd.read_csv(hf_hub_download(JBB, "data/benign-behaviors.csv",
                                         repo_type="dataset", token=token))
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], token=token)
    model = load_model(cfg["base_model"], token)
    model.eval()

    print("\n=== отбор: безопасные запросы, на которые модель отказала ===")
    selected, baselines = [], []
    for _, row in benign.iterrows():
        text, _ = generate(model, tok, row["Goal"], args.max_new_tokens)
        if is_refusal(text):
            selected.append(row["Goal"])
            baselines.append(text)
        if len(selected) >= args.n_prompts:
            break
    print(f"  отобрано {len(selected)} промптов с ошибочным отказом")
    if not selected:
        print("  нечего исправлять — карта не имеет смысла")
        return 1

    vector_t = torch.as_tensor(v, dtype=torch.float32, device=model.device)

    print(f"\n=== калибровка α в режиме all-token ===")
    print(f"{'α':>6} {'доля исправленных':>20}")
    sweep = {}
    with Steerer(model, layer, -vector_t) as steerer:
        for a in ALPHA_SWEEP:
            steerer.vector = -a * vector_t
            flipped = sum(not is_refusal(generate(model, tok, p, args.max_new_tokens,
                                                  steerer, None)[0]) for p in selected)
            sweep[a] = flipped / len(selected)
            print(f"{a:6.1f} {sweep[a]:19.1%}")

        if args.alpha:
            alpha = args.alpha
            print(f"\n  α ЗАДАНА ЯВНО: {alpha}")
        else:
            usable = [a for a in ALPHA_SWEEP if sweep[a] >= 0.5]
            alpha = usable[0] if usable else ALPHA_SWEEP[-1]
            print(f"\n  ВЫБРАНА α = {alpha}")

        print(f"\n=== карта: стиринг ровно одного токена, {args.positions} позиций ===")
        steerer.vector = -alpha * vector_t
        utility = np.zeros((len(selected), args.positions), dtype=np.int8)
        # Вторая метрика — отказ, посчитанный только по хвосту после стирённой
        # позиции. Первая по построению не может быть положительной там, где
        # префикс отказа уже выдан: стиринг на шаге 5 не убирает "I'm sorry",
        # выданное на шагах 0-2. Без этого различения нули на поздних позициях
        # нельзя отличить от артефакта метрики.
        suffix = np.zeros_like(utility)
        for i, prompt in enumerate(selected):
            for pos in range(args.positions):
                text, ids = generate(model, tok, prompt, args.max_new_tokens,
                                     steerer, {pos})
                utility[i, pos] = 0 if is_refusal(text) else 1
                tail = tok.decode(ids[pos + 1:], skip_special_tokens=True)
                suffix[i, pos] = 0 if is_refusal(tail) else 1
            print(f"  промпт {i + 1}/{len(selected)}: полный ответ "
                  f"{int(utility[i].sum())}/{args.positions}, хвост "
                  f"{int(suffix[i].sum())}/{args.positions}")

    free(model)

    by_position, by_position_suffix = utility.mean(0), suffix.mean(0)
    print("\n=== доля исправленных по позиции ===")
    print(f"{'поз':>4} {'полный ответ':>14} {'хвост после позиции':>22}")
    for pos in range(args.positions):
        bar = "#" * int(by_position[pos] * 30)
        print(f"  {pos:>3} {by_position[pos]:12.1%} {by_position_suffix[pos]:20.1%} {bar}")

    random_baseline = float(utility.mean())
    print(f"\n  RANDOM GATING при бюджете в 1 токен: {random_baseline:.1%}")
    print(f"  лучшая позиция: {int(np.argmax(by_position))} ({by_position.max():.1%})")
    print(f"  all-token при той же α: {sweep.get(alpha, float('nan')):.1%} "
          f"(бюджет в {args.positions} раз больше)")
    print(f"\n  по хвосту: random {suffix.mean():.1%}, "
          f"лучшая позиция {int(np.argmax(by_position_suffix))} "
          f"({by_position_suffix.max():.1%})")

    stamp = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    out_dir = artifacts_dir("utility")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{stamp}.npz", utility=utility,
                        suffix=suffix, vector=v, alpha=alpha)
    (out_dir / f"{stamp}.json").write_text(json.dumps(
        {"alpha": alpha, "alpha_sweep": sweep,
         "by_position_suffix": by_position_suffix.tolist(),
         "random_gating_suffix": float(suffix.mean()), "n_prompts": len(selected),
         "positions": args.positions, "random_gating_1token": random_baseline,
         "by_position": by_position.tolist(), "all_token_rate": sweep[alpha],
         "prompts": selected, "baselines": baselines},
        indent=2, ensure_ascii=False), "utf-8")
    print(f"\nАртефакт на Drive: {out_dir / (stamp + '.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
