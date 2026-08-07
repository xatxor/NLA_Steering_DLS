"""Reconstruction score: замыкаем автоэнкодер и выбираем правильный тензор.

Прогоняем полный цикл `активация -> av -> текст -> ar -> активация` и меряем
косинус между исходной активацией и восстановленной. Это даёт сразу три вещи.

**Выбор тензора.** `02_activation_roundtrip.py` показал, что и `hidden_states[24]`
(вход слоя 24), и `hidden_states[25]` (выход) дают осмысленные вербализации, на
глаз неразличимые. Reconstruction score различает их количественно. Со стороны
`ar` тоже два кандидата: выход слоя 24 до финального RMSNorm (берётся хуком) и
после него (последний элемент `hidden_states`) — HF кладёт в список уже
отнормированный тензор, до-нормовый достаётся только хуком. Итого матрица 2x2.

**Контроль, без которого число ничего не значит.** Высокий косинус сам по себе
не доказывает ничего: в высокой размерности активации одной модели и так могут
лежать в общем конусе. Поэтому кроме «своих» пар считаем «чужие» (объяснение
текста i против активации текста j) — информативна именно разница между ними.

**Первый сигнал по концепту.** Тексты берутся двумя кластерами, отказы и
нейтральные утверждения. Если reconstruction score или латент их разделяет,
это и есть кандидат в гейтинг-сигнал.

Три фазы с выгрузкой между ними: база (~6.2 ГБ), av (~6.2 ГБ), ar (~4.3 ГБ).

Запуск:
    python scripts/03_reconstruction_score.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering import hf  # noqa: E402
from nla_steering.paths import artifacts_dir  # noqa: E402

# Кластер отказа — это формулировки самой модели, а не вредные запросы:
# гейтинг должен срабатывать на токенах, где модель формирует отказ.
REFUSAL = [
    "I'm sorry, but I can't help with that request",
    "I cannot provide that information",
    "I'm unable to assist with this particular question",
    "Unfortunately, I am not able to comply with that",
    "I must decline to answer this question",
    "That's not something I can help you with",
    "I won't be able to provide instructions for that",
    "I'm not going to answer that request",
]

NEUTRAL = [
    "The weather in London is often rainy in November",
    "Photosynthesis converts light energy into chemical energy",
    "The train arrives at the station at half past four",
    "She opened the book and started reading the first chapter",
    "Copper is an excellent conductor of electricity",
    "The recipe calls for two cups of flour and one egg",
    "Mount Everest is the highest mountain above sea level",
    "The library closes at eight in the evening on weekdays",
]


def load_model(repo: str, token: str | None):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs = dict(token=token, attn_implementation="sdpa", device_map="cuda")
    try:
        return AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float16, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, **kwargs)


def free(*objects) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


def phase_capture(cfg: dict, token: str | None, texts: list[str]) -> dict:
    """Фаза 1: активации базовой модели, оба кандидата."""
    import torch
    from transformers import AutoTokenizer

    layer = cfg["layer"]
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], token=token)
    model = load_model(cfg["base_model"], token)
    model.eval()

    out = {f"hs{layer}_in": [], f"hs{layer + 1}_out": []}
    for text in texts:
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            res = model(**ids, output_hidden_states=True)
        last = ids["input_ids"].shape[1] - 1
        out[f"hs{layer}_in"].append(res.hidden_states[layer][0, last].float().cpu())
        out[f"hs{layer + 1}_out"].append(res.hidden_states[layer + 1][0, last].float().cpu())

    print(f"=== фаза 1: снято {len(texts)} активаций, варианты {list(out)} ===")
    free(model, res)
    return {k: torch.stack(v) for k, v in out.items()}


def phase_verbalize(cfg: dict, token: str | None, activations: dict, max_new_tokens: int) -> dict:
    """Фаза 2: активация -> объяснение, для каждого варианта тензора."""
    import torch
    from transformers import AutoTokenizer

    nla, inj = cfg["nla"], cfg["nla"]["injection"]
    tok = AutoTokenizer.from_pretrained(nla["verbalizer"], token=token)
    model = load_model(nla["verbalizer"], token)
    model.eval()

    enc = tok.apply_chat_template(
        [{"role": "user", "content": nla["prompts"]["av"].format(injection_char=inj["char"])}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    input_ids = enc["input_ids"].to(model.device)
    mask = enc["attention_mask"].to(model.device) if "attention_mask" in enc else None
    pos = int((input_ids[0] == inj["token_id"]).nonzero().flatten()[0])

    holder = {}

    def hook(module, args, output):
        if output.shape[1] > pos and "v" in holder:
            output = output.clone()
            output[:, pos, :] = holder["v"].to(output.dtype)
        return output

    handle = model.get_input_embeddings().register_forward_hook(hook)
    explanations = {}
    try:
        for variant, tensor in activations.items():
            texts = []
            for i in range(tensor.shape[0]):
                holder["v"] = tensor[i].to(model.device)
                with torch.no_grad():
                    gen = model.generate(input_ids, attention_mask=mask,
                                         max_new_tokens=max_new_tokens, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                texts.append(tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True).strip())
            explanations[variant] = texts
            print(f"=== фаза 2: {variant} — {len(texts)} объяснений ===")
    finally:
        handle.remove()

    free(model)
    return explanations


def phase_reconstruct(cfg: dict, token: str | None, explanations: dict) -> dict:
    """Фаза 3: объяснение -> активация. Оба варианта считывания сразу."""
    import torch
    from transformers import AutoTokenizer

    nla = cfg["nla"]
    tok = AutoTokenizer.from_pretrained(nla["reconstructor"], token=token)
    model = load_model(nla["reconstructor"], token)
    model.eval()

    layer_idx = model.config.num_hidden_layers - 1
    grab = {}

    def hook(module, args, output):
        # выход блока до финального RMSNorm; output — кортеж
        grab["prenorm"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.model.layers[layer_idx].register_forward_hook(hook)

    out = {}
    try:
        for variant, texts in explanations.items():
            prenorm, postnorm = [], []
            for text in texts:
                prompt = nla["prompts"]["ar"].format(explanation=text)
                ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    res = model(**ids, output_hidden_states=True)
                # critic_suffix_ids = 'text> <summary>' — хвост шаблона, значит
                # считывание идёт на последнем токене промпта
                last = ids["input_ids"].shape[1] - 1
                prenorm.append(grab["prenorm"][0, last].float().cpu())
                postnorm.append(res.hidden_states[-1][0, last].float().cpu())
            out[f"{variant}|ar_prenorm"] = torch.stack(prenorm)
            out[f"{variant}|ar_postnorm"] = torch.stack(postnorm)
            print(f"=== фаза 3: {variant} — восстановлено {len(prenorm)} ===")
    finally:
        handle.remove()

    free(model)
    return out


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text("utf-8"))
    token = hf.token()
    texts = REFUSAL + NEUTRAL
    labels = ["refusal"] * len(REFUSAL) + ["neutral"] * len(NEUTRAL)

    activations = phase_capture(cfg, token, texts)
    explanations = phase_verbalize(cfg, token, activations, args.max_new_tokens)
    reconstructions = phase_reconstruct(cfg, token, explanations)

    print("\n=== reconstruction score ===")
    print(f"{'комбинация':38} {'свои':>8} {'чужие':>8} {'разрыв':>8}")
    scores = {}
    for key, recon in reconstructions.items():
        base_variant = key.split("|")[0]
        original = activations[base_variant].numpy()
        matrix = cosine_matrix(original, recon.numpy())
        matched = float(np.mean(np.diag(matrix)))
        off = matrix.copy()
        np.fill_diagonal(off, np.nan)
        mismatched = float(np.nanmean(off))
        scores[key] = {"matched": matched, "mismatched": mismatched,
                       "gap": matched - mismatched}
        print(f"{key:38} {matched:8.4f} {mismatched:8.4f} {matched - mismatched:8.4f}")

    best = max(scores, key=lambda k: scores[k]["gap"])
    print(f"\n  ЛУЧШАЯ КОМБИНАЦИЯ: {best}  (разрыв {scores[best]['gap']:.4f})")

    # Разделяет ли сам латент два кластера — первый намёк на гейтинг-сигнал.
    print("\n=== разделение кластеров в пространстве активаций ===")
    n_ref = len(REFUSAL)
    separation = {}
    for variant, tensor in activations.items():
        arr = tensor.numpy()
        matrix = cosine_matrix(arr, arr)
        within_ref = np.mean(matrix[:n_ref, :n_ref][np.triu_indices(n_ref, 1)])
        within_neu = np.mean(matrix[n_ref:, n_ref:][np.triu_indices(len(NEUTRAL), 1)])
        between = np.mean(matrix[:n_ref, n_ref:])
        separation[variant] = {"within_refusal": float(within_ref),
                               "within_neutral": float(within_neu),
                               "between": float(between),
                               "gap": float((within_ref + within_neu) / 2 - between)}
        print(f"  {variant:14} внутри-отказ={within_ref:.4f} внутри-нейтр={within_neu:.4f} "
              f"между={between:.4f} разрыв={separation[variant]['gap']:.4f}")

    print("\n=== примеры объяснений (лучший вариант тензора) ===")
    best_variant = best.split("|")[0]
    for i in (0, 1, len(REFUSAL), len(REFUSAL) + 1):
        print(f"\n  [{labels[i]}] {texts[i]}")
        print(f"    -> {explanations[best_variant][i][:220]}")

    out = artifacts_dir("reconstruction") / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"scores": scores, "best": best, "cluster_separation": separation,
         "texts": texts, "labels": labels, "explanations": explanations},
        indent=2, ensure_ascii=False), "utf-8")
    print(f"\nОтчёт на Drive: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
