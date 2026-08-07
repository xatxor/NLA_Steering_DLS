"""Первый настоящий прогон NLA: работает ли вербализация активаций.

Отвечает на два вопроса, без которых нельзя писать ни гейтинг, ни стиринг.

**Какой именно тензор считать «активацией слоя 24».** В `nla_meta.yaml` стоит
`extraction_layer_index: 24`, но этого мало. В HF `hidden_states[i]` — это вход
слоя `i`, то есть выход слоя 24 лежит в `hidden_states[25]`. Вдобавок последний
элемент `hidden_states` у Qwen2 отдаётся уже после финального RMSNorm, а
промежуточные — нет. Для декодера `ar`, обрезанного ровно до 25 слоёв, «выход
последнего слоя» может означать любой из этих вариантов, и перепутать их —
значит молча считать всё остальное на неправильном тензоре.

**В какую сторону применять `mse_scale = 45.2548`.** В метаданных сказано только
что он есть. Активация подставляется вместо эмбеддинга, поэтому её масштаб
должен быть сопоставим с эмбеддингами; отношение норм это и покажет.

Метод — прямой перебор: берём тексты с заведомо разной семантикой, снимаем с них
активации всеми вариантами, подставляем в `av` со всеми масштабами и смотрим, в
какой комбинации объяснения осмысленны. Правильная комбинация должна назвать
Париж для текста про Париж и отказ для текста с отказом; неправильные дадут
мусор или шаблонную воду.

Двухфазный прогон: база (~6.2 ГБ) выгружается перед загрузкой `av` (~6.2 ГБ),
потому что на T4 держать обе одновременно почти не остаётся места на KV-кэш.

Запуск:
    python scripts/02_activation_roundtrip.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering import hf  # noqa: E402
from nla_steering.paths import artifacts_dir  # noqa: E402

# Тексты подобраны так, чтобы объяснение можно было проверить глазами за секунду:
# каждый про своё, и «правильная» вербализация обязана это своё назвать.
PROBES = [
    ("paris", "The capital of France is Paris"),
    ("refusal", "I'm sorry, but I cannot help you with that request"),
    ("code", "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)"),
    ("medical", "The patient was diagnosed with bacterial pneumonia and prescribed antibiotics"),
]


def load_model(repo: str, token: str | None):
    """Загрузка в fp16 + sdpa. Имя аргумента dtype в transformers менялось."""
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


def phase_capture(cfg: dict, token: str | None) -> dict:
    """Фаза 1: снять с базовой модели активации на слое 24, все варианты."""
    import torch
    from transformers import AutoTokenizer

    layer = cfg["layer"]
    tok = AutoTokenizer.from_pretrained(cfg["base_model"], token=token)
    model = load_model(cfg["base_model"], token)
    model.eval()

    # Норма эмбеддингов — точка отсчёта: активацию подставляют именно вместо
    # эмбеддинга, поэтому масштабы должны быть сопоставимы.
    emb = model.get_input_embeddings().weight
    emb_norm = emb.float().norm(dim=-1).mean().item()

    captured, stats = {}, {}
    for name, text in PROBES:
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states
        last = ids["input_ids"].shape[1] - 1

        # hidden_states[i] — вход слоя i, значит выход слоя 24 лежит в [25].
        # Третий вариант — не кандидат, а контроль: у 36-слойной базы hs[-1]
        # это выход ПОСЛЕДНЕГО слоя (35) после финального RMSNorm, к слою 24
        # отношения не имеет. Если все три дадут одинаковый текст, значит
        # инъекция вообще не влияет на генерацию.
        variants = {
            f"hs{layer}_in": hs[layer][0, last],
            f"hs{layer + 1}_out": hs[layer + 1][0, last],
            "control_final_layer": hs[-1][0, last],
        }
        captured[name] = {k: v.float().cpu() for k, v in variants.items()}
        stats[name] = {k: round(v.float().norm().item(), 3) for k, v in variants.items()}

    report = {
        "embedding_norm_mean": round(emb_norm, 4),
        "activation_norms": stats,
        "n_hidden_states": len(hs),
        "num_layers": model.config.num_hidden_layers,
    }

    print("=== фаза 1: снятие активаций с базовой модели ===")
    print(f"  средняя норма эмбеддинга: {emb_norm:.4f}")
    print(f"  hidden_states: {len(hs)} штук при {model.config.num_hidden_layers} слоях")
    for name, norms in stats.items():
        print(f"  {name:9} " + "  ".join(f"{k}={v}" for k, v in norms.items()))

    scale = cfg["nla"]["mse_scale"]
    any_norm = next(iter(stats.values()))[f"hs{layer + 1}_out"]
    print(f"\n  отношения к норме эмбеддинга ({emb_norm:.3f}):")
    print(f"    как есть  : {any_norm / emb_norm:.2f}x")
    print(f"    / {scale:.2f} : {any_norm / scale / emb_norm:.2f}x")
    print(f"    * {scale:.2f} : {any_norm * scale / emb_norm:.2f}x")

    free(model, out)
    return {"captured": captured, "report": report}


def phase_verbalize(cfg: dict, token: str | None, captured: dict, max_new_tokens: int) -> dict:
    """Фаза 2: подставить активации в av и посмотреть на объяснения."""
    import torch
    from transformers import AutoTokenizer

    nla = cfg["nla"]
    inj = nla["injection"]
    scale = nla["mse_scale"]

    tok = AutoTokenizer.from_pretrained(nla["verbalizer"], token=token)
    model = load_model(nla["verbalizer"], token)
    model.eval()

    prompt = nla["prompts"]["av"].format(injection_char=inj["char"])
    # apply_chat_template в свежих версиях отдаёт BatchEncoding, а не голый
    # тензор, даже при return_tensors='pt' — берём input_ids явно.
    enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    positions = (input_ids[0] == inj["token_id"]).nonzero().flatten().tolist()
    if len(positions) != 1:
        raise RuntimeError(f"позиция инъекции не единственна: {positions}")
    pos = positions[0]
    print(f"\n=== фаза 2: вербализация (позиция инъекции {pos}) ===")

    vector_holder = {}

    def hook(module, args, output):
        # Во время генерации с KV-кэшем embed_tokens зовётся и для одиночных
        # новых токенов — подменять нужно только на первом полном проходе.
        if output.shape[1] > pos and "vector" in vector_holder:
            output = output.clone()
            output[:, pos, :] = vector_holder["vector"].to(output.dtype)
        return output

    handle = model.get_input_embeddings().register_forward_hook(hook)

    scalings = {"raw": 1.0, "div": 1.0 / scale, "mul": scale}
    results = {}
    try:
        for probe_name, variants in captured.items():
            for variant_name, vector in variants.items():
                for scaling_name, factor in scalings.items():
                    key = f"{probe_name}|{variant_name}|{scaling_name}"
                    vector_holder["vector"] = (vector * factor).to(model.device)
                    with torch.no_grad():
                        out = model.generate(
                            input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=tok.eos_token_id,
                        )
                    text = tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
                    results[key] = text.strip()
    finally:
        handle.remove()

    free(model)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text("utf-8"))
    token = hf.token()

    capture = phase_capture(cfg, token)
    results = phase_verbalize(cfg, token, capture["captured"], args.max_new_tokens)

    print("\n=== объяснения ===")
    for key, text in results.items():
        probe, variant, scaling = key.split("|")
        print(f"\n--- {probe} / {variant} / {scaling}")
        print(f"    {text[:300]}")

    out = artifacts_dir("roundtrip") / f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"report": capture["report"], "explanations": results},
                   indent=2, ensure_ascii=False),
        "utf-8",
    )
    print(f"\nОтчёт на Drive: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
