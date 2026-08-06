"""Разведка интерфейса NLA — задача №1 проекта.

Ни у одной из опубликованных NLA-моделей нет model card, поэтому формат подачи
активации пришлось восстанавливать из метаданных. Скрипт качает только конфиги
и метаданные (десятки килобайт, не веса), поэтому запускается локально без GPU.

Ключ ко всему — файл `nla_meta.yaml` в каждом репозитории: там лежат и роль
модели, и токен инъекции, и промпт-шаблоны. Найденное зафиксировано в
`configs/base.yaml`; скрипт остаётся, чтобы проверять новые чекпоинты.

Что уже установлено:
  * `av` — полная 36-слойная модель, активация -> текстовое объяснение;
  * `ar` — та же модель, **обрезанная до 25 слоёв** (0..24), текст -> активация
    на слое 24, то есть декодер автоэнкодера;
  * спец-токенов не добавлено (vocab_delta=0): активация подставляется хуком
    вместо эмбеддинга символа U+320E (id 149705), а не через новый токен;
  * `stage` в метаданных — штамп упаковщика, а не признак обученности; реально
    чекпоинты различает поле `trained_on`.

Запуск:
    uv run python scripts/00_inspect_nla.py
    uv run python scripts/00_inspect_nla.py --repo therem/nla-qwen2.5-3b-L24-rl-v13-av
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering.paths import results_dir  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# SFT и RL версии обеих голов на слое 24 плюс базовая модель для сверки.
DEFAULT_REPOS = [
    BASE_MODEL,
    "therem/nla-qwen2.5-3b-L24-sft-av",
    "therem/nla-qwen2.5-3b-L24-sft-ar",
    "therem/nla-qwen2.5-3b-L24-rl-v13-av",
    "therem/nla-qwen2.5-3b-L24-rl-v13-ar",
    "therem/nla-qwen2.5-3b-L24-rl-iter5000-av",
]

SMALL_FILES = [
    "config.json",
    "tokenizer_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "generation_config.json",
]


def fetch_json(repo: str, filename: str, token: str | None) -> dict | None:
    try:
        path = hf_hub_download(repo, filename, token=token)
    except Exception:
        return None
    try:
        return json.loads(Path(path).read_text("utf-8"))
    except json.JSONDecodeError:
        return None


def fetch_meta(repo: str, token: str | None) -> dict:
    """nla_meta.yaml — то, что должно было быть в model card."""
    try:
        path = hf_hub_download(repo, "nla_meta.yaml", token=token)
    except Exception:
        return {}
    return yaml.safe_load(Path(path).read_text("utf-8")) or {}


def inspect(repo: str, token: str | None) -> dict:
    try:
        files = sorted(list_repo_files(repo, token=token))
    except Exception as exc:
        return {"repo": repo, "error": str(exc)}

    config = fetch_json(repo, "config.json", token) or {}
    tok_config = fetch_json(repo, "tokenizer_config.json", token) or {}
    added = fetch_json(repo, "added_tokens.json", token) or {}
    meta = fetch_meta(repo, token)

    # Токены из tokenizer_config, которых нет в стандартном наборе Qwen —
    # главный кандидат на роль слота для инъекции активации.
    added_decoder = tok_config.get("added_tokens_decoder", {})
    token_names = {int(k): v.get("content", "") for k, v in added_decoder.items()}
    token_names.update({v: k for k, v in added.items()})

    return {
        "repo": repo,
        # --- главное: то, что нашлось в nla_meta.yaml ---
        "role": meta.get("role"),
        "trained_on": meta.get("trained_on"),
        "extraction_layer_index": meta.get("extraction_layer_index"),
        "d_model": meta.get("d_model"),
        "injection": meta.get("tokens"),
        "mse_scale": (meta.get("extraction") or {}).get("mse_scale"),
        "injection_scale": (meta.get("extraction") or {}).get("injection_scale"),
        "prompt_templates": meta.get("prompt_templates"),
        "base_checkpoint": meta.get("base_checkpoint"),
        # --- сверка с базовой моделью ---
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "hidden_size": config.get("hidden_size"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "vocab_size": config.get("vocab_size"),
        "torch_dtype": config.get("torch_dtype"),
        # всё, чего нет в ванильном Qwen2Config — следы кастомной архитектуры
        "extra_config_keys": sorted(
            k for k in config if k.startswith(("nla", "act", "inject", "probe"))
        ),
        "auto_map": config.get("auto_map"),
        "added_tokens": dict(sorted(token_names.items())[-24:]),
        "has_chat_template": "chat_template" in tok_config,
        "chat_template_head": (tok_config.get("chat_template") or "")[:400],
        "custom_code": [f for f in files if f.endswith(".py")],
        "non_weight_files": [
            f for f in files if not f.endswith((".safetensors", ".bin", ".gguf"))
        ],
    }


def diff_vs_base(reports: list[dict]) -> dict:
    base = next((r for r in reports if r.get("repo") == BASE_MODEL), None)
    if not base or "error" in base:
        return {}
    out = {}
    for r in reports:
        if r.get("repo") == BASE_MODEL or "error" in r:
            continue
        out[r["repo"]] = {
            "vocab_delta": (r.get("vocab_size") or 0) - (base.get("vocab_size") or 0),
            "hidden_matches": r.get("hidden_size") == base.get("hidden_size"),
            "layers_match": r.get("num_hidden_layers") == base.get("num_hidden_layers"),
            "new_tokens": sorted(
                set(r.get("added_tokens", {}).values())
                - set(base.get("added_tokens", {}).values())
            ),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", dest="repos", help="повторяемый")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    load_dotenv()
    token = os.environ.get("HF_TOKEN") or None
    if not token:
        print("! HF_TOKEN не найден в .env — публичные репозитории всё равно откроются\n")

    repos = args.repos or DEFAULT_REPOS
    reports = []
    for repo in repos:
        print(f"→ {repo}")
        report = inspect(repo, token)
        reports.append(report)
        if "error" in report:
            print(f"  ОШИБКА: {report['error']}")
            continue
        print(f"  {report['architectures']}  hidden={report['hidden_size']} "
              f"layers={report['num_hidden_layers']} vocab={report['vocab_size']}")
        if report["role"]:
            print(f"  role={report['role']}  L={report['extraction_layer_index']}  "
                  f"d_model={report['d_model']}  mse_scale={report['mse_scale']}")
            print(f"  trained_on: {report['trained_on']}")
            inj = report["injection"] or {}
            print(f"  инъекция: char={inj.get('injection_char')!r} "
                  f"id={inj.get('injection_token_id')} "
                  f"соседи=({inj.get('injection_left_neighbor_id')}, "
                  f"{inj.get('injection_right_neighbor_id')})")
        if report["custom_code"]:
            print(f"  кастомный код: {report['custom_code']}")
        if report["extra_config_keys"]:
            print(f"  нестандартные ключи config: {report['extra_config_keys']}")

    deltas = diff_vs_base(reports)
    if deltas:
        print("\n=== отличия от базовой модели ===")
        for repo, d in deltas.items():
            print(f"{repo}")
            print(f"  vocab_delta={d['vocab_delta']}  hidden_ok={d['hidden_matches']}  "
                  f"layers_ok={d['layers_match']}")
            if d["new_tokens"]:
                print(f"  новые токены: {d['new_tokens']}")

    out = Path(args.out) if args.out else results_dir("00_nla_interface.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"reports": reports, "deltas": deltas}, indent=2, ensure_ascii=False),
        "utf-8",
    )
    print(f"\nОтчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
