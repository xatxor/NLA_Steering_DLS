"""Обёртки над двумя половинами NLA.

`Verbalizer` (av) переводит активацию в текст, `Reconstructor` (ar) — текст
обратно в активацию. Обе загружают по ~6 ГБ, поэтому используются как
контекстные менеджеры и освобождают память на выходе: на T4 держать их
одновременно с базовой моделью нельзя.

Протокол восстановлен из `nla_meta.yaml`, подробности — в EXPERIMENTS.md (00–03). Коротко:
активация подставляется хуком вместо эмбеддинга символа U+320E в промпте,
никакого масштабирования к ней не применяется.
"""

from __future__ import annotations

import gc

import numpy as np


def load_model(repo: str, token: str | None):
    """fp16 + sdpa. Имя аргумента dtype в transformers со временем менялось."""
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


class Verbalizer:
    """av: активация -> текстовое объяснение."""

    def __init__(self, cfg: dict, token: str | None):
        self.cfg, self.token = cfg, token
        self.model = self.tok = None

    def __enter__(self):
        from transformers import AutoTokenizer

        nla, inj = self.cfg["nla"], self.cfg["nla"]["injection"]
        self.tok = AutoTokenizer.from_pretrained(nla["verbalizer"], token=self.token)
        self.model = load_model(nla["verbalizer"], self.token)
        self.model.eval()

        enc = self.tok.apply_chat_template(
            [{"role": "user",
              "content": nla["prompts"]["av"].format(injection_char=inj["char"])}],
            add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
        self.input_ids = enc["input_ids"].to(self.model.device)
        self.mask = enc["attention_mask"].to(self.model.device) if "attention_mask" in enc else None

        positions = (self.input_ids[0] == inj["token_id"]).nonzero().flatten().tolist()
        if len(positions) != 1:
            raise RuntimeError(f"позиция инъекции не единственна: {positions}")
        self.pos = positions[0]
        self._holder = {}

        def hook(module, args, output):
            # при генерации с KV-кэшем embed_tokens зовётся и для одиночных
            # новых токенов — подменяем только на первом полном проходе
            if output.shape[1] > self.pos and "v" in self._holder:
                output = output.clone()
                output[:, self.pos, :] = self._holder["v"].to(output.dtype)
            return output

        self._handle = self.model.get_input_embeddings().register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        free(self.model)
        self.model = None
        return False

    def verbalize(self, vectors: np.ndarray, max_new_tokens: int = 60) -> list[str]:
        import torch

        texts = []
        for vector in np.atleast_2d(vectors):
            self._holder["v"] = torch.as_tensor(vector, dtype=torch.float32,
                                                device=self.model.device)
            with torch.no_grad():
                out = self.model.generate(self.input_ids, attention_mask=self.mask,
                                          max_new_tokens=max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
            texts.append(self.tok.decode(out[0, self.input_ids.shape[1]:],
                                         skip_special_tokens=True).strip())
        return texts


class Reconstructor:
    """ar: текст -> активация слоя 24.

    Модель обрезана до 25 слоёв, и в чекпоинте нет `model.norm.weight` — HF
    инициализирует его единицами, поэтому `hidden_states[-1]` отличается от
    выхода блока только скалярным масштабом. Читаем выход блока хуком: для
    косинуса разницы нет, а для норм и MSE есть.
    """

    def __init__(self, cfg: dict, token: str | None):
        self.cfg, self.token = cfg, token
        self.model = self.tok = None

    def __enter__(self):
        from transformers import AutoTokenizer

        repo = self.cfg["nla"]["reconstructor"]
        self.tok = AutoTokenizer.from_pretrained(repo, token=self.token)
        self.model = load_model(repo, self.token)
        self.model.eval()
        self._grab = {}

        def hook(module, args, output):
            self._grab["h"] = (output[0] if isinstance(output, tuple) else output).detach()

        last = self.model.config.num_hidden_layers - 1
        self._handle = self.model.model.layers[last].register_forward_hook(hook)
        return self

    def __exit__(self, *exc):
        self._handle.remove()
        free(self.model)
        self.model = None
        return False

    def reconstruct(self, texts: list[str]) -> np.ndarray:
        import torch

        template = self.cfg["nla"]["prompts"]["ar"]
        out = []
        for text in texts:
            ids = self.tok(template.format(explanation=text), return_tensors="pt",
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                self.model(**ids)
            # critic_suffix_ids = 'text> <summary>' — хвост шаблона, значит
            # считывание идёт на последнем токене промпта
            out.append(self._grab["h"][0, -1].float().cpu().numpy())
        return np.stack(out)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.atleast_2d(a) / np.linalg.norm(np.atleast_2d(a), axis=1, keepdims=True)
    b = np.atleast_2d(b) / np.linalg.norm(np.atleast_2d(b), axis=1, keepdims=True)
    return (a * b).sum(1)
