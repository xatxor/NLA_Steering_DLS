"""Оценка вариантов ответа правдоподобием самой модели.

Единственный измеритель в проекте, у которого нет собственного шума: судья не
нужен, классификатор не нужен, число получается точное. К нему пришли после
того, как эмбеддинговый классификатор дал 58.6% на задаче с двумя классами и
обрушил выводы целого пилота (эксперименты 10 и 10b).

Логарифмы нормируются на длину варианта — иначе метрика вырождается в
предпочтение коротких.
"""

from __future__ import annotations

import numpy as np


def mean_logprobs(model, tok, context: list[int], candidates: list[list[int]]) -> np.ndarray:
    """Средний логарифм вероятности каждого варианта как продолжения контекста."""
    import torch

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs = [context + c for c in candidates]
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    ids, mask = ids.to(model.device), mask.to(model.device)

    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)

    out = []
    start = len(context)
    for i, candidate in enumerate(candidates):
        # логиты в позиции j предсказывают токен j+1
        window = logprobs[i, start - 1:start - 1 + len(candidate)]
        picked = window.gather(-1, torch.tensor(candidate, device=model.device)
                               .unsqueeze(-1)).squeeze(-1)
        out.append(float(picked.mean()))
    return np.array(out)
