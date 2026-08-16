"""Применение steering-вектора на выбранных шагах генерации.

Позиция задаётся в терминах шага генерации, а не позиции в тензоре: шаг 0 — это
prefill, где активация последней позиции порождает первый токен ответа; шаг g>0
обрабатывает один токен и порождает следующий. Именно это и есть «стирить
токен t» в терминах проекта.
"""

from __future__ import annotations


class Steerer:
    """Хук на блок слоя: добавляет вектор на заданных шагах генерации.

    `targets=None` — стирить везде (режим all-token). Пустое множество —
    не стирить нигде, что удобно как контроль с идентичным кодовым путём.
    """

    def __init__(self, model, layer: int, vector=None):
        self.model = model
        self.vector = vector
        self.block = model.model.layers[layer]
        self.targets: set[int] | None = set()
        self.step = 0
        self._handle = None

    def __enter__(self):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hit = self.targets is None or self.step in self.targets
            if hit and self.vector is not None:
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

    def reset(self, targets: set[int] | None) -> None:
        """Задать шаги для следующей генерации и обнулить счётчик."""
        self.targets, self.step = targets, 0
