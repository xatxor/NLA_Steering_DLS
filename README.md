# NLA Steering

На каких токенах нужно применять activation steering? Проверяем, умеет ли
Natural Language Autoencoder находить токены, где модель действительно «думает»
о концепте, и выигрывает ли такой гейтинг у стиринга на всех токенах подряд.

Модель — Qwen2.5-3B-Instruct, NLA на слое 24, концепт — refusal.
Постановка задачи: [`постановка_задачи.txt`](постановка_задачи.txt).

## Установка (локально)

```bash
uv sync
cp .env.example .env   # и вписать HF_TOKEN
```

Локальное окружение — только анализ: torch и веса моделей не ставятся.

## Запуск GPU-экспериментов

1. Открыть [`colab/runner.ipynb`](colab/runner.ipynb) в Colab, поставить runtime **T4 GPU**,
   добавить в Secrets `HF_TOKEN` (веса) и `GH_TOKEN` (клонирование приватного
   репозитория; не нужен, если сделать репозиторий публичным).
2. Выполнить ячейки сверху вниз; последняя — воркер, её оставить работать.
3. Ставить задачи в очередь:

```python
from nla_steering import jobs
job = jobs.submit("inspect", "scripts/00_inspect_nla.py")
jobs.wait(job.id)
```

Задачи и результаты ходят через Google Drive (`nla_steering/` в корне диска),
который на Windows виден как обычная папка — скачивать ничего не нужно.

## Структура

```
src/nla_steering/   библиотека: пути, очередь задач, хуки, векторы, метрики
scripts/            нумерованные точки входа с argparse
configs/            один yaml = один эксперимент
results/            мелкие агрегаты, коммитятся
artifacts/          тяжёлые дампы, живут на Drive
colab/runner.ipynb  GPU-воркер
```
