"""Графики для отчёта. Локально, без GPU.

Собирает данные из артефактов на Drive, пересчитывает итоговую таблицу
сравнения из кэша фазы 3 (все режимы × все α) и рисует пять рисунков в PDF.

Правила оформления взяты из руководства по визуализации: категориальные цвета
назначаются по фиксированному порядку слотов и не циклятся; для одной серии
легенды нет, её роль играет заголовок; сетка и оси приглушены; на диаграммах
рассеяния не больше трёх цветов, потому что при попарном сравнении всех восьми
слотов различимость не гарантируется.

Запуск:
    uv run python scripts/19_report_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nla_steering.metrics import is_refusal, refusal_onset  # noqa: E402
from nla_steering.paths import artifacts_dir, repo_root  # noqa: E402

# Палитра: слоты 1–3 категориальные, остальное — хром и чернила.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

MODES = ["none", "all", "random", "first_k", "cosine", "probe", "nla_latent",
         "cosine_pos_matched", "nla_latent_pos_matched"]
LABEL = {"none": "без стиринга", "all": "все токены", "random": "случайно",
         "first_k": "первые k", "cosine": "cosine (CAST)", "probe": "probe",
         "nla_latent": "NLA-латент", "cosine_pos_matched": "cosine, контроль",
         "nla_latent_pos_matched": "NLA, контроль"}


def save(fig, out, name):
    """PDF для LaTeX и PNG для визуального контроля перед сдачей."""
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=150)


def style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)


def latest(sub, pattern="*.json"):
    files = sorted(artifacts_dir(sub).glob(pattern))
    if not files:
        raise SystemExit(f"нет артефактов в {sub}")
    return files[-1]


def refusal_onsets_from_dump(tok) -> np.ndarray:
    """Момент отказа по всему дампу генерации JBB, а не по 11 промптам пилота."""
    path = sorted(artifacts_dir("generation").glob("*_texts.json"))[-1]
    payload = json.loads(path.read_text("utf-8"))
    onsets = []
    for text in payload["responses"]:
        if not is_refusal(text):
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"]
        onset = refusal_onset(tok, ids)
        if onset is not None:
            onsets.append(onset)
    return np.array(onsets)


def comparison_table(budget=2):
    """Пересчёт таблицы из кэша фазы 3: все режимы, все α."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from nla_steering import hf

    cache = artifacts_dir("comparison") / "cache"
    key = "150_32_0"
    df = pd.read_parquet(hf_hub_download(
        "natolambert/xstest-v2-copy", "data/prompts-00000-of-00001.parquet",
        repo_type="dataset", token=hf.token()))
    df = df.assign(unsafe=df["type"].str.startswith("contrast_").astype(int))
    rng = np.random.default_rng(0)
    parts = []
    for label in (0, 1):
        sub = df[df.unsafe == label]
        parts.append(sub.iloc[rng.permutation(len(sub))[:150]])
    data = pd.concat(parts, ignore_index=True)
    safe = data.unsafe.to_numpy() == 0

    phase1 = np.load(cache / f"phase1_{key}.npz", allow_pickle=True)
    rows = {("none", 0.0): list(phase1["baseline_text"])}
    for path in sorted(cache.glob("phase3_*.npz")):
        stem = path.stem[len("phase3_"):]
        mode, _, tail = stem.partition(f"_{key}_{budget}_")
        if not tail:
            continue
        rows[(mode, float(tail))] = list(np.load(path, allow_pickle=True)["text"])

    table = {}
    for (mode, alpha), texts in rows.items():
        ref = np.array([is_refusal(t) for t in texts])
        table[(mode, alpha)] = {"over": float(ref[safe].mean()),
                                "unsafe": float(ref[~safe].mean())}
    return table


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    from nla_steering import hf

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "pdf.fonttype": 42})
    out = repo_root() / "report" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", token=hf.token())
    numbers = {}

    # ---------- рисунок 1: когда фиксируется решение ----------
    refusal = refusal_onsets_from_dump(tok) / 48.0
    tqa = np.load(sorted(artifacts_dir("truthfulness").glob("*_ll.npz"))[-1])
    truth = tqa["onsets"] / 32.0
    cot = np.array(json.loads(latest("cot").read_text("utf-8"))["collapse_wrong"])

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.7), sharey=True)
    panels = [("refusal (JBB)", refusal), ("truthfulness (TruthfulQA)", truth),
              ("рассуждение (GSM8K)", cot)]
    bins = np.linspace(0, 1, 11)
    for ax, (name, values) in zip(axes, panels):
        ax.hist(values, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.8,
                weights=np.ones(len(values)) / len(values))
        style(ax, xlabel="доля ответа", title=f"{name}\nn = {len(values)}")
        ax.set_xlim(0, 1)
    axes[0].set_ylabel("доля примеров", color=INK2, fontsize=9)
    for ax in axes:
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    save(fig, out, "fig1_timing")
    plt.close(fig)
    numbers["timing"] = {name: {"n": int(len(v)), "median": float(np.median(v)),
                                "std": float(v.std()),
                                "frac_first_10pct": float((v <= 0.1).mean())}
                         for name, v in panels}

    # ---------- рисунок 2: карта полезности ----------
    lift = np.array(json.loads(
        (repo_root() / "results" / "08b_utility_baseline.json").read_text("utf-8"))["lift"])
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    colors = [BLUE if i <= 2 else MUTED for i in range(len(lift))]
    ax.bar(range(len(lift)), lift * 100, color=colors, width=0.7, edgecolor=SURFACE,
           linewidth=0.6)
    ax.axhline(0, color=BASE, linewidth=0.8)
    style(ax, xlabel="позиция стирённого токена",
          ylabel="прирост исправлений, п.п.",
          title="Полезность вмешательства локализована в первых трёх токенах")
    for i in range(3):
        ax.annotate(f"+{lift[i] * 100:.0f}", (i, lift[i] * 100), ha="center",
                    va="bottom", fontsize=8, color=INK)
    fig.tight_layout()
    save(fig, out, "fig2_utility")
    plt.close(fig)
    numbers["utility_lift"] = lift.tolist()

    # ---------- рисунок 3: гейт против позиционного контроля ----------
    # Диаграмма рассеяния читалась плохо: точки гейта и его контроля почти
    # совпадают, подписи налезали, а главное сообщение терялось. Парные точки на
    # общей оси показывают совпадение прямо.
    table = comparison_table()
    pairs = [("cosine", 0.5), ("cosine", 1.0), ("nla_latent", 0.5), ("nla_latent", 1.0)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9), sharey=True)
    for ax, (key, axis_label) in zip(axes, [("over", "over-refusal на безопасных, %"),
                                            ("unsafe", "отказ на опасных, %")]):
        for row, (gate, alpha) in enumerate(pairs):
            g = table[(gate, alpha)][key] * 100
            c = table[(f"{gate}_pos_matched", alpha)][key] * 100
            ax.plot([g, c], [row, row], color=MUTED, linewidth=1.4, zorder=2)
            ax.scatter([g], [row], s=80, color=BLUE, edgecolor=SURFACE, linewidth=1.2,
                       zorder=3, label="гейт" if row == 0 else None)
            # контрольная точка меньше: там, где значения совпадают ровно, синяя
            # остаётся видна кольцом вокруг оранжевой
            ax.scatter([c], [row], s=26, color=ORANGE, edgecolor=SURFACE, linewidth=0.9,
                       zorder=4, label="позиционный контроль" if row == 0 else None)
        ax.axvline(table[("none", 0.0)][key] * 100, color=BASE, linewidth=1.0,
                   linestyle="--", zorder=1)
        style(ax, xlabel=axis_label)
        ax.set_yticks(range(len(pairs)))
        ax.set_ylim(-0.6, len(pairs) - 0.4)
    axes[0].set_yticklabels([LABEL[g] + "\n" + f"α={a:g}" for g, a in pairs],
                            fontsize=8, color=INK2)
    axes[0].set_title("Содержательный гейт не отличается от контроля\n"
                      "с тем же распределением позиций", color=INK, fontsize=10,
                      loc="left", pad=8)
    # легенда вынесена под оси: внутри панели она перекрывала точки
    handles, labels = axes[1].get_legend_handles_labels()
    leg = fig.legend(handles, labels, frameon=False, fontsize=8, ncol=2,
                     loc="lower center", bbox_to_anchor=(0.5, -0.06))
    for text in leg.get_texts():
        text.set_color(INK2)
    fig.tight_layout()
    save(fig, out, "fig3_control")
    plt.close(fig)
    numbers["table"] = {f"{m}@{a:g}": v for (m, a), v in table.items()}

    # ---------- рисунок 4: конфаунд ----------
    picks = json.loads(
        (repo_root() / "results" / "13_verification.json").read_text("utf-8"))["pick_positions"]
    base_over = table[("none", 0.0)]["over"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    xs, ys, names = [], [], []
    for gate, stats in picks.items():
        cell = table.get((gate, 0.5))
        if cell is None:
            continue
        xs.append(stats["frac_first3"] * 100)
        ys.append((base_over - cell["over"]) * 100)
        names.append(LABEL[gate])
    ax.scatter(xs, ys, s=52, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    # ручные смещения: probe и random дают почти нулевое снижение и их подписи
    # иначе накладываются друг на друга
    offsets = {"probe": (7, 6), "случайно": (7, -10)}
    for x, y, name in zip(xs, ys, names):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=offsets.get(name, (7, -3)), fontsize=8, color=INK2)
    style(ax, xlabel="доля выборов в позициях 0–2, %",
          ylabel="снижение over-refusal, п.п.",
          title="Сила гейта объясняется тем, как часто он\nвыбирает первые токены")
    fig.tight_layout()
    save(fig, out, "fig4_confound")
    plt.close(fig)
    numbers["confound"] = {n: {"frac_first3": x, "reduction": y}
                           for n, x, y in zip(names, xs, ys)}

    # ---------- рисунок 5: свип по α для truthfulness ----------
    per_q = np.load(sorted(artifacts_dir("truth_inform").glob("*_per_question.npz"))[-1])
    alphas = per_q["alphas"]
    sig = json.loads(
        (repo_root() / "results" / "18_truth_significance.json").read_text("utf-8"))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for key, color, legend in (("mc2_all", BLUE, "все верные ответы"),
                               ("mc2_committal", ORANGE, "только содержательные")):
        means = per_q[key].mean(axis=1)
        err = per_q[key].std(axis=1) / np.sqrt(per_q[key].shape[1])
        ax.errorbar(alphas, means, yerr=err, color=color, linewidth=2, marker="o",
                    markersize=5, capsize=2.5, elinewidth=0.9, label=legend, zorder=3)
    ax.axvline(0, color=BASE, linewidth=0.8)
    for alpha_str, row in sig["by_alpha"].items():
        if row["mcnemar_p"] < 0.05:
            a = float(alpha_str)
            i = int(np.where(alphas == a)[0][0])
            ax.annotate("*", (a, per_q["mc2_committal"][i].mean() - 0.14),
                        ha="center", fontsize=13, color=INK)
    style(ax, xlabel="α (сила стиринга вдоль вектора правдивости)",
          ylabel="MC2, средний score",
          title="Вектор значимо ухудшает правдивость и не улучшает её")
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right")
    for text in leg.get_texts():
        text.set_color(INK2)
    fig.tight_layout()
    save(fig, out, "fig5_truth_alpha")
    plt.close(fig)

    (repo_root() / "results" / "19_report_numbers.json").write_text(
        json.dumps(numbers, indent=2, ensure_ascii=False), "utf-8")
    print("рисунки:", *(p.name for p in sorted(out.glob("*.pdf"))), sep="\n  ")
    print("\nсводка чисел: results/19_report_numbers.json")
    for name, stats in numbers["timing"].items():
        print(f"  {name}: медиана {stats['median']:.3f}, std {stats['std']:.3f}, "
              f"в первых 10% — {stats['frac_first_10pct']:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
