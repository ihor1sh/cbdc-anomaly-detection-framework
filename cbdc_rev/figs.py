"""Figures for the manuscript (static PNG, 300 dpi, Liberation Sans ~ Arial metrics, colour-blind-safe categorical palette,
direct labels so that identity never depends on colour alone)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "figure.dpi": 100, "savefig.dpi": 300})

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
COL = {"xgb": "#2a78d6", "lgbm": "#eb6834", "rf": "#1baf7a", "iforest_o": "#eda100", "iforest": "#9a9993",
       "mlp": "#e87ba4", "gcn": "#008300", "sage": "#6250d6", "gat": "#e34948"}
NAME_EN = {"iforest": "Isolation Forest (raw score)", "iforest_o": "Isolation Forest (oriented)", "xgb": "XGBoost", "lgbm": "LightGBM",
           "rf": "Random Forest", "mlp": "MLP (no graph)", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}
NAME_UA = {"iforest": "Isolation Forest (сирий бал)", "iforest_o": "Isolation Forest (з орієнтацією)", "xgb": "XGBoost", "lgbm": "LightGBM",
           "rf": "Random Forest", "mlp": "MLP (без графа)", "gcn": "GCN", "sage": "GraphSAGE", "gat": "GAT"}
CM = 1 / 2.54


def t_ci(x):
    x = np.asarray(x, float); n = len(x); m = x.mean()
    if n < 2: return m, m, m
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, m - h, m + h


def _comma(ax, axis="x"):
    """Ukrainian decimal comma on the value axis."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, p: (f"{v:g}").replace(".", ","))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def _style(ax, xlabel=None):
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    if xlabel: ax.set_xlabel(xlabel, color=INK2)
    ax.tick_params(colors=INK2)


def fig_detection(df, out, lang="ua", cond="none"):
    names = NAME_UA if lang == "ua" else NAME_EN
    kinds = [k for k in ["iforest", "iforest_o", "xgb", "lgbm", "rf", "mlp", "gcn", "sage", "gat"] if ((df.kind == k) & (df.cond == cond)).any()]
    fig, axs = plt.subplots(1, 2, figsize=(17 * CM, 6.2 * CM), sharey=True)
    for ax, met, title in ((axs[0], "auc_pr", "AUC-PR"), (axs[1], "rec_at_fpr5", "recall @ FPR ≤ 5 %" if lang == "en" else "повнота @ FPR ≤ 5 %")):
        for i, k in enumerate(kinds[::-1]):
            v = df[(df.kind == k) & (df.cond == cond)][met].values
            m, lo, hi = t_ci(v)
            ax.plot([lo, hi], [i, i], color=COL[k], lw=1.6, solid_capstyle="butt")
            ax.plot(v, np.full_like(v, i, dtype=float), "o", ms=2.2, color=COL[k], alpha=0.45, mec="none")
            ax.plot([m], [i], "o", ms=5.5, color=COL[k], mec="white", mew=0.8)
            ax.text(min(hi + 0.012, 0.985), i, (f"{m:.3f}".replace(".", ",") if lang == "ua" else f"{m:.3f}"), va="center", ha="left", fontsize=7, color=INK)
        ax.set_yticks(range(len(kinds))); ax.set_yticklabels([names[k] for k in kinds[::-1]], color=INK)
        ax.set_xlim(0, 1.0); ax.set_title(title, loc="left", color=INK, fontsize=9); _style(ax)
        if lang == "ua": _comma(ax)
    fig.tight_layout(w_pad=1.2); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)



COND_STYLE = {"none": dict(c="#52514e", m="o", ua="без оверлею", en="no overlay"),
              "raw42": dict(c="#e34948", m="s", ua="оверлей, сирий масштаб (як у початковому пакеті)", en="overlay, raw scale (as in the initial package)"),
              "std42": dict(c="#2a78d6", m="D", ua="оверлей, стандартизований", en="overlay, standardised"),
              "noise": dict(c="#eda100", m="^", ua="5 шумових колонок N(0,1)", en="5 noise columns N(0,1)")}


def fig_ablation(df, out, lang="ua", kinds=("xgb", "lgbm", "rf", "mlp", "gcn", "sage", "gat")):
    names = NAME_UA if lang == "ua" else NAME_EN
    kinds = [k for k in kinds if ((df.kind == k) & (df.cond == "raw42")).any() or ((df.kind == k) & (df.cond == "std42")).any()]
    conds = ["none", "raw42", "std42", "noise"]
    fig, axs = plt.subplots(1, 2, figsize=(17 * CM, 7.4 * CM), sharey=True)
    panels = ((axs[0], "auc_pr", "AUC-PR" if lang == "en" else "AUC-PR", (0, 1.0)),
              (axs[1], "flag_rate_t05", "share of test transactions flagged at τ = 0.5" if lang == "en" else "частка позначених нелегальними, τ = 0,5", (0, 1.0)))
    nC = len(conds); step = 0.17
    for ax, met, title, xl in panels:
        seen = set()                                   # the legend label goes on the first marker of each condition
        for i, k in enumerate(kinds[::-1]):
            for j, c in enumerate(conds):
                g = df[(df.kind == k) & (df.cond == c)]
                if not len(g): continue
                y0 = i + (j - (nC - 1) / 2) * step
                v = g[met].values; m, lo, hi = t_ci(v); st = COND_STYLE[c]
                ax.plot([lo, hi], [y0, y0], color=st["c"], lw=1.0, solid_capstyle="butt", zorder=2)
                ax.plot(v, np.full_like(v, y0, dtype=float), st["m"], ms=2.0, color=st["c"], alpha=0.35, mec="none", zorder=2)
                ax.plot([m], [y0], st["m"], ms=4.6, color=st["c"], mec="white", mew=0.6, zorder=3, label=(st[lang] if c not in seen else None)); seen.add(c)
        ax.set_yticks(range(len(kinds))); ax.set_yticklabels([names[k] for k in kinds[::-1]], color=INK)
        ax.set_xlim(*xl); ax.set_title(title, loc="left", color=INK, fontsize=8.5); _style(ax)
        ax.set_ylim(-0.6, len(kinds) - 0.4)
        if lang == "ua": _comma(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.0, rect=(0, 0.13, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=7.5, handletextpad=0.4, columnspacing=1.4)
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_latency(lat, out, lang="ua", budgets=(1, 5, 20, 100)):
    names = NAME_UA if lang == "ua" else NAME_EN
    kinds = [k for k in ["iforest", "xgb", "lgbm", "rf", "mlp", "gcn", "sage", "gat"] if k in lat]
    fig, ax = plt.subplots(figsize=(17 * CM, 6.0 * CM))
    for i, k in enumerate(kinds[::-1]):
        s = lat[k]["single"]; col = COL[k]
        ax.plot([s["p95_ci"][0], s["p95_ci"][1]], [i, i], color=col, lw=1.4, solid_capstyle="butt", zorder=2)
        ax.plot([s["p50_ms"]], [i], "o", ms=3.2, color=col, alpha=0.55, mec="none", zorder=3)
        ax.plot([s["p95_ms"]], [i], "o", ms=5.5, color=col, mec="white", mew=0.7, zorder=4)
        ax.plot([s["p99_ms"]], [i], "|", ms=7, color=col, mew=1.3, zorder=3)
        ax.plot([lat[k]["batch_amortised_ms"]], [i], "D", ms=3.2, mfc="white", mec=col, mew=0.9, zorder=3)
        ax.text(s["p95_ms"] * 1.18, i + 0.22, f"{s['p95_ms']:.2f}".replace(".", ",") if lang == "ua" else f"{s['p95_ms']:.2f}", fontsize=6.8, color=INK, va="bottom")
    ax.set_xscale("log"); ax.set_yticks(range(len(kinds))); ax.set_yticklabels([names[k] for k in kinds[::-1]], color=INK)
    for b in budgets: ax.axvline(b, color=GRID, lw=0.8, zorder=1); ax.text(b, len(kinds) + 0.02, f"{b} мс" if lang == "ua" else f"{b} ms", fontsize=6.5, color=INK2, ha="center", va="bottom")
    ax.set_xlabel("затримка оцінювання однієї транзакції, мс (логарифмічна шкала)" if lang == "ua" else "single-transaction scoring latency, ms (log scale)", color=INK2)
    ax.set_ylim(-0.6, len(kinds) + 0.3); ax.tick_params(colors=INK2); ax.grid(axis="x", color=GRID, lw=0.4, which="minor", alpha=0.5)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=INK2, ms=5.5, mec="white", label="p95 (95 % ДІ бутстрепу)" if lang == "ua" else "p95 (bootstrap 95 % CI)"),
          Line2D([], [], marker="o", ls="", color=INK2, ms=3.2, alpha=0.55, label="p50"), Line2D([], [], marker="|", ls="", color=INK2, ms=7, mew=1.3, label="p99"),
          Line2D([], [], marker="D", ls="", mfc="white", mec=INK2, ms=3.2, label="середнє у пакеті 10 000" if lang == "ua" else "mean in a 10,000 batch")]
    if lang == "ua":
        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: (f"{v:g}").replace(".", ",")))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=hs, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=7)
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def _sel_matrix(by_budget, kinds, budgets):
    M = np.zeros((len(kinds), len(budgets))); F = np.zeros_like(M, dtype=bool)
    for j, B in enumerate(budgets):
        d = by_budget[str(B)] if str(B) in by_budget else by_budget[B]
        for i, k in enumerate(kinds):
            M[i, j] = d["summary"]["appears"].get(k, 0); F[i, j] = k in d["front"]
    return M, F


def fig_selection(sel, out, kinds, lang="ua", budgets=(1.0, 5.0, 20.0, 100.0)):
    """Robustness of the choice to the order of criteria.  Panel (a): main analysis (native TreeSHAP); panel (b), when the attribution-space
    control exists: S of the three tree ensembles taken from the common attribution space (selection.json['common_space'])."""
    names = NAME_UA if lang == "ua" else NAME_EN
    cs = sel.get("common_space")
    panels = [(sel["by_budget"], "(а) основний аналіз (рідний S)" if lang == "ua" else "(a) main analysis (native S)")]
    if cs: panels.append((cs["selection"], "(б) S зі спільного простору" if lang == "ua" else "(b) S from the common space"))
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=((11.5 if n == 1 else 16.5) * CM, 6.4 * CM))
    axes = np.atleast_1d(axes)
    for p, (ax, (byb, title)) in enumerate(zip(axes, panels)):
        M, F = _sel_matrix(byb, kinds, budgets)
        ax.imshow(M, cmap="Blues", vmin=0, vmax=24, aspect="auto")
        for i in range(len(kinds)):
            for j in range(len(budgets)):
                ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 14 else INK, fontweight="bold" if M[i, j] > 0 else "normal")
                if not F[i, j]: ax.text(j, i + 0.34, "поза фронтом" if lang == "ua" else "off-front", ha="center", va="center", fontsize=5.2, color=INK2)
        ax.set_xticks(range(len(budgets))); ax.set_xticklabels([f"{int(b)} мс" if lang == "ua" else f"{int(b)} ms" for b in budgets], color=INK)
        ax.set_yticks(range(len(kinds)))
        if p == 0: ax.set_yticklabels([names[k] for k in kinds], color=INK)
        else: ax.tick_params(labelleft=False)
        ax.set_xlabel("бюджет затримки B (p95)" if lang == "ua" else "latency budget B (p95)", color=INK2)
        ax.set_title(title, loc="left", fontsize=8, color=INK)
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(length=0)
    sub = ("число в клітинці — у скількох із 24 порядків пріоритетів модель входить до множини переможців" if lang == "ua"
           else "cell value = in how many of the 24 priority orders the model is among the winners")
    fig.text(0.0, -0.02, sub, ha="left", va="top", fontsize=7, color=INK2)
    fig.tight_layout(w_pad=2.0); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    pass
