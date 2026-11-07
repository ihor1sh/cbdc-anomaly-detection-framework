"""Renders the manuscript figures from the result files.   usage: python -m cbdc_rev.make_figs"""
import os, json
from . import analysis as A, figs as F
from .common import Elliptic
from .report import S, R, TB, PRIMARY, CAND

FIG = os.path.join(S, "figures")


def main(lang="ua"):
    os.makedirs(FIG, exist_ok=True)
    df = A.load(R); df = A.add_derived(df, int(Elliptic().m_test.sum()))
    F.fig_detection(df, f"{FIG}/fig1_detection_{lang}.png", lang=lang, cond=PRIMARY)
    F.fig_ablation(df, f"{FIG}/fig2_ablation_{lang}.png", lang=lang)
    lp = f"{R}/latency.json"
    if os.path.exists(lp):
        lat = json.load(open(lp)); F.fig_latency(lat, f"{FIG}/fig3_latency_{lang}.png", lang=lang)
    sp = f"{TB}/selection.json"
    if os.path.exists(sp):
        sel = json.load(open(sp)); F.fig_selection(sel, f"{FIG}/fig4_selection_{lang}.png", [k for k in CAND], lang=lang)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
