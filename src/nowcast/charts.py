"""Static README figures (matplotlib PNG). Palette: dataviz reference slots 1-2 (validated), gray for references."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nowcast.evaluate import summarize  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#9a9893"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2, "lines.solid_capstyle": "round",
})


def _title(fig, title: str, subtitle: str):
    fig.text(0.01, 0.985, title, fontsize=14, fontweight="bold", va="top", color=INK)
    fig.text(0.01, 0.935, subtitle, fontsize=10, va="top", color=INK2)


def signal_vs_target(quarters: pd.DataFrame, features: pd.DataFrame, out: Path):
    """Small multiples: YoY revenue growth vs YoY pageview growth per ticker (full-quarter cut)."""
    f = features[features["cut"] == "full"][["ticker", "end", "views_yoy"]]
    d = quarters.merge(f, on=["ticker", "end"]).dropna(subset=["revenue_yoy", "views_yoy"])
    d["end"] = pd.to_datetime(d["end"])
    tickers = sorted(d["ticker"].unique())
    fig, axes = plt.subplots(3, 4, figsize=(13, 8.2), sharex=True)
    for ax, t in zip(axes.flat, tickers):
        g = d[d["ticker"] == t]
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.plot(g["end"], g["revenue_yoy"], color=BLUE, label="Revenue YoY")
        ax.plot(g["end"], g["views_yoy"], color=ORANGE, label="Wikipedia pageviews YoY")
        corr = g["revenue_yoy"].corr(g["views_yoy"])
        ax.set_title(f"{t}   ρ = {corr:+.2f}")
        vals = np.r_[g["revenue_yoy"], g["views_yoy"]]
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        pad = 0.08 * (hi - lo)
        ax.set_ylim(min(lo, 0) - pad, max(hi, 0) + pad)
        ax.tick_params(axis="x", labelrotation=0)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(4))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    for ax in axes.flat[len(tickers):]:
        ax.axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", bbox_to_anchor=(0.995, 0.985), ncol=2)
    _title(fig, "Does consumer attention track revenue?",
           "YoY log growth per fiscal quarter, pageviews summed over the exact XBRL fiscal period. ρ = full-sample correlation.")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _spec_label(r) -> str:
    return f"{'Full quarter' if r.cut == 'full' else '30 days early'} · {r.variant} · {r.sample.replace('_', '-')}"


def skill_by_spec(preds: pd.DataFrame, out: Path):
    s = summarize(preds, by=("cut", "variant", "sample"))
    s = s[s["model"].isin(["M1", "M2"])].copy()
    s["spec"] = [_spec_label(r) for r in s.itertuples()]
    order = list(dict.fromkeys(s.sort_values(["cut", "variant", "sample"])["spec"]))[::-1]
    y = {k: i for i, k in enumerate(order)}
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axvline(0, color=GRAY, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(0.005, len(order) - 0.35, "naive forecast", color=INK2, fontsize=9)
    for model, color, off, name in [("M1", ORANGE, 0.13, "M1 per-ticker OLS"), ("M2", BLUE, -0.13, "M2 pooled OLS + FE")]:
        g = s[s["model"] == model]
        ys = [y[k] + off for k in g["spec"]]
        ax.hlines(ys, 0, g["skill"], color=color, linewidth=2, alpha=0.5)
        ax.scatter(g["skill"], ys, color=color, s=60, zorder=3, edgecolor=SURFACE, linewidth=2, label=name)
    ax.set_yticks(range(len(order)), order)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Skill vs naive  = 1 − MAE(model) / MAE(naive)    (right of the line = better than naive)")
    ax.set_xlim(min(s["skill"].min() - 0.05, -0.5), 0.15)
    ax.legend(loc="lower left")
    _title(fig, "No specification beats “growth persists”",
           "Out-of-sample walk-forward, every pre-registered spec. raw/site = feature variant; all/ex-covid = sample.")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def hit_vs_base(preds: pd.DataFrame, out: Path):
    s = summarize(preds, by=("cut", "variant", "sample"))
    m3 = s[s["model"] == "M3"].copy()
    mb = s[s["model"] == "MB"].set_index(["cut", "variant", "sample"])["hit_rate"]
    m3["base"] = [mb.loc[(r.cut, r.variant, r.sample)] for r in m3.itertuples()]
    m3["spec"] = [_spec_label(r) for r in m3.itertuples()]
    m3 = m3.sort_values(["cut", "variant", "sample"]).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.axvline(0.5, color=GRAY, linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(0.502, len(m3) - 0.4, "coin flip", color=INK2, fontsize=9)
    ys = np.arange(len(m3))
    ax.hlines(ys, 0.5, m3["hit_rate"], color=GRID, linewidth=6, zorder=1)
    ax.scatter(m3["base"], ys, color=GRAY, s=150, zorder=2, label="Always predict the majority direction (MB)")
    ax.scatter(m3["hit_rate"], ys, color=BLUE, s=60, zorder=3, edgecolor=SURFACE, linewidth=2,
               label="M3 pageview direction classifier")
    for yv, r in zip(ys, m3.itertuples()):
        ax.text(max(r.hit_rate, r.base) + 0.006, yv, f"{r.hit_rate - r.base:+.1%} vs base", va="center",
                fontsize=9, color=INK2)
    ax.set_yticks(ys, m3["spec"])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0.48, 0.68)
    ax.set_xticks(np.arange(0.48, 0.681, 0.02))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Directional hit rate on revenue acceleration")
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.005, 0.89), ncol=2)
    _title(fig, "The 57–61% hit rate is the base rate",
           "Only ~40% of quarters accelerate, so “always decelerate” scores the same. Edge over base: ≈ 0.")
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def wikipedia_decline(features: pd.DataFrame, out: Path):
    f = features[(features["cut"] == "full")].dropna(subset=["views_yoy", "site_yoy"]).copy()
    f["year"] = pd.to_datetime(f["end"]).dt.year
    y = f.groupby("year").agg(article=("views_yoy", "median"), site=("site_yoy", "median")).reset_index()
    y = y[y["year"] >= 2016]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.plot(y["year"], y["article"], color=ORANGE, marker="o", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2, label="Company articles (median across tickers)")
    ax.plot(y["year"], y["site"], color=BLUE, marker="o", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=2, label="All of en.wikipedia (agent = user)")
    for col, color in [("article", ORANGE), ("site", BLUE)]:
        last = y.iloc[-1]
        ax.annotate(f"{last[col]:+.0%}", (last["year"], last[col]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xticks(y["year"])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("YoY log change in pageviews")
    ax.legend(loc="lower left")
    _title(fig, "A structural break in the signal: 2025–26",
           "Pageviews to company articles fell far faster than Wikipedia overall; site-normalizing removes only part of it.")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def make_all(con, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    q = con.execute("SELECT * FROM quarters").df()
    f = con.execute("SELECT * FROM features").df()
    p = con.execute("SELECT * FROM predictions").df()
    for df, cols in [(q, ["start", "end", "filed"]), (f, ["end", "asof"]), (p, ["end", "asof"])]:
        for c in cols:
            df[c] = pd.to_datetime(df[c]).dt.date
    paths = [out_dir / n for n in ["signal_vs_revenue.png", "skill_vs_naive.png", "hit_rate_vs_base.png",
                                   "wikipedia_decline.png"]]
    signal_vs_target(q, f, paths[0])
    skill_by_spec(p, paths[1])
    hit_vs_base(p, paths[2])
    wikipedia_decline(f, paths[3])
    return paths
