"""Run every pre-registered M3 spec (DESIGN.md §16) and write predictions + reports/results_m3.md."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm

from nowcast.evaluate import build_panel, exclude_covid, ic_frame, mean_ic, summarize, walk_forward

SPECS = list(itertools.product(["full", "m30"], ["raw", "site"], ["all", "ex_covid"]))
PRIMARY = ("full", "raw", "all")


def run_all(quarters: pd.DataFrame, features: pd.DataFrame):
    preds, ics = [], []
    for cut, variant, sample in SPECS:
        panel = build_panel(quarters, features, cut, variant)
        if sample == "ex_covid":
            panel = exclude_covid(panel)
        p = walk_forward(panel).assign(cut=cut, variant=variant, sample=sample)
        preds.append(p)
        keys = set(zip(p["ticker"], p["end"]))
        evald = panel[[k in keys for k in zip(panel["ticker"], panel["end"])]]
        ic, t, n = mean_ic(ic_frame(evald))
        ics.append(dict(cut=cut, variant=variant, sample=sample, mean_ic=ic, ic_t=t, periods=n))
    return pd.concat(preds, ignore_index=True), pd.DataFrame(ics)


def leave_one_out(quarters, features, tickers) -> pd.DataFrame:
    cut, variant, _ = PRIMARY
    panel = build_panel(quarters, features, cut, variant)
    rows = []
    for t in tickers:
        s = summarize(walk_forward(panel[panel["ticker"] != t])).set_index("model")
        rows.append(dict(dropped=t, m2_skill=s.loc["M2", "skill"], m2_hit=s.loc["M2", "hit_rate"],
                         m3_hit=s.loc["M3", "hit_rate"]))
    return pd.DataFrame(rows)


def insample(quarters, features) -> pd.DataFrame:
    """Full-sample coefficient on x (in-sample only; out-of-sample metrics decide)."""
    cut, variant, _ = PRIMARY
    panel = build_panel(quarters, features, cut, variant).dropna(subset=["y", "y_lag", "x"])
    rows = []
    for t, g in panel.groupby("ticker"):
        res = sm.OLS(g["y"], sm.add_constant(g[["y_lag", "x"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
        rows.append(dict(ticker=t, n=len(g), b_ylag=res.params["y_lag"], c_x=res.params["x"],
                         t_x=res.tvalues["x"], r2=res.rsquared))
    fe = pd.get_dummies(panel["ticker"], dtype=float)
    X = pd.concat([panel[["y_lag", "x"]], fe], axis=1)
    codes = panel["ticker"].astype("category").cat.codes
    res = sm.OLS(panel["y"], X).fit(cov_type="cluster", cov_kwds={"groups": codes})
    rows.append(dict(ticker="POOLED (FE, clustered)", n=len(panel), b_ylag=res.params["y_lag"],
                     c_x=res.params["x"], t_x=res.tvalues["x"], r2=res.rsquared))
    return pd.DataFrame(rows)


def _fmt(df: pd.DataFrame, pct: tuple = (), dec: int = 3) -> str:
    def cell(c, v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "–"
            return f"{v:.0%}" if c in pct else f"{v:.{dec}f}"
        return str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(cell(c, v) for c, v in zip(df.columns, r)) + " |"
                            for r in df.itertuples(index=False)) + "\n"


def verdict(head: pd.DataFrame, ics: pd.DataFrame) -> list[str]:
    reg = head[head["model"].isin(["M1", "M2"])]
    wins = reg[(reg["skill"] > 0) & (reg["dm_p"] < 0.05)]
    dirm = head[head["model"].isin(["M1", "M2", "M3"])]
    best = dirm.loc[dirm["hit_vs_base"].idxmax()]
    ic_sig = ics[ics["ic_t"].abs() > 2]
    return [
        "## Verdict",
        f"- **Level forecasts:** {len(wins)} of {len(reg)} regression specs beat the naive forecast with "
        f"positive skill and DM p < 0.05. Best skill: {reg['skill'].max():.0%}.",
        f"- **Direction:** the best edge over the base-rate benchmark (MB) is {best['hit_vs_base']:+.1%} "
        f"({best['model']}, {best['cut']}/{best['variant']}/{best['sample']}). Raw hit rates above 50% mostly "
        "reflect that decelerations outnumber accelerations.",
        f"- **Cross-sectional IC:** {len(ic_sig)} of {len(ics)} specs have |t| > 2. Mean IC ranges "
        f"{ics['mean_ic'].min():.2f} to {ics['mean_ic'].max():.2f}.",
        "",
    ]


def report(preds, ics, loo, ins) -> str:
    head = summarize(preds, by=("cut", "variant", "sample"))
    prim = preds[(preds.cut == "full") & (preds.variant == "raw") & (preds["sample"] == "all")]
    per_t = summarize(prim, by=("ticker",))
    wide = per_t.pivot(index="ticker", columns="model", values=["skill", "hit_rate"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide[["skill_M1", "skill_M2", "hit_rate_M1", "hit_rate_M2", "hit_rate_M3", "hit_rate_MB"]].reset_index()
    wide.insert(1, "n", per_t[per_t.model == "M0"].set_index("ticker").loc[wide["ticker"], "n"].to_numpy())
    pct = ("skill", "hit_rate", "skill_M1", "skill_M2", "hit_rate_M1", "hit_rate_M2", "hit_rate_M3", "hit_rate_MB", "hit_vs_base",
           "m2_skill", "m2_hit", "m3_hit")
    return "\n".join([
        "# Milestone 3 results: walk-forward evaluation",
        "",
        "Pre-registered in DESIGN.md §16. Target `y` = YoY log revenue growth (first-filed); feature `x` = YoY "
        "log pageviews over the exact fiscal period. Skill = 1 − MAE/MAE(naive). Hit rate = acceleration "
        "direction; p one-sided vs 50%. DM < 0 means the model beats naive. All models are scored on the "
        "identical sample within each spec.",
        "",
        *verdict(head, ics),
        "## All specs",
        _fmt(head, pct=pct),
        "## Cross-sectional IC of Δx vs Δy (per calendar quarter, ≥5 names)",
        _fmt(ics),
        "## Per ticker (primary spec: full cut, raw, all quarters)",
        _fmt(wide, pct=pct),
        "## Leave one ticker out (primary spec)",
        _fmt(loo, pct=pct),
        "## In-sample coefficients (primary spec; HAC per ticker, clustered pooled)",
        _fmt(ins),
    ])
