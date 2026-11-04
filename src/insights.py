"""
One-sentence takeaway per research question (RQ1-RQ8), shown under each chart on
the website. Every number is recomputed from the current selection, so the text
stays true whatever the filters are. Wording is descriptive: these are
correlations, not causes.
"""

from __future__ import annotations

import pandas as pd

from figures import split_years
from pipeline import lagged_correlations, ols_line


def _strength(r: float) -> str:
    size = abs(r)
    word = ("little or no" if size < 0.2 else "a weak" if size < 0.4
            else "a moderate" if size < 0.6 else "a strong")
    if size < 0.2:
        return word + " link"
    return f"{word} {'positive' if r > 0 else 'negative'} link"


def _within(df: pd.DataFrame, x: str, y: str) -> str:
    """Range of the per-borough r, to tell borough size apart from change over time."""
    rs = [ols_line(g[x], g[y])[2] for _, g in df.groupby("borough")]
    rs = [r for r in rs if not pd.isna(r)]
    if len(rs) < 2:
        return ""
    return f" Within single boroughs r ranges from {min(rs):+.2f} to {max(rs):+.2f}."


def _scatter(df: pd.DataFrame, x: str, y: str, what: str) -> str:
    df = df.dropna(subset=[x, y])
    _, _, r = ols_line(df[x], df[y])
    if pd.isna(r):
        return "Not enough borough-years in this selection to measure a link."
    return (f"Across {len(df)} borough-years, {what} show {_strength(r)} (r = {r:+.2f})."
            + _within(df, x, y))


def rq1(yearly):
    return _scatter(yearly, "units_completed", "incidents", "units completed and bus incidents")


def rq2(monthly):
    rs = []
    for _, g in monthly.dropna(subset=["incidents"]).groupby("borough"):
        r = ols_line(g["units_last_12m"], g["incidents"])[2]
        if not pd.isna(r):
            rs.append(r)
    if not rs:
        return "Not enough school months in this selection to compare the two series."
    mean = sum(rs) / len(rs)
    return (f"Month by month, the 12-month housing pipeline and bus incidents show {_strength(mean)} "
            f"(mean r within boroughs = {mean:+.2f}, school months only).")


def rq3(yearly, bus):
    df = yearly[yearly["units_completed"] > 0].assign(
        deep_share=lambda d: d["deeply_affordable_units"] / d["units_completed"])
    text = _scatter(df, "deep_share", "median_delay_min", "the deeply affordable share and median delay")
    med = bus.groupby("borough", observed=True)["delay_minutes"].median().dropna()
    if len(med) >= 2 and med.min() == med.max():
        text += f" Median delay is {med.min():.0f} min in every selected borough."
    elif len(med) >= 2:
        text += (f" Median delay is {med.min():.0f} min in {med.idxmin().title()} "
                 f"and {med.max():.0f} min in {med.idxmax().title()}.")
    return text


def rq4(monthly):
    lags = lagged_correlations(monthly).dropna(subset=["r"])
    if lags.empty:
        return "Not enough months in this selection to compare lags."
    mean = lags.groupby("lag_months")["r"].mean()
    best = mean.abs().idxmax()
    if abs(mean[best]) < 0.4:
        tail = "no lag reaches a moderate link."
    elif best == 0:
        tail = "the link is strongest within the same month, not after a delay."
    else:
        tail = "completions tend to come before incidents by about that long."
    return f"The strongest average link is at a {best}-month lag (mean r = {mean[best]:+.2f}); {tail}"


def rq5(housing, yearly):
    units = housing.groupby("borough")["total_units"].sum()
    incidents = yearly.groupby("borough")["incidents"].sum().astype(float)
    if units.sum() == 0 or incidents.sum() == 0 or len(incidents) < 2:
        return "Pick at least two boroughs to compare where housing and incidents concentrate."
    u, i = units / units.sum(), incidents / incidents.sum()
    text = (f"Most new units: {u.idxmax().title()} ({u.max():.0%}). "
            f"Most bus incidents: {i.idxmax().title()} ({i.max():.0%}).")
    more = [b.title() for b in i.index if i[b] > u.get(b, 0) + 0.05]
    if more:
        text += f" Clearly more incidents than new units: {', '.join(more)}."
    return text


def rq6(yearly):
    df = split_years(yearly.dropna(subset=["heavy_traffic_share"]), "new_construction_units", "high", "low")
    if df.empty:
        return "No borough has both high and low new-construction years in this selection."
    share = (df.groupby(["borough", "group"])[["heavy_traffic_incidents", "incidents"]].sum()
               .pipe(lambda d: d["heavy_traffic_incidents"] / d["incidents"]).unstack())
    diff = share["high"] - share["low"]
    text = (f"Heavy traffic is a larger share of incidents in high new-construction years in "
            f"{int((diff > 0).sum())} of {len(share)} borough(s)")
    if diff.max() > 0:
        gap = diff.idxmax()
        text += (f"; the biggest rise is in {gap.title()} "
                 f"({share.loc[gap, 'low']:.0%} to {share.loc[gap, 'high']:.0%})")
    return text + "."


def rq7(yearly):
    return _scatter(yearly, "family_units", "avg_students", "family-sized units and students on board")


def rq8(bus, yearly):
    groups = split_years(yearly, "units_completed", "high", "low")[["borough", "school_year", "group"]]
    df = bus.dropna(subset=["delay_minutes"]).merge(groups, on=["borough", "school_year"])
    if df.empty:
        return "Not enough delays in this selection to compare high- and low-completion years."
    med = df.groupby([df["is_prek"].map({True: "Pre-K", False: "School-age"}), "group"])["delay_minutes"].median()
    parts = [f"{age}: median {med[(age, 'high')]:.0f} min in high-completion years vs "
             f"{med[(age, 'low')]:.0f} in low" for age in ["School-age", "Pre-K"]
             if (age, "high") in med and (age, "low") in med]
    return ("; ".join(parts) + ".") if parts else "Not enough delays in this selection to compare."


def takeaways(housing, bus, yearly, monthly) -> list[str]:
    """In RQ1-RQ8 order, matching the chart cards."""
    return [rq1(yearly), rq2(monthly), rq3(yearly, bus), rq4(monthly),
            rq5(housing, yearly), rq6(yearly), rq7(yearly), rq8(bus, yearly)]
