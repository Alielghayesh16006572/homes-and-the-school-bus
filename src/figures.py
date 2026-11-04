"""
One figure function per research question (RQ1-RQ8).

Every function takes already-filtered tables and returns a plotly Figure, so the
notebook and the website draw exactly the same charts.

Design rules applied to all charts
- Borough colour is fixed per borough (never re-assigned when a filter removes one),
  and every borough also gets its own marker symbol so colour is never the only cue.
- No dual-axis charts: two measures with different units get separate panels.
- Diverging scale (correlations) is blue <-> red with a neutral grey at r = 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pipeline import BOROUGHS, lagged_correlations, ols_line

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
BOROUGH_COLORS = {
    "BRONX": "#2a78d6", "BROOKLYN": "#eb6834", "MANHATTAN": "#1baf7a",
    "QUEENS": "#6250d6", "STATEN ISLAND": "#e34948",
}
BOROUGH_SYMBOLS = {
    "BRONX": "circle", "BROOKLYN": "square", "MANHATTAN": "diamond",
    "QUEENS": "triangle-up", "STATEN ISLAND": "x",
}
# Approximate geographic centres, used to place borough-level bus bubbles on the map.
BOROUGH_CENTROIDS = {
    "BRONX": (40.8448, -73.8648), "BROOKLYN": (40.6782, -73.9442),
    "MANHATTAN": (40.7831, -73.9712), "QUEENS": (40.7282, -73.7949),
    "STATEN ISLAND": (40.5795, -74.1502),
}
CONSTRUCTION_COLORS = {"New Construction": "#2a78d6", "Preservation": "#eb6834"}
GROUP_COLORS = ["#2a78d6", "#eb6834"]
DIVERGING = [[0, "#c53030"], [0.5, "#f0efec"], [1, "#1c5cab"]]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e7e6e2"


def _title(text: str, subtitle: str = "") -> dict:
    sub = f"<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>" if subtitle else ""
    return {"text": f"{text}{sub}", "x": 0, "xanchor": "left", "font": {"size": 15, "color": INK}}


def _style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=60, r=20, t=80, b=120),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color=INK),
        legend=dict(orientation="h", yanchor="top", y=-0.2, x=0, title_text=""),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def empty_figure(message: str = "No data for these filters — try widening the selection") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color=MUTED))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _style(fig, 300)


def split_years(yearly: pd.DataFrame, col: str, high: str, low: str) -> pd.DataFrame:
    """Label each borough's years high/low at that borough's own median of `col`, so
    big and small boroughs are compared fairly. Boroughs whose years all land in
    one group (e.g. no new construction under a "Preservation" filter) have nothing
    to compare and are left out."""
    med = yearly.groupby("borough")[col].transform("median")
    df = yearly.assign(group=np.where(yearly[col] > med, high, low))
    return df[df.groupby("borough")["group"].transform("nunique") == 2]


def _r_text(r: float) -> str:
    return "r = n/a" if pd.isna(r) else f"Pearson r = {r:+.2f}"


def _borough_scatter(df: pd.DataFrame, x: str, y: str, size: str | None = None,
                     hover: str = "") -> go.Figure:
    """Scatter with one trace per borough (fixed colour + symbol) and a pooled OLS line."""
    fig = go.Figure()
    sizes = None
    if size is not None and df[size].max() > 0:
        sizes = 8 + 30 * np.sqrt(df[size].astype(float) / df[size].astype(float).max())
    for borough in BOROUGHS:
        g = df[df["borough"] == borough]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g[x], y=g[y], mode="markers", name=borough.title(),
            marker=dict(color=BOROUGH_COLORS[borough], symbol=BOROUGH_SYMBOLS[borough],
                        size=sizes.loc[g.index] if sizes is not None else 11,
                        line=dict(color="white", width=2), opacity=0.9),
            customdata=g[["school_year"]],
            hovertemplate=f"<b>{borough.title()}</b> %{{customdata[0]}}<br>{hover}<extra></extra>",
        ))
    slope, intercept, r = ols_line(df[x], df[y])
    if not pd.isna(slope):
        xs = np.linspace(df[x].astype(float).min(), df[x].astype(float).max(), 50)
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Linear fit (all)",
                                 line=dict(color=MUTED, width=2, dash="dash"), hoverinfo="skip"))
    fig._r = r  # stored so the caller can put it in the subtitle
    return fig


# ---------------------------------------------------------------------------
# RQ1: Ali Sherif Elghayesh · RQ2: Asser Ehab Khattab
# ---------------------------------------------------------------------------
def rq1_units_vs_incidents(yearly: pd.DataFrame) -> go.Figure:
    """RQ1: Do borough-years that delivered more affordable units also log more
    school-bus incidents?"""
    df = yearly.dropna(subset=["units_completed", "incidents"])
    if df.empty:
        return empty_figure()
    fig = _borough_scatter(df, "units_completed", "incidents",
                           hover="Units completed: %{x:,}<br>Bus incidents: %{y:,}")
    fig.update_layout(title=_title("RQ1 · Affordable units completed vs. school-bus incidents",
                                   f"One point per borough × school year · {_r_text(fig._r)}"))
    fig.update_xaxes(title="Affordable units completed in the school year")
    fig.update_yaxes(title="Bus breakdowns + delays")
    return _style(fig)


def rq2_timeline(monthly: pd.DataFrame) -> go.Figure:
    """RQ2: Over time, does the 12-month pipeline of new affordable units move together
    with monthly bus incidents? Two panels on a shared time axis (no dual y-axis)."""
    if monthly.empty:
        return empty_figure()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Affordable units completed, rolling 12 months",
                                        "School-bus incidents per month (gaps = no school service)"))
    for borough in BOROUGHS:
        g = monthly[monthly["borough"] == borough].sort_values("month")
        if g.empty:
            continue
        common = dict(name=borough.title(), legendgroup=borough, mode="lines",
                      line=dict(color=BOROUGH_COLORS[borough], width=2))
        fig.add_trace(go.Scatter(x=g["month"], y=g["units_last_12m"], **common,
                                 hovertemplate="%{x|%b %Y}: %{y:,} units<extra>" + borough.title() + "</extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=g["month"], y=g["incidents"], **common, showlegend=False,
                                 connectgaps=False,
                                 hovertemplate="%{x|%b %Y}: %{y:,} incidents<extra>" + borough.title() + "</extra>"),
                      row=2, col=1)
    fig.update_layout(title=_title("RQ2 · Housing pipeline and bus disruptions over time"),
                      hovermode="x unified")
    fig.update_yaxes(title="Units", row=1, col=1)
    fig.update_yaxes(title="Incidents", row=2, col=1)
    fig.update_xaxes(title="Month", row=2, col=1)
    return _style(fig, 560)


# ---------------------------------------------------------------------------
# RQ3: Ali Ahmed · RQ4: Ahmed Rashad
# ---------------------------------------------------------------------------
def rq3_deep_affordability_vs_delay(yearly: pd.DataFrame) -> go.Figure:
    """RQ3: Where a larger share of new units is reserved for the lowest-income
    households, are school-bus delays longer?"""
    df = yearly[yearly["units_completed"] > 0].copy()
    df = df.dropna(subset=["median_delay_min"])
    if df.empty:
        return empty_figure()
    df["deep_share_pct"] = 100 * df["deeply_affordable_units"] / df["units_completed"]
    fig = _borough_scatter(df, "deep_share_pct", "median_delay_min", size="units_completed",
                           hover="Extremely/very low income share: %{x:.1f}%<br>Median delay: %{y:.0f} min")
    fig.update_layout(title=_title("RQ3 · Deeply affordable share vs. median delay",
                                   f"Bubble size = units completed · {_r_text(fig._r)}"))
    fig.update_xaxes(title="% of units for extremely/very low income", ticksuffix="%")
    fig.update_yaxes(title="Median reported delay (minutes)")
    return _style(fig)


def rq4_lag_heatmap(monthly: pd.DataFrame, max_lag: int = 12) -> go.Figure:
    """RQ4: Do housing completions *precede* bus disruptions, and by how many months?"""
    lags = lagged_correlations(monthly, max_lag=max_lag)
    if lags["r"].notna().sum() == 0:
        return empty_figure("Not enough months to compute lagged correlations")
    grid = lags.pivot(index="borough", columns="lag_months", values="r").reindex(
        [b for b in BOROUGHS if b in lags["borough"].unique()])
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=grid.columns, y=[b.title() for b in grid.index],
        zmin=-1, zmax=1, colorscale=DIVERGING, xgap=2, ygap=2,
        text=np.vectorize(lambda v: "" if pd.isna(v) else f"{v:+.1f}")(grid.values),
        texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title="r", thickness=12),
        hovertemplate="%{y}, completions %{x} months earlier<br>r = %{z:+.2f}<extra></extra>",
    ))
    fig.update_layout(title=_title("RQ4 · Do completions lead bus incidents?",
                                   "Pearson r between units completed (shifted by lag) and monthly incidents"))
    fig.update_xaxes(title="Lag (months between completion and bus month)", dtick=1)
    return _style(fig, 380)


# ---------------------------------------------------------------------------
# RQ5: Asser Ehab Khattab · RQ6: Ali Sherif Elghayesh
# ---------------------------------------------------------------------------
def rq5_map(housing: pd.DataFrame, yearly: pd.DataFrame) -> go.Figure:
    """RQ5: Is new affordable housing concentrated in the same boroughs where school-bus
    disruptions concentrate? Buildings as points, bus incidents as borough bubbles."""
    pts = housing.dropna(subset=["latitude", "longitude"])
    if pts.empty and yearly.empty:
        return empty_figure()
    fig = go.Figure()
    bus = yearly.groupby("borough", as_index=False)["incidents"].sum()
    if not bus.empty and bus["incidents"].max() > 0:
        bus["lat"] = bus["borough"].map(lambda b: BOROUGH_CENTROIDS[b][0])
        bus["lon"] = bus["borough"].map(lambda b: BOROUGH_CENTROIDS[b][1])
        fig.add_trace(go.Scattermap(
            lat=bus["lat"], lon=bus["lon"], mode="markers", name="Bus incidents (borough total)",
            marker=dict(size=20 + 60 * np.sqrt(bus["incidents"] / bus["incidents"].max()),
                        color="#52514e", opacity=0.25),
            customdata=bus[["borough", "incidents"]],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,} bus incidents<extra></extra>",
        ))
    for ctype, color in CONSTRUCTION_COLORS.items():
        g = pts[pts["reporting_construction_type"] == ctype]
        if g.empty:
            continue
        fig.add_trace(go.Scattermap(
            lat=g["latitude"], lon=g["longitude"], mode="markers", name=ctype,
            marker=dict(size=np.clip(4 + np.sqrt(g["total_units"]), 4, 22), color=color, opacity=0.7),
            customdata=g[["project_name", "total_units", "school_year"]],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:,} units · completed %{customdata[2]}<extra></extra>",
        ))
    fig.update_layout(map=dict(style="open-street-map", center=dict(lat=40.71, lon=-73.95), zoom=9.2),
                      title=_title("RQ5 · Where new affordable housing and bus disruptions concentrate",
                                   "Dots = buildings (size = units) · grey bubbles = borough bus incidents"))
    fig = _style(fig, 560)
    fig.update_layout(margin=dict(l=10, r=10, t=80, b=10), legend=dict(y=0.02, x=0.01, bgcolor="rgba(255,255,255,.85)"))
    return fig


def rq6_heavy_traffic_share(yearly: pd.DataFrame) -> go.Figure:
    """RQ6: In a borough's high new-construction years, is 'Heavy Traffic' a larger
    share of bus delay reasons than in its low new-construction years?"""
    df = yearly.dropna(subset=["heavy_traffic_share"]).copy()
    if df.empty:
        return empty_figure()
    df = split_years(df, "new_construction_units", "High new-construction years", "Low new-construction years")
    if df.empty:
        return empty_figure("No borough has both high and low new-construction years in this selection")
    agg = (df.groupby(["borough", "group"])
             .agg(ht=("heavy_traffic_incidents", "sum"), inc=("incidents", "sum"), years=("school_year", "count"))
             .reset_index())
    agg["share"] = 100 * agg["ht"] / agg["inc"]
    fig = go.Figure()
    for color, group in zip(GROUP_COLORS, ["Low new-construction years", "High new-construction years"]):
        g = agg[agg["group"] == group].set_index("borough").reindex([b for b in BOROUGHS if b in agg["borough"].values])
        fig.add_trace(go.Bar(
            x=[b.title() for b in g.index], y=g["share"], name=group, marker_color=color,
            marker=dict(cornerradius=4, line=dict(color="white", width=2)),
            customdata=g[["years", "inc"]],
            hovertemplate="%{x}: %{y:.1f}% heavy traffic<br>%{customdata[0]} school years, %{customdata[1]:,} incidents<extra>" + group + "</extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.3,
                      title=_title("RQ6 · Heavy-traffic share: high vs. low construction years",
                                   "Years split at each borough's median of new-construction units"))
    fig.update_yaxes(title="% of incidents caused by heavy traffic", ticksuffix="%")
    return _style(fig)


# ---------------------------------------------------------------------------
# RQ7: Ali Ahmed · RQ8: Ahmed Rashad
# ---------------------------------------------------------------------------
def rq7_family_units_vs_students(yearly: pd.DataFrame) -> go.Figure:
    """RQ7: Do borough-years that add more family-sized units (2+ bedrooms) see more
    students on board when a bus is disrupted?"""
    df = yearly.dropna(subset=["avg_students"])
    if df.empty:
        return empty_figure()
    fig = _borough_scatter(df, "family_units", "avg_students",
                           hover="Family-sized units: %{x:,}<br>Avg students on disrupted bus: %{y:.1f}")
    fig.update_layout(title=_title("RQ7 · Family-sized units vs. students on board",
                                   f"One point per borough × school year · {_r_text(fig._r)}"))
    fig.update_xaxes(title="New units with 2+ bedrooms")
    fig.update_yaxes(title="Avg students on the bus per incident")
    return _style(fig)


def rq8_prek_delay_box(bus: pd.DataFrame, yearly: pd.DataFrame) -> go.Figure:
    """RQ8: Are Pre-K and school-age delays affected differently in borough-years with
    high vs. low affordable housing completions? Box statistics are pre-computed so
    the browser never has to receive hundreds of thousands of points."""
    if bus.empty or yearly.empty:
        return empty_figure()
    growth = (split_years(yearly, "units_completed", "High-completion years", "Low-completion years")
              .rename(columns={"group": "growth"})[["borough", "school_year", "growth"]])
    df = bus.dropna(subset=["delay_minutes"]).merge(growth, on=["borough", "school_year"], how="inner")
    if df.empty:
        return empty_figure()
    df["age_group"] = np.where(df["is_prek"], "Pre-K", "School-age")
    fig = go.Figure()
    for color, group in zip(GROUP_COLORS, ["Low-completion years", "High-completion years"]):
        g = df[df["growth"] == group]
        stats = g.groupby("age_group")["delay_minutes"].describe(percentiles=[.25, .5, .75])
        stats = stats.reindex(["School-age", "Pre-K"]).dropna(subset=["count"])
        iqr = stats["75%"] - stats["25%"]
        fig.add_trace(go.Box(
            name=group, x=stats.index, q1=stats["25%"], median=stats["50%"], q3=stats["75%"],
            lowerfence=np.maximum(stats["min"], stats["25%"] - 1.5 * iqr),
            upperfence=np.minimum(stats["max"], stats["75%"] + 1.5 * iqr),
            marker_color=color, line=dict(width=2), boxpoints=False,
        ))
    fig.update_layout(boxmode="group",
                      title=_title("RQ8 · Delay length for Pre-K vs. school-age riders",
                                   "Borough-years split at each borough's median of affordable units completed"))
    fig.update_yaxes(title="Reported delay (minutes)")
    return _style(fig)
