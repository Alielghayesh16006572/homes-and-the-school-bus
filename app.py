"""
Dash website: "Homes and the School Bus" — NYC affordable housing x school-bus disruptions.

Run locally:   python app.py              -> http://127.0.0.1:8050
Production:    gunicorn app:server        (Render start command)

The app reads the cleaned row-level tables written by the notebook
(data/processed/*.csv.gz) and, for every report, re-runs the same aggregation and
integration functions the notebook used (src/pipeline.py) on the filtered rows.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, State, dcc, html

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import figures as F  # noqa: E402
from pipeline import BOROUGHS, add_flags, build_integrated, lagged_correlations, ols_line  # noqa: E402

DATA = ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not (DATA / "housing_clean.csv.gz").exists() or not (DATA / "bus_clean.csv.gz").exists():
        raise FileNotFoundError("data/processed/ is empty: run notebook.ipynb first (it writes the "
                                "cleaned tables the website reads), then commit data/processed/.")
    housing = pd.read_csv(DATA / "housing_clean.csv.gz",
                          parse_dates=["building_completion_date", "completion_month"])
    bus = pd.read_csv(
        DATA / "bus_clean.csv.gz", parse_dates=["month"],
        usecols=lambda c: c != "occurred_on",
        dtype={"reason": "category", "breakdown_or_running_late": "category",
               "school_age_or_prek": "category", "borough": "category", "school_year": "category"},
    )
    # The bus table's repeated text values are read as categories (same results, a
    # fraction of the memory: Render's free plan allows 512 MB), and the unused
    # occurred_on column is skipped.
    housing["borough"] = housing["borough"].astype("string")
    housing["school_year"] = housing["school_year"].astype("string")
    # Rebuild the flags from the text columns instead of trusting the saved
    # True/False values, so a change to the matching rules applies without
    # re-running the notebook.
    add_flags(bus)
    return housing, bus


HOUSING, BUS = load_data()

SCHOOL_YEARS = sorted(set(HOUSING["school_year"].dropna()) & set(BUS["school_year"].dropna()))
CONSTRUCTION_TYPES = sorted(HOUSING["reporting_construction_type"].dropna().unique())
INCIDENT_TYPES = sorted(BUS["breakdown_or_running_late"].dropna().unique())
REASONS = sorted(BUS["reason"].dropna().unique())


# ---------------------------------------------------------------------------
# Search mode: free text -> filter values
# ---------------------------------------------------------------------------
BOROUGH_WORDS = {
    "bronx": "BRONX", "brooklyn": "BROOKLYN", "manhattan": "MANHATTAN",
    "queens": "QUEENS", "staten island": "STATEN ISLAND", "staten": "STATEN ISLAND",
}
CONSTRUCTION_WORDS = {
    "new construction": "New Construction", "new building": "New Construction",
    "new buildings": "New Construction", "new": "New Construction",
    "preservation": "Preservation", "preserved": "Preservation", "renovat": "Preservation",
}
INCIDENT_WORDS = {"breakdown": "Breakdown", "running late": "Running Late", "late": "Running Late"}


def parse_search(query: str) -> dict:
    """Turn e.g. 'Brooklyn 2022 new buildings traffic' into filter values.

    Borough names, 4-digit years (matched to every school year containing that
    calendar year), construction type, incident type and any delay-reason word are
    recognised; anything not recognised is reported back to the user.
    """
    q = " " + query.lower().strip() + " "
    found = {"borough": [], "school_year": [], "construction": [], "incident": [], "reason": []}

    for word, value in sorted(BOROUGH_WORDS.items(), key=lambda kv: -len(kv[0])):
        if word in q and value not in found["borough"]:
            found["borough"].append(value)
            q = q.replace(word, " ")
    for year in re.findall(r"\b(20\d{2})\b", q):
        found["school_year"] += [sy for sy in SCHOOL_YEARS if year in sy.split("-") and sy not in found["school_year"]]
        q = q.replace(year, " ")
    for word, value in sorted(INCIDENT_WORDS.items(), key=lambda kv: -len(kv[0])):
        if word in q and value in INCIDENT_TYPES and value not in found["incident"]:
            found["incident"].append(value)
            q = q.replace(word, " ")
    for reason in REASONS:
        key = re.sub(r"[^a-z ]", "", reason.lower())
        main_word = max(key.split(), key=len) if key.split() else ""
        if main_word and len(main_word) > 3 and main_word in q:
            found["reason"].append(reason)
            q = q.replace(main_word, " ")
    for word, value in sorted(CONSTRUCTION_WORDS.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{word}", q) and value in CONSTRUCTION_TYPES and value not in found["construction"]:
            found["construction"].append(value)
            q = re.sub(rf"\b{word}\w*", " ", q)

    found["unmatched"] = [w for w in q.split() if w not in {"and", "in", "the", "of", "for", "units", "housing", "bus", "buses"}]
    return found


# ---------------------------------------------------------------------------
# Filtering + report building
# ---------------------------------------------------------------------------
def filter_rows(boroughs, school_years, construction, incidents, reasons):
    h, b = HOUSING, BUS
    if boroughs:
        h, b = h[h["borough"].isin(boroughs)], b[b["borough"].isin(boroughs)]
    if school_years:
        h, b = h[h["school_year"].isin(school_years)], b[b["school_year"].isin(school_years)]
    if construction:
        h = h[h["reporting_construction_type"].isin(construction)]
    if incidents:
        b = b[b["breakdown_or_running_late"].isin(incidents)]
    if reasons:
        b = b[b["reason"].isin(reasons)]
    return h, b


def build_report(h: pd.DataFrame, b: pd.DataFrame):
    if h.empty or b.empty:
        empty = F.empty_figure()
        return [empty] * 8, "No rows match these filters — widen the selection.", []

    yearly = build_integrated(h, b, "school_year")
    monthly = build_integrated(h, b, "month")
    if yearly.empty or monthly.empty:
        empty = F.empty_figure("No complete school year in this selection")
        return [empty] * 8, ("No complete school year in this selection — the current school year is "
                             "still in progress, so it is left out of the comparisons."), []

    if b["borough"].nunique() < len(BOROUGHS):     # keep only boroughs actually selected
        yearly = yearly[yearly["borough"].isin(b["borough"].unique())]
        monthly = monthly[monthly["borough"].isin(b["borough"].unique())]

    figs = [
        F.rq1_units_vs_incidents(yearly), F.rq2_timeline(monthly),
        F.rq3_deep_affordability_vs_delay(yearly), F.rq4_lag_heatmap(monthly),
        F.rq5_map(h, yearly), F.rq6_heavy_traffic_share(yearly),
        F.rq7_family_units_vs_students(yearly), F.rq8_prek_delay_box(b, yearly),
    ]
    summary = (f"{len(h):,} completed buildings · {int(h['total_units'].sum()):,} affordable units · "
               f"{len(b):,} school-bus incidents · {yearly['school_year'].nunique()} school years · "
               f"{yearly['borough'].nunique()} borough(s)")

    # Live numbers for the story section, recomputed for the current filters.
    _, _, r_units = ols_line(yearly["units_completed"], yearly["incidents"])
    lags = lagged_correlations(monthly)
    best = lags.dropna(subset=["r"]).groupby("lag_months")["r"].mean()
    live = [
        html.Li(f"Borough-year link between units completed and bus incidents: "
                f"{'n/a' if pd.isna(r_units) else f'r = {r_units:+.2f}'} "
                f"({len(yearly)} borough-years)."),
        html.Li("Lag with the strongest average correlation across boroughs: "
                + ("n/a" if best.empty else f"{int(best.abs().idxmax())} months (mean r = {best.loc[best.abs().idxmax()]:+.2f})") + "."),
        html.Li(f"Share of incidents caused by heavy traffic: {100 * b['is_heavy_traffic'].mean():.1f}%."),
    ]
    return figs, summary, live


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def dropdown(id_, label, options, placeholder="All"):
    return html.Div(className="filter", children=[
        html.Label(label, htmlFor=id_),
        dcc.Dropdown(id=id_, options=options, multi=True, placeholder=placeholder),
    ])


RQ_CARDS = [
    ("rq1", "Ali Elghayesh"), ("rq2", "Asser Khattab"), ("rq3", "Ali Ahmed"), ("rq4", "Ahmed Rashad"),
    ("rq5", "Asser Khattab"), ("rq6", "Ali Elghayesh"), ("rq7", "Ali Ahmed"), ("rq8", "Ahmed Rashad"),
]

app = Dash(__name__, title="Homes and the School Bus",
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
server = app.server

app.layout = html.Div(className="page", children=[
    html.Header(className="hero", children=[
        html.P("NYC Data Detectives · Data Engineering & Visualization", className="eyebrow"),
        html.H1("Homes and the School Bus"),
        html.P("When New York finishes new affordable housing, families move in and children need to get to "
               "school. Do the boroughs that build the most also have the most disrupted school-bus service? "
               "This site joins HPD's Affordable Housing Production by Building with DOE's Bus Breakdown and "
               "Delays at the borough × school-year and borough × month level.", className="lede"),
    ]),

    html.Section(className="controls", children=[
        html.Div(className="search-row", children=[
            dcc.Input(id="search", type="text", debounce=False, className="search",
                      placeholder='Search, e.g. "Brooklyn 2022 new buildings traffic"'),
            html.Button("Apply search", id="search-btn", className="btn secondary"),
        ]),
        html.Div(id="search-status", className="status"),
        html.Div(className="filters", children=[
            dropdown("f-borough", "Borough", [{"label": b.title(), "value": b} for b in BOROUGHS]),
            dropdown("f-year", "School year", SCHOOL_YEARS),
            dropdown("f-construction", "Housing: construction type", CONSTRUCTION_TYPES),
            dropdown("f-incident", "Bus: incident type", INCIDENT_TYPES),
            dropdown("f-reason", "Bus: delay reason", REASONS),
        ]),
        html.Div(className="report-row", children=[
            html.Button("Generate Report", id="generate", className="btn primary"),
            dcc.Loading(html.Div(id="summary", className="summary"), type="dot"),
        ]),
    ]),

    html.Section(className="grid", children=[
        html.Div(className="card" + (" wide" if rq in ("rq2", "rq5") else ""), children=[
            html.Span(member, className="badge"),
            dcc.Loading(dcc.Graph(id=f"fig-{rq}", config={"displaylogo": False}), type="circle"),
        ]) for rq, member in RQ_CARDS
    ]),

    html.Section(className="story", children=[
        html.H2("The story"),
        html.H3("The question"),
        html.P("New affordable buildings bring new households, many with school-age children. If the school-bus "
               "system does not grow with them, the children in those buildings feel it first. We ask whether "
               "affordable-housing completions and school-bus breakdowns/delays rise and fall together across "
               "boroughs and over time, and whether the lowest-income and youngest riders are affected differently."),
        html.H3("What the current selection shows"),
        html.Ul(id="live-findings"),
        html.H3("Key findings"),
        html.Ul([
            html.Li("Across 60 borough-years, more affordable units completed went with more school-bus incidents "
                    "(r = +0.73). Borough size explains part of this, but the link also holds within every borough "
                    "(r = +0.43 to +0.67), so a borough's high-completion years tended to be its high-incident years. "
                    "Citywide shifts that affect both, such as the 2020-21 closures, may explain much of it."),
            html.Li("A larger share of units for the lowest-income households is not linked to longer delays (r = -0.19); "
                    "median delays differ more by borough (mostly 20-40 min in the Bronx and Brooklyn vs up to about "
                    "50 min in Queens and Manhattan). Lagging completions by 0-12 months gives only weak correlations "
                    "(best mean r = +0.17)."),
            html.Li("New affordable housing clusters in the Bronx, Upper Manhattan and central Brooklyn. Heavy traffic "
                    "causes 62.7% of incidents, and its share rises in high-construction years in Brooklyn and Staten "
                    "Island but not in Manhattan or Queens."),
            html.Li("More family-sized units goes with more students on disrupted buses (r = +0.40), mostly a borough "
                    "difference (Bronx about 3-4 students per incident vs under 1 in Manhattan and Staten Island). "
                    "In high-completion years the median school-age delay is longer (about 38 vs 30 min), and Pre-K "
                    "delays keep a similar median but more long delays (upper quartile about 53 vs 38 min)."),
        ]),
        html.H3("Limits"),
        html.Ul([
            html.Li("Correlation, not causation: both series follow citywide trends (enrolment, COVID-19 closures, "
                    "contractor changes) that neither dataset records."),
            html.Li("The bus file only identifies the borough, so the analysis cannot see neighbourhoods; a "
                    "building and a delayed bus in the same borough may be miles apart."),
            html.Li("The housing file covers only HPD-financed affordable units, not all new housing."),
            html.Li("Delay length is self-reported free text; values that could not be parsed, or were "
                    "implausible, were left blank rather than guessed."),
            html.Li("Bus incident counts are not divided by the number of routes or riders, which the dataset "
                    "does not provide, so boroughs are compared to themselves over time rather than to each other."),
        ]),
        html.H3("Why it matters for New York"),
        html.P("HPD and the DOE Office of Pupil Transportation plan separately. If disruptions concentrate where "
               "affordable housing is being delivered, OPT could add route capacity and contractor oversight in "
               "those boroughs ahead of large completions, and HPD could share its completion pipeline with OPT "
               "so school transport is planned with the housing rather than after it."),
    ]),

    html.Footer(children=[
        "Data: NYC Open Data — ",
        html.A("Affordable Housing Production by Building", href="https://data.cityofnewyork.us/d/hg8x-zxpr", target="_blank"),
        " · ",
        html.A("Bus Breakdown and Delays", href="https://data.cityofnewyork.us/d/ez4e-fazm", target="_blank"),
    ]),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("f-borough", "value"), Output("f-year", "value"), Output("f-construction", "value"),
    Output("f-incident", "value"), Output("f-reason", "value"), Output("search-status", "children"),
    Input("search-btn", "n_clicks"), Input("search", "n_submit"),
    State("search", "value"),
    prevent_initial_call=True,
)
def apply_search(_clicks, _submit, query):
    """Search mode: fill the dropdowns from the typed query (Enter or the button)."""
    if not query or not query.strip():
        return [], [], [], [], [], "Search cleared — all filters reset. Press Generate Report."
    f = parse_search(query)
    parts = []
    if f["borough"]:      parts.append("borough: " + ", ".join(b.title() for b in f["borough"]))
    if f["school_year"]:  parts.append("school years: " + ", ".join(f["school_year"]))
    if f["construction"]: parts.append("construction: " + ", ".join(f["construction"]))
    if f["incident"]:     parts.append("incident: " + ", ".join(f["incident"]))
    if f["reason"]:       parts.append("reason: " + ", ".join(f["reason"]))
    msg = ("Applied " + "; ".join(parts) + ". Press Generate Report.") if parts else "No filter words recognised."
    if f["unmatched"]:
        msg += f" Ignored: {' '.join(f['unmatched'])}."
    return f["borough"], f["school_year"], f["construction"], f["incident"], f["reason"], msg


@app.callback(
    [Output(f"fig-{rq}", "figure") for rq, _ in RQ_CARDS]
    + [Output("summary", "children"), Output("live-findings", "children")],
    Input("generate", "n_clicks"),
    State("f-borough", "value"), State("f-year", "value"), State("f-construction", "value"),
    State("f-incident", "value"), State("f-reason", "value"),
)
def generate_report(_n, boroughs, years, construction, incidents, reasons):
    """Generate Report: every chart and the live story numbers update together.
    Runs once on page load with no filters so the page is never empty."""
    h, b = filter_rows(boroughs, years, construction, incidents, reasons)
    figs, summary, live = build_report(h, b)
    return figs + [summary, live]


if __name__ == "__main__":
    app.run(debug=False, port=8050)
