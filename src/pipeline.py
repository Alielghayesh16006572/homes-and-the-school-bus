"""
Data pipeline for the NYC Housing x Transportation project.

Datasets
--------
Housing & Development : Affordable Housing Production by Building  (hg8x-zxpr)
Transportation         : Bus Breakdown and Delays                   (ez4e-fazm)

The same functions are used by the notebook (to document every step) and by the
Dash website (to re-aggregate the cleaned data whenever the user changes a filter),
so the website always shows exactly what the notebook produced.

Pipeline stages
---------------
1. load_*            download from NYC Open Data (or read the local raw cache)
2. normalize_columns standardise column names (CSV export and SODA API differ)
3. clean_housing     pre-integration cleaning of the housing dataset
   clean_bus         pre-integration cleaning of the bus dataset
4. aggregate_*       roll both datasets up to a common level (borough x month
                     or borough x school year)
5. integrate         join the two aggregates on the shared key
6. post_integration_clean   fix join NaNs, duplicated columns and dtypes
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HOUSING_ID = "hg8x-zxpr"
BUS_ID = "ez4e-fazm"
HOUSING_URL = f"https://data.cityofnewyork.us/d/{HOUSING_ID}"
BUS_URL = f"https://data.cityofnewyork.us/d/{BUS_ID}"
API_URL = "https://data.cityofnewyork.us/resource/{}.csv"

BOROUGHS = ["BRONX", "BROOKLYN", "MANHATTAN", "QUEENS", "STATEN ISLAND"]

# Spelling variants seen in the raw data, mapped to the five official boroughs.
# Values outside NYC (Nassau County, Westchester, New Jersey, ...) map to None.
BOROUGH_ALIASES = {
    "bronx": "BRONX", "the bronx": "BRONX",
    "brooklyn": "BROOKLYN", "kings": "BROOKLYN",
    "manhattan": "MANHATTAN", "new york": "MANHATTAN",
    "queens": "QUEENS",
    "staten island": "STATEN ISLAND", "richmond": "STATEN ISLAND",
}

# Rough bounding box of NYC; coordinates outside it are data-entry errors.
NYC_LAT = (40.49, 40.92)
NYC_LON = (-74.27, -73.68)

# Domain rules (documented in the notebook)
MAX_PLAUSIBLE_DELAY_MIN = 240   # > 4 hours is not a "delay" but a cancelled run or a typo
MAX_STUDENTS_ON_BUS = 90        # largest NYC school buses seat ~ 66-90 pupils

INCOME_UNIT_COLS = [
    "extremely_low_income_units", "very_low_income_units", "low_income_units",
    "moderate_income_units", "middle_income_units", "other_income_units",
]
BEDROOM_UNIT_COLS = [
    "studio_units", "br1_units", "br2_units", "br3_units", "br4_units",
    "br5_units", "br6_units", "unknown_br_units",
]
YES_NO_COLS = [
    "has_contractor_notified_schools", "has_contractor_notified_parents",
    "have_you_alerted_opt",
]


# ---------------------------------------------------------------------------
# Cleaning log: every step records rows before/after so the notebook can show it
# ---------------------------------------------------------------------------
@dataclass
class CleaningLog:
    entries: list = field(default_factory=list)

    def add(self, dataset: str, step: str, before: int, after: int, note: str = ""):
        self.entries.append(
            {"dataset": dataset, "step": step, "rows_before": before,
             "rows_after": after, "rows_removed": before - after, "note": note}
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------
def load_dataset(dataset_id: str, cache_path: Path | None = None,
                 limit: int = 2_000_000, where: str | None = None) -> pd.DataFrame:
    """Download a dataset through the SODA API, caching the raw CSV locally.

    The API returns only 1,000 rows unless $limit is set, so we set it above the
    size of both datasets. `$order=:id` keeps the download order stable.
    """
    if cache_path is not None and Path(cache_path).exists():
        return pd.read_csv(cache_path, low_memory=False)

    params = {"$limit": limit, "$order": ":id"}
    if where:
        params["$where"] = where
    df = pd.read_csv(API_URL.format(dataset_id) + "?" + urlencode(params), low_memory=False)

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# 2. Column names
# ---------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case snake_case names that are identical for the CSV export and the API.

    Examples: 'Project Completion Date' -> 'project_completion_date',
              '1-BR Units' / '_1_br_units' -> 'br1_units',
              'NTA - Neighborhood Tabulation Area' -> 'nta'.
    """
    def clean(name: str) -> str:
        s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
        s = re.sub(r"^(\d)_br_units$", r"br\1_units", s)          # 1_br_units -> br1_units
        if s.startswith("nta"):
            s = "nta"
        return s

    out = df.copy()
    out.columns = [clean(c) for c in out.columns]
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def standardize_borough(series: pd.Series) -> pd.Series:
    """Map free-text borough values to the five official names (None if outside NYC)."""
    key = series.astype("string").str.strip().str.lower()
    return key.map(BOROUGH_ALIASES).astype("string")


def school_year_of(dates: pd.Series) -> pd.Series:
    """NYC school years run July -> June: 2019-10-03 belongs to '2019-2020'."""
    start = dates.dt.year - (dates.dt.month < 7).astype("Int64")
    return (start.astype("string") + "-" + (start + 1).astype("string")).astype("string")


def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Tukey fences used to *detect* outliers; the notebook decides what to do with them."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


_DELAY_CACHE: dict = {}


def parse_delay_minutes(text) -> float:
    """Convert the free-text How_Long_Delayed field to minutes.

    Handles: '25 minutes', '30MINS', '20', '1hr', '1 hr 15 min', '1:30',
    '45-60 min' (midpoint), '1/2 hour', '1.5 hrs'. Returns NaN when nothing parses.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return np.nan
    if text in _DELAY_CACHE:
        return _DELAY_CACHE[text]

    s = str(text).lower().strip()
    s = s.replace("1/2", "0.5").replace("half", "0.5")
    result = np.nan

    hhmm = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    rng = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)", s)
    hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)\b", s)
    mins = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|min|mins|minute|minutes)\b", s)
    bare = re.fullmatch(r"(\d+(?:\.\d+)?)", s)

    if hhmm:                                            # '1:30' -> 90
        result = int(hhmm.group(1)) * 60 + int(hhmm.group(2))
    elif rng:                                           # '45-60 min' -> 52.5
        mid = (float(rng.group(1)) + float(rng.group(2))) / 2
        result = mid * 60 if ("h" in s and "m" not in s) else mid
    elif hours or mins:                                 # '1 hr 15 min' -> 75
        result = (float(hours.group(1)) * 60 if hours else 0) + \
                 (float(mins.group(1)) if mins else 0)
    elif bare:                                          # '20' -> 20 (unit-less = minutes)
        result = float(bare.group(1))
    else:                                               # '30mins' with no space, etc.
        num = re.search(r"(\d+(?:\.\d+)?)", s)
        if num:
            result = float(num.group(1)) * (60 if "h" in s else 1)

    _DELAY_CACHE[text] = result
    return result


def _to_bool(series: pd.Series) -> pd.Series:
    key = series.astype("string").str.strip().str.lower()
    return key.map({"yes": True, "y": True, "true": True,
                    "no": False, "n": False, "false": False}).astype("boolean")


# ---------------------------------------------------------------------------
# 3a. Housing cleaning
# ---------------------------------------------------------------------------
def clean_housing(raw: pd.DataFrame, log: CleaningLog | None = None) -> pd.DataFrame:
    log = log or CleaningLog()
    df = normalize_columns(raw)
    n = len(df)

    # Duplicates: exact copies first, then the same building reported twice for a project.
    df = df.drop_duplicates()
    log.add("housing", "drop exact duplicates", n, len(df)); n = len(df)
    has_bid = df["building_id"].notna()
    df = pd.concat([df[has_bid].drop_duplicates(["project_id", "building_id"], keep="last"),
                    df[~has_bid]])
    log.add("housing", "drop duplicate (project_id, building_id)", n, len(df),
            "confidential rows have no building_id, so they are only deduplicated exactly"); n = len(df)

    # Standardise formats ----------------------------------------------------
    df["borough"] = standardize_borough(df["borough"])
    for col in ["project_start_date", "project_completion_date", "building_completion_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["postcode"] = (pd.to_numeric(df["postcode"], errors="coerce").astype("Int64")
                      .astype("string").str.zfill(5))
    for col in ["latitude", "longitude"] + INCOME_UNIT_COLS + BEDROOM_UNIT_COLS + \
               ["counted_rental_units", "counted_homeownership_units", "all_counted_units", "total_units"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["reporting_construction_type", "extended_affordability_only", "prevailing_wage_status"]:
        df[col] = df[col].astype("string").str.strip().str.title()
    df["project_name"] = df["project_name"].astype("string").str.strip()

    # Missing values ---------------------------------------------------------
    # Borough is the integration key: a row without it cannot be used -> drop.
    df = df[df["borough"].notna()]
    log.add("housing", "drop rows with missing/invalid borough", n, len(df)); n = len(df)

    # Completion date: a building with no building date but a project date was
    # delivered with its project -> impute from the project date.
    imputed = df["building_completion_date"].isna() & df["project_completion_date"].notna()
    df.loc[imputed, "building_completion_date"] = df.loc[imputed, "project_completion_date"]
    df["completion_imputed"] = imputed
    log.add("housing", f"impute {int(imputed.sum())} building dates from project date", n, len(df))

    # Still no completion date -> the building is not finished yet. It has no
    # completion month to join on, so it leaves the integrated analysis.
    df["is_completed"] = df["building_completion_date"].notna()
    df = df[df["is_completed"]]
    log.add("housing", "drop buildings still in progress (no completion date)", n, len(df)); n = len(df)

    # Unit counts: missing count of a category means 0 units in that category.
    unit_cols = [c for c in INCOME_UNIT_COLS + BEDROOM_UNIT_COLS if c in df]
    df[unit_cols] = df[unit_cols].fillna(0)

    # Inconsistent data ------------------------------------------------------
    # Coordinates outside NYC are wrong -> blank them (row still has a borough).
    bad_xy = ~(df["latitude"].between(*NYC_LAT) & df["longitude"].between(*NYC_LON))
    df.loc[bad_xy, ["latitude", "longitude"]] = np.nan
    log.add("housing", f"blank {int(bad_xy.sum())} coordinates outside NYC / missing", n, len(df))

    # Completion before start is impossible -> drop.
    bad_order = df["project_start_date"].notna() & \
                (df["building_completion_date"] < df["project_start_date"])
    df = df[~bad_order]
    log.add("housing", "drop completion date earlier than start date", n, len(df)); n = len(df)

    # Dates in the future are typos.
    df = df[df["building_completion_date"] <= pd.Timestamp.today()]
    log.add("housing", "drop completion dates in the future", n, len(df)); n = len(df)

    # Income-band units should add up to total units; flag (do not drop) mismatches.
    df["income_units_sum"] = df[[c for c in INCOME_UNIT_COLS if c in df]].sum(axis=1)
    df["units_mismatch"] = (df["income_units_sum"] - df["total_units"]).abs() > 0

    # total_units <= 0 is not a real building record for this analysis.
    df = df[df["total_units"] > 0]
    log.add("housing", "drop buildings with total_units <= 0", n, len(df)); n = len(df)

    # Outliers ---------------------------------------------------------------
    # Large buildings are real (e.g. 1,000-unit towers), so IQR is used to FLAG
    # them, not to drop them; charts use them as-is.
    lo, hi = iqr_bounds(df["total_units"])
    df["units_outlier_iqr"] = ~df["total_units"].between(lo, hi)

    # Derived columns used by the research questions -------------------------
    df["deeply_affordable_units"] = df["extremely_low_income_units"] + df["very_low_income_units"]
    df["family_units"] = df[["br2_units", "br3_units", "br4_units", "br5_units", "br6_units"]].sum(axis=1)
    df["completion_month"] = df["building_completion_date"].dt.to_period("M").dt.to_timestamp()
    df["school_year"] = school_year_of(df["building_completion_date"])
    return df.reset_index(drop=True)


def add_flags(bus: pd.DataFrame) -> pd.DataFrame:
    """Boolean flags derived from the bus text columns.

    Matching is by keyword, case-insensitive, so spelling variants in the raw data
    ('Heavy Traffic', 'HEAVY TRAFFIC', 'Heavy traffic ') are all recognised.
    """
    bus["is_breakdown"] = bus["breakdown_or_running_late"].str.contains("breakdown", case=False, na=False)
    bus["is_heavy_traffic"] = bus["reason"].str.contains("traffic", case=False, na=False)
    bus["is_prek"] = bus["school_age_or_prek"].str.contains("pre", case=False, na=False)
    return bus


# ---------------------------------------------------------------------------
# 3b. Bus cleaning
# ---------------------------------------------------------------------------
def clean_bus(raw: pd.DataFrame, log: CleaningLog | None = None) -> pd.DataFrame:
    log = log or CleaningLog()
    df = normalize_columns(raw)
    n = len(df)

    # Standardise formats ----------------------------------------------------
    for col in ["occurred_on", "created_on", "informed_on", "last_updated_on"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["reason", "run_type", "breakdown_or_running_late", "school_age_or_prek",
                "bus_company_name"]:
        df[col] = df[col].astype("string").str.strip()
    df["reason"] = df["reason"].str.title()
    df["breakdown_or_running_late"] = df["breakdown_or_running_late"].str.title()
    df["school_age_or_prek"] = df["school_age_or_prek"].str.replace(r"\s+", " ", regex=True)
    df["bus_company_name"] = df["bus_company_name"].str.upper().str.replace(r"\s+", " ", regex=True)
    df["school_year"] = df["school_year"].astype("string").str.strip()
    for col in YES_NO_COLS:
        df[col] = _to_bool(df[col])
    df["number_of_students_on_the_bus"] = pd.to_numeric(df["number_of_students_on_the_bus"],
                                                        errors="coerce")

    # Duplicates: the same incident edited several times -> keep the latest edit.
    df = df.drop_duplicates()
    log.add("bus", "drop exact duplicates", n, len(df)); n = len(df)
    df = (df.sort_values("last_updated_on")
            .drop_duplicates("busbreakdown_id", keep="last"))
    log.add("bus", "keep latest version of each busbreakdown_id", n, len(df)); n = len(df)

    # Missing / out-of-scope borough ----------------------------------------
    df["boro_raw"] = df["boro"].astype("string")
    df["borough"] = standardize_borough(df["boro"])
    df = df[df["borough"].notna()]
    log.add("bus", "drop incidents outside the 5 boroughs or with no borough", n, len(df),
            "Nassau, Westchester, NJ, ... have no matching NYC housing records"); n = len(df)

    # Missing / impossible date --------------------------------------------
    df = df[df["occurred_on"].notna()
            & (df["occurred_on"] >= "2015-07-01")
            & (df["occurred_on"] <= pd.Timestamp.today())]
    log.add("bus", "drop missing or out-of-range occurred_on", n, len(df)); n = len(df)

    # School year must agree with the date; recompute it from the date when it does not.
    computed_sy = school_year_of(df["occurred_on"])
    sy_fixed = (df["school_year"] != computed_sy).fillna(True)
    df.loc[sy_fixed, "school_year"] = computed_sy[sy_fixed]
    log.add("bus", f"recompute {int(sy_fixed.sum())} school_year values from occurred_on", n, len(df))

    # Free-text delay -> minutes ---------------------------------------------
    df["delay_minutes"] = df["how_long_delayed"].map(parse_delay_minutes).astype(float)

    # Outliers (domain rules; IQR fences are reported in the notebook) --------
    # Unparseable or implausible delays are set to NaN rather than imputed: a
    # filled-in value would be invented data and would bias the median delay.
    implausible = (df["delay_minutes"] <= 0) | (df["delay_minutes"] > MAX_PLAUSIBLE_DELAY_MIN)
    df.loc[implausible, "delay_minutes"] = np.nan
    log.add("bus", f"set {int(implausible.sum())} implausible delays (<=0 or >{MAX_PLAUSIBLE_DELAY_MIN} min) to NaN",
            n, len(df))
    too_many = df["number_of_students_on_the_bus"] > MAX_STUDENTS_ON_BUS
    df.loc[too_many, "number_of_students_on_the_bus"] = np.nan
    log.add("bus", f"set {int(too_many.sum())} student counts > {MAX_STUDENTS_ON_BUS} to NaN", n, len(df))

    # Derived columns ---------------------------------------------------------
    add_flags(df)
    df["month"] = df["occurred_on"].dt.to_period("M").dt.to_timestamp()

    keep = ["busbreakdown_id", "school_year", "occurred_on", "month", "borough", "boro_raw",
            "run_type", "reason", "breakdown_or_running_late", "school_age_or_prek",
            "bus_company_name", "route_number", "how_long_delayed", "delay_minutes",
            "number_of_students_on_the_bus", "is_breakdown", "is_heavy_traffic", "is_prek"] + YES_NO_COLS
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Aggregation to a common level
# ---------------------------------------------------------------------------
def aggregate_housing(housing: pd.DataFrame, period: str = "month") -> pd.DataFrame:
    """Completions per borough x period. period = 'month' or 'school_year'."""
    key = "completion_month" if period == "month" else "school_year"
    h = housing.assign(
        new_construction_units=np.where(housing["reporting_construction_type"].eq("New Construction"),
                                        housing["total_units"], 0),
        preservation_units=np.where(housing["reporting_construction_type"].eq("Preservation"),
                                    housing["total_units"], 0),
    )
    out = (h.groupby(["borough", key], observed=True)
             .agg(buildings_completed=("total_units", "size"),
                  units_completed=("total_units", "sum"),
                  deeply_affordable_units=("deeply_affordable_units", "sum"),
                  family_units=("family_units", "sum"),
                  new_construction_units=("new_construction_units", "sum"),
                  preservation_units=("preservation_units", "sum"))
             .reset_index())
    return out.rename(columns={key: period})


def aggregate_bus(bus: pd.DataFrame, period: str = "month") -> pd.DataFrame:
    """Incidents and delay statistics per borough x period."""
    out = (bus.groupby(["borough", period], observed=True)
              .agg(incidents=("busbreakdown_id", "size"),
                   breakdowns=("is_breakdown", "sum"),
                   heavy_traffic_incidents=("is_heavy_traffic", "sum"),
                   prek_incidents=("is_prek", "sum"),
                   median_delay_min=("delay_minutes", "median"),
                   delay_obs=("delay_minutes", "count"),
                   avg_students=("number_of_students_on_the_bus", "mean"))
              .reset_index())
    out["running_late"] = out["incidents"] - out["breakdowns"]
    out["heavy_traffic_share"] = out["heavy_traffic_incidents"] / out["incidents"]
    if period == "month":
        # The bus file has its own school_year column; carrying it through lets the
        # post-integration step check it against the one derived on the housing side.
        sy = bus.groupby(["borough", "month"], observed=True)["school_year"].agg(
            lambda s: s.mode().iat[0]).reset_index()
        out = out.merge(sy, on=["borough", "month"], how="left")
    return out


# ---------------------------------------------------------------------------
# 5. Integration
# ---------------------------------------------------------------------------
def integrate(housing: pd.DataFrame, bus: pd.DataFrame, period: str = "month") -> pd.DataFrame:
    """Join housing and bus aggregates on (borough, period) over their common time span.

    A full borough x period grid is built first so that a borough with zero
    completions in a month still appears (as 0) instead of silently vanishing,
    which an inner join would do.
    """
    h = aggregate_housing(housing, period)
    b = aggregate_bus(bus, period)

    if period == "month":
        start = max(h["month"].min(), b["month"].min())
        end = min(h["month"].max(), b["month"].max())
        periods = pd.date_range(start, end, freq="MS")
        grid = pd.MultiIndex.from_product([BOROUGHS, periods], names=["borough", "month"]).to_frame(index=False)
        grid["school_year"] = school_year_of(grid["month"])
    else:
        common = sorted(set(h["school_year"]) & set(b["school_year"]))
        grid = pd.MultiIndex.from_product([BOROUGHS, common], names=["borough", "school_year"]).to_frame(index=False)

    grid["borough"] = grid["borough"].astype("string")
    merged = grid.merge(h, on=["borough", period], how="left") \
                 .merge(b, on=["borough", period], how="left", suffixes=("", "_bus"))
    return merged


# ---------------------------------------------------------------------------
# 6. Post-integration cleaning
# ---------------------------------------------------------------------------
HOUSING_COUNT_COLS = ["buildings_completed", "units_completed", "deeply_affordable_units",
                      "family_units", "new_construction_units", "preservation_units"]
BUS_COUNT_COLS = ["incidents", "breakdowns", "running_late", "heavy_traffic_incidents",
                  "prek_incidents", "delay_obs"]


def post_integration_clean(merged: pd.DataFrame, period: str = "month",
                           log: CleaningLog | None = None) -> pd.DataFrame:
    log = log or CleaningLog()
    df = merged.copy()
    n = len(df)

    # (a) Redundant / conflicting columns --------------------------------------
    # Monthly join brings two school_year columns: the grid's (from the date) and
    # the bus file's. Check they agree, then keep one.
    if "school_year_bus" in df:
        both = df["school_year_bus"].notna()
        disagree = int((df.loc[both, "school_year"] != df.loc[both, "school_year_bus"]).sum())
        log.add("integrated", f"school_year columns disagree on {disagree} rows -> keep date-derived one",
                n, len(df))
        df = df.drop(columns="school_year_bus")
    # running_late is breakdowns' complement and heavy_traffic_incidents is
    # captured by heavy_traffic_share; keep both counts (needed for re-aggregation)
    # but drop columns that duplicate information exactly.
    if {"units_completed", "new_construction_units", "preservation_units"} <= set(df.columns):
        other = df["units_completed"] - df["new_construction_units"] - df["preservation_units"]
        df["other_construction_units"] = other

    # (b) Missing values created by the join ------------------------------------
    # Housing side: no row in the aggregate means no building was completed -> 0.
    df[HOUSING_COUNT_COLS] = df[HOUSING_COUNT_COLS].fillna(0)
    df["other_construction_units"] = df["other_construction_units"].fillna(0)

    # Bus side: a month with (almost) no incidents in the whole city is a month
    # without school bus service (summer, COVID closure from March 2020), NOT a
    # perfect month. Mark it and leave its bus metrics NaN so it is excluded.
    if period == "month":
        city = df.groupby("month")["incidents"].transform("sum")
        threshold = 0.05 * df.groupby("month")["incidents"].sum().median()
        df["school_in_session"] = city.fillna(0) > threshold
        off = ~df["school_in_session"]
        df.loc[off, BUS_COUNT_COLS + ["median_delay_min", "avg_students", "heavy_traffic_share"]] = np.nan
        df.loc[df["school_in_session"], BUS_COUNT_COLS] = df.loc[df["school_in_session"], BUS_COUNT_COLS].fillna(0)
        log.add("integrated", f"flag {int(off.sum())} borough-months with no school bus service",
                n, len(df), "bus metrics left NaN, excluded from correlations")
    else:
        df[BUS_COUNT_COLS] = df[BUS_COUNT_COLS].fillna(0)

    # (c) Data types: the left joins turned integer counts into float -> back to Int64.
    for col in HOUSING_COUNT_COLS + ["other_construction_units"] + BUS_COUNT_COLS:
        df[col] = df[col].round().astype("Int64")
    df["borough"] = df["borough"].astype("string")
    df["school_year"] = df["school_year"].astype("string")
    if period == "month":
        df["month"] = pd.to_datetime(df["month"])
        df["year"] = df["month"].dt.year.astype("Int64")
        # Cumulative pipeline: a building keeps housing people after the month it opens.
        df = df.sort_values(["borough", "month"])
        df["units_last_12m"] = (df.groupby("borough")["units_completed"]
                                  .transform(lambda s: s.rolling(12, min_periods=1).sum())
                                  .astype("Int64"))

    df = df.drop_duplicates(subset=["borough", period])
    log.add("integrated", "drop duplicate (borough, period) keys", n, len(df))
    return df.reset_index(drop=True)


def build_integrated(housing: pd.DataFrame, bus: pd.DataFrame, period: str = "month",
                     log: CleaningLog | None = None) -> pd.DataFrame:
    """integrate + post_integration_clean in one call (used by the website)."""
    return post_integration_clean(integrate(housing, bus, period), period, log)


# ---------------------------------------------------------------------------
# Analysis helpers shared by notebook and website
# ---------------------------------------------------------------------------
def lagged_correlations(monthly: pd.DataFrame, x: str = "units_completed",
                        y: str = "incidents", max_lag: int = 12) -> pd.DataFrame:
    """Pearson r between housing completions shifted by `lag` months and bus incidents,
    per borough, using only months with school bus service."""
    rows = []
    for borough, g in monthly.sort_values("month").groupby("borough"):
        xs = g[x].astype(float)
        ys = g[y].astype(float)
        for lag in range(max_lag + 1):
            pair = pd.concat([xs.shift(lag), ys], axis=1).dropna()
            # A correlation needs variation on both sides and a handful of months.
            usable = len(pair) > 6 and pair.iloc[:, 0].std() > 0 and pair.iloc[:, 1].std() > 0
            r = pair.iloc[:, 0].corr(pair.iloc[:, 1]) if usable else np.nan
            rows.append({"borough": borough, "lag_months": lag, "r": r, "n": len(pair)})
    return pd.DataFrame(rows)


def ols_line(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """Slope, intercept and Pearson r of a simple least-squares fit (NaNs dropped)."""
    pair = pd.concat([x.astype(float), y.astype(float)], axis=1).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(pair.iloc[:, 0], pair.iloc[:, 1], 1)
    return slope, intercept, pair.iloc[:, 0].corr(pair.iloc[:, 1])