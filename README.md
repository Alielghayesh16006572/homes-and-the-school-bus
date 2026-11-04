# Homes and the School Bus

**NYC affordable housing completions × school-bus breakdowns and delays**
Data Engineering and Visualization, Winter 2026 — Project Milestone 1

- **Live website:** https://homes-and-the-school-bus.onrender.com/
- **Repository:** https://github.com/Alielghayesh16006572/homes-and-the-school-bus
- **Datasets (NYC Open Data):**
  - Housing & Development — [Affordable Housing Production by Building](https://data.cityofnewyork.us/d/hg8x-zxpr) (`hg8x-zxpr`)
  - Transportation — [Bus Breakdown and Delays](https://data.cityofnewyork.us/d/ez4e-fazm) (`ez4e-fazm`)

## The question

Do the boroughs and periods that deliver the most affordable housing also see more, or longer, school-bus
disruptions — and are the lowest-income and youngest riders affected differently?

The two datasets share no key. The bus data only records the borough, so they are integrated by aggregating
both to **borough × school year** (main level) and **borough × month** (time-series and lag questions).
The notebook justifies this choice and the alternatives considered.

## Repository layout

```
├── notebook.ipynb            EDA, cleaning, integration, post-integration cleaning, 8 research questions
├── app.py                    Dash website (filters, search mode, Generate Report, 8 charts, story)
├── src/
│   ├── pipeline.py           loading, cleaning, aggregation, integration, post-integration cleaning
│   ├── figures.py            one Plotly figure function per research question (used by notebook AND website)
│   └── insights.py           one-sentence takeaway per research question, recomputed for the current filters
├── assets/style.css          website styling (loaded automatically by Dash)
├── data/
│   ├── raw/                  raw downloads (git-ignored, re-created by the notebook)
│   └── processed/            cleaned tables written by the notebook and read by the website
├── requirements.txt
└── render.yaml               Render deployment config
```

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/Alielghayesh16006572/homes-and-the-school-bus.git
cd homes-and-the-school-bus
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
```

### 1. Run the notebook (creates the processed data)

```bash
jupyter notebook notebook.ipynb    # then Run All
```

The first run downloads both datasets from the NYC Open Data API into `data/raw/` (a few minutes for the bus
file) and writes the cleaned tables to `data/processed/`. In Colab, upload the repository folder (or clone it)
and run all cells from the repository root so `src/` can be imported.

### 2. Run the website locally

```bash
python app.py                      # http://127.0.0.1:8050
```

## Deployment (Render, free tier)

1. Run the notebook so `data/processed/` contains `housing_clean.csv.gz` and `bus_clean.csv.gz`, then commit and
   push them — the website reads these files and does not download anything at runtime.
2. On [render.com](https://render.com): **New → Blueprint**, connect the GitHub repository. Render reads
   `render.yaml` (build: `pip install -r requirements.txt`, start: `gunicorn app:server`).
   Alternatively **New → Web Service** and enter those two commands by hand.
3. Open the generated `onrender.com` URL and check: dropdowns, search (try `Brooklyn 2022 new buildings`),
   **Generate Report**, hover/zoom on every chart, and the map.

The free tier sleeps after inactivity; the first request after that takes about a minute.

## Using the website

- **Filters:** Borough, School year, Housing construction type, Bus incident type, Bus delay reason.
- **Search mode:** type e.g. `Queens 2019 breakdown traffic` and press Enter. Recognised words fill the
  dropdowns (boroughs, 4-digit years → the school years containing that year, `new` / `preservation`,
  `breakdown` / `late`, delay-reason words); unrecognised words are reported.
- **Generate Report:** recomputes the integration on the filtered rows and redraws all eight charts and the
  live numbers in the story section. A line above the charts shows which filters the report uses, and each
  chart has a one-sentence takeaway computed from the same rows.

## Research questions and contributions

| Member (ID) | Research questions | Visualizations | Data cleaning | Integration | Website |
|---|---|---|---|---|---|
| Asser Ehab Khattab (16003477) | **RQ2** Does the 12-month housing pipeline move with monthly bus incidents over time? **RQ5** Is new affordable housing concentrated where bus disruptions concentrate? | Two-panel time series; map | Housing: duplicates, borough, dates | Monthly grid + join | Search mode |
| Ali Sherif Elghayesh (16006572) | **RQ1** Do borough-years with more affordable units completed log more school-bus incidents? **RQ6** Is heavy traffic a larger share of delay causes in high new-construction years? | Scatter with fit; grouped bar | Housing: coordinates, unit consistency | School-year aggregation | Map, layout and CSS |
| Ali Ahmed (16001891) | **RQ3** Where more new units are for extremely/very low-income households, are delays longer? **RQ7** Do borough-years adding more family-sized units see more students on disrupted buses? | Bubble scatter; scatter | Bus: borough mapping, duplicates, student counts | Post-integration cleaning | Story section, deployment |
| Ahmed Rashad (16004574) | **RQ4** Do completions precede bus incidents, and by how many months? **RQ8** Are Pre-K and school-age delays different in high- vs low-completion years? | Lag-correlation heatmap; box plot | Bus: delay-text parsing, outliers | Lagged correlation | Generate Report callback |


## Limitations

Correlation is not causation; the bus data is borough-level only; only HPD-financed affordable units are
included; the bus data has no route or rider counts to normalise by; delay lengths are self-reported free text.
Details are in the notebook and on the website.
