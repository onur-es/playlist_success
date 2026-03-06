# Playlist Success Exploration

Repository for two main workflows:

- notebook-based exploration in [`notebooks/playlist_success.ipynb`](notebooks/playlist_success.ipynb)
- a Streamlit dashboard in [`dashboard/app.py`](dashboard/app.py)

## Setup

- Python: `3.12`
- Install dependencies from the repo root:

```bash
uv sync
```

If you do not use `uv`, create a virtual environment and install from [`pyproject.toml`](pyproject.toml).

## Environment

Copy [`.env.example`](.env.example) to `.env`.

The only variable needed for the current notebook + dashboard workflow is:

```bash
ANTHROPIC_API_KEY=your_key_here
```

If the key is missing, the dashboard still runs, but AI explanations are disabled.

## Notebook

Main notebook:

- [`notebooks/playlist_success.ipynb`](notebooks/playlist_success.ipynb)

Run Jupyter from the repo root:

```bash
uv run jupyter lab
```

## Streamlit App

App entrypoint:

- [`dashboard/app.py`](dashboard/app.py)

Run from the repo root:

```bash
uv run streamlit run dashboard/app.py
```

The dashboard reads from:

- [`dashboard/data/dashboard.duckdb`](dashboard/data/dashboard.duckdb)

## Data

Main data files:

- [`dashboard/data/dashboard.duckdb`](dashboard/data/dashboard.duckdb): data source used by the Streamlit app
- [`notebooks/data/dashboard_all_in_one.parquet`](notebooks/data/dashboard_all_in_one.parquet): notebook-side dashboard table
- [`notebooks/data/README.md`](notebooks/data/README.md): column guide and dataset notes

Approximate sizes:

- `dashboard/data/dashboard.duckdb`: `87 MB`
- `notebooks/data/dashboard_all_in_one.parquet`: `99 MB`

## Deployment

For Streamlit Community Cloud:

- repo: `onur-es/playlist_success`
- branch: `main`
- app file: `dashboard/app.py`
- dependency file: `dashboard/requirements.txt`

Add this secret in the Streamlit app settings:

```toml
ANTHROPIC_API_KEY = "your_key_here"
```
