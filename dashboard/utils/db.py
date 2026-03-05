import duckdb
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "dashboard.duckdb"


def get_connection():
    return duckdb.connect(str(DB_PATH), read_only=True)


def get_playlist_list(
    owner_type: str | None = None,
    mau_group: str | None = None,
    pred_label: int | None = None,
    search_uri: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> pd.DataFrame:
    con = get_connection()
    where_clauses = []
    params = []

    if owner_type:
        where_clauses.append("owner_type = ?")
        params.append(owner_type)
    if mau_group:
        where_clauses.append("mau_group = ?")
        params.append(mau_group)
    if pred_label is not None:
        where_clauses.append("pred_label = ?")
        params.append(pred_label)
    if search_uri:
        where_clauses.append("playlist_uri ILIKE ?")
        params.append(f"%{search_uri}%")

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    query = f"""
        SELECT row_id, playlist_uri, pred_proba, pred_label,
               owner_type, mau_group, mau
        FROM dashboard
        {where}
        ORDER BY pred_proba DESC
        LIMIT {limit} OFFSET {offset}
    """
    df = con.execute(query, params).fetchdf()
    con.close()
    return df


def get_playlist_detail(row_id: str) -> dict:
    con = get_connection()
    row = con.execute(
        "SELECT * FROM dashboard WHERE row_id = ?", [row_id]
    ).fetchdf()
    con.close()
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def get_total_count(
    owner_type: str | None = None,
    mau_group: str | None = None,
    pred_label: int | None = None,
    search_uri: str | None = None,
) -> int:
    con = get_connection()
    where_clauses = []
    params = []

    if owner_type:
        where_clauses.append("owner_type = ?")
        params.append(owner_type)
    if mau_group:
        where_clauses.append("mau_group = ?")
        params.append(mau_group)
    if pred_label is not None:
        where_clauses.append("pred_label = ?")
        params.append(pred_label)
    if search_uri:
        where_clauses.append("playlist_uri ILIKE ?")
        params.append(f"%{search_uri}%")

    where = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    count = con.execute(
        f"SELECT COUNT(*) FROM dashboard {where}", params
    ).fetchone()[0]
    con.close()
    return count


def get_global_shap_importance(top_n: int = 20) -> pd.DataFrame:
    con = get_connection()
    shap_cols = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'dashboard' AND column_name LIKE 'shap__%' "
        "AND column_name != 'shap_base_value_raw'"
    ).fetchdf()["column_name"].tolist()

    agg_parts = [
        f"AVG(ABS(\"{c}\")) AS \"{c.replace('shap__', '')}\"" for c in shap_cols
    ]
    query = f"SELECT {', '.join(agg_parts)} FROM dashboard"
    result = con.execute(query).fetchdf()
    con.close()

    importance = result.T.reset_index()
    importance.columns = ["feature", "mean_abs_shap"]
    importance = importance.sort_values("mean_abs_shap", ascending=False).head(top_n)
    return importance


def get_segment_stats() -> pd.DataFrame:
    con = get_connection()
    query = """
        SELECT
            owner_type,
            mau_group,
            COUNT(*) AS n,
            AVG(pred_proba) AS avg_pred_proba,
            AVG(CAST(pred_label AS DOUBLE)) AS pct_predicted_success
        FROM dashboard
        GROUP BY owner_type, mau_group
        ORDER BY owner_type, mau_group
    """
    df = con.execute(query).fetchdf()
    con.close()
    return df


def get_pred_distribution() -> pd.DataFrame:
    con = get_connection()
    df = con.execute(
        "SELECT pred_proba, pred_label FROM dashboard"
    ).fetchdf()
    con.close()
    return df


def get_feature_stats() -> dict:
    """Compute population stats (mean, p25, p50, p75) for every feat__ column.

    Used to contextualise individual SHAP driver values so a PM can see
    whether a value is low / typical / high relative to the full dataset.
    """
    con = get_connection()
    feat_cols = con.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'dashboard' AND column_name LIKE 'feat__%'"
    ).fetchdf()["column_name"].tolist()

    agg_parts = []
    for c in feat_cols:
        safe = f'"{c}"'
        name = c.replace("feat__", "")
        agg_parts.extend([
            f'AVG({safe}) AS "{name}__mean"',
            f'PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {safe}) AS "{name}__p25"',
            f'PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY {safe}) AS "{name}__p50"',
            f'PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {safe}) AS "{name}__p75"',
        ])

    result = con.execute(f"SELECT {', '.join(agg_parts)} FROM dashboard").fetchdf()
    con.close()

    stats: dict = {}
    for c in feat_cols:
        name = c.replace("feat__", "")
        stats[name] = {
            "mean": float(result[f"{name}__mean"].iloc[0]),
            "p25": float(result[f"{name}__p25"].iloc[0]),
            "p50": float(result[f"{name}__p50"].iloc[0]),
            "p75": float(result[f"{name}__p75"].iloc[0]),
        }
    return stats
