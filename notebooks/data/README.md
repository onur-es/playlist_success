# Dashboard Dataset Guide

## File
- `dashboard_all_in_one.parquet`
- Location: `notebooks/data/dashboard_all_in_one.parquet`
- Current shape: `74,835 rows x 182 columns`

## Goal
This dataset is the final, dashboard-ready test-set table for model output + explainability.
Use it to build interactive views such as:
- Playlist-level prediction cards
- Global and filtered performance slices
- SHAP-based "why" explanations per playlist

## Row Grain
Each row = one playlist from the model test set.

## Column Groups

### 1) Core prediction/evaluation columns
- `row_id`: row key used for joins/lookup in UI
- `playlist_uri`: playlist identifier
- `y_true`: actual test label (0/1)
- `pred_proba`: predicted probability of success
- `pred_label`: predicted class after thresholding
- `is_correct`: `1` if `pred_label == y_true`, else `0`
- `threshold_used`: threshold used to create `pred_label` (best F1 threshold from LightGBM workflow)
- `shap_base_value_raw`: SHAP base value in raw model output space

### 2) Human-readable explanation columns
- `top_positive_json`: JSON list of top positive SHAP contributors for that row
- `top_negative_json`: JSON list of top negative SHAP contributors for that row

Each JSON element has:
- `feature`
- `feature_value`
- `shap_value`

### 3) Metadata columns (business context)
Examples:
- `owner_type`
- `mau_group`
- `mau`
- `engagement_rate`
- `retention_rate`
- `monthly_stream30s_per_mau`
- `engagement_median`
- `retention_median`
- `is_high_engagement`
- `is_high_retention`
- `is_high_both`
- `segment`

### 3b) Benchmark/segment columns (added)
These columns carry the group benchmark each playlist was compared against (by `mau_group`):
- `engagement_median`: median engagement benchmark for that playlist's MAU group
- `retention_median`: median retention benchmark for that playlist's MAU group
- `is_high_engagement`: `1/0` flag for playlist engagement >= group median
- `is_high_retention`: `1/0` flag for playlist retention >= group median
- `is_high_both`: `1/0` flag where both high engagement and high retention are true
- `segment`: readable segment label (for example, "High Engagement + High Retention")

### 4) Model input feature columns
Prefix: `feat__`
- Count: `80`
- These are exactly the features used by the final model input matrix (`X_test`)
- Includes numeric features, token flags, purpose flags, genre/mood tag dummies, owner dummy

### 5) SHAP per-feature columns
Prefix: `shap__`
- Count: `80`
- One-to-one aligned with `feat__` columns
- `shap__<feature>` explains contribution of `feat__<feature>` for that row

## Important Interpretation Notes
1. SHAP values are feature contributions in model output space (raw margin/log-odds style for this tree model setup).
2. `pred_proba` is the final probability output used for prediction and UI scoring.
3. `top_positive_json` / `top_negative_json` are truncated top-k lists; they are for readability, not full exact decomposition.
4. For exact per-row decomposition, use all `shap__*` columns plus `shap_base_value_raw`.

## Recommended Dashboard Usage

### Playlist detail page
Show:
- `playlist_uri`, `pred_proba`, `pred_label`, `y_true`, `is_correct`
- Top positive/negative drivers from JSON fields
- Optional table of selected `feat__*` values

### Global pages
Use:
- Distribution of `pred_proba`
- Precision/recall style slices by threshold
- Error analysis with `is_correct`, split by `owner_type`, `mau_group`, `segment`, etc.
- Benchmark views: compare `engagement_rate` vs `engagement_median` and `retention_rate` vs `retention_median`

### Explainability views
- Top global drivers: mean absolute `shap__*`
- Segment-level drivers: average `shap__*` by `owner_type`, `mau_group`, predicted class, correctness

## What Not To Do
- Do not use `y_true` as an input signal in any scoring UI component; it is only for evaluation.
- Do not compare SHAP values from this model directly against SHAP values from a different model run without recalibration.
- Do not assume top-k JSON lists contain all contribution mass.

## Quick Load
```python
import pandas as pd

dashboard_df = pd.read_parquet("notebooks/data/dashboard_all_in_one.parquet")
```

## Quick Column Helpers
```python
feat_cols = [c for c in dashboard_df.columns if c.startswith("feat__")]
shap_cols = [c for c in dashboard_df.columns if c.startswith("shap__")]
```
