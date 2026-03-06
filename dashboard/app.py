import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from dotenv import load_dotenv
from pathlib import Path
from utils.db import (
    get_playlist_list,
    get_playlist_detail,
    get_global_shap_importance,
    get_total_count,
    get_feature_stats,
)
from utils.explainer import get_explanation
from utils.explorer import (
    EXPLORER_PAGE_SIZE,
    clamp_page,
    get_dashboard_tabs,
    get_page_offset,
    get_total_pages,
)

load_dotenv(Path(__file__).parent.parent / ".env")


@st.cache_data(show_spinner=False)
def cached_explanation(row_id: str, playlist_json: str, feat_stats_json: str) -> str:
    return get_explanation(json.loads(playlist_json), json.loads(feat_stats_json))


@st.cache_data(show_spinner=False)
def cached_feature_stats() -> dict:
    return get_feature_stats()


_BINARY_PREFIXES = ("tok_", "genre_tag__", "mood_tag__", "purpose_", "is_positive_local_tracks", "owner_type")


def _short_name(feature: str) -> str:
    """Compact, PM-friendly display name for a feature."""
    if feature.startswith("genre_tag__"):
        return "genre: " + feature[len("genre_tag__"):].replace("_", " ")
    if feature.startswith("mood_tag__"):
        return "mood: " + feature[len("mood_tag__"):].replace("_", " ")
    if feature.startswith("tok_"):
        return 'title: "' + feature[len("tok_"):] + '"'
    if feature.startswith("purpose_"):
        return "purpose: " + feature[len("purpose_"):].replace("_", " ")
    return feature.replace("_", " ")


def _fmt_prevalence(prevalence: float) -> str:
    pct = prevalence * 100
    if pct < 1:
        return "<1%"
    return f"{pct:.0f}%"


def format_driver_label(feature: str, value, stats: dict | None = None) -> str:
    """Label showing feature value + population context (prevalence or quartile bucket)."""
    display = _short_name(feature)
    is_binary = any(feature.startswith(p) for p in _BINARY_PREFIXES)
    feat_stats = stats.get(feature, {}) if stats else {}

    if is_binary:
        present = float(value) >= 1
        return f"{display} = {'Yes' if present else 'No'}"

    try:
        v = float(value)
        val_str = str(int(v)) if v == int(v) and abs(v) < 1e6 else f"{v:.2f}"
    except (ValueError, TypeError):
        return f"{display} = {value}"

    p25 = feat_stats.get("p25")
    p75 = feat_stats.get("p75")
    if p25 is not None and p75 is not None:
        if v <= p25:
            context = "low"
        elif v >= p75:
            context = "high"
        else:
            context = "typical"
        return f"{display} = {val_str} ({context})"

    return f"{display} = {val_str}"


def format_driver_hover(feature: str, value, shap_value: float, stats: dict | None = None) -> str:
    """Rich hover text for SHAP driver bar charts."""
    feat_stats = stats.get(feature, {}) if stats else {}
    is_binary = any(feature.startswith(p) for p in _BINARY_PREFIXES)

    if is_binary:
        present = float(value) >= 1
        prevalence = feat_stats.get("mean")
        lines = [
            f"{'Has' if present else 'Lacks'} this flag",
        ]
        if prevalence is not None:
            lines.append(f"{_fmt_prevalence(prevalence)} of all playlists have it")
        lines.append(f"SHAP contribution: {shap_value:+.4f}")
        return "<br>".join(lines)

    try:
        v = float(value)
    except (ValueError, TypeError):
        return f"SHAP: {shap_value:+.4f}"

    median = feat_stats.get("p50")
    lines = [f"Value: {v:.2f}"]
    if median is not None:
        lines.append(f"Dataset median: {median:.2f}")
    lines.append(f"SHAP contribution: {shap_value:+.4f}")
    return "<br>".join(lines)


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Playlist X-Ray",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS overrides ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Sora:wght@300;400;500;600;700&display=swap');

/* Typography */
html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

h1, h2, h3 {
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
}

code, .stCode, div[data-testid="stMetric"] label {
    font-family: 'IBM Plex Mono', monospace !important;
}

/* App background */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #121212 !important;
}
.stMainBlockContainer, [data-testid="stMain"] {
    background-color: #121212 !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: #181818;
    border: 1px solid #282828;
    border-radius: 12px;
    padding: 20px;
}

div[data-testid="stMetric"] label {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #a7a7a7 !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    color: #e6edf3 !important;
}

/* Header area */
.hero-subtitle {
    color: #a7a7a7;
    font-size: 1rem;
    font-weight: 300;
    margin-top: -8px;
    margin-bottom: 24px;
}

/* Detail card */
.detail-header {
    background: #181818;
    border: 1px solid #282828;
    border-radius: 12px;
    padding: 20px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

.detail-uri {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #a7a7a7;
    word-break: break-all;
}

.pill {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 5px 12px;
    border-radius: 20px;
    margin-left: 8px;
}

.pill-green { background: #1a3a2a; color: #1DB954; border: 1px solid #1DB954; }
.pill-red { background: #3a1a1a; color: #f85149; border: 1px solid #88232a; }
.pill-blue { background: #1a2a3a; color: #58a6ff; border: 1px solid #1f6feb; }
.pill-gray { background: #282828; color: #a7a7a7; border: 1px solid #383838; }

/* AI explanation box */
.ai-explanation {
    background: linear-gradient(135deg, #181818 0%, #1a2420 100%);
    border: 1px solid #282828;
    border-left: 3px solid #1DB954;
    border-radius: 12px;
    padding: 24px 28px;
    font-size: 0.92rem;
    line-height: 1.75;
    color: #e6edf3;
    margin-top: 8px;
}

.ai-explanation strong, .ai-explanation b {
    color: #1DB954;
}
.ai-explanation ul {
    margin: 8px 0 8px 4px;
    padding-left: 18px;
}
.ai-explanation li {
    margin-bottom: 6px;
}
.ai-explanation .feat {
    background: #1db95418;
    border: 1px solid #1db95440;
    color: #58d68d;
    padding: 1px 7px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82em;
    white-space: nowrap;
}
.ai-explanation .val {
    color: #f0c674;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.92em;
}
.ai-explanation .pos {
    color: #1DB954;
    font-weight: 600;
}
.ai-explanation .neg {
    color: #f85149;
    font-weight: 600;
}

.ai-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #1DB954;
    margin-bottom: 8px;
}

/* Section dividers */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #535353;
    margin-bottom: 4px;
}

/* Sidebar polish */
section[data-testid="stSidebar"] {
    background-color: #0b0b0b !important;
    border-right: 1px solid #282828 !important;
}

/* Tab styling */
button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* Button override */
.stButton > button[kind="primary"] {
    background-color: #1DB954 !important;
    color: #000 !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #1ed760 !important;
}

/* Pagination bar */
.pagination-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 12px 0;
    margin: 4px 0 12px 0;
}
.pagination-bar .page-info {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #8b949e;
    letter-spacing: 0.03em;
    min-width: 140px;
    text-align: center;
}
.pagination-bar .page-info strong {
    color: #e6edf3;
    font-weight: 600;
}

/* Row-click hint */
.table-hint {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #484f58;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 6px;
}

/* Pagination buttons */
.stButton > button[kind="secondary"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.04em !important;
    border: 1px solid #383838 !important;
    border-radius: 8px !important;
    padding: 6px 18px !important;
    color: #b3b3b3 !important;
    background: #181818 !important;
    transition: all 0.15s ease !important;
}
.stButton > button[kind="secondary"]:hover:not(:disabled) {
    border-color: #1DB954 !important;
    color: #1DB954 !important;
    background: #121212 !important;
}
.stButton > button[kind="secondary"]:disabled {
    opacity: 0.3 !important;
}

/* Force ProgressColumn bars to green */
td[data-testid="column-P(Success)"] progress {
    accent-color: #1DB954 !important;
}
td[data-testid="column-P(Success)"] [role="progressbar"] div {
    background-color: #1DB954 !important;
}
/* Glide data grid progress bar override */
[data-testid="stDataFrame"] canvas + div [style*="background"] {
    background-color: #1DB954 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Plotly template ──────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Sora, sans-serif", color="#a7a7a7"),
)


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">MODEL EXPLAINABILITY</p>', unsafe_allow_html=True)
st.title("Playlist X-Ray")
st.markdown(
    '<p class="hero-subtitle">'
    "Understand what drives playlist success — powered by SHAP and Claude"
    "</p>",
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")

    search_uri = st.text_input(
        "Search by URI",
        placeholder="spotify:playlist:...",
        label_visibility="collapsed",
    )
    st.caption("Search by playlist URI")

    owner_filter = st.selectbox("Owner type", ["All", "user", "spotify"])
    mau_filter = st.selectbox("MAU group", ["All", "Low MAU (<=10)", "High MAU (>10)"])
    pred_label_filter = st.selectbox("Predicted class", ["All", "1 — Success", "0 — Fail"])

    owner_val = None if owner_filter == "All" else owner_filter
    mau_val = None if mau_filter == "All" else mau_filter
    pred_label_val = None if pred_label_filter == "All" else int(pred_label_filter[0])
    search_val = search_uri.strip() or None

    total = get_total_count(owner_val, mau_val, pred_label=pred_label_val, search_uri=search_val)

    st.markdown("---")
    st.metric("Matching playlists", f"{total:,}")
    st.markdown("---")
    st.caption("Built with Streamlit and Claude")


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_explore, tab_global = st.tabs(list(get_dashboard_tabs()))


# ════════════════════════════════════════════════════════════════════════════
# TAB 1: PLAYLIST EXPLORER
# ════════════════════════════════════════════════════════════════════════════
with tab_explore:
    current_page = clamp_page(
        page=st.session_state.get("explorer_page", 1),
        total_count=total,
        page_size=EXPLORER_PAGE_SIZE,
    )
    st.session_state["explorer_page"] = current_page
    total_pages = get_total_pages(total_count=total, page_size=EXPLORER_PAGE_SIZE)
    offset = get_page_offset(page=current_page, total_count=total, page_size=EXPLORER_PAGE_SIZE)

    playlists = get_playlist_list(
        owner_type=owner_val,
        mau_group=mau_val,
        pred_label=pred_label_val,
        search_uri=search_val,
        limit=EXPLORER_PAGE_SIZE,
        offset=offset,
    )

    if playlists.empty:
        st.info("No playlists match the current filters.")
    else:
        page_start = offset + 1
        page_end = min(offset + EXPLORER_PAGE_SIZE, total)

        # ── Pagination bar ────────────────────────────────────────────
        pg_left, pg_mid, pg_right = st.columns([1, 2, 1])
        with pg_left:
            if st.button("← Prev", disabled=(current_page <= 1), key="pg_prev", use_container_width=True):
                st.session_state["explorer_page"] = current_page - 1
                st.rerun()
        with pg_mid:
            st.markdown(
                f'<div class="pagination-bar">'
                f'<span class="page-info">Page <strong>{current_page}</strong> of <strong>{total_pages:,}</strong>'
                f' &nbsp;·&nbsp; {page_start:,}–{page_end:,} of {total:,}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with pg_right:
            if st.button("Next →", disabled=(current_page >= total_pages), key="pg_next", use_container_width=True):
                st.session_state["explorer_page"] = current_page + 1
                st.rerun()

        # ── Clickable table ───────────────────────────────────────────
        display_df = playlists.copy()
        display_df["P(Success)"] = display_df["pred_proba"] * 100  # 0-100 for ProgressColumn
        display_df["Predicted"] = display_df["pred_label"].map({1: "✓ Success", 0: "✗ Fail"})
        display_df["URI"] = display_df["playlist_uri"].str[-22:]
        display_df = display_df[["URI", "P(Success)", "Predicted", "owner_type", "mau"]]
        display_df.columns = ["URI", "P(Success)", "Predicted", "Owner", "MAU"]

        st.markdown('<p class="table-hint"><span style="font-size:1.3em">☑</span> Select the checkbox next to a playlist to inspect it</p>', unsafe_allow_html=True)
        event = st.dataframe(
            display_df,
            column_config={
                "P(Success)": st.column_config.ProgressColumn(
                    "P(Success)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
            },
            use_container_width=True,
            height=400,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )

        # Determine selected playlist from row click
        selected_rows = event.selection.rows if event.selection else []
        selected_id = str(playlists.iloc[selected_rows[0]]["row_id"]) if selected_rows else None

        # ── Playlist detail ──────────────────────────────────────────────
        if selected_id:

            st.markdown("---")
            st.markdown('<p class="section-label">PLAYLIST DETAIL</p>', unsafe_allow_html=True)

            detail = get_playlist_detail(selected_id)
            if not detail:
                st.error("Playlist not found.")
            else:
                # Status pill
                pred_pill = "pill-green" if detail["pred_label"] == 1 else "pill-red"
                pred_text = "Likely to succeed" if detail["pred_label"] == 1 else "Unlikely to succeed"

                st.markdown(f"""
                <div class="detail-header">
                    <span class="detail-uri">{detail['playlist_uri']}</span>
                    <div>
                        <span class="pill {pred_pill}">{pred_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Metrics row 1
                m1, m2, m3 = st.columns(3)
                m1.metric("P(Success)", f"{detail['pred_proba']:.1%}")
                m2.metric("Owner", detail.get("owner_type", "—"))
                m3.metric("MAU", f"{detail.get('mau', 0):,}")

                # Metrics row 2 — engagement & retention vs benchmarks
                eng_rate = detail.get("engagement_rate")
                eng_med = detail.get("engagement_median")
                ret_rate = detail.get("retention_rate")
                ret_med = detail.get("retention_median")
                segment = detail.get("segment", "—")

                e1, e2 = st.columns(2)
                if eng_rate is not None and eng_med is not None:
                    eng_delta = eng_rate - eng_med
                    e1.metric(
                        "Engagement (streams/user)",
                        f"{eng_rate:.1f}",
                        delta=f"{eng_delta:+.1f} vs median",
                        delta_color="normal",
                    )
                else:
                    e1.metric("Engagement", "N/A")
                if ret_rate is not None and ret_med is not None:
                    ret_delta = ret_rate - ret_med
                    e2.metric(
                        "Retention",
                        f"{ret_rate:.0%}",
                        delta=f"{ret_delta:+.0%} vs median",
                        delta_color="normal",
                    )
                else:
                    e2.metric("Retention", "N/A")

                st.markdown("")

                # SHAP driver charts
                feat_stats = cached_feature_stats()
                top_pos = json.loads(detail.get("top_positive_json", "[]") or "[]")
                top_neg = json.loads(detail.get("top_negative_json", "[]") or "[]")

                st.caption("Each bar shows how much a feature contributes to the model's prediction, as a % of the total signal across all features.")

                # Compute total absolute SHAP across ALL features for unified % scale
                total_abs_shap = sum(
                    abs(float(v)) for k, v in detail.items()
                    if k.startswith("shap__") and k != "shap_base_value_raw" and v is not None
                )
                if total_abs_shap == 0:
                    total_abs_shap = 1  # avoid division by zero

                col_pos, col_neg = st.columns(2)

                def _pct_text(v):
                    if v < 0.5:
                        return "<1%"
                    return f"{v:.0f}%"

                # Find the max % across both sides to set a shared x-axis range
                all_pcts = []

                col_pos, col_neg = st.columns(2)

                with col_pos:
                    st.markdown('<p class="section-label">POSITIVE DRIVERS</p>', unsafe_allow_html=True)
                    if top_pos:
                        pos_df = pd.DataFrame(top_pos[:8])
                        pos_df["pct"] = pos_df["shap_value"].apply(lambda v: abs(v) / total_abs_shap * 100)
                        all_pcts.extend(pos_df["pct"].tolist())
                        pos_df["label"] = pos_df.apply(
                            lambda r: format_driver_label(r["feature"], r["feature_value"], feat_stats), axis=1
                        )
                        pos_df["hover"] = pos_df.apply(
                            lambda r: format_driver_hover(r["feature"], r["feature_value"], r["shap_value"], feat_stats), axis=1
                        )

                with col_neg:
                    st.markdown('<p class="section-label">NEGATIVE DRIVERS</p>', unsafe_allow_html=True)
                    if top_neg:
                        neg_df = pd.DataFrame(top_neg[:8])
                        neg_df["pct"] = neg_df["shap_value"].apply(lambda v: abs(v) / total_abs_shap * 100)
                        all_pcts.extend(neg_df["pct"].tolist())
                        neg_df["label"] = neg_df.apply(
                            lambda r: format_driver_label(r["feature"], r["feature_value"], feat_stats), axis=1
                        )
                        neg_df["hover"] = neg_df.apply(
                            lambda r: format_driver_hover(r["feature"], r["feature_value"], r["shap_value"], feat_stats), axis=1
                        )

                x_max = max(all_pcts) * 1.25 if all_pcts else 10  # 25% padding for text

                with col_pos:
                    if top_pos:
                        fig_pos = go.Figure(go.Bar(
                            x=pos_df["pct"],
                            y=pos_df["label"],
                            orientation="h",
                            marker_color="#1DB954",
                            text=pos_df["pct"].apply(_pct_text),
                            textposition="outside",
                            textfont=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
                            hovertext=pos_df["hover"],
                            hoverinfo="text",
                        ))
                        fig_pos.update_layout(
                            **PLOTLY_LAYOUT,
                            height=300,
                            margin=dict(l=10, r=10, t=10, b=10),
                            yaxis=dict(autorange="reversed", automargin=True, tickfont=dict(family="IBM Plex Mono", size=11, color="#c9d1d9")),
                            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, x_max]),
                        )
                        st.plotly_chart(fig_pos, use_container_width=True)

                with col_neg:
                    if top_neg:
                        fig_neg = go.Figure(go.Bar(
                            x=neg_df["pct"],
                            y=neg_df["label"],
                            orientation="h",
                            marker_color="#f85149",
                            text=neg_df["pct"].apply(_pct_text),
                            textposition="outside",
                            textfont=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
                            hovertext=neg_df["hover"],
                            hoverinfo="text",
                        ))
                        fig_neg.update_layout(
                            **PLOTLY_LAYOUT,
                            height=300,
                            margin=dict(l=10, r=10, t=10, b=10),
                            yaxis=dict(autorange="reversed", automargin=True, tickfont=dict(family="IBM Plex Mono", size=11, color="#c9d1d9")),
                            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, x_max]),
                        )
                        st.plotly_chart(fig_neg, use_container_width=True)

                # Claude AI explanation
                st.markdown('<p class="ai-label">AI EXPLANATION — CLAUDE</p>', unsafe_allow_html=True)
                with st.spinner("Generating explanation..."):
                    explanation = cached_explanation(
                        str(selected_id),
                        json.dumps(detail, default=str),
                        json.dumps(feat_stats, default=str),
                    )
                st.markdown(
                    f'<div class="ai-explanation">{explanation}</div>',
                    unsafe_allow_html=True,
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2: GLOBAL IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════
with tab_global:
    st.markdown('<p class="section-label">FEATURE IMPORTANCE</p>', unsafe_allow_html=True)
    st.markdown("### Global Feature Importance")
    st.caption("How much each feature influences predictions on average, as a % of the total signal across all features and playlists.")

    importance = get_global_shap_importance(top_n=20)
    total_importance = importance["mean_abs_shap"].sum()
    if total_importance == 0:
        total_importance = 1
    importance["pct"] = importance["mean_abs_shap"] / total_importance * 100

    fig_global = go.Figure(go.Bar(
        x=importance["pct"],
        y=importance["feature"].str.replace("_", " "),
        orientation="h",
        marker=dict(
            color=importance["pct"],
            colorscale=[[0, "#282828"], [0.3, "#1a5c34"], [1, "#1DB954"]],
        ),
        text=importance["pct"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11, color="#8b949e"),
    ))
    fig_global.update_layout(
        **PLOTLY_LAYOUT,
        height=550,
        margin=dict(l=10, r=70, t=10, b=10),
        yaxis=dict(autorange="reversed", tickfont=dict(family="IBM Plex Mono", size=11, color="#c9d1d9")),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
    )
    st.plotly_chart(fig_global, use_container_width=True)
