import json
import os
import anthropic

try:
    import streamlit as st
except ImportError:  # pragma: no cover - streamlit is installed for the app runtime
    st = None

SYSTEM_PROMPT = (
    "You are a music analytics expert explaining ML model predictions to a "
    "non-technical stakeholder at a music streaming company. Write in a warm, "
    "insightful tone — like a senior analyst briefing a product manager."
)

FEATURE_GLOSSARY = """## Feature glossary
Below are the features that appear in the SHAP drivers. Use this to interpret them correctly.

**Success definition:** A playlist is "successful" if it has BOTH high engagement (monthly streams per user >= median for its MAU group) AND high retention (returning users / previous month users >= median for its MAU group).

**Playlist composition features:**
- n_tracks: Number of tracks in the playlist
- n_artists: Number of unique artists in the playlist
- n_albums: Number of unique albums in the playlist
- n_local_tracks: Change in number of tracks since yesterday (positive = tracks were added)
- pct_local_tracks: Ratio of n_local_tracks to n_tracks (how much the playlist changed recently)
- track_per_album: n_tracks / n_albums (higher = more tracks per album, i.e. deeper album picks)
- track_per_artist: n_tracks / n_artists (higher = more tracks per artist, i.e. more focused on fewer artists)
- album_per_artist: n_albums / n_artists
- is_positive_local_tracks: Binary flag, 1 if tracks were added recently (n_local_tracks > 0)

**Title token features (tok_*):**
Binary flags (0 or 1) indicating whether a specific word appears in the playlist title.
- tok_new, tok_study, tok_chill, tok_worship, tok_christian, tok_workout, tok_party, tok_dance, tok_love, tok_fall, tok_summer, tok_wedding
- Example: tok_fall = 1 means the word "fall" appears in the playlist title (likely a seasonal/autumn playlist)
- token_count: Total number of meaningful words in the playlist title

**Purpose features (purpose_*):**
Binary flags indicating the playlist's apparent purpose, derived from title tokens:
- purpose_workout: title contains any of {workout, gym, running, cardio, fitness, exercise}
- purpose_worship: title contains any of {worship, gospel, church, praise, christian}
- purpose_party: title contains any of {party, dance, club, edm}
- purpose_study_focus: title contains any of {study, focus, coding, work}
- purpose_sleep_relax: title contains any of {sleep, relax, relaxing, calm, ambient}
- purpose_discovery_fresh: title contains any of {new, discover, discovery, latest, fresh, release, releases}

**Genre tag features (genre_tag__*):**
Binary flags (0 or 1) for whether a genre appears in the playlist's top 3 genres (from Gracenote metadata).
- Examples: genre_tag__indie_rock, genre_tag__latin, genre_tag__pop, genre_tag__rock, genre_tag__alternative, genre_tag__country_folk, genre_tag__rap

**Mood tag features (mood_tag__*):**
Binary flags (0 or 1) for whether a mood appears in the playlist's top 3 moods (from Gracenote metadata).
- Examples: mood_tag__yearning, mood_tag__cool, mood_tag__defiant, mood_tag__excited, mood_tag__energizing, mood_tag__brooding, mood_tag__upbeat, mood_tag__lively, mood_tag__aggressive
"""

_BINARY_PREFIXES = (
    "tok_", "genre_tag__", "mood_tag__", "purpose_",
    "is_positive_local_tracks", "owner_type",
)

_client: anthropic.Anthropic | None = None


def _get_api_key() -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key

    if st is None:
        return None

    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None


def _get_client() -> anthropic.Anthropic | None:
    global _client
    api_key = _get_api_key()
    if not api_key:
        return None
    if _client is None:
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _parse_json_safe(raw: str | None) -> list[dict]:
    if not raw:
        return []
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []


def _quartile_label(feature: str, value, feat_stats: dict) -> str:
    """Return 'low', 'typical', or 'high' relative to the population, or '' for binary features."""
    is_binary = any(feature.startswith(p) for p in _BINARY_PREFIXES)
    if is_binary:
        return ""
    stats = feat_stats.get(feature, {})
    p25 = stats.get("p25")
    p75 = stats.get("p75")
    if p25 is None or p75 is None:
        return ""
    try:
        v = float(value)
    except (ValueError, TypeError):
        return ""
    if v <= p25:
        return "low"
    elif v >= p75:
        return "high"
    return "typical"


def build_explanation_prompt(playlist: dict, feat_stats: dict | None = None) -> str:
    top_pos = _parse_json_safe(playlist.get("top_positive_json"))
    top_neg = _parse_json_safe(playlist.get("top_negative_json"))

    pred_proba = playlist["pred_proba"]
    pred_label = playlist["pred_label"]
    owner_type = playlist.get("owner_type", "unknown")
    mau = playlist.get("mau", "N/A")
    mau_group = playlist.get("mau_group", "N/A")

    def format_drivers(drivers: list[dict]) -> str:
        lines = []
        for d in drivers[:5]:
            feat = d["feature"]
            val = d["feature_value"]
            shap = d["shap_value"]
            direction = "+" if shap > 0 else ""

            # Add quartile context for numeric features
            q_label = _quartile_label(feat, val, feat_stats or {})
            if q_label:
                val_display = f"{val} ({q_label} relative to population)"
            else:
                val_display = str(val)

            lines.append(
                f"  - {feat}: value={val_display}, SHAP impact={direction}{shap:.4f}"
            )
        return "\n".join(lines) if lines else "  (none)"

    prediction_str = "likely to succeed" if pred_label == 1 else "unlikely to succeed"

    return f"""A playlist was analyzed by a machine learning model that predicts whether a playlist will be "successful" (high engagement AND high retention).

{FEATURE_GLOSSARY}

## How to interpret the drivers below
- Each driver has a feature name, its value for this playlist, a population context label (low / typical / high), and a SHAP impact score.
- **"low / typical / high" tells you where this playlist's value sits relative to the full population** (quartiles). Trust these labels — do NOT judge whether a value is large or small based on the raw number alone.
- **SHAP impact tells you the direction**: positive means the feature pushes TOWARD success, negative means it pushes AGAINST success.
- For binary features (Yes/No), there is no quartile label — just whether the feature is present or absent.
- **SHAP values are marginal contributions, not standalone verdicts.** No single feature determines success or failure — the prediction comes from the combination of all features together. A feature with a negative SHAP value doesn't mean "this trait = failure"; it means "in the context of this playlist's other traits, this one nudges the score slightly lower."
- **Do NOT overstate any single feature.** Avoid language like "this single factor accounts for half the failure" or "this alone causes the prediction." Every feature works in concert with the others.
- **Do NOT invent causal explanations.** Describe the model's learned patterns, not speculative reasons. Say "the model associates X with Y" rather than claiming why.

**Playlist info:**
- Owner type: {owner_type}
- Monthly active users (MAU): {mau} ({mau_group})
- Model prediction: {pred_proba:.1%} probability of success → **{prediction_str}**

**Top signals nudging TOWARD success:**
{format_drivers(top_pos)}

**Top signals nudging AGAINST success:**
{format_drivers(top_neg)}

Write a 2-3 sentence explanation of why the model thinks this playlist is {prediction_str}. Focus on the **overall pattern** — how the combination of features shapes the prediction, not a feature-by-feature breakdown. Reference the population context labels (low/typical/high) for numeric features. Keep the tone balanced: no single feature is decisive, the prediction emerges from the mix."""


def get_explanation(playlist: dict, feat_stats: dict | None = None) -> str:
    client = _get_client()
    if client is None:
        return "_Set ANTHROPIC_API_KEY in your .env file or Streamlit secrets to enable AI explanations._"

    prompt = build_explanation_prompt(playlist, feat_stats)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        if not response.content:
            return "_No explanation returned._"
        return response.content[0].text
    except anthropic.APIError as e:
        return f"_Explanation unavailable: {e}_"
