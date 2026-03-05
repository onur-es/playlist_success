#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

logging.getLogger("spotipy").setLevel(logging.ERROR)


# CLIENT_ID = 
# CLIENT_SECRET =


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spotify API playlist enrichment")
    parser.add_argument("--data-path", default="playlist_revision_v05.txt")
    parser.add_argument("--output-path", default="data/playlist_sample_enriched.parquet")
    parser.add_argument("--log-path", default="data/spotify_enrichment_run_log.parquet")
    parser.add_argument("--n-playlists", type=int, default=240)
    parser.add_argument("--max-tracks", type=int, default=100)
    parser.add_argument("--pause-sec", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--force", action="store_true", help="Overwrite output if it already exists")
    return parser.parse_args()


def playlist_uri_to_id(playlist_uri: str) -> str:
    if pd.isna(playlist_uri):
        return ""
    uri = str(playlist_uri).strip()
    if ":playlist:" in uri:
        return uri.split(":playlist:")[-1].split("?")[0]
    if uri.startswith("spotify:playlist:"):
        return uri.split(":")[-1]
    if "open.spotify.com/playlist/" in uri:
        return uri.split("playlist/")[-1].split("?")[0]
    return uri


def safe_call(callable_obj, *args, max_retries: int = 5, **kwargs):
    for attempt in range(max_retries):
        try:
            return callable_obj(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            status = getattr(e, "http_status", None)
            if status == 429:
                retry_after = 1
                headers = getattr(e, "headers", {}) or {}
                retry_after = int(headers.get("Retry-After", 1))
                time.sleep(retry_after + 0.2)
                continue
            # Fail fast for non-retryable errors.
            if status in {400, 401, 403, 404}:
                return None
            if attempt == max_retries - 1:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def release_year_from_str(s):
    if not s or pd.isna(s):
        return np.nan
    x = str(s)
    if len(x) >= 4 and x[:4].isdigit():
        return int(x[:4])
    return np.nan


def parse_token_list(v):
    if pd.isna(v):
        return []
    try:
        parsed = json.loads(v)
        if isinstance(parsed, list):
            return sorted(set([str(x).strip().lower() for x in parsed if str(x).strip()]))
    except Exception:  # noqa: BLE001
        return []
    return []


def combine_cols_to_unique_tags(row, cols):
    tags = []
    for c in cols:
        v = row[c]
        if pd.notna(v):
            x = str(v).strip().lower()
            if x and x != "nan":
                tags.append(x)
    return sorted(set(tags))


def fetch_playlist_enrichment(sp, playlist_uri: str, max_tracks: int = 100):
    playlist_id = playlist_uri_to_id(playlist_uri)
    if not playlist_id:
        return {"playlist_uri": playlist_uri, "api_ok": False, "api_error": "missing_playlist_id"}

    pl = safe_call(
        sp.playlist,
        playlist_id,
        fields="id,name,description,public,collaborative,followers.total,owner.id,owner.display_name,owner.type,tracks.total,snapshot_id",
    )
    if pl is None:
        return {"playlist_uri": playlist_uri, "api_ok": False, "api_error": "playlist_fetch_failed"}

    popularity_vals = []
    duration_vals = []
    explicit_vals = []
    release_year_vals = []
    artist_ids = set()
    added_times = []

    fetched = 0
    limit = min(100, max_tracks)
    offset = 0

    while fetched < max_tracks:
        items_resp = safe_call(
            sp.playlist_items,
            playlist_id,
            limit=limit,
            offset=offset,
            fields="items(added_at,track(id,popularity,duration_ms,explicit,album(release_date),artists(id,name))),next,total",
        )
        if items_resp is None:
            break

        items = items_resp.get("items", [])
        if not items:
            break

        for item in items:
            track = item.get("track")
            if track is None:
                continue

            pop = track.get("popularity")
            if pop is not None:
                popularity_vals.append(pop)

            dur = track.get("duration_ms")
            if dur is not None:
                duration_vals.append(dur)

            exp = track.get("explicit")
            if exp is not None:
                explicit_vals.append(int(bool(exp)))

            rel = release_year_from_str(track.get("album", {}).get("release_date"))
            if not pd.isna(rel):
                release_year_vals.append(rel)

            for a in track.get("artists", []):
                aid = a.get("id")
                if aid:
                    artist_ids.add(aid)

            added_at = item.get("added_at")
            if added_at:
                added_times.append(pd.to_datetime(added_at, errors="coerce"))

        fetched += len(items)
        if items_resp.get("next") is None:
            break
        offset += limit
        if fetched >= max_tracks:
            break

    added_times = [x for x in added_times if not pd.isna(x)]
    first_added = min(added_times) if len(added_times) else pd.NaT
    last_added = max(added_times) if len(added_times) else pd.NaT
    span_days = (last_added - first_added).days if len(added_times) >= 2 else np.nan

    return {
        "playlist_uri": playlist_uri,
        "api_ok": True,
        "api_error": None,
        "api_playlist_id": pl.get("id"),
        "api_playlist_name": pl.get("name"),
        "api_playlist_public": pl.get("public"),
        "api_collaborative": pl.get("collaborative"),
        "api_followers_total": pl.get("followers", {}).get("total"),
        "api_owner_id": pl.get("owner", {}).get("id"),
        "api_owner_display_name": pl.get("owner", {}).get("display_name"),
        "api_owner_type": pl.get("owner", {}).get("type"),
        "api_tracks_total": pl.get("tracks", {}).get("total"),
        "api_description_len": len(pl.get("description") or ""),
        "api_snapshot_id_len": len(pl.get("snapshot_id") or ""),
        "api_track_rows_sampled": fetched,
        "api_track_popularity_mean": np.mean(popularity_vals) if len(popularity_vals) else np.nan,
        "api_track_popularity_median": np.median(popularity_vals) if len(popularity_vals) else np.nan,
        "api_track_duration_min_mean": (np.mean(duration_vals) / 60000) if len(duration_vals) else np.nan,
        "api_track_explicit_rate": np.mean(explicit_vals) if len(explicit_vals) else np.nan,
        "api_artist_unique_count": len(artist_ids),
        "api_artist_diversity": (len(artist_ids) / fetched) if fetched > 0 else np.nan,
        "api_release_year_median": np.median(release_year_vals) if len(release_year_vals) else np.nan,
        "api_first_added_at": first_added,
        "api_last_added_at": last_added,
        "api_added_span_days": span_days,
    }


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    log_path = Path(args.log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.force:
        print(f"[skip] Output already exists: {output_path}")
        print("[skip] Use --force to rerun enrichment.")
        return

    print(f"[start] Loading data: {args.data_path}")
    df = pd.read_csv(args.data_path, sep="\t")
    print(f"[info] Raw shape: {df.shape}")

    # Minimal derived columns for sampling and downstream analysis merge.
    df["owner_type"] = np.where(df["owner"] == "spotify", "spotify", "user")
    df["target_success_engagement"] = np.where(df["mau"] > 0, df["monthly_stream30s"] / df["mau"], np.nan)
    df["target_success_retention"] = np.where(
        df["mau_previous_month"] > 0,
        df["mau_both_months"] / df["mau_previous_month"],
        np.nan,
    )
    df["target_success_reach"] = df["mau"]
    df["genre_tags"] = df.apply(lambda r: combine_cols_to_unique_tags(r, ["genre_1", "genre_2", "genre_3"]), axis=1)
    df["mood_tags"] = df.apply(lambda r: combine_cols_to_unique_tags(r, ["mood_1", "mood_2", "mood_3"]), axis=1)
    df["token_tags"] = df["tokens"].apply(parse_token_list)

    sample_cols = [
        "playlist_uri",
        "owner_type",
        "genre_tags",
        "mood_tags",
        "token_tags",
        "n_tracks",
        "target_success_engagement",
        "target_success_retention",
        "target_success_reach",
    ]
    base = df[sample_cols].drop_duplicates("playlist_uri")

    n_each = args.n_playlists // 2
    sample_spotify = base[base["owner_type"] == "spotify"].sample(
        min(n_each, (base["owner_type"] == "spotify").sum()),
        random_state=args.random_state,
    )
    sample_user = base[base["owner_type"] == "user"].sample(
        min(n_each, (base["owner_type"] == "user").sum()),
        random_state=args.random_state,
    )
    sample = pd.concat([sample_spotify, sample_user], ignore_index=True)
    remaining_n = args.n_playlists - len(sample)
    if remaining_n > 0:
        remaining_pool = base[~base["playlist_uri"].isin(sample["playlist_uri"])]
        extra = remaining_pool.sample(min(remaining_n, len(remaining_pool)), random_state=args.random_state)
        sample = pd.concat([sample, extra], ignore_index=True)

    print(f"[info] Sample size: {len(sample)}")
    print(sample["owner_type"].value_counts())

    print("[start] Initializing Spotify client")
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
        )
    )

    total_n = len(sample)
    start = time.time()
    enrichment_records = []
    run_log_records = []

    for idx, playlist_uri in enumerate(sample["playlist_uri"], start=1):
        t0 = time.time()
        rec = fetch_playlist_enrichment(sp, playlist_uri, max_tracks=args.max_tracks)
        item_elapsed = time.time() - t0
        elapsed = time.time() - start
        avg = elapsed / idx
        eta = (total_n - idx) * avg

        enrichment_records.append(rec)
        run_log_records.append(
            {
                "idx": idx,
                "playlist_uri": playlist_uri,
                "api_ok": bool(rec.get("api_ok", False)),
                "api_error": rec.get("api_error"),
                "item_elapsed_sec": item_elapsed,
                "total_elapsed_min": elapsed / 60,
                "eta_min": eta / 60,
                "timestamp_utc": datetime.now(timezone.utc),
            }
        )

        if (idx % args.log_every == 0) or (idx == 1) or (idx == total_n):
            print(
                f"[{idx:>4}/{total_n}] ok={bool(rec.get('api_ok', False))} "
                f"item={item_elapsed:,.2f}s elapsed={elapsed/60:,.2f}m eta={eta/60:,.2f}m"
            )

        time.sleep(args.pause_sec)

    enrichment_df = pd.DataFrame(enrichment_records)
    run_log_df = pd.DataFrame(run_log_records)
    merged = sample.merge(enrichment_df, on="playlist_uri", how="left")

    print(f"[save] Writing enriched sample: {output_path}")
    merged.to_parquet(output_path, index=False)
    print(f"[save] Writing run log: {log_path}")
    run_log_df.to_parquet(log_path, index=False)

    success_rate = enrichment_df["api_ok"].fillna(False).mean() * 100 if len(enrichment_df) else 0
    print("[done] Enrichment complete")
    print(f"[done] Rows: {len(merged)} | API success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    main()
