from math import ceil
from typing import Mapping

EXPLORER_PAGE_SIZE = 25


def get_dashboard_tabs() -> tuple[str, ...]:
    return ("EXPLORER", "GLOBAL IMPORTANCE")


def get_total_pages(total_count: int, page_size: int = EXPLORER_PAGE_SIZE) -> int:
    if total_count <= 0:
        return 1
    return ceil(total_count / page_size)


def clamp_page(page: int, total_count: int, page_size: int = EXPLORER_PAGE_SIZE) -> int:
    total_pages = get_total_pages(total_count=total_count, page_size=page_size)
    return min(max(int(page), 1), total_pages)


def get_page_offset(page: int, total_count: int, page_size: int = EXPLORER_PAGE_SIZE) -> int:
    current_page = clamp_page(page=page, total_count=total_count, page_size=page_size)
    return (current_page - 1) * page_size


def format_playlist_option(row: Mapping[str, object]) -> str:
    row_id = str(row["row_id"])
    playlist_uri = str(row["playlist_uri"])
    pred_pct = float(row["pred_proba"]) * 100
    pred_label = "Success" if int(row["pred_label"]) == 1 else "Fail"
    owner_type = str(row["owner_type"])
    mau = f"{int(row['mau']):,}"
    uri_tail = playlist_uri.split(":")[-1]
    return f"{row_id} | {pred_pct:.1f}% {pred_label} | {owner_type} | MAU {mau} | {uri_tail}"
