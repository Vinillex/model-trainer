"""Which pictures must stay together in one train/val/test split, beyond "same page".

Shared by the scrapers (the `group_key` stored on a new picture) and by dataset versions (the group a split keeps together), so
the two cannot drift apart.
"""

import re
from urllib.parse import unquote, urlsplit


def file_series(path: str) -> str | None:
    """The photo series a Wikimedia `File:` page belongs to, or None for any other page.

    Commons files come in runs ("MQ-9 Takes Flight Over Central New York 161019-Z-QU230-002.jpg", "... -005.jpg"): the frames of
    one sortie are near-identical, so they must share a split. Trailing words with a digit in them (serial numbers, "(9267734)",
    "2") are dropped, which leaves the title of the series.
    """
    m = re.search(r"/File:([^/]+)$", unquote(path), re.I)
    if not m:
        return None
    words = re.sub(r"\.\w{2,5}$", "", m.group(1)).replace("_", " ").split()
    while len(words) > 1 and re.search(r"\d", words[-1]):
        words.pop()
    return " ".join(words).lower()


def series_group(page_url: str | None) -> str | None:
    """`host/series` for a Wikimedia `File:` page, None for every other address (those keep their own page as the group)."""
    parts = urlsplit(page_url or "")
    series = file_series(parts.path)
    return f"{parts.hostname}/{series}" if series and parts.hostname else None
