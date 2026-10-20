"""Image scraping: SerpAPI Google Images search -> download -> validate/dedupe -> store as `scraped` images."""

import ipaddress
import socket
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import imagehash
from .api.images import NearDuplicate, ingest_image
from .config import settings
from .models import DetClass, Image, ImageHash, Job, ScrapeRecord, Source
from .storage import get_storage

USER_AGENT = "Mozilla/5.0 (compatible; model-trainer-dataset-builder)"
MAX_REDIRECTS = 3


class SearchError(Exception):
    """The search provider refused or failed. Messages never contain the API key."""


class FetchError(Exception):
    """An image URL could not be downloaded (blocked, HTTP error, too large, not an image...)."""


@dataclass
class SearchResult:
    original: str
    page_url: str | None = None
    title: str | None = None
    width: int | None = None
    height: int | None = None


class SerpApiClient:
    """Google Images via SerpAPI. One `search()` call is one search (= one page of results)."""

    def __init__(self, api_key: str, url: str | None = None, http: httpx.Client | None = None):
        self.api_key = api_key
        self.url = url or settings.serpapi_url
        self.http = http or httpx.Client(timeout=30)

    def search(self, query: str, page: int) -> list[SearchResult]:
        params = {"engine": "google_images", "q": query, "ijn": page, "api_key": self.api_key}
        try:
            resp = self.http.get(self.url, params=params)
        except httpx.HTTPError as e:
            # Don't chain or print `e`: some httpx errors embed the full URL, including the key.
            raise SearchError(f"SerpAPI request failed ({type(e).__name__})") from None
        try:
            body = resp.json()
        except ValueError:
            body = {}
        error = body.get("error") if isinstance(body, dict) else None
        if resp.status_code != 200:
            raise SearchError(f"SerpAPI HTTP {resp.status_code}: {error or 'no details'}")
        if error:
            if "hasn't returned any results" in str(error).lower():
                return []  # ran past the last page
            raise SearchError(f"SerpAPI: {error}")

        results = []
        for r in body.get("images_results", []):
            if not r.get("original"):
                continue
            results.append(SearchResult(original=r["original"], page_url=r.get("link"), title=r.get("title"),
                                        width=r.get("original_width"), height=r.get("original_height")))
        return results


def _check_public(url: str) -> None:
    """Reject non-http(s) URLs and hosts that resolve to non-public addresses (SSRF guard).

    Search results are untrusted URLs; without this a result could point the server at localhost or the LAN.
    """
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError("not an http(s) URL")
    if settings.scrape_allow_private_hosts:
        return
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror:
        raise FetchError("host does not resolve") from None
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise FetchError("host resolves to a non-public address")


def fetch_image(url: str, http: httpx.Client, max_bytes: int | None = None) -> bytes:
    """Download one image, following up to MAX_REDIRECTS redirects (each hop is re-checked) with a size cap."""
    max_bytes = max_bytes or settings.scrape_max_bytes
    for _ in range(MAX_REDIRECTS + 1):
        _check_public(url)
        try:
            with http.stream("GET", url, headers={"User-Agent": USER_AGENT}) as r:
                if r.is_redirect:
                    location = r.headers.get("location")
                    if not location:
                        raise FetchError("redirect without location")
                    url = urljoin(url, location)
                    continue
                if r.status_code != 200:
                    raise FetchError(f"HTTP {r.status_code}")
                ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype and not (ctype.startswith("image/") or ctype.endswith("octet-stream")):
                    raise FetchError(f"not an image ({ctype})")
                if int(r.headers.get("content-length") or 0) > max_bytes:
                    raise FetchError("too large")
                buf = bytearray()
                for chunk in r.iter_bytes():
                    buf += chunk
                    if len(buf) > max_bytes:
                        raise FetchError("too large")
                return bytes(buf)
        except httpx.HTTPError as e:
            raise FetchError(type(e).__name__) from None
    raise FetchError("too many redirects")


COUNTERS = ("pages", "candidates", "added", "duplicates", "near_duplicates", "already_scraped",
            "too_small", "download_failed", "rejected")


def _load_hashes(db: Session, project_id: int) -> list[int]:
    """All dHashes in the project, computing any that are missing (images stored before hashing existed)."""
    for image in db.scalars(select(Image).outerjoin(ImageHash).where(Image.project_id == project_id, ImageHash.image_id.is_(None))):
        image.hash = ImageHash(dhash=imagehash.to_hex(imagehash.dhash(get_storage().get(image.storage_key))))
    db.flush()
    rows = db.scalars(select(ImageHash.dhash).join(Image).where(Image.project_id == project_id))
    return [int(h, 16) for h in rows]


def run_scrape(db: Session, job: Job, search: Callable[[str, int], list[SearchResult]],
               fetch: Callable[[str], bytes]) -> None:
    """Execute a scrape job. Progress counters are written to job.progress after every image, so the UI can poll."""
    params = job.params
    cls = db.get(DetClass, params["class_id"])
    max_images, max_pages = params["max_images"], params["max_pages"]
    min_side, near_dist = params["min_side"], params["near_dup_distance"]
    counters = dict.fromkeys(COUNTERS, 0)

    seen_urls = set(db.scalars(select(ScrapeRecord.original_url).join(Image).where(Image.project_id == job.project_id)))
    hashes = _load_hashes(db, job.project_id)
    near = (lambda h: any(imagehash.distance(h, x) <= near_dist for x in hashes)) if near_dist > 0 else None

    def publish():
        job.progress = {**counters, "target": max_images}
        db.commit()

    def cancelled() -> bool:
        db.refresh(job)
        return job.cancel_requested

    publish()
    page = 0
    while counters["added"] < max_images and page < max_pages and not cancelled():
        results = search(params["query"], page)
        page += 1
        counters["pages"] = page
        if not results:
            break  # ran out of results
        counters["candidates"] += len(results)

        todo = []
        for r in results:
            if r.original in seen_urls:
                counters["already_scraped"] += 1
            elif r.width and r.height and min(r.width, r.height) < min_side:
                counters["too_small"] += 1  # cheap pre-filter from the search metadata: skip the download
            else:
                todo.append(r)
                seen_urls.add(r.original)
        publish()

        chunk_size = settings.scrape_download_workers * 2
        with ThreadPoolExecutor(settings.scrape_download_workers) as pool:
            for i in range(0, len(todo), chunk_size):
                if counters["added"] >= max_images or cancelled():
                    break
                chunk = todo[i:i + chunk_size]
                futures = [pool.submit(fetch, r.original) for r in chunk]  # download in parallel, ingest in order
                for r, future in zip(chunk, futures):
                    if counters["added"] >= max_images:
                        break
                    try:
                        data = future.result()
                    except FetchError:
                        counters["download_failed"] += 1
                        publish()
                        continue
                    host = urlparse(r.page_url or "").hostname
                    try:
                        image = ingest_image(db, job.project_id, cls, Source.scraped, data, group_key=host,
                                             min_side=min_side, is_near_duplicate=near)
                    except NearDuplicate:
                        counters["near_duplicates"] += 1
                    except ValueError:
                        counters["rejected"] += 1
                    else:
                        if image is None:
                            counters["duplicates"] += 1
                        else:
                            image.scrape_record = ScrapeRecord(job_id=job.id, query=params["query"][:300],
                                                               original_url=r.original[:2048], title=(r.title or None) and r.title[:500],
                                                               page_url=(r.page_url or None) and r.page_url[:2048])
                            hashes.append(int(image.hash.dhash, 16))
                            counters["added"] += 1
                    publish()
    publish()  # flush counters changed after the last in-loop publish (e.g. the empty page that ended the search)


def handle_scrape(db: Session, job: Job) -> None:
    if not settings.serpapi_key:
        raise RuntimeError("MT_SERPAPI_KEY is not set")
    client = SerpApiClient(settings.serpapi_key)
    with httpx.Client(follow_redirects=False, timeout=httpx.Timeout(10, connect=5)) as http:
        run_scrape(db, job, client.search, lambda url: fetch_image(url, http))
