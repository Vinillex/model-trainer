"""Image scraping: SerpAPI Google Images search -> download -> validate/dedupe -> store as `scraped` images."""

import hashlib
import ipaddress
import socket
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse, urlsplit, urlunsplit

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import imagehash
from .api.images import NearDuplicate, ingest_image
from .config import settings
from .grouping import file_series
from .models import DetClass, Image, ImageHash, Job, ScrapedUrl, ScrapeRecord, SearchLog, Source, UrlOutcome
from .storage import get_storage

USER_AGENT = "Mozilla/5.0 (compatible; model-trainer-dataset-builder)"
MAX_REDIRECTS = 3


class SearchError(Exception):
    """The search provider refused or failed. Messages never contain the API key."""


class FetchError(Exception):
    """An image URL could not be downloaded (blocked, HTTP error, too large, not an image...)."""


class SearchDone(Exception):
    """Raised by a `search` function that has nothing more to offer. An empty list only means "this page had nothing",
    which ends a SerpAPI search (it ran past the last results) but not a crawl of web pages."""


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
        # SerpAPI can take over a minute on a slow search (one real search took 82s) and still charges for it,
        # so a short client timeout turns a paid search into a failed job.
        self.http = http or httpx.Client(timeout=httpx.Timeout(180, connect=10))

    def search(self, query: str, page: int) -> list[SearchResult]:
        params = {"engine": "google_images", "q": query, "ijn": page, "api_key": self.api_key}
        try:
            resp = self.http.get(self.url, params=params)
        except httpx.TimeoutException:
            raise SearchError("SerpAPI timed out. The search may still have been counted; repeating the identical "
                              "search within an hour is served from SerpAPI's cache at no cost") from None
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


ACCOUNT_FIELDS = ("plan_name", "searches_per_month", "this_month_usage", "plan_searches_left",
                  "total_searches_left", "account_rate_limit_per_hour")


def fetch_account(api_key: str, http: httpx.Client | None = None, url: str | None = None) -> dict:
    """Plan and quota from SerpAPI's Account API. It's free and doesn't count as a search.

    Only whitelisted plan fields are returned (the raw response also holds the key and account email).
    """
    http = http or httpx.Client(timeout=20)
    try:
        resp = http.get(url or settings.serpapi_account_url, params={"api_key": api_key})
    except httpx.HTTPError as e:
        raise SearchError(f"SerpAPI account request failed ({type(e).__name__})") from None
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if resp.status_code != 200 or not isinstance(body, dict) or body.get("error"):
        detail = body.get("error") if isinstance(body, dict) else None
        raise SearchError(f"SerpAPI account HTTP {resp.status_code}: {detail or 'no details'}")
    return {k: body.get(k) for k in ACCOUNT_FIELDS}


def url_key(url: str) -> str:
    """Stable key for 'have we seen this image URL': scheme/host lower-cased, fragment dropped."""
    parts = urlsplit(url.strip())
    return hashlib.sha256(urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, "")).encode()).hexdigest()


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


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


def fetch_image(url: str, http: httpx.Client, max_bytes: int | None = None, user_agent: str | None = None) -> bytes:
    """Download one image, following up to MAX_REDIRECTS redirects (each hop is re-checked) with a size cap."""
    max_bytes = max_bytes or settings.scrape_max_bytes
    for _ in range(MAX_REDIRECTS + 1):
        _check_public(url)
        try:
            with http.stream("GET", url, headers={"User-Agent": user_agent or USER_AGENT}) as r:
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


def _known_url_keys(db: Session, project_id: int) -> set[str]:
    """URLs not worth downloading again: everything tried before except transient download failures."""
    keys = set(db.scalars(select(ScrapedUrl.url_sha).where(ScrapedUrl.project_id == project_id,
                                                          ScrapedUrl.outcome != UrlOutcome.download_failed)))
    # Images scraped before the ledger existed only have a provenance record.
    keys.update(url_key(u) for u in db.scalars(select(ScrapeRecord.original_url).join(Image).where(Image.project_id == project_id)))
    return keys


def group_key_for(page_url: str | None, group_by: str) -> str | None:
    """What decides which train/val/test split an image lands in: its site (default) or its article.

    The article key is the host plus a hash of the page URL, so it always fits the 200-character column and two different
    articles with a long common prefix never share a key.
    """
    parts = urlsplit(page_url or "")
    if group_by == "article" and parts.hostname and page_url:
        series = file_series(parts.path)
        basis = f"{parts.hostname}/{series}" if series else urlunsplit(parts._replace(fragment=""))
        return f"{parts.hostname}/{hashlib.sha1(basis.encode()).hexdigest()[:16]}"
    return parts.hostname


def run_scrape(db: Session, job: Job, search: Callable[[str, int], list[SearchResult]],
               fetch: Callable[[str], bytes], *, log_searches: bool = True,
               extra_progress: Callable[[], dict] | None = None) -> None:
    """Execute a scrape job. Progress counters are written to job.progress after every image, so the UI can poll.

    `log_searches` is for SerpAPI jobs: each page is a search that was spent. Page crawls (app/collect.py) switch it off and
    use `SearchDone` to say they are finished; `extra_progress` adds their own numbers (pages read, failures) to the progress.
    """
    params = job.params
    group_by = params.get("group_by", "site")
    cls = db.get(DetClass, params["class_id"])
    query = normalize_query(params["query"])
    max_images, max_pages = params["max_images"], params["max_pages"]
    start_page = params.get("start_page", 0)
    min_side, near_dist = params["min_side"], params["near_dup_distance"]
    counters = dict.fromkeys(COUNTERS, 0)

    known = _known_url_keys(db, job.project_id) if params.get("skip_known_urls", True) else set()
    hashes = _load_hashes(db, job.project_id)
    near = (lambda h: any(imagehash.distance(h, x) <= near_dist for x in hashes)) if near_dist > 0 else None

    def publish():
        job.progress = {**counters, "target": max_images, "start_page": start_page, **(extra_progress() if extra_progress else {})}
        db.commit()

    def cancelled() -> bool:
        db.refresh(job)
        return job.cancel_requested

    def record(r: SearchResult, outcome: UrlOutcome) -> None:
        key = url_key(r.original)
        row = db.scalar(select(ScrapedUrl).where(ScrapedUrl.project_id == job.project_id, ScrapedUrl.url_sha == key))
        if row is None:
            db.add(ScrapedUrl(project_id=job.project_id, url_sha=key, url=r.original, outcome=outcome, job_id=job.id))
        else:  # a retried failure
            row.outcome, row.job_id = outcome, job.id

    publish()
    page = start_page
    while counters["added"] < max_images and page < start_page + max_pages and not cancelled():
        try:
            results = search(params["query"], page)
        except SearchDone:
            break
        if log_searches:
            db.add(SearchLog(project_id=job.project_id, query=query[:300], page=page, results=len(results), job_id=job.id))
        page += 1
        counters["pages"] += 1
        if not results:
            if log_searches:
                break  # ran out of results
            publish()
            continue  # a page without usable pictures; the next one may have some
        counters["candidates"] += len(results)

        todo = []
        for r in results:
            key = url_key(r.original)
            if key in known:
                counters["already_scraped"] += 1
            elif r.width and r.height and min(r.width, r.height) < min_side:
                counters["too_small"] += 1  # cheap pre-filter from the search metadata: skip the download
            else:
                todo.append(r)
                known.add(key)  # also catches the same URL appearing twice in one job
        publish()

        chunk_size = settings.scrape_download_workers * 2
        pool = ThreadPoolExecutor(settings.scrape_download_workers)
        try:
            for i in range(0, len(todo), chunk_size):
                if counters["added"] >= max_images or cancelled():
                    break
                chunk = todo[i:i + chunk_size]
                futures = [pool.submit(fetch, r.original) for r in chunk]  # download in parallel, ingest in order
                for r, future in zip(chunk, futures):
                    if counters["added"] >= max_images or cancelled():  # checked per image so Cancel takes effect promptly
                        break
                    try:
                        data = future.result()
                    except FetchError:
                        counters["download_failed"] += 1
                        record(r, UrlOutcome.download_failed)
                        publish()
                        continue
                    try:
                        image = ingest_image(db, job.project_id, cls, Source.scraped, data, group_key=group_key_for(r.page_url, group_by),
                                             min_side=min_side, is_near_duplicate=near)
                    except NearDuplicate:
                        counters["near_duplicates"] += 1
                        record(r, UrlOutcome.near_duplicate)
                    except ValueError:
                        counters["rejected"] += 1
                        record(r, UrlOutcome.rejected)
                    else:
                        if image is None:
                            counters["duplicates"] += 1
                            record(r, UrlOutcome.duplicate)
                        else:
                            image.scrape_record = ScrapeRecord(job_id=job.id, query=params["query"][:300],
                                                               original_url=r.original[:2048], title=(r.title or None) and r.title[:500],
                                                               page_url=(r.page_url or None) and r.page_url[:2048])
                            hashes.append(int(image.hash.dhash, 16))
                            counters["added"] += 1
                            record(r, UrlOutcome.added)
                    publish()
        finally:
            # Don't wait for downloads we no longer want (stop, cancel, or reached max_images).
            pool.shutdown(wait=False, cancel_futures=True)
    publish()  # flush counters changed after the last in-loop publish (e.g. the empty page that ended the search)


def handle_scrape(db: Session, job: Job) -> None:
    if not settings.serpapi_key:
        raise RuntimeError("MT_SERPAPI_KEY is not set")
    client = SerpApiClient(settings.serpapi_key)
    with httpx.Client(follow_redirects=False, timeout=httpx.Timeout(10, connect=5)) as http:
        run_scrape(db, job, client.search, lambda url: fetch_image(url, http))
