"""Collect pictures from web pages: article / gallery URLs -> the pictures on them -> the scrape pipeline.

The scrape runner (`scrape.run_scrape`) already knows how to download, size-check, de-duplicate and store candidate images and
how to keep their provenance. This module is its other front end: instead of SerpAPI results the candidates come from reading
pages. It is polite by construction: robots.txt is obeyed (pages and pictures), every host gets at most one request per
`collect_request_delay` seconds, a job reads at most `max_pages` pages, and every request goes through the same SSRF guard as
the image downloads. No HTML library: the stdlib parser is enough for finding `<img>` tags.
"""

import io
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
from PIL import Image as PILImage
from PIL import ImageOps
from sqlalchemy.orm import Session

from .config import settings
from .models import Job
from .scrape import MAX_REDIRECTS, FetchError, SearchDone, SearchResult, _check_public, fetch_image, run_scrape

ROBOTS_AGENT = "model-trainer-dataset-builder"  # the product token robots.txt rules are matched against
# Says who is asking, with a link, instead of pretending to be a browser. Wikimedia answers a "Mozilla/5.0 (compatible; ...)" agent
# with HTTP 403 and asks crawlers for exactly this (agent name, version, contact); other sites that care read it the same way.
COLLECT_USER_AGENT = f"{ROBOTS_AGENT}/0.1 (personal dataset research; +https://github.com/Vinillex/model-trainer)"
ROBOTS_MAX_BYTES = 512 * 1024
SKIP_EXTENSIONS = {".svg", ".gif", ".ico"}  # logos, spinners, tracking pixels, animated ads
NOT_A_PAGE = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg", ".pdf", ".zip", ".mp4", ".mp3", ".mov", ".avi", ".css", ".js", ".xml", ".json", ".ico"}
JUNK_IN_URL = re.compile(r"favicon|sprite|spacer|1x1|/logo|avatar|emoji|/icons?/|facebook\.com/tr\b|doubleclick|googleads|/pixel\b", re.I)
WP_SIZE_SUFFIX = re.compile(r"-(\d{2,4})x(\d{2,4})\.(?:jpe?g|png|webp)$", re.I)  # WordPress thumbnails: photo-128x71.jpg
LAZY_SRC_ATTRS = ("data-src", "data-lazy-src", "data-original", "data-full-src", "data-large-file", "src")
SRCSET_ATTRS = ("srcset", "data-srcset", "data-lazy-srcset")
IMAGE_LINK = re.compile(r"\.(jpe?g|png|webp)$", re.I)
PAGE_ABOUT_A_FILE = re.compile(r"/wiki/|/File:|/Image:", re.I)


# ---------- politeness ----------

class Throttle:
    """At most one request per `delay` seconds to each host, also when several download threads ask at once."""

    def __init__(self, delay: float, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.delay, self._clock, self._sleep = delay, clock, sleep
        self._next: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, host: str) -> None:
        if self.delay <= 0:
            return
        with self._lock:  # reserve a slot under the lock, sleep outside it
            now = self._clock()
            slot = max(now, self._next.get(host, 0.0))
            self._next[host] = slot + self.delay
        if slot > now:
            self._sleep(slot - now)


@dataclass
class Fetched:
    url: str  # after redirects
    status: int
    content_type: str
    charset: str | None
    body: bytes


def get(url: str, http: httpx.Client, *, max_bytes: int, accept: str = "*/*") -> Fetched:
    """One guarded GET: every redirect hop is re-checked against the SSRF guard, the body is size-capped while it streams.

    Raises FetchError for anything that stops the request (blocked host, too many redirects, too large, network error).
    HTTP error statuses are returned, not raised: robots.txt treats 404 and 500 differently.
    """
    for _ in range(MAX_REDIRECTS + 1):
        _check_public(url)
        try:
            with http.stream("GET", url, headers={"User-Agent": COLLECT_USER_AGENT, "Accept": accept}) as r:
                if r.is_redirect:
                    location = r.headers.get("location")
                    if not location:
                        raise FetchError("redirect without location")
                    url = urljoin(url, location)
                    continue
                if int(r.headers.get("content-length") or 0) > max_bytes:
                    raise FetchError("too large")
                buf = bytearray()
                for chunk in r.iter_bytes():
                    buf += chunk
                    if len(buf) > max_bytes:
                        raise FetchError("too large")
                ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
                return Fetched(url, r.status_code, ctype, r.charset_encoding, bytes(buf))
        except httpx.HTTPError as e:
            raise FetchError(type(e).__name__) from None
    raise FetchError("too many redirects")


def shrink(data: bytes, max_side: int) -> bytes:
    """Agency photos can be 6000 px wide and 10 MB; the detector sees 640 px. Anything larger than `max_side` is scaled down
    (EXIF rotation applied first) and saved as JPEG. Smaller pictures are returned untouched, byte for byte.
    Raises FetchError for bytes that cannot be opened at all, so one broken file is a failed download, not a failed job."""
    try:
        with PILImage.open(io.BytesIO(data)) as im:
            if max(im.size) <= max_side:
                return data
            im = ImageOps.exif_transpose(im)
            im.thumbnail((max_side, max_side), PILImage.Resampling.LANCZOS)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            out = io.BytesIO()
            im.save(out, "JPEG", quality=92)
            return out.getvalue()
    except Exception as e:  # noqa: BLE001 - Pillow raises many kinds (decompression bombs, truncated files, SyntaxError for bad PNGs)
        raise FetchError(f"not a usable image ({type(e).__name__})") from None


def decode(body: bytes, charset: str | None) -> str:
    if not charset:
        m = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", body[:4096], re.I)
        charset = m.group(1).decode("ascii", "ignore") if m else "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


class Robots:
    """robots.txt per site, read through the same guarded client (never `RobotFileParser.read()`, which would skip the SSRF guard).

    404 and other client errors mean "no rules". A server error or an unreachable file means the site is left alone.
    `check` raises FetchError for an address the SSRF guard refuses, so that shows up as what it is, not as a robots.txt problem.
    """

    def __init__(self, http: httpx.Client, throttle: Throttle):
        self.http, self.throttle = http, throttle
        self._rules: dict[str, RobotFileParser | str] = {}  # origin -> parsed rules, or "allow" / "deny"
        self._lock = threading.Lock()

    def check(self, url: str) -> tuple[bool, str | None]:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}".lower()
        with self._lock:
            if origin not in self._rules:
                _check_public(origin)
                self._rules[origin] = self._load(origin)
            rules = self._rules[origin]
        if rules == "allow":
            return True, None
        if rules == "deny":
            return False, f"robots.txt of {parts.netloc} could not be read, so the site is left alone"
        assert isinstance(rules, RobotFileParser)
        if rules.can_fetch(ROBOTS_AGENT, url):
            return True, None
        return False, f"robots.txt of {parts.netloc} disallows {parts.path or '/'}"

    def _load(self, origin: str) -> RobotFileParser | str:
        self.throttle.wait(urlsplit(origin).netloc)
        try:
            got = get(f"{origin}/robots.txt", self.http, max_bytes=ROBOTS_MAX_BYTES, accept="text/plain,*/*")
        except FetchError:
            return "deny"
        if 400 <= got.status < 500:
            return "allow"
        if got.status != 200:
            return "deny"
        rules = RobotFileParser()
        rules.parse(decode(got.body, got.charset).splitlines())
        rules.modified()
        return rules


# ---------- reading a page ----------

@dataclass
class _Candidate:
    url: str
    in_article: bool
    in_skipped: bool


@dataclass
class Page:
    title: str | None = None
    images: list[str] = field(default_factory=list)  # best candidates first, at most one per picture
    links: list[str] = field(default_factory=list)


def largest_from_srcset(value: str) -> str | None:
    """`a.jpg 480w, b.jpg 1200w` -> `b.jpg`. Only the largest, so the four sizes of one photo are not all downloaded."""
    best, best_size = None, -1.0
    for part in value.split(","):
        bits = part.split()
        if not bits:
            continue
        size = 1.0
        if len(bits) > 1 and bits[1][-1:] in ("w", "x"):
            try:
                size = float(bits[1][:-1])
            except ValueError:
                pass
        if size > best_size:
            best, best_size = bits[0], size
    return best


class _PageParser(HTMLParser):
    SKIPPED = {"nav", "footer", "aside", "form"}  # menus, related stories, comment forms: not the article's own pictures
    ARTICLE = {"article", "main"}
    LEAD_META = {"og:image", "og:image:secure_url", "twitter:image", "twitter:image:src"}

    def __init__(self, page_url: str):
        super().__init__(convert_charrefs=True)
        self.base = page_url
        self.page = Page()
        self.candidates: list[_Candidate] = []
        self.lead: list[str] = []
        self.links: list[str] = []
        self._title_parts: list[str] | None = None
        self._og_title: str | None = None
        self._article = 0
        self._skipped = 0
        self._anchors: list[str | None] = []
        self._picture: list[str] | None = None  # <source> pictures of the <picture> being read, used if it has no <img>
        self._picture_has_img = False

    def _url(self, value: str | None) -> str | None:
        value = (value or "").strip()
        if not value or value.startswith(("data:", "javascript:", "about:", "#")):
            return None
        try:
            return urljoin(self.base, value)
        except ValueError:
            return None

    def _add(self, url: str | None) -> None:
        if url:
            self.candidates.append(_Candidate(url, self._article > 0, self._skipped > 0))

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): v for k, v in attrs}
        if tag in self.ARTICLE:
            self._article += 1
        elif tag in self.SKIPPED:
            self._skipped += 1
        elif tag == "title" and self.page.title is None:
            self._title_parts = []
        elif tag == "base" and a.get("href"):
            self.base = urljoin(self.base, a["href"])
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or "").lower()
            if key in self.LEAD_META and (u := self._url(a.get("content"))):
                self.lead.append(u)
            elif key == "og:title" and a.get("content"):
                self._og_title = a["content"].strip()
        elif tag == "a":
            href = self._url(a.get("href"))
            self._anchors.append(href)
            if href:
                self.links.append(urlunsplit(urlsplit(href)._replace(fragment="")))
        elif tag == "picture":
            self._picture, self._picture_has_img = [], False
        elif tag == "source" and self._picture is not None:
            if (srcset := next((a[k] for k in SRCSET_ATTRS if a.get(k)), None)) and (best := largest_from_srcset(srcset)):
                if u := self._url(best):
                    self._picture.append(u)
        elif tag == "img":
            self._picture_has_img = True
            self._add(self._best_of_img(a))

    def _best_of_img(self, a: dict) -> str | None:
        # A picture wrapped in a link to its own image file is the full-size one in most galleries.
        wrapper = next((h for h in reversed(self._anchors) if h), None)
        # ...but Wikipedia and similar sites link a thumbnail to the *page* about the file ("/wiki/File:Name.jpg"), which only ends like a picture.
        if wrapper and IMAGE_LINK.search(urlsplit(wrapper).path) and not PAGE_ABOUT_A_FILE.search(urlsplit(wrapper).path):
            return wrapper
        for key in SRCSET_ATTRS:
            if a.get(key) and (best := largest_from_srcset(a[key])) and (u := self._url(best)):
                return u
        for key in LAZY_SRC_ATTRS:  # lazy-loading sites keep the real address in a data- attribute and a placeholder in src
            if u := self._url(a.get(key)):
                return u
        return None

    def handle_endtag(self, tag):
        if tag in self.ARTICLE:
            self._article = max(0, self._article - 1)
        elif tag in self.SKIPPED:
            self._skipped = max(0, self._skipped - 1)
        elif tag == "title" and self._title_parts is not None:
            self.page.title = " ".join("".join(self._title_parts).split()) or None
            self._title_parts = None
        elif tag == "a" and self._anchors:
            self._anchors.pop()
        elif tag == "picture" and self._picture is not None:
            if not self._picture_has_img and self._picture:
                self._add(self._picture[0])
            self._picture = None

    def handle_data(self, data):
        if self._title_parts is not None:
            self._title_parts.append(data)


def usable_image_url(url: str, min_side: int = 0) -> bool:
    """False for addresses that are not worth a download: not http(s), logos/icons/tracking pixels, or a thumbnail the
    address itself says is smaller than `min_side` (WordPress names them photo-128x71.jpg)."""
    parts = urlsplit(url)
    if (m := WP_SIZE_SUFFIX.search(parts.path)) and min(int(m.group(1)), int(m.group(2))) < min_side:
        return False
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return False
    if "." + parts.path.rsplit(".", 1)[-1].lower() in SKIP_EXTENSIONS:
        return False
    return not JUNK_IN_URL.search(url)


def parse_page(html: str, page_url: str, per_page: int = 12, min_side: int = 0) -> Page:
    """Titles, the pictures worth downloading (at most `per_page`) and every link of a page.

    Pictures inside <article> or <main> win when the page has any (the rest is menus, related stories and ads); without them
    everything outside menus/footers/forms counts. The page's lead picture (og:image) is used only when nothing else was found,
    because it is usually the same photo again, cropped.
    """
    parser = _PageParser(page_url)
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # noqa: BLE001 - broken markup must cost one page, not the job
        pass
    page = parser.page
    page.title = page.title or parser._og_title
    candidates = [c for c in parser.candidates if not c.in_skipped]
    if any(c.in_article for c in candidates):
        candidates = [c for c in candidates if c.in_article]
    urls = [c.url for c in candidates] or parser.lead
    seen: set[str] = set()
    for u in urls:
        u = urlunsplit(urlsplit(u)._replace(fragment=""))
        if u not in seen and usable_image_url(u, min_side):
            seen.add(u)
            page.images.append(u)
    page.images = page.images[:per_page]
    page.links = list(dict.fromkeys(parser.links))
    return page


# ---------- the crawl ----------

def _site(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _page_key(url: str) -> str:
    return urlunsplit(urlsplit(url.strip())._replace(fragment=""))


class Collector:
    """Reads the pages of one job in order and hands their pictures to `run_scrape`, one page per `search` call."""

    def __init__(self, params: dict, http: httpx.Client, stop: Callable[[], bool] = lambda: False,
                 throttle: Throttle | None = None):
        self.params = params
        self.http = http
        self.stop = stop
        self.throttle = throttle or Throttle(settings.collect_request_delay)
        self.robots = Robots(http, self.throttle)
        self.max_pages = params["max_pages"]
        self.per_page = params.get("images_per_page", 12)
        self.min_side = params.get("min_side", 0)
        self.follow = params.get("follow_links", False)
        self.take_start_pages = params.get("include_start_pages", True)
        self.contains = [s.lower() for s in params.get("link_contains", []) if s.strip()]
        self.seen = {_page_key(u) for u in params["urls"]}
        self.queue: deque[tuple[str, int]] = deque((_page_key(u), 0) for u in dict.fromkeys(params["urls"]))
        self.attempted = self.pages = self.pages_failed = 0
        self.robots_blocked = 0
        self.last_failure: str | None = None
        self._lock = threading.Lock()

    # -- what the job shows --

    def progress(self) -> dict:
        with self._lock:
            out: dict = {"pages": self.pages, "pages_failed": self.pages_failed, "robots_blocked": self.robots_blocked,
                         "queued_pages": len(self.queue)}
            if self.last_failure:
                out["last_failure"] = self.last_failure
            return out

    def _fail(self, message: str, *, robots: bool = False, page: bool = False) -> None:
        with self._lock:
            self.last_failure = message[:300]
            self.robots_blocked += robots
            self.pages_failed += page

    # -- the two functions run_scrape needs --

    def search(self, query: str, page_index: int) -> list[SearchResult]:
        """The pictures of the next page that has any. An empty list means that page had none; SearchDone means no pages are left."""
        if not self.queue or self.attempted >= self.max_pages or self.stop():
            raise SearchDone
        url, depth = self.queue.popleft()
        self.attempted += 1
        return self._read(url, depth)  # [] for a page that failed or had no pictures: the runner publishes and asks again

    def fetch(self, url: str) -> bytes:
        """One picture, with the same manners as the pages."""
        allowed, why = self.robots.check(url)
        if not allowed:
            self._fail(why or "blocked by robots.txt", robots=True)
            raise FetchError("blocked by robots.txt")
        self.throttle.wait(urlsplit(url).netloc)
        return shrink(fetch_image(url, self.http, user_agent=COLLECT_USER_AGENT), settings.collect_max_side)

    # -- one page --

    def _read(self, url: str, depth: int) -> list[SearchResult]:
        try:
            allowed, why = self.robots.check(url)
            if not allowed:
                self._fail(why or "blocked by robots.txt", robots=True)
                return []
            self.throttle.wait(urlsplit(url).netloc)
            got = get(url, self.http, max_bytes=settings.collect_page_max_bytes, accept="text/html,application/xhtml+xml")
        except FetchError as e:
            self._fail(f"{url}: {e}", page=True)
            return []
        if got.status != 200:
            hint = " (many sites turn automated readers away)" if got.status in (401, 403, 429) else ""
            self._fail(f"{url}: HTTP {got.status}{hint}", page=True)
            return []
        if got.content_type not in ("text/html", "application/xhtml+xml"):
            self._fail(f"{url}: not a web page ({got.content_type or 'unknown type'})", page=True)
            return []
        page = parse_page(decode(got.body, got.charset), got.url, self.per_page, self.min_side)
        with self._lock:
            self.pages += 1
        if depth == 0 and self.follow:
            self._queue_links(url, page.links)
        if depth == 0 and self.follow and not self.take_start_pages:
            return []
        final = _page_key(got.url)
        return [SearchResult(original=u, page_url=final, title=page.title) for u in page.images]

    def _queue_links(self, origin: str, links: list[str]) -> None:
        for link in links:
            if self.attempted + len(self.queue) >= self.max_pages:
                return
            parts = urlsplit(link)
            if parts.scheme not in ("http", "https") or _site(link) != _site(origin):
                continue
            if "." + parts.path.rsplit(".", 1)[-1].lower() in NOT_A_PAGE and not PAGE_ABOUT_A_FILE.search(parts.path):
                continue  # a link to a picture, pdf or video file; but "/wiki/File:Name.jpg" is a web page about one
            key = _page_key(link)
            if key in self.seen or (self.contains and not any(s in key.lower() for s in self.contains)):
                continue
            self.seen.add(key)
            try:
                if not self.robots.check(key)[0]:
                    continue  # a page the site asks robots to leave alone is not worth queueing, or counting against the page cap
            except FetchError:
                continue
            self.queue.append((key, 1))


def handle_collect(db: Session, job: Job) -> None:
    def stop() -> bool:
        db.refresh(job)
        return job.cancel_requested

    with httpx.Client(follow_redirects=False, timeout=httpx.Timeout(15, connect=5)) as http:
        collector = Collector(job.params, http, stop)
        run_scrape(db, job, collector.search, collector.fetch, log_searches=False, extra_progress=collector.progress)
