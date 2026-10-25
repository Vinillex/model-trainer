"""Still frames from videos: a job that downloads video files and hands sparse frames to the scrape pipeline.

Small targets seen from the air (a Shahed over a field, an interceptor closing in) are rare as photographs but common in video.
A video is downloaded politely (robots.txt, one request a second, honest User-Agent, the same SSRF guard as page crawls), ffmpeg
cuts it into frames at a low rate, and `run_scrape` does the rest: size filter, exact and near-duplicate check, provenance.
Frames of one video share one group key, so they stay together in one train/val/test split.
"""

import json
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

import httpx
from sqlalchemy.orm import Session

from .collect import COLLECT_USER_AGENT, Robots, Throttle
from .config import settings
from .models import Job
from .scrape import MAX_REDIRECTS, FetchError, SearchDone, SearchResult, _check_public, run_scrape

VIDEO_SUFFIXES = (".webm", ".ogv", ".ogg", ".mp4", ".m4v", ".mov", ".mkv")
FRAME_MAX_WIDTH = 1920  # wider frames are scaled down; the detector sees 640 px
FFMPEG_TIMEOUT = 900


def ffmpeg_tools() -> tuple[str, str] | None:
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    return (ffmpeg, ffprobe) if ffmpeg and ffprobe else None


def frame_url(video_url: str, seconds: float) -> str:
    """A made-up address that is unique per frame, so the URL ledger can tell one frame of a video from another."""
    parts = urlsplit(video_url)
    query = f"{parts.query}&mt_frame={seconds:.2f}" if parts.query else f"mt_frame={seconds:.2f}"
    return urlunsplit(parts._replace(query=query, fragment=""))


def is_commons_file_page(url: str) -> bool:
    parts = urlsplit(url)
    return (parts.hostname or "").endswith("wikimedia.org") and "/wiki/File:" in unquote(parts.path)


def resolve(url: str, http: httpx.Client, throttle: Throttle) -> tuple[str, str]:
    """(address of the video file, title). A Wikimedia `File:` page is looked up through the Commons API; any other address is
    taken to be the video file itself."""
    if not is_commons_file_page(url):
        return url, unquote(urlsplit(url).path.rsplit("/", 1)[-1]) or url
    parts = urlsplit(url)
    title = unquote(parts.path.split("/wiki/", 1)[1]).replace("_", " ")
    api = f"https://{parts.hostname}/w/api.php"
    throttle.wait(parts.hostname)
    try:
        _check_public(api)
        r = http.get(api, params={"action": "query", "format": "json", "titles": title, "prop": "imageinfo", "iiprop": "url|mime"},
                     headers={"User-Agent": COLLECT_USER_AGENT})
        r.raise_for_status()
        info = next(iter(r.json()["query"]["pages"].values()))["imageinfo"][0]
    except (httpx.HTTPError, KeyError, IndexError, ValueError, StopIteration):
        raise FetchError("could not look up the file on Wikimedia") from None
    if not (info.get("mime", "").startswith("video/") or info["url"].lower().endswith(VIDEO_SUFFIXES)):
        raise FetchError(f"not a video ({info.get('mime') or 'unknown type'})")
    return info["url"], title.removeprefix("File:")


def download(url: str, http: httpx.Client, dest: Path, max_bytes: int) -> None:
    """One guarded GET streamed to disk: every redirect hop is re-checked, the size cap is enforced while it streams."""
    for _ in range(MAX_REDIRECTS + 1):
        _check_public(url)
        try:
            with http.stream("GET", url, headers={"User-Agent": COLLECT_USER_AGENT, "Accept": "video/*,*/*"}) as r:
                if r.is_redirect:
                    location = r.headers.get("location")
                    if not location:
                        raise FetchError("redirect without location")
                    url = urljoin(url, location)
                    continue
                if r.status_code != 200:
                    raise FetchError(f"HTTP {r.status_code}")
                if int(r.headers.get("content-length") or 0) > max_bytes:
                    raise FetchError("video is larger than the limit")
                size = 0
                with dest.open("wb") as out:
                    for chunk in r.iter_bytes(1024 * 256):
                        size += len(chunk)
                        if size > max_bytes:
                            raise FetchError("video is larger than the limit")
                        out.write(chunk)
                return
        except httpx.HTTPError as e:
            raise FetchError(type(e).__name__) from None
    raise FetchError("too many redirects")


def duration_seconds(path: Path, ffprobe: str) -> float | None:
    try:
        out = subprocess.run([ffprobe, "-v", "error", "-print_format", "json", "-show_format", str(path)],
                             capture_output=True, text=True, timeout=60, check=True).stdout
        value = float(json.loads(out)["format"]["duration"])
        return value if value > 0 else None
    except (subprocess.SubprocessError, OSError, KeyError, ValueError):
        return None


def extract_frames(path: Path, out_dir: Path, *, fps: float, max_frames: int, ffmpeg: str, ffprobe: str) -> list[tuple[float, Path]]:
    """Frames at `fps`, spread over the whole video: a long video gets a lower rate so `max_frames` cover all of it, not just the
    first minute. Returns (second, file) pairs. Raises FetchError when ffmpeg cannot read the file."""
    length = duration_seconds(path, ffprobe)
    if length:
        fps = min(fps, max_frames / length)
    fps = max(fps, 0.01)
    scale = f"scale='min({FRAME_MAX_WIDTH},iw)':-2"
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-an", "-sn", "-vf", f"fps={fps:.6f},{scale}",
           "-frames:v", str(max_frames), "-q:v", "2", str(out_dir / "f_%05d.jpg")]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT, check=True)
    except subprocess.TimeoutExpired:
        raise FetchError("ffmpeg took too long") from None
    except (subprocess.CalledProcessError, OSError) as e:
        detail = (getattr(e, "stderr", "") or "").strip().splitlines()
        raise FetchError(f"ffmpeg could not read the video ({detail[-1][:120] if detail else type(e).__name__})") from None
    files = sorted(out_dir.glob("f_*.jpg"))
    return [(i / fps, f) for i, f in enumerate(files)]


class VideoSource:
    """One video per `search` call: downloads it, cuts frames, and serves them to `run_scrape` through `fetch`."""

    def __init__(self, params: dict, http: httpx.Client, workdir: Path, stop: Callable[[], bool] = lambda: False,
                 throttle: Throttle | None = None):
        tools = ffmpeg_tools()
        if tools is None:
            raise RuntimeError("ffmpeg is not installed on this machine (brew install ffmpeg)")
        self.ffmpeg, self.ffprobe = tools
        self.params = params
        self.http = http
        self.workdir = workdir
        self.stop = stop
        self.throttle = throttle or Throttle(settings.collect_request_delay)
        self.robots = Robots(http, self.throttle)
        self.fps = params.get("frames_per_second", 1.0)
        self.max_frames = params.get("max_frames_per_video", 120)
        self.todo = list(dict.fromkeys(params["urls"]))
        self.frames: dict[str, Path] = {}
        self.videos = self.videos_failed = self.frames_cut = 0
        self.last_failure: str | None = None
        self._current: Path | None = None
        self._lock = threading.Lock()

    def progress(self) -> dict:
        with self._lock:
            out = {"videos": self.videos, "videos_failed": self.videos_failed, "frames_cut": self.frames_cut,
                   "queued_videos": len(self.todo)}
            if self.last_failure:
                out["last_failure"] = self.last_failure
            return out

    def _fail(self, message: str) -> None:
        with self._lock:
            self.last_failure = message[:300]
            self.videos_failed += 1

    def _cleanup(self) -> None:
        self.frames = {}
        if self._current is not None:
            shutil.rmtree(self._current, ignore_errors=True)
            self._current = None

    def close(self) -> None:
        self._cleanup()

    def search(self, query: str, page_index: int) -> list[SearchResult]:
        """The frames of the next video. An empty list means that video failed or had none; SearchDone means no videos are left."""
        self._cleanup()  # the previous video's frames have all been taken by now
        if not self.todo or self.stop():
            raise SearchDone
        page_url = self.todo.pop(0)
        try:
            return self._cut(page_url)
        except FetchError as e:
            self._fail(f"{page_url}: {e}")
            self._cleanup()
            return []

    def _cut(self, page_url: str) -> list[SearchResult]:
        file_url, title = resolve(page_url, self.http, self.throttle)
        allowed, why = self.robots.check(file_url)
        if not allowed:
            raise FetchError(why or "blocked by robots.txt")
        self._current = Path(tempfile.mkdtemp(prefix="video-", dir=self.workdir))
        video = self._current / "video"
        self.throttle.wait(urlsplit(file_url).netloc)
        download(file_url, self.http, video, settings.video_max_bytes)
        frames_dir = self._current / "frames"
        frames_dir.mkdir()
        cut = extract_frames(video, frames_dir, fps=self.fps, max_frames=self.max_frames, ffmpeg=self.ffmpeg, ffprobe=self.ffprobe)
        video.unlink(missing_ok=True)  # only the frames are needed from here on
        if not cut:
            raise FetchError("no frames could be cut from it")
        with self._lock:
            self.videos += 1
            self.frames_cut += len(cut)
        results = []
        for seconds, path in cut:
            url = frame_url(file_url, seconds)
            self.frames[url] = path
            results.append(SearchResult(original=url, page_url=page_url, title=f"{title} @ {seconds:.1f}s"))
        return results

    def fetch(self, url: str) -> bytes:
        path = self.frames.get(url)
        if path is None:
            raise FetchError("frame is gone")
        try:
            return path.read_bytes()
        except OSError:
            raise FetchError("frame is gone") from None


def handle_video(db: Session, job: Job) -> None:
    def stop() -> bool:
        db.refresh(job)
        return job.cancel_requested

    with tempfile.TemporaryDirectory(prefix="mt-video-") as work, \
            httpx.Client(follow_redirects=False, timeout=httpx.Timeout(30, connect=5)) as http:
        source = VideoSource(job.params, http, Path(work), stop)
        try:
            run_scrape(db, job, source.search, source.fetch, log_searches=False, extra_progress=source.progress)
        finally:
            source.close()
