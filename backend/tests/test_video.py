"""Still frames from videos (app/video.py): the job end to end against a fake internet, with a real (tiny) video made by ffmpeg."""

import json
import re
import subprocess
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

from app import jobs
from app.collect import COLLECT_USER_AGENT
from app.config import settings
from app.models import MIXED, DetClass, Image, Job
from app.scrape import SearchDone
from app.video import VideoSource, ffmpeg_tools, frame_url, is_commons_file_page

from .test_collect import PUBLIC, Web

pytestmark = pytest.mark.skipif(ffmpeg_tools() is None, reason="ffmpeg is not installed")

VIDEO = f"{PUBLIC}/clips/flight.mp4"
REAL_CLIENT = httpx.Client  # captured before any test patches it, so one test can run two jobs


def make_video(path: Path, seconds: int = 6, source: str = "testsrc2=size=320x240:rate=10") -> bytes:
    subprocess.run([ffmpeg_tools()[0], "-v", "error", "-y", "-f", "lavfi", "-i", f"{source}", "-t", str(seconds), "-c:v", "mpeg4",
                    "-pix_fmt", "yuv420p", str(path)], check=True)
    return path.read_bytes()


@pytest.fixture(scope="session")
def video_bytes(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("video") / "moving.mp4")


@pytest.fixture(scope="session")
def still_video_bytes(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("still") / "still.mp4", 4, "color=c=blue:size=320x240:rate=10")


@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def serve(web: Web, url: str, body: bytes, ctype="video/mp4", status=200):
    web.routes[url] = (status, ctype, body)


def frames(client, pid, class_id, web, monkeypatch, **body):
    """Create a video-frames job through the API and run it through the worker's own path, against the fake internet."""
    r = client.post(f"/projects/{pid}/jobs/video-frames", json={"class_id": class_id, "urls": [VIDEO], "near_dup_distance": 0,
                                                                  "min_side": 100, **body})
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    monkeypatch.setattr(settings, "collect_request_delay", 0)
    monkeypatch.setattr(httpx, "Client", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(web.handler), **kw))
    assert jobs.claim_next(client.session_factory) == job_id
    jobs.run_job(job_id, client.session_factory)
    with client.session_factory() as db:
        return db.get(Job, job_id)


def stored(client):
    with client.session_factory() as db:
        return [{"url": i.scrape_record.original_url, "page": i.scrape_record.page_url, "title": i.scrape_record.title,
                 "query": i.scrape_record.query, "group": i.group_key, "class_id": i.class_id, "review": i.review_status.value,
                 "w": i.width, "h": i.height}
                for i in db.scalars(select(Image).order_by(Image.id))]


def seconds_of(title: str) -> float:
    return float(re.search(r"@ ([\d.]+)s$", title).group(1))


# ---------- the job ----------

def test_frames_of_a_video_are_filed_pending_in_one_group_with_provenance(client, project, video_bytes, monkeypatch):
    pid, ids = project
    web = Web()
    serve(web, VIDEO, video_bytes)
    job = frames(client, pid, ids["delta-wing"], web, monkeypatch, frames_per_second=1)
    assert job.status.value == "succeeded", job.error
    rows = stored(client)
    assert 5 <= len(rows) <= 7 and job.progress["added"] == len(rows)
    assert job.progress["videos"] == 1 and job.progress["frames_cut"] == len(rows)
    assert {r["class_id"] for r in rows} == {ids["delta-wing"]} and {r["review"] for r in rows} == {"pending"}
    assert len({r["group"] for r in rows}) == 1  # one video = one train/val/test split
    assert all(r["page"] == VIDEO and r["query"].startswith("video: 93.184.216.34") for r in rows)
    assert len({r["url"] for r in rows}) == len(rows) and all("mt_frame=" in r["url"] for r in rows)  # each frame has its own address
    seconds = [seconds_of(r["title"]) for r in rows]
    assert seconds == sorted(seconds) and seconds[0] == 0 and 4 <= seconds[-1] <= 6
    assert web.fetched("flight.mp4").count(VIDEO) == 1  # downloaded once


def test_a_long_video_is_sampled_across_its_whole_length(client, project, video_bytes, monkeypatch):
    pid, ids = project
    web = Web()
    serve(web, VIDEO, video_bytes)
    job = frames(client, pid, ids["fixed-wing"], web, monkeypatch, frames_per_second=5, max_frames_per_video=3)
    assert job.status.value == "succeeded", job.error
    seconds = [seconds_of(r["title"]) for r in stored(client)]
    assert len(seconds) == 3
    assert seconds[-1] >= 3  # not the first 0.6 s of a 6 s video


def test_a_slow_scene_collapses_to_near_duplicates(client, project, still_video_bytes, monkeypatch):
    pid, ids = project
    web = Web()
    serve(web, VIDEO, still_video_bytes)
    job = frames(client, pid, ids["delta-wing"], web, monkeypatch, frames_per_second=2, near_dup_distance=5)
    assert job.status.value == "succeeded", job.error
    assert job.progress["added"] == 1 and job.progress["frames_cut"] >= 6
    assert job.progress["duplicates"] + job.progress["near_duplicates"] == job.progress["frames_cut"] - 1


def test_frames_of_two_videos_get_two_groups(client, project, video_bytes, monkeypatch):
    pid, ids = project
    web = Web()
    second = f"{PUBLIC}/clips/other.mp4"
    serve(web, VIDEO, video_bytes)
    serve(web, second, video_bytes)
    job = frames(client, pid, ids["delta-wing"], web, monkeypatch, urls=[VIDEO, second], frames_per_second=0.5)
    assert job.status.value == "succeeded", job.error
    rows = stored(client)
    # the second video is the same pictures, so byte-identical frames are refused as duplicates: its frames add nothing
    assert job.progress["videos"] == 2 and job.progress["duplicates"] >= 1
    assert {r["page"] for r in rows} == {VIDEO}


def test_the_job_says_who_it_is_and_obeys_robots(client, project, video_bytes, monkeypatch):
    pid, ids = project
    web = Web()
    serve(web, VIDEO, video_bytes)
    web.robots(PUBLIC, "User-agent: *\nDisallow: /clips/")
    seen = []
    handler = web.handler
    web.handler = lambda request: (seen.append(request.headers.get("user-agent")), handler(request))[1]
    job = frames(client, pid, ids["delta-wing"], web, monkeypatch)
    assert job.status.value == "succeeded"
    assert job.progress["added"] == 0 and job.progress["videos_failed"] == 1 and "robots.txt" in job.progress["last_failure"]
    assert web.fetched("flight.mp4") == []  # the video itself was never requested
    assert set(seen) == {COLLECT_USER_AGENT}


@pytest.mark.parametrize("what,route,reason", [
    ("404", (404, "text/plain", b"nope"), "HTTP 404"),
    ("not a video", (200, "video/mp4", b"this is not a video at all" * 100), "ffmpeg could not read"),
])
def test_a_video_that_cannot_be_used_fails_that_video_not_the_job(client, project, monkeypatch, what, route, reason):
    pid, ids = project
    web = Web()
    web.routes[VIDEO] = route
    job = frames(client, pid, ids["delta-wing"], web, monkeypatch)
    assert job.status.value == "succeeded", job.error
    assert job.progress["added"] == 0 and job.progress["videos_failed"] == 1 and reason in job.progress["last_failure"]


def test_a_video_over_the_size_limit_is_not_kept(client, project, video_bytes, monkeypatch):
    pid, ids = project
    web = Web()
    serve(web, VIDEO, video_bytes)
    monkeypatch.setattr(settings, "video_max_bytes", len(video_bytes) // 2)
    job = frames(client, pid, ids["delta-wing"], web, monkeypatch)
    assert job.progress["added"] == 0 and "larger than the limit" in job.progress["last_failure"]


def test_a_private_address_is_refused(client, project, monkeypatch):
    pid, ids = project
    web = Web()
    job = frames(client, pid, ids["delta-wing"], web, monkeypatch, urls=["http://10.0.0.5/clip.mp4"])
    assert job.progress["videos_failed"] == 1 and "non-public" in job.progress["last_failure"]
    assert web.requests == []


def test_a_rerun_skips_frames_already_seen(client, project, video_bytes, monkeypatch):
    pid, ids = project
    web = Web()
    serve(web, VIDEO, video_bytes)
    first = frames(client, pid, ids["delta-wing"], web, monkeypatch, frames_per_second=1)
    n = first.progress["added"]
    second = frames(client, pid, ids["delta-wing"], web, monkeypatch, frames_per_second=1)
    assert second.progress["added"] == 0 and second.progress["already_scraped"] == n


# ---------- a Wikimedia Commons File: page ----------

def test_a_commons_file_page_is_resolved_to_the_video_file(client, project, video_bytes, monkeypatch):
    pid, ids = project
    monkeypatch.setattr(settings, "scrape_allow_private_hosts", True)  # no DNS in tests; the real hosts are public
    page = "https://commons.wikimedia.org/wiki/File:Shahed_drone_in_Dnipropetrovsk_Oblast_(2024-02-09).webm"
    file_url = "https://upload.wikimedia.org/wikipedia/commons/a/ab/Shahed_drone_in_Dnipropetrovsk_Oblast_(2024-02-09).webm"
    asked = []

    def handler(request):
        asked.append(str(request.url))
        if request.url.path == "/w/api.php":
            assert request.url.params["titles"] == "File:Shahed drone in Dnipropetrovsk Oblast (2024-02-09).webm"
            return httpx.Response(200, json={"query": {"pages": {"1": {"imageinfo": [{"url": file_url, "mime": "video/webm"}]}}}})
        if str(request.url) == file_url:
            return httpx.Response(200, content=video_bytes, headers={"content-type": "video/webm"})
        return httpx.Response(404)

    web = Web()
    web.handler = handler
    job = frames(client, pid, ids["delta-wing"], web, monkeypatch, urls=[page], frames_per_second=1)
    assert job.status.value == "succeeded", job.error
    rows = stored(client)
    assert rows and all(r["page"] == page and r["url"].startswith(file_url) for r in rows)
    assert rows[0]["title"].startswith("Shahed drone in Dnipropetrovsk Oblast (2024-02-09).webm @ ")
    assert len({r["group"] for r in rows}) == 1


def test_a_commons_file_page_that_is_a_photo_is_not_a_video(client, project, monkeypatch):
    pid, ids = project
    monkeypatch.setattr(settings, "scrape_allow_private_hosts", True)
    web = Web()
    web.handler = lambda request: httpx.Response(200, json={"query": {"pages": {"1": {"imageinfo": [
        {"url": "https://upload.wikimedia.org/x/Photo.jpg", "mime": "image/jpeg"}]}}}})
    job = frames(client, pid, ids["delta-wing"], web, monkeypatch, urls=["https://commons.wikimedia.org/wiki/File:Photo.jpg"])
    assert job.progress["videos_failed"] == 1 and "not a video" in job.progress["last_failure"]


def test_address_helpers():
    assert is_commons_file_page("https://commons.wikimedia.org/wiki/File:X.webm")
    assert not is_commons_file_page("https://example.com/wiki/File:X.webm")
    assert not is_commons_file_page("https://commons.wikimedia.org/wiki/Category:X")
    a, b = frame_url("http://h/v.mp4", 1.5), frame_url("http://h/v.mp4?x=1", 2)
    assert a == "http://h/v.mp4?mt_frame=1.50" and b == "http://h/v.mp4?x=1&mt_frame=2.00"


# ---------- the source on its own ----------

def test_frames_are_removed_from_disk_when_the_next_video_starts(tmp_path, video_bytes, monkeypatch):
    web = Web()
    serve(web, VIDEO, video_bytes)
    monkeypatch.setattr(settings, "collect_request_delay", 0)
    http = web.client()
    source = VideoSource({"urls": [VIDEO], "frames_per_second": 1, "max_frames_per_video": 10}, http, tmp_path)
    results = source.search("q", 0)
    assert results and all(source.fetch(r.original)[:2] == b"\xff\xd8" for r in results)  # JPEG
    assert any(tmp_path.iterdir())
    with pytest.raises(SearchDone):
        source.search("q", 1)
    assert list(tmp_path.iterdir()) == []  # the downloaded video and its frames are gone


def test_cancelling_stops_before_the_next_video(tmp_path):
    source = VideoSource({"urls": [VIDEO]}, httpx.Client(), tmp_path, stop=lambda: True)
    with pytest.raises(SearchDone):
        source.search("q", 0)


# ---------- the API ----------

def test_video_job_api(client, project, monkeypatch):
    pid, ids = project
    other = client.post("/projects", json={"name": "other"}).json()
    url = f"/projects/{pid}/jobs/video-frames"
    body = {"class_id": ids["delta-wing"], "urls": [f" {VIDEO} ", VIDEO, "https://commons.wikimedia.org/wiki/File:X.webm"]}
    r = client.post(url, json=body)
    assert r.status_code == 201, r.text
    job = r.json()
    assert job["type"] == "video_frames" and job["status"] == "queued"
    p = job["params"]
    assert p["urls"] == [VIDEO, "https://commons.wikimedia.org/wiki/File:X.webm"] and p["max_pages"] == 2
    assert p["group_by"] == "article" and p["frames_per_second"] == 1.0 and p["query"] == "video: 93.184.216.34 (+1 more)"
    assert client.post(url, json={**body, "class_id": other["classes"][0]["id"]}).status_code == 404  # another project's class
    assert client.post(url, json={**body, "class_id": 99999}).status_code == 404
    with client.session_factory() as db:  # the folder normally appears when a reviewed image shows two classes
        db.add(DetClass(project_id=pid, name=MIXED, is_negative=False))
        db.commit()
        mixed = db.scalar(select(DetClass.id).where(DetClass.project_id == pid, DetClass.name == MIXED))
    assert client.post(url, json={**body, "class_id": mixed}).status_code == 422
    assert client.post(url, json={**body, "urls": ["ftp://x/y.mp4"]}).status_code == 422
    assert client.post(url, json={**body, "urls": []}).status_code == 422
    assert client.post(url, json={**body, "urls": [f"{PUBLIC}/{i}.mp4" for i in range(21)]}).status_code == 422
    assert client.post(url, json={**body, "frames_per_second": 0}).status_code == 422
    assert client.post(url, json={**body, "frames_per_second": 9}).status_code == 422
    assert client.post("/projects/9999/jobs/video-frames", json=body).status_code == 404


def test_video_job_is_refused_when_ffmpeg_is_missing(client, project, monkeypatch):
    pid, ids = project
    monkeypatch.setattr("app.api.jobs.ffmpeg_tools", lambda: None)
    r = client.post(f"/projects/{pid}/jobs/video-frames", json={"class_id": ids["delta-wing"], "urls": [VIDEO]})
    assert r.status_code == 503 and "ffmpeg" in r.json()["detail"]
    with client.session_factory() as db:
        assert db.scalars(select(Job)).all() == []
