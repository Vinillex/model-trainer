import httpx
import pytest
from sqlalchemy import select

from app import jobs
from app.config import settings
from app.models import Image, ImageHash, Job, JobStatus, ScrapeRecord, Source
from app.scrape import FetchError, SearchError, SearchResult, SerpApiClient, fetch_image, run_scrape

from .helpers import encode, photo, photo_image, png

SECRET = "SECRET-KEY-123"


# ---------- SerpAPI client ----------

def serp(handler) -> SerpApiClient:
    return SerpApiClient(SECRET, "https://serp.test/search", httpx.Client(transport=httpx.MockTransport(handler)))


def test_serpapi_parses_results_and_sends_expected_params():
    seen = {}

    def handler(request):
        seen.update(request.url.params)
        return httpx.Response(200, json={"images_results": [
            {"original": "https://a.test/1.jpg", "link": "https://a.test/page", "title": "Quad", "original_width": 800, "original_height": 600},
            {"thumbnail": "https://a.test/t.jpg"},  # no original: unusable
            {"original": "https://b.test/2.png"},
        ]})

    results = serp(handler).search("drone in flight", 2)
    assert seen == {"engine": "google_images", "q": "drone in flight", "ijn": "2", "api_key": SECRET}
    assert [r.original for r in results] == ["https://a.test/1.jpg", "https://b.test/2.png"]
    assert (results[0].page_url, results[0].title, results[0].width, results[0].height) == ("https://a.test/page", "Quad", 800, 600)
    assert results[1].width is None


def test_serpapi_end_of_results_is_empty_not_an_error():
    assert serp(lambda r: httpx.Response(200, json={"error": "Google hasn't returned any results for this query."})).search("q", 9) == []


def test_serpapi_errors_never_leak_the_key():
    with pytest.raises(SearchError, match="HTTP 401: Invalid API key") as e:
        serp(lambda r: httpx.Response(401, json={"error": "Invalid API key."})).search("q", 0)
    assert SECRET not in str(e.value)

    with pytest.raises(SearchError, match="Your account has run out of searches"):
        serp(lambda r: httpx.Response(200, json={"error": "Your account has run out of searches."})).search("q", 0)

    def boom(request):
        raise httpx.ConnectError(f"cannot reach {request.url}")  # httpx embeds the URL, including api_key

    with pytest.raises(SearchError) as e:
        serp(boom).search("q", 0)
    assert SECRET not in str(e.value) and SECRET not in repr(e.value.__cause__)
    assert str(e.value) == "SerpAPI request failed (ConnectError)"

    with pytest.raises(SearchError, match="HTTP 500"):
        serp(lambda r: httpx.Response(500, text="<html>oops</html>")).search("q", 0)


# ---------- image download / SSRF guard ----------

def fetcher(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)


PUBLIC = "http://93.184.216.34"  # a public IP literal, so no DNS lookup is needed


def test_fetch_image_ok_and_follows_public_redirect():
    def handler(request):
        if request.url.path == "/old.jpg":
            return httpx.Response(302, headers={"location": "/new.jpg"})
        return httpx.Response(200, content=b"imgbytes", headers={"content-type": "image/jpeg"})

    assert fetch_image(f"{PUBLIC}/old.jpg", fetcher(handler)) == b"imgbytes"


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/x.jpg", "http://localhost/x.jpg", "http://10.0.0.5/x.jpg", "http://192.168.1.1/x.jpg",
    "http://169.254.169.254/latest/meta-data", "http://[::1]/x.jpg", "file:///etc/passwd", "ftp://93.184.216.34/x.jpg",
])
def test_fetch_image_blocks_non_public_and_non_http(url):
    called = []
    with pytest.raises(FetchError):
        fetch_image(url, fetcher(lambda r: called.append(r) or httpx.Response(200, content=b"x")))
    assert called == []  # blocked before any request went out


def test_fetch_image_rechecks_every_redirect_hop():
    def handler(request):
        if request.url.host == "93.184.216.34":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})
        raise AssertionError("must not reach the private host")

    with pytest.raises(FetchError, match="non-public"):
        fetch_image(f"{PUBLIC}/a.jpg", fetcher(handler))


def test_fetch_image_rejects_bad_responses():
    with pytest.raises(FetchError, match="HTTP 404"):
        fetch_image(f"{PUBLIC}/a", fetcher(lambda r: httpx.Response(404)))
    with pytest.raises(FetchError, match="not an image"):
        fetch_image(f"{PUBLIC}/a", fetcher(lambda r: httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"})))
    with pytest.raises(FetchError, match="too large"):
        fetch_image(f"{PUBLIC}/a", fetcher(lambda r: httpx.Response(200, content=b"x" * 100, headers={"content-type": "image/png"})), max_bytes=50)
    # no Content-Length to trust: the streamed size cap must still stop it
    streaming = lambda r: httpx.Response(200, stream=httpx.ByteStream(b"x" * 100), headers={"content-type": "image/png"})
    with pytest.raises(FetchError, match="too large"):
        fetch_image(f"{PUBLIC}/a", fetcher(streaming), max_bytes=50)
    with pytest.raises(FetchError, match="too many redirects"):
        fetch_image(f"{PUBLIC}/a", fetcher(lambda r: httpx.Response(302, headers={"location": "/b"})))
    with pytest.raises(FetchError, match="ConnectError"):
        fetch_image(f"{PUBLIC}/a", fetcher(lambda r: (_ for _ in ()).throw(httpx.ConnectError("x"))))


# ---------- the scrape runner ----------

@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def make_job(client, project_id, class_id, **over):
    params = {"class_id": class_id, "query": "multirotor drone", "max_images": 10, "max_pages": 3,
              "min_side": 100, "near_dup_distance": 4, **over}
    with client.session_factory() as db:
        job = Job(project_id=project_id, type="scrape", params=params, status=JobStatus.running)
        db.add(job)
        db.commit()
        return job.id


def run(client, job_id, pages, downloads):
    """Run the scraper against canned search pages and a canned {url: bytes | Exception} download table."""
    searched, fetched = [], []

    def search(query, page):
        searched.append(page)
        return pages[page] if page < len(pages) else []

    def fetch(url):
        fetched.append(url)
        out = downloads[url]
        if isinstance(out, Exception):
            raise out
        return out

    with client.session_factory() as db:
        run_scrape(db, db.get(Job, job_id), search, fetch)
    return searched, fetched


def sr(url, **kw):
    return SearchResult(original=url, page_url=f"https://{url.split('/')[2]}/gallery", title=f"title {url}", **kw)


def test_run_scrape_filters_and_stores(client, project):
    pid, ids = project
    # An uploaded image already in the project: the scraper must dedupe against it too.
    uploaded = photo(1)
    r = client.post(f"/projects/{pid}/classes/{ids['multirotor']}/images", files=[("files", ("u.jpg", uploaded, "image/jpeg"))]).json()
    assert len(r["added"]) == 1
    with client.session_factory() as db:  # simulate an image stored before hashing existed
        db.query(ImageHash).delete()
        db.commit()

    new2, new3 = photo(2), photo(3)
    downloads = {
        "https://a.test/new2.jpg": new2,                                                       # added
        "https://b.test/same-bytes.jpg": new2,                                                 # exact duplicate (other URL)
        "https://c.test/resized.jpg": photo(2, (160, 120), quality=40),                        # near-duplicate of new2
        "https://d.test/broken.jpg": FetchError("HTTP 404"),                                   # download failed
        "https://f.test/garbage.jpg": b"this is not an image",                                 # rejected
        "https://g.test/new3.jpg": new3,                                                       # added
        "https://h.test/upload-copy.jpg": photo(1, (160, 120), quality=50),                    # near-duplicate of the upload
        "https://i.test/tiny-decoded.png": png(size=(40, 40)),                                 # decodes but below min_side
    }
    pages = [[
        sr("https://a.test/new2.jpg"), sr("https://b.test/same-bytes.jpg"), sr("https://c.test/resized.jpg"),
        sr("https://d.test/broken.jpg"), sr("https://e.test/small-meta.jpg", width=50, height=50),
        sr("https://f.test/garbage.jpg"), sr("https://g.test/new3.jpg"), sr("https://h.test/upload-copy.jpg"),
        sr("https://i.test/tiny-decoded.png"),
    ]]
    job_id = make_job(client, pid, ids["multirotor"])
    searched, fetched = run(client, job_id, pages, downloads)

    assert searched == [0, 1]  # page 1 came back empty, so it stopped
    assert "https://e.test/small-meta.jpg" not in fetched  # metadata said too small: never downloaded

    with client.session_factory() as db:
        job = db.get(Job, job_id)
        assert job.progress == {"pages": 2, "candidates": 9, "added": 2, "duplicates": 1, "near_duplicates": 2,
                                "already_scraped": 0, "too_small": 1, "download_failed": 1, "rejected": 2, "target": 10, "start_page": 0}
        scraped = db.scalars(select(Image).where(Image.source == Source.scraped).order_by(Image.id)).all()
        assert [i.scrape_record.original_url for i in scraped] == ["https://a.test/new2.jpg", "https://g.test/new3.jpg"]
        first = scraped[0]
        assert (first.scrape_record.page_url, first.scrape_record.title, first.scrape_record.query, first.scrape_record.job_id) == \
               ("https://a.test/gallery", "title https://a.test/new2.jpg", "multirotor drone", job_id)
        assert first.group_key == "a.test" and first.review_status.value == "pending" and first.class_id == ids["multirotor"]
        assert db.query(ImageHash).count() == 3  # the upload was backfilled, both scraped images hashed

    # A re-run never downloads a URL it already tried, whatever happened to it the first time:
    # stored, exact duplicate, near-duplicate or rejected. Only the failed download is retried.
    job2 = make_job(client, pid, ids["multirotor"])
    _, fetched2 = run(client, job2, pages, downloads)
    assert fetched2 == ["https://d.test/broken.jpg"]
    with client.session_factory() as db:
        p2 = db.get(Job, job2).progress
        assert (p2["already_scraped"], p2["download_failed"], p2["too_small"], p2["added"]) == (7, 1, 1, 0)


def test_run_scrape_stops_at_max_images_and_max_pages(client, project):
    pid, ids = project
    pages = [[sr(f"https://s{p}{i}.test/x.jpg") for i in range(5)] for p in range(5)]
    downloads = {r.original: photo(100 + n) for n, r in enumerate(r for page in pages for r in page)}

    job_id = make_job(client, pid, ids["multirotor"], max_images=3)
    searched, _ = run(client, job_id, pages, downloads)
    with client.session_factory() as db:
        assert db.get(Job, job_id).progress["added"] == 3
    assert searched == [0]  # three images came from the first page, so no second search was spent

    job_id = make_job(client, pid, ids["multirotor"], max_images=500, max_pages=2)
    searched, _ = run(client, job_id, pages, downloads)
    assert searched == [0, 1]  # capped by max_pages even though more results exist


def test_run_scrape_near_dup_distance_zero_keeps_near_duplicates(client, project):
    pid, ids = project
    downloads = {"https://a.test/1.jpg": photo(7), "https://b.test/2.jpg": photo(7, (160, 120), quality=40)}
    pages = [[sr(u) for u in downloads]]
    job_id = make_job(client, pid, ids["multirotor"], near_dup_distance=0)
    run(client, job_id, pages, downloads)
    with client.session_factory() as db:
        assert db.get(Job, job_id).progress["added"] == 2


def test_run_scrape_negative_class_images_need_no_review(client, project):
    pid, ids = project
    downloads = {"https://a.test/bird.jpg": photo(9)}
    job_id = make_job(client, pid, ids["birds"])
    run(client, job_id, [[sr("https://a.test/bird.jpg")]], downloads)
    with client.session_factory() as db:
        img = db.scalars(select(Image)).one()
        assert img.review_status.value == "reviewed" and img.storage_key.startswith(f"projects/{pid}/raw/birds/scraped/")


def test_run_scrape_honours_cancel(client, project):
    pid, ids = project
    job_id = make_job(client, pid, ids["multirotor"])
    downloads = {"https://a.test/1.jpg": photo(5)}

    def cancel_during_search(query, page):
        with client.session_factory() as other:  # what POST /jobs/{id}/cancel does while the job runs
            other.get(Job, job_id).cancel_requested = True
            other.commit()
        return [sr("https://a.test/1.jpg")]

    with client.session_factory() as db:
        run_scrape(db, db.get(Job, job_id), cancel_during_search, lambda u: downloads[u])
    with client.session_factory() as db:
        assert db.get(Job, job_id).progress["added"] == 0  # stopped before downloading anything


def test_run_scrape_cancel_takes_effect_between_images_not_batches(client, project):
    pid, ids = project
    job_id = make_job(client, pid, ids["multirotor"], max_images=50)
    urls = [f"https://h{i}.test/x.jpg" for i in range(12)]  # all in one download batch
    downloads = {u: photo(200 + i) for i, u in enumerate(urls)}

    def fetch(url):
        with client.session_factory() as other:  # the user presses Cancel as soon as downloading starts
            other.get(Job, job_id).cancel_requested = True
            other.commit()
        return downloads[url]

    with client.session_factory() as db:
        run_scrape(db, db.get(Job, job_id), lambda q, p: [sr(u) for u in urls] if p == 0 else [], fetch)
    with client.session_factory() as db:
        added = db.get(Job, job_id).progress["added"]
    assert added <= 1, f"{added} of 12 images were ingested after Cancel"


def test_deleting_a_scraped_image_removes_its_hash_and_record(client, project):
    pid, ids = project
    job_id = make_job(client, pid, ids["multirotor"])
    run(client, job_id, [[sr("https://a.test/1.jpg")]], {"https://a.test/1.jpg": photo(5)})
    image_id = client.get(f"/projects/{pid}/images", params={"source": "scraped"}).json()[0]["id"]
    assert client.delete(f"/images/{image_id}").status_code == 204
    with client.session_factory() as db:
        assert db.query(ImageHash).count() == 0 and db.query(ScrapeRecord).count() == 0


# ---------- worker ----------

def test_worker_claim_run_and_failure_paths(client, project, monkeypatch):
    pid, ids = project
    Session = client.session_factory

    def add(**kw):
        with Session() as db:
            j = Job(project_id=pid, type="scrape", params={}, **kw)
            db.add(j)
            db.commit()
            return j.id

    first, second = add(), add()
    assert jobs.claim_next(Session) == first  # oldest first
    with Session() as db:
        assert db.get(Job, first).status == JobStatus.running and db.get(Job, first).started_at
    assert jobs.claim_next(Session) == second
    assert jobs.claim_next(Session) is None

    def ok(db, job):
        job.progress = {"done": True}
        db.commit()

    def boom(db, job):
        job.progress = {"half": True}
        raise RuntimeError("search exploded")

    monkeypatch.setitem(jobs.HANDLERS, "scrape", ok)
    jobs.run_job(first, Session)
    monkeypatch.setitem(jobs.HANDLERS, "scrape", boom)
    jobs.run_job(second, Session)
    with Session() as db:
        a, b = db.get(Job, first), db.get(Job, second)
        assert (a.status, a.progress, a.error) == (JobStatus.succeeded, {"done": True}, None) and a.finished_at
        assert b.status == JobStatus.failed and b.error == "RuntimeError: search exploded" and b.finished_at

    # a cancel request that arrives while the handler runs ends as `cancelled`, not `succeeded`
    third = add()
    jobs.claim_next(Session)

    def cancelled_midway(db, job):
        with Session() as other:
            other.get(Job, third).cancel_requested = True
            other.commit()

    monkeypatch.setitem(jobs.HANDLERS, "scrape", cancelled_midway)
    jobs.run_job(third, Session)
    with Session() as db:
        assert db.get(Job, third).status == JobStatus.cancelled


def test_recover_interrupted_fails_orphaned_running_jobs(client, project):
    pid, _ = project
    with client.session_factory() as db:
        db.add_all([Job(project_id=pid, type="scrape", status=JobStatus.running),
                    Job(project_id=pid, type="scrape", status=JobStatus.queued)])
        db.commit()
    assert jobs.recover_interrupted(client.session_factory) == 1
    with client.session_factory() as db:
        statuses = sorted(j.status.value for j in db.scalars(select(Job)))
        assert statuses == ["failed", "queued"]


def test_worker_thread_processes_a_queued_job(client, project, monkeypatch):
    pid, _ = project
    with client.session_factory() as db:
        db.add(Job(project_id=pid, type="scrape", params={}))
        db.commit()

    def ok(db, job):
        job.progress = {"ran": True}
        db.commit()

    monkeypatch.setitem(jobs.HANDLERS, "scrape", ok)
    worker = jobs.Worker(client.session_factory, poll_seconds=0.05)
    worker.start()
    try:
        for _ in range(100):
            with client.session_factory() as db:
                if db.scalars(select(Job)).one().status == JobStatus.succeeded:
                    break
            worker.stop_event.wait(0.05)
    finally:
        worker.stop()
        worker.join(timeout=3)
    with client.session_factory() as db:
        assert db.scalars(select(Job)).one().progress == {"ran": True}
    assert not worker.is_alive()


# ---------- jobs API ----------

def test_scrape_job_api(client, project, monkeypatch):
    pid, ids = project
    other, other_ids = (lambda p: (p, {c["name"]: c["id"] for c in p["classes"]}))(client.post("/projects", json={"name": "other"}).json())
    url = f"/projects/{pid}/jobs/scrape"
    body = {"class_id": ids["multirotor"], "query": "  fpv drone  "}

    monkeypatch.setattr(settings, "serpapi_key", None)
    r = client.post(url, json=body)
    assert r.status_code == 503 and "MT_SERPAPI_KEY" in r.json()["detail"]

    monkeypatch.setattr(settings, "serpapi_key", SECRET)
    assert client.post(url, json={**body, "class_id": other_ids["multirotor"]}).status_code == 404
    for bad in ({"max_images": 0}, {"max_images": 501}, {"max_pages": 0}, {"min_side": 5}, {"near_dup_distance": 99}, {"query": ""}):
        assert client.post(url, json={**body, **bad}).status_code == 422, bad

    job = client.post(url, json=body).json()
    assert job["status"] == "queued" and job["type"] == "scrape"
    assert job["params"] == {"class_id": ids["multirotor"], "query": "fpv drone", "max_images": 50, "max_pages": 3,
                             "min_side": 200, "near_dup_distance": 4, "start_page": 0, "skip_known_urls": True}
    assert SECRET not in str(job)

    assert [j["id"] for j in client.get(f"/projects/{pid}/jobs").json()] == [job["id"]]
    assert client.get(f"/projects/{other['id']}/jobs").json() == []
    assert client.get(f"/jobs/{job['id']}").json()["id"] == job["id"]
    assert client.get("/jobs/999").status_code == 404

    cancelled = client.post(f"/jobs/{job['id']}/cancel").json()
    assert cancelled["status"] == "cancelled"
    assert client.post(f"/jobs/{job['id']}/cancel").status_code == 409

    running = client.post(url, json=body).json()
    with client.session_factory() as db:
        db.get(Job, running["id"]).status = JobStatus.running
        db.commit()
    r = client.post(f"/jobs/{running['id']}/cancel").json()
    assert r["status"] == "running" and r["cancel_requested"] is True  # asked to stop at its next checkpoint


# ---------- "never download the same image twice" ----------

def test_known_urls_option_off_reattempts_but_never_stores_twice(client, project):
    pid, ids = project
    downloads = {"https://a.test/1.jpg": photo(11), "https://b.test/2.jpg": photo(12)}
    pages = [[sr(u) for u in downloads]]
    run(client, make_job(client, pid, ids["multirotor"]), pages, downloads)

    second = make_job(client, pid, ids["multirotor"], skip_known_urls=False)
    _, fetched = run(client, second, pages, downloads)
    assert sorted(fetched) == sorted(downloads)  # explicitly asked to look again...
    with client.session_factory() as db:
        assert db.get(Job, second).progress["duplicates"] == 2 and db.query(Image).count() == 2  # ...but nothing doubled up


def test_url_stays_known_after_its_image_is_deleted(client, project):
    pid, ids = project
    downloads = {"https://a.test/1.jpg": photo(21)}
    pages = [[sr("https://a.test/1.jpg")]]
    run(client, make_job(client, pid, ids["multirotor"]), pages, downloads)
    image_id = client.get(f"/projects/{pid}/images").json()[0]["id"]
    client.delete(f"/images/{image_id}")  # you deleted it because you didn't want it...

    _, fetched = run(client, make_job(client, pid, ids["multirotor"]), pages, downloads)
    assert fetched == []  # ...so it must not come back on the next run


def test_images_scraped_before_the_ledger_existed_are_still_skipped(client, project):
    pid, ids = project
    downloads = {"https://a.test/1.jpg": photo(31)}
    pages = [[sr("https://a.test/1.jpg")]]
    run(client, make_job(client, pid, ids["multirotor"]), pages, downloads)
    from app.models import ScrapedUrl

    with client.session_factory() as db:
        db.query(ScrapedUrl).delete()  # as in a database from before the ledger
        db.commit()
    _, fetched = run(client, make_job(client, pid, ids["multirotor"]), pages, downloads)
    assert fetched == []  # the provenance record still says we have it


def test_url_key_ignores_case_and_fragment_but_not_query():
    from app.scrape import url_key

    assert url_key("HTTPS://Example.COM/a.jpg#frag") == url_key("https://example.com/a.jpg")
    assert url_key("https://example.com/a.jpg?w=1") != url_key("https://example.com/a.jpg?w=2")


# ---------- start page, search history ----------

def test_start_page_and_search_log(client, project):
    pid, ids = project
    pages = {3: [sr("https://p3.test/1.jpg")], 4: [sr("https://p4.test/1.jpg")]}  # page 5 returns nothing
    downloads = {"https://p3.test/1.jpg": photo(41), "https://p4.test/1.jpg": photo(42)}
    searched = []

    def search(query, page):
        searched.append(page)
        return pages.get(page, [])

    job_id = make_job(client, pid, ids["multirotor"], start_page=3, max_pages=5, query="  Quad   Copter ")
    with client.session_factory() as db:
        run_scrape(db, db.get(Job, job_id), search, lambda u: downloads[u])
    assert searched == [3, 4, 5]  # began at the requested page and stopped at the first empty one
    with client.session_factory() as db:
        progress = db.get(Job, job_id).progress
        assert (progress["pages"], progress["start_page"], progress["added"]) == (3, 3, 2)

    h = client.get(f"/projects/{pid}/searches", params={"query": "quad copter"}).json()
    assert h == {"query": "quad copter", "pages_searched": [3, 4, 5], "next_page": 6, "exhausted_at": 5}
    assert client.get(f"/projects/{pid}/searches", params={"query": "never searched"}).json() == \
           {"query": "never searched", "pages_searched": [], "next_page": 0, "exhausted_at": None}


def test_start_page_validation(client, project, monkeypatch):
    pid, ids = project
    monkeypatch.setattr(settings, "serpapi_key", SECRET)
    url = f"/projects/{pid}/jobs/scrape"
    body = {"class_id": ids["multirotor"], "query": "drone"}
    assert client.post(url, json={**body, "start_page": 97, "max_pages": 3}).status_code == 201  # ijn 97..99
    assert client.post(url, json={**body, "start_page": 98, "max_pages": 3}).status_code == 422  # would reach ijn 100
    assert client.post(url, json={**body, "start_page": 100}).status_code == 422
    assert client.post(url, json={**body, "start_page": -1}).status_code == 422


# ---------- account / quota ----------

def test_fetch_account_returns_only_plan_fields():
    from app.scrape import fetch_account

    body = {"account_email": "me@example.com", "api_key": SECRET, "plan_name": "Free Plan", "searches_per_month": 250,
            "this_month_usage": 3, "plan_searches_left": 247, "total_searches_left": 247, "account_rate_limit_per_hour": 250}
    seen = {}

    def handler(request):
        seen.update(request.url.params)
        return httpx.Response(200, json=body)

    out = fetch_account(SECRET, httpx.Client(transport=httpx.MockTransport(handler)), "https://serp.test/account.json")
    assert seen == {"api_key": SECRET}
    assert out["plan_name"] == "Free Plan" and out["total_searches_left"] == 247
    assert "me@example.com" not in str(out) and SECRET not in str(out)

    with pytest.raises(SearchError, match="HTTP 401: Invalid API key") as e:
        fetch_account(SECRET, httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401, json={"error": "Invalid API key."}))))
    assert SECRET not in str(e.value)

    def boom(request):
        raise httpx.ConnectError(f"cannot reach {request.url}")

    with pytest.raises(SearchError) as e:
        fetch_account(SECRET, httpx.Client(transport=httpx.MockTransport(boom)))
    assert SECRET not in str(e.value) and SECRET not in repr(e.value.__cause__)


def test_account_endpoint(client, monkeypatch):
    from app.api import jobs as jobs_api

    monkeypatch.setattr(settings, "serpapi_key", None)
    assert client.get("/serpapi/account").status_code == 503

    monkeypatch.setattr(settings, "serpapi_key", SECRET)
    monkeypatch.setattr(jobs_api, "fetch_account", lambda key: {"plan_name": "Free Plan", "searches_per_month": 250,
                        "this_month_usage": 2, "plan_searches_left": 248, "total_searches_left": 248, "account_rate_limit_per_hour": 250})
    assert client.get("/serpapi/account").json()["total_searches_left"] == 248

    def fail(key):
        raise SearchError("SerpAPI account HTTP 401: Invalid API key.")

    monkeypatch.setattr(jobs_api, "fetch_account", fail)
    r = client.get("/serpapi/account")
    assert r.status_code == 502 and "Invalid API key" in r.json()["detail"] and SECRET not in r.text
