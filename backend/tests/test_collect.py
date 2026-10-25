"""Collecting pictures from web pages (app/collect.py): the page reader, the manners, and the job end to end."""

import threading

import httpx
import pytest
from sqlalchemy import select

from app import jobs
from app.collect import COLLECT_USER_AGENT, Collector, Robots, Throttle, _site, get, largest_from_srcset, parse_page, shrink
from app.config import settings
from app.models import MIXED, DetClass, Image, Job, JobStatus, SearchLog
from app.scrape import FetchError, SearchDone, group_key_for

from .helpers import photo

PUBLIC = "http://93.184.216.34"  # public IP literals, so no DNS lookup is needed (as in test_scrape.py)
OTHER = "http://8.8.8.8"


def page(body: str, head: str = "", title: str = "A story") -> str:
    return f"<html><head><title>{title}</title>{head}</head><body>{body}</body></html>"


class Web:
    """A tiny fake internet behind MockTransport: {url: (status, content type, body)}. Every request is recorded."""

    def __init__(self):
        self.routes: dict[str, tuple[int, str, bytes]] = {}
        self.requests: list[str] = []

    def page(self, url, html, status=200, ctype="text/html; charset=utf-8"):
        self.routes[url] = (status, ctype, html.encode())

    def image(self, url, seed):
        self.routes[url] = (200, "image/jpeg", photo(seed, (400, 300)))

    def robots(self, origin, text, status=200):
        self.routes[f"{origin}/robots.txt"] = (status, "text/plain", text.encode())

    def handler(self, request):
        url = str(request.url)
        self.requests.append(url)
        if url in self.routes:
            status, ctype, body = self.routes[url]
            return httpx.Response(status, content=body, headers={"content-type": ctype})
        return httpx.Response(404)

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self.handler), follow_redirects=False)

    def fetched(self, fragment):
        return [u for u in self.requests if fragment in u]


@pytest.fixture()
def project(client):
    p = client.post("/projects", json={"name": "drones"}).json()
    return p["id"], {c["name"]: c["id"] for c in p["classes"]}


def collect(client, pid, class_id, web, monkeypatch, **body):
    """Create a collect-pages job through the API and run it through the worker's own path, against the fake internet."""
    r = client.post(f"/projects/{pid}/jobs/collect-pages", json={"class_id": class_id, "min_side": 100, **body})
    assert r.status_code == 201, r.text
    job_id = r.json()["id"]
    monkeypatch.setattr(settings, "collect_request_delay", 0)
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(web.handler), **kw))
    assert jobs.claim_next(client.session_factory) == job_id
    jobs.run_job(job_id, client.session_factory)
    with client.session_factory() as db:
        return db.get(Job, job_id)


def stored(client):
    with client.session_factory() as db:
        return [{"url": i.scrape_record.original_url, "page": i.scrape_record.page_url, "title": i.scrape_record.title,
                 "query": i.scrape_record.query, "group": i.group_key, "class_id": i.class_id, "review": i.review_status.value}
                for i in db.scalars(select(Image).order_by(Image.id))]


# ---------- reading a page ----------

def test_largest_from_srcset_keeps_only_the_biggest():
    assert largest_from_srcset("a.jpg 480w, b.jpg 1200w, c.jpg 800w") == "b.jpg"
    assert largest_from_srcset("a.jpg 1x, b.jpg 2x") == "b.jpg"
    assert largest_from_srcset("only.jpg") == "only.jpg"
    assert largest_from_srcset("") is None


def test_parse_page_finds_pictures_the_way_news_sites_hide_them():
    html = page('''
        <base href="http://93.184.216.34/gallery/">
        <article>
          <img src="one.jpg">
          <img src="data:image/gif;base64,R0lGOD" data-src="/lazy/two.jpg">
          <img src="/small/three.jpg" srcset="/small/three.jpg 400w, /big/three.jpg 1600w, /mid/three.jpg 800w">
          <a href="/full/four.jpg"><img src="/thumb/four.jpg"></a>
          <picture><source srcset="/pic/five-s.webp 500w, /pic/five-l.webp 1500w"><img src="/pic/five.jpg"></picture>
          <picture><source srcset="/pic/six.webp 900w"></picture>
          <img src="/img/logo.png"><img src="/img/spinner.gif"><img src="/img/mark.svg"><img src="one.jpg">
          <noscript><img src="/noscript/seven.jpg"></noscript>
        </article>''', title="  Kizilelma \n flies ")
    p = parse_page(html, "http://93.184.216.34/news/story")
    assert p.title == "Kizilelma flies"
    assert p.images == [
        "http://93.184.216.34/gallery/one.jpg",   # relative to <base>
        "http://93.184.216.34/lazy/two.jpg",      # the real address was in data-src, src was a placeholder
        "http://93.184.216.34/big/three.jpg",     # largest srcset entry only
        "http://93.184.216.34/full/four.jpg",     # the link around the thumbnail points at the full-size file
        "http://93.184.216.34/pic/five.jpg",      # an <img> inside <picture> wins over its <source>s
        "http://93.184.216.34/pic/six.webp",      # a <picture> without an <img> uses its <source>
        "http://93.184.216.34/noscript/seven.jpg",
    ]  # logo, gif and svg dropped; the repeated one.jpg listed once


def test_a_thumbnail_linked_to_a_file_page_uses_the_thumbnail_not_the_page():
    """Wikipedia links every thumbnail to /wiki/File:Name.jpg, a web page whose address merely ends like a picture."""
    html = page('''<main><a href="/wiki/File:Anka-3_in_flight.jpg" class="mw-file-description">
        <img src="//upload.wikimedia.org/thumb/a/ab/Anka-3.jpg/330px-Anka-3.jpg"
             srcset="//upload.wikimedia.org/thumb/a/ab/Anka-3.jpg/330px-Anka-3.jpg 1x, //upload.wikimedia.org/thumb/a/ab/Anka-3.jpg/660px-Anka-3.jpg 2x"></a>
        <a href="https://cdn.test/full/gallery-one.jpg"><img src="https://cdn.test/small/gallery-one.jpg"></a></main>''')
    assert parse_page(html, "https://en.wikipedia.org/wiki/TAI_Anka-3").images == [
        "https://upload.wikimedia.org/thumb/a/ab/Anka-3.jpg/660px-Anka-3.jpg",   # the picture, not the File: page
        "https://cdn.test/full/gallery-one.jpg"]                                 # a real gallery link to the full-size file still wins


def test_addresses_that_say_they_are_small_or_tracking_are_not_downloaded():
    body = ('''<article><img src="/wp/photo-128x71.jpg"><img src="/wp/photo-1200x675.jpg"><img src="/wp/related-460x230.png">
        <noscript><img src="https://www.facebook.com/tr?id=123&ev=PageView&noscript=1"></noscript><img src="/ads/pixel"><img src="/wp/plain.jpg"></article>''')
    got = parse_page(page(body), "http://93.184.216.34/s", min_side=300).images
    assert got == ["http://93.184.216.34/wp/photo-1200x675.jpg", "http://93.184.216.34/wp/plain.jpg"]
    assert len(parse_page(page(body), "http://93.184.216.34/s", min_side=0).images) == 4  # without a minimum only the tracking addresses go


def test_parse_page_prefers_the_article_and_ignores_menus_and_related_stories():
    html = page('''
        <img src="/header-banner.jpg">
        <nav><img src="/nav.jpg"></nav>
        <main><article><img src="/story/a.jpg"><footer><img src="/story/footer.jpg"></footer></article></main>
        <aside><img src="/related.jpg"></aside><footer><img src="/foot.jpg"></footer><form><img src="/captcha.jpg"></form>''')
    assert parse_page(html, "http://93.184.216.34/s").images == ["http://93.184.216.34/story/a.jpg"]
    # no <article>/<main>: everything outside menus, related stories, footers and forms counts
    flat = page('<img src="/a.jpg"><nav><img src="/nav.jpg"></nav><img src="/b.jpg"><aside><img src="/rel.jpg"></aside>')
    assert parse_page(flat, "http://93.184.216.34/s").images == ["http://93.184.216.34/a.jpg", "http://93.184.216.34/b.jpg"]


def test_parse_page_lead_picture_only_when_nothing_else_was_found():
    head = '<meta property="og:image" content="/lead.jpg"><meta name="twitter:image" content="http://93.184.216.34/tw.jpg">'
    assert parse_page(page("<p>text only</p>", head), "http://93.184.216.34/s").images == [
        "http://93.184.216.34/lead.jpg", "http://93.184.216.34/tw.jpg"]
    assert parse_page(page('<img src="/body.jpg">', head), "http://93.184.216.34/s").images == ["http://93.184.216.34/body.jpg"]
    assert parse_page(page("<p>x</p>", '<meta property="og:title" content="Og title">').replace("<title>A story</title>", ""),
                      "http://93.184.216.34/s").title == "Og title"


def test_parse_page_caps_pictures_per_page_and_survives_broken_markup():
    body = "".join(f'<img src="/p/{i}.jpg">' for i in range(30))
    assert len(parse_page(page(body), "http://93.184.216.34/s", per_page=5).images) == 5
    junk = '<article><img src="/ok.jpg"><div <<>> <img src= <a href=>' + "<b>" * 500
    assert parse_page(junk, "http://93.184.216.34/s").images[:1] == ["http://93.184.216.34/ok.jpg"]


def test_site_ignores_www_and_case():
    assert _site("https://WWW.Example.com/a") == _site("http://example.com/b") == "example.com"


def test_shrink_scales_big_pictures_down_and_leaves_the_rest_alone():
    from io import BytesIO

    from PIL import Image as PILImage

    small = photo(1, (400, 300))
    assert shrink(small, 2560) is small  # untouched, the very same bytes
    big = shrink(photo(2, (3000, 1500)), 1000)
    with PILImage.open(BytesIO(big)) as im:
        assert im.size == (1000, 500) and im.format == "JPEG"  # longest side capped, aspect kept
    tall = BytesIO()
    PILImage.new("RGBA", (900, 2000), (10, 20, 30, 128)).save(tall, "PNG")
    with PILImage.open(BytesIO(shrink(tall.getvalue(), 1000))) as im:
        assert im.size == (450, 1000) and im.mode == "RGB"  # transparency flattened, so it can be stored as JPEG
    with pytest.raises(FetchError, match="not a usable image"):
        shrink(b"definitely not a picture", 1000)


def test_collected_pictures_are_shrunk_before_they_are_stored(client, project, monkeypatch):
    pid, ids = project
    web = Web()
    web.page(f"{PUBLIC}/news/a", page('<article><img src="/img/huge.jpg"></article>'))
    web.routes[f"{PUBLIC}/img/huge.jpg"] = (200, "image/jpeg", photo(1, (1600, 1200)))
    monkeypatch.setattr(settings, "collect_max_side", 800)

    collect(client, pid, ids["delta-wing"], web, monkeypatch, urls=[f"{PUBLIC}/news/a"])

    with client.session_factory() as db:
        image = db.scalars(select(Image)).one()
        assert (image.width, image.height) == (800, 600)


# ---------- manners ----------

def test_throttle_spaces_requests_per_host():
    now, slept = [0.0], []
    t = Throttle(1.0, clock=lambda: now[0], sleep=slept.append)
    for host in ("a", "a", "a", "b"):
        t.wait(host)
    assert slept == [1.0, 2.0]  # the first request to each host goes at once
    now[0] = 10.0
    t.wait("a")
    assert slept == [1.0, 2.0]  # long since idle again
    Throttle(0, sleep=lambda s: pytest.fail("a zero delay never sleeps")).wait("a")


def test_throttle_holds_when_many_threads_ask_at_once():
    slept = []
    t = Throttle(1.0, clock=lambda: 0.0, sleep=slept.append)
    threads = [threading.Thread(target=t.wait, args=("same-host",)) for _ in range(8)]
    [x.start() for x in threads]
    [x.join() for x in threads]
    assert sorted(slept) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]  # eight requests, eight distinct slots


def robots_for(web):
    return Robots(web.client(), Throttle(0))


def test_robots_rules_and_what_a_missing_or_broken_file_means():
    web = Web()
    web.robots(PUBLIC, "User-agent: *\nDisallow: /private/\nAllow: /private/open\n")
    web.robots(OTHER, "", status=500)
    robots = robots_for(web)
    assert robots.check(f"{PUBLIC}/news/a")[0] is True
    assert robots.check(f"{PUBLIC}/private/a")[0] is False
    assert robots.check(f"{PUBLIC}/private/open")[0] is True
    allowed, why = robots.check(f"{OTHER}/news/a")
    assert allowed is False and "could not be read" in why  # a server error: leave the site alone
    assert robots_for(Web()).check(f"{PUBLIC}/anything")[0] is True  # 404: no rules
    with pytest.raises(FetchError, match="non-public"):  # the SSRF guard's own reason, and nothing reaches the network
        robots.check("http://127.0.0.1/x")
    only_us = Web()
    only_us.robots(PUBLIC, "User-agent: model-trainer-dataset-builder\nDisallow: /\n")
    assert robots_for(only_us).check(f"{PUBLIC}/a")[0] is False
    assert len(web.fetched("robots.txt")) == 2  # read once per site (the private address never got that far), then remembered


def test_get_guards_every_hop_caps_size_and_returns_error_statuses():
    web = Web()
    handler = lambda r: httpx.Response(302, headers={"location": "http://127.0.0.1/admin"}) if r.url.path == "/hop" else web.handler(r)
    http = httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=False)
    with pytest.raises(FetchError, match="non-public"):
        get(f"{PUBLIC}/hop", http, max_bytes=1000)
    assert web.requests == []
    web.page(f"{PUBLIC}/big", "x" * 5000)
    with pytest.raises(FetchError, match="too large"):
        get(f"{PUBLIC}/big", web.client(), max_bytes=1000)
    got = get(f"{PUBLIC}/missing", web.client(), max_bytes=1000)
    assert got.status == 404  # returned, not raised: robots.txt treats 404 and 500 differently


# ---------- the job, end to end ----------

def test_collect_job_reads_pictures_off_article_pages(client, project, monkeypatch):
    pid, ids = project
    web = Web()
    web.page(f"{PUBLIC}/news/a", page('''
        <nav><img src="/img/nav.jpg"></nav>
        <article><img src="/img/a1.jpg"><img data-src="/img/a2.jpg" src="data:image/gif;base64,AAAA">
        <img srcset="/img/a3-small.jpg 400w, /img/a3-big.jpg 1200w" src="/img/a3-small.jpg"></article>
        <aside><img src="/img/related.jpg"></aside>''', title="Kizilelma flies"))
    web.page(f"{PUBLIC}/news/b", page('<img src="/img/b1.jpg">', '<meta property="og:image" content="/img/b-lead.jpg">', title="Anka-3"))
    for seed, name in enumerate(["a1", "a2", "a3-big", "a3-small", "nav", "related", "b1", "b-lead"], 1):
        web.image(f"{PUBLIC}/img/{name}.jpg", seed)

    job = collect(client, pid, ids["delta-wing"], web, monkeypatch, urls=[f"{PUBLIC}/news/a", f"{PUBLIC}/news/b"])

    assert job.status == JobStatus.succeeded and job.type == "collect_pages" and job.error is None
    assert job.progress["added"] == 4 and job.progress["pages"] == 2 and job.progress["pages_failed"] == 0 and job.progress["robots_blocked"] == 0
    rows = stored(client)
    assert [r["url"].rsplit("/", 1)[1] for r in rows] == ["a1.jpg", "a2.jpg", "a3-big.jpg", "b1.jpg"]
    assert all(r["class_id"] == ids["delta-wing"] and r["review"] == "pending" for r in rows)
    assert {r["page"] for r in rows[:3]} == {f"{PUBLIC}/news/a"} and rows[0]["title"] == "Kizilelma flies"
    assert rows[0]["query"] == f"pages: 93.184.216.34 (+1 more)"
    # one article = one group: its pictures can never be split between train and test, and another article can land elsewhere
    assert len({r["group"] for r in rows[:3]}) == 1 and rows[3]["group"] != rows[0]["group"]
    assert all(r["group"].startswith("93.184.216.34/") and len(r["group"]) < 200 for r in rows)
    for never in ("nav.jpg", "related.jpg", "a3-small.jpg", "b-lead.jpg"):
        assert web.fetched(never) == [], never
    with client.session_factory() as db:
        assert db.scalars(select(SearchLog)).all() == []  # no SerpAPI search was spent, none is claimed


def test_a_failed_or_empty_page_in_the_middle_does_not_end_the_job(client, project, monkeypatch):
    pid, ids = project
    web = Web()
    web.page(f"{PUBLIC}/news/a", page('<img src="/img/a.jpg">'))
    web.page(f"{PUBLIC}/news/empty", page("<p>no pictures here</p>"))
    web.page(f"{PUBLIC}/news/pdf", "%PDF-1.4", ctype="application/pdf")
    web.page(f"{PUBLIC}/news/forbidden", "no", status=403)
    web.page(f"{PUBLIC}/news/b", page('<img src="/img/b.jpg">'))
    web.image(f"{PUBLIC}/img/a.jpg", 1)
    web.image(f"{PUBLIC}/img/b.jpg", 2)
    urls = [f"{PUBLIC}/news/a", f"{PUBLIC}/news/empty", f"{PUBLIC}/news/gone", f"{PUBLIC}/news/pdf",
            "http://127.0.0.1/secret", f"{PUBLIC}/news/forbidden", f"{PUBLIC}/news/b"]

    job = collect(client, pid, ids["delta-wing"], web, monkeypatch, urls=urls, max_pages=20)

    assert job.status == JobStatus.succeeded
    assert [r["url"].rsplit("/", 1)[1] for r in stored(client)] == ["a.jpg", "b.jpg"]
    assert job.progress["pages"] == 3  # a, empty, b were read
    assert job.progress["pages_failed"] == 4  # gone (404), pdf, the private address, forbidden (403)
    assert "HTTP 403" in job.progress["last_failure"] and "automated readers" in job.progress["last_failure"]
    assert job.progress["robots_blocked"] == 0  # the private address is a refused address, not a robots.txt matter
    assert web.fetched("127.0.0.1") == []  # the private address never left the process


def test_robots_txt_is_obeyed_for_pages_and_for_pictures(client, project, monkeypatch):
    pid, ids = project
    web = Web()
    web.robots(PUBLIC, "User-agent: *\nDisallow: /private/\nDisallow: /img/blocked/\n")
    web.robots(OTHER, "", status=503)
    web.page(f"{PUBLIC}/private/story", page('<img src="/img/secret.jpg">'))
    web.page(f"{PUBLIC}/news/ok", page('<img src="/img/ok.jpg"><img src="/img/blocked/x.jpg">'))
    web.page(f"{OTHER}/news/x", page('<img src="/img/y.jpg">'))
    for seed, url in enumerate([f"{PUBLIC}/img/secret.jpg", f"{PUBLIC}/img/ok.jpg", f"{PUBLIC}/img/blocked/x.jpg", f"{OTHER}/img/y.jpg"], 1):
        web.image(url, seed)

    job = collect(client, pid, ids["delta-wing"], web, monkeypatch, urls=[f"{PUBLIC}/private/story", f"{PUBLIC}/news/ok", f"{OTHER}/news/x"])

    assert [r["url"].rsplit("/", 1)[1] for r in stored(client)] == ["ok.jpg"]
    assert web.fetched("/private/story") == [] and web.fetched("blocked/x.jpg") == [] and web.fetched("8.8.8.8/news") == []
    assert job.progress["robots_blocked"] == 3 and job.progress["download_failed"] == 1  # a page, a picture and an unreadable robots.txt
    assert "could not be read" in job.progress["last_failure"] or "disallows" in job.progress["last_failure"]


def test_the_collector_says_who_it_is_on_every_request(client, project, monkeypatch):
    pid, ids = project
    web = Web()
    seen = []
    handler = web.handler
    web.handler = lambda request: (seen.append(request.headers["user-agent"]), handler(request))[1]
    web.page(f"{PUBLIC}/news/a", page('<article><img src="/img/a.jpg"></article>'))
    web.image(f"{PUBLIC}/img/a.jpg", 1)

    collect(client, pid, ids["delta-wing"], web, monkeypatch, urls=[f"{PUBLIC}/news/a"])

    assert len(seen) == 3  # robots.txt, the page, the picture
    assert set(seen) == {COLLECT_USER_AGENT}
    assert not COLLECT_USER_AGENT.startswith("Mozilla") and "github.com/Vinillex/model-trainer" in COLLECT_USER_AGENT  # honest, with a way to reach us


def test_a_picture_pointing_at_a_private_address_is_refused(client, project, monkeypatch):
    pid, ids = project
    web = Web()
    web.page(f"{PUBLIC}/news/a", page('<article><img src="http://127.0.0.1/admin.jpg"><img src="http://[::1]/a.jpg"><img src="/img/ok.jpg"></article>'))
    web.image(f"{PUBLIC}/img/ok.jpg", 1)

    job = collect(client, pid, ids["delta-wing"], web, monkeypatch, urls=[f"{PUBLIC}/news/a"])

    assert job.progress["added"] == 1 and job.progress["download_failed"] == 2 and job.progress["robots_blocked"] == 0
    assert web.fetched("127.0.0.1") == [] and web.fetched("::1") == []


def test_an_oversized_page_is_refused_not_read(client, project, monkeypatch):
    pid, ids = project
    web = Web()
    web.page(f"{PUBLIC}/news/huge", page('<img src="/img/a.jpg">' + "<p>filler</p>" * 2000))
    web.image(f"{PUBLIC}/img/a.jpg", 1)
    monkeypatch.setattr(settings, "collect_page_max_bytes", 2000)

    job = collect(client, pid, ids["delta-wing"], web, monkeypatch, urls=[f"{PUBLIC}/news/huge"])

    assert job.status == JobStatus.succeeded and job.progress["added"] == 0
    assert job.progress["pages_failed"] == 1 and "too large" in job.progress["last_failure"]


INDEX = page('''<article>
    <a href="/story/1">one</a> <a href="/story/2">two</a> <a href="/story/2#comments">two again</a> <a href="/story/1?utm=x">one with a tag</a>
    <a href="/about">about us</a> <a href="http://8.8.8.8/story/3">another site</a> <a href="/pics/photo.jpg">a picture</a>
    <a href="/story/4">four</a> <img src="/img/thumb.jpg"></article>''', title="Index")


def follow_web():
    web = Web()
    web.page(f"{PUBLIC}/index", INDEX)
    for n in (1, 2, 4):
        web.page(f"{PUBLIC}/story/{n}", page(f'<article><img src="/img/s{n}.jpg"></article>', title=f"Story {n}"))
    web.page(f"{PUBLIC}/story/1?utm=x", page('<article><img src="/img/s1.jpg"></article>'))
    web.page(f"{PUBLIC}/about", page('<img src="/img/about.jpg">'))
    web.page(f"{OTHER}/story/3", page('<img src="/img/s3.jpg">'))
    for seed, name in enumerate(["thumb", "s1", "s2", "s4", "about", "s3"], 1):
        web.image(f"{PUBLIC}/img/{name}.jpg", seed)
    web.image(f"{OTHER}/img/s3.jpg", 9)
    return web


def test_following_links_stays_on_the_site_one_level_deep_and_uses_the_start_page_only_for_links(client, project, monkeypatch):
    pid, ids = project
    web = follow_web()

    job = collect(client, pid, ids["delta-wing"], web, monkeypatch, urls=[f"{PUBLIC}/index"], follow_links=True,
                  include_start_pages=False, link_contains=["STORY"], max_pages=20)

    rows = stored(client)
    assert sorted(r["url"].rsplit("/", 1)[1] for r in rows) == ["s1.jpg", "s2.jpg", "s4.jpg"]
    assert {r["title"] for r in rows} == {"Story 1", "Story 2", "Story 4"}
    # the index, stories 1, 2 and 4, and "story/1?utm=x": a different address with another query is another page, though its picture is known already
    assert job.progress["pages"] == 5 and job.progress["already_scraped"] == 1 and job.progress["added"] == 3
    assert web.fetched("thumb.jpg") == [] and web.fetched("/about") == [] and web.fetched("8.8.8.8/story") == [] and web.fetched("photo.jpg") == []
    assert len({r["group"] for r in rows}) == 3  # each article its own group


def test_following_links_reads_file_pages_that_merely_end_like_a_picture(client, project, monkeypatch):
    """Wikimedia Commons: a category lists /wiki/File:Name.jpg pages, and the full-size picture is on each of those pages."""
    pid, ids = project
    web = Web()
    web.page(f"{PUBLIC}/wiki/Category:Flying_wings", page('''<main>
        <a href="/wiki/File:Wing_one.jpg"><img src="/thumb/120px-Wing_one.jpg"></a> <a href="/wiki/File:Wing_two.jpg">two</a>
        <a href="/files/direct.jpg">a real picture file</a> <a href="/wiki/Category:Sub">a subcategory</a></main>'''))
    for n, seed in (("one", 1), ("two", 2)):
        web.page(f"{PUBLIC}/wiki/File:Wing_{n}.jpg", page(f'<main><a href="/orig/Wing_{n}.jpg"><img src="/preview/1280px-Wing_{n}.jpg"></a></main>', title=f"File:Wing {n}"))
        web.image(f"{PUBLIC}/orig/Wing_{n}.jpg", seed)

    job = collect(client, pid, ids["delta-wing"], web, monkeypatch, urls=[f"{PUBLIC}/wiki/Category:Flying_wings"], follow_links=True,
                  include_start_pages=False, link_contains=["/wiki/File:"])

    assert sorted(r["url"].rsplit("/", 1)[1] for r in stored(client)) == ["Wing_one.jpg", "Wing_two.jpg"]  # the originals, from the file pages
    assert job.progress["pages"] == 3 and web.fetched("direct.jpg") == [] and web.fetched("Category:Sub") == []


def test_links_that_robots_txt_forbids_are_not_queued_and_do_not_use_up_the_page_cap(client, project, monkeypatch):
    pid, ids = project
    web = Web()
    web.robots(PUBLIC, "User-agent: *\nDisallow: /w/\n")
    web.page(f"{PUBLIC}/wiki/Category:X", page('''<main>''' + "".join(f'<a href="/w/index.php?title=flies{n}&action=edit">edit</a>' for n in range(6))
             + '''<a href="/wiki/File:Wing.jpg">file</a></main>'''))
    web.page(f"{PUBLIC}/wiki/File:Wing.jpg", page('<main><a href="/orig/Wing.jpg"><img src="/p/Wing.jpg"></a></main>'))
    web.image(f"{PUBLIC}/orig/Wing.jpg", 1)

    job = collect(client, pid, ids["delta-wing"], web, monkeypatch, urls=[f"{PUBLIC}/wiki/Category:X"], follow_links=True,
                  include_start_pages=False, link_contains=["flies", "/wiki/File:"], max_pages=2)

    assert job.progress["added"] == 1 and job.progress["pages"] == 2 and job.progress["robots_blocked"] == 0
    assert web.fetched("/w/index.php") == []


def test_page_cap_counts_every_page_including_start_pages(client, project, monkeypatch):
    pid, ids = project
    web = follow_web()

    job = collect(client, pid, ids["delta-wing"], web, monkeypatch, urls=[f"{PUBLIC}/index"], follow_links=True,
                  include_start_pages=True, link_contains=["story"], max_pages=2)

    assert job.progress["pages"] == 2  # the index and the first story, then it stopped
    assert [r["url"].rsplit("/", 1)[1] for r in stored(client)] == ["thumb.jpg", "s1.jpg"]
    assert web.fetched("/story/2") == []


def test_cancelling_stops_before_the_next_page():
    web = Web()
    web.page(f"{PUBLIC}/news/a", page('<img src="/img/a.jpg">'))
    params = {"urls": [f"{PUBLIC}/news/a"], "max_pages": 5}
    with pytest.raises(SearchDone):
        Collector(params, web.client(), stop=lambda: True, throttle=Throttle(0)).search("", 0)
    assert web.requests == []
    c = Collector(params, web.client(), throttle=Throttle(0))
    assert len(c.search("", 0)) == 1
    with pytest.raises(SearchDone):
        c.search("", 1)  # nothing left


# ---------- keeping an article together in one split ----------

def test_group_key_by_article_is_short_stable_and_distinguishes_long_urls():
    base = "https://news.test/" + "section/" * 40
    a, b = group_key_for(base + "story-one", "article"), group_key_for(base + "story-two", "article")
    assert a != b and len(a) < 200 and len(b) < 200
    assert a == group_key_for(base + "story-one#comments", "article")  # a fragment is the same page
    assert a.startswith("news.test/")
    assert group_key_for(base + "story-one", "site") == group_key_for(base + "story-two", "site") == "news.test"
    assert group_key_for(None, "article") is None and group_key_for("not a url", "article") is None


def test_commons_file_pages_of_one_photo_series_share_a_group():
    f = "https://commons.wikimedia.org/wiki/File:"
    one = group_key_for(f + "MQ-9_Takes_Flight_Over_Central_New_York_161019-Z-QU230-002.jpg", "article")
    two = group_key_for(f + "MQ-9_Takes_Flight_Over_Central_New_York_161019-Z-QU230-014.jpg", "article")
    assert one == two  # frames of one sortie must land in one split
    refuel = lambda n: group_key_for(f"{f}USMC_refuels_California_Air_Guard_MQ-9_at_Gwangju_({n}).jpg", "article")
    assert refuel(9267680) == refuel(9267734)
    assert one != refuel(9267680)  # a different series
    assert one != group_key_for(f + "MQ-9_Reaper_in_flight.jpg", "article")
    assert group_key_for(f + "2023.jpg", "article") == group_key_for(f + "2023.png", "article")  # a title that is all digits stays whole
    assert one.startswith("commons.wikimedia.org/")
    # an ordinary page keeps one group per page, whatever its last words look like
    assert group_key_for("https://news.test/a/story-1", "article") != group_key_for("https://news.test/a/story-2", "article")


# ---------- the API ----------

def test_collect_job_api(client, project, monkeypatch):
    pid, ids = project
    other = client.post("/projects", json={"name": "other"}).json()
    url = f"/projects/{pid}/jobs/collect-pages"
    body = {"class_id": ids["delta-wing"], "urls": [f" {PUBLIC}/a ", f"{PUBLIC}/a", "https://news.test/b"]}
    monkeypatch.setattr(settings, "serpapi_key", None)

    job = client.post(url, json=body)
    assert job.status_code == 201  # no SerpAPI key needed
    job = job.json()
    assert job["status"] == "queued" and job["type"] == "collect_pages"
    assert job["params"]["urls"] == [f"{PUBLIC}/a", "https://news.test/b"]  # trimmed, the repeat dropped
    assert job["params"]["query"] == "pages: 93.184.216.34 (+1 more)"
    assert (job["params"]["group_by"], job["params"]["follow_links"], job["params"]["include_start_pages"],
            job["params"]["max_pages"], job["params"]["min_side"]) == ("article", False, True, 30, 300)

    assert client.post(url, json={**body, "class_id": other["classes"][0]["id"]}).status_code == 404
    with client.session_factory() as db:  # the folder normally appears when a reviewed image shows two classes
        db.add(DetClass(project_id=pid, name=MIXED, is_negative=False))
        db.commit()
        mixed = db.scalar(select(DetClass.id).where(DetClass.project_id == pid, DetClass.name == MIXED))
    assert client.post(url, json={**body, "class_id": mixed}).status_code == 422
    for bad in ({"urls": []}, {"urls": ["ftp://x.test/a"]}, {"urls": ["javascript:alert(1)"]}, {"urls": ["/relative"]},
                {"urls": [f"https://x.test/{i}" for i in range(51)]}, {"max_pages": 0}, {"max_pages": 301}, {"max_images": 1001},
                {"images_per_page": 0}, {"group_by": "planet"}, {"link_contains": ["a"] * 11}):
        assert client.post(url, json={**body, **bad}).status_code == 422, bad

    # each panel lists only its own kind of job
    assert [j["id"] for j in client.get(f"/projects/{pid}/jobs?type=collect_pages").json()] == [job["id"]]
    assert client.get(f"/projects/{pid}/jobs?type=scrape").json() == []
    assert client.post(f"/jobs/{job['id']}/cancel").json()["status"] == "cancelled"
