import httpx
import pytest

from app.config import settings
from app.runpod_client import RunPodError, fetch_account

SECRET = "rpa_SECRET-KEY-123"


def rp(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_account_sends_key_in_header_and_returns_only_balance_fields():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers["authorization"]
        seen["query"] = request.url.query
        return httpx.Response(200, json={"data": {"myself": {"clientBalance": 12.5, "currentSpendPerHr": 0.4,
                                                              "spendLimit": 80, "pods": [{"id": "a"}, {"id": "b"}],
                                                              "email": "me@example.com"}}})

    out = fetch_account(SECRET, rp(handler), "https://rp.test/graphql")
    assert seen == {"auth": f"Bearer {SECRET}", "query": b""}  # header only, nothing in the URL
    assert out == {"balance": 12.5, "spend_per_hr": 0.4, "spend_limit_per_hr": 80, "pod_count": 2}


def test_fetch_account_errors_never_contain_the_key():
    with pytest.raises(RunPodError, match="HTTP 401: Unauthorized") as e:
        fetch_account(SECRET, rp(lambda r: httpx.Response(401, json={"errors": [{"message": "Unauthorized"}]})))
    assert SECRET not in str(e.value)

    with pytest.raises(RunPodError, match="HTTP 200: Not allowed"):  # GraphQL reports errors with a 200 too
        fetch_account(SECRET, rp(lambda r: httpx.Response(200, json={"errors": [{"message": "Not allowed"}]})))

    with pytest.raises(RunPodError, match="HTTP 502: no details"):
        fetch_account(SECRET, rp(lambda r: httpx.Response(502, text="bad gateway")))

    def boom(request):
        raise httpx.ConnectError(f"cannot reach {request.url}")

    with pytest.raises(RunPodError) as e:
        fetch_account(SECRET, rp(boom))
    assert SECRET not in str(e.value) and SECRET not in repr(e.value.__cause__)


def test_account_endpoint(client, monkeypatch):
    from app.api import jobs as jobs_api

    monkeypatch.setattr(settings, "runpod_api_key", None)
    assert client.get("/runpod/account").status_code == 503

    monkeypatch.setattr(settings, "runpod_api_key", SECRET)
    monkeypatch.setattr(jobs_api, "fetch_runpod_account", lambda key: {"balance": 0.0, "spend_per_hr": 0.0,
                        "spend_limit_per_hr": 80.0, "pod_count": 0})
    assert client.get("/runpod/account").json() == {"balance": 0.0, "spend_per_hr": 0.0, "spend_limit_per_hr": 80.0, "pod_count": 0}

    def fail(key):
        raise RunPodError("RunPod HTTP 401: Unauthorized")

    monkeypatch.setattr(jobs_api, "fetch_runpod_account", fail)
    r = client.get("/runpod/account")
    assert r.status_code == 502 and "Unauthorized" in r.json()["detail"] and SECRET not in r.text
