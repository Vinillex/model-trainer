import httpx
import pytest

from app.config import settings
from app.runpod_client import RunPodError, create_pod, delete_pod, fetch_account, fetch_gpus, get_pod, list_pods

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


# ---------- pods (REST) and GPU prices ----------

REST = "https://rest.test/v1"


def test_rest_calls_send_the_key_in_a_header_only_and_decode_the_answers():
    seen = []

    def handler(request):
        seen.append((request.method, str(request.url), request.headers["authorization"]))
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.method == "POST":
            return httpx.Response(201, json={"id": "p1", "costPerHr": 0.27})
        return httpx.Response(200, json=[{"id": "p1"}] if request.url.path == "/v1/pods" else {"id": "p1", "desiredStatus": "RUNNING"})

    http = rp(handler)
    assert create_pod(SECRET, {"name": "mt-x"}, http, REST) == {"id": "p1", "costPerHr": 0.27}
    assert list_pods(SECRET, http, REST) == [{"id": "p1"}]
    assert get_pod(SECRET, "p1", http, REST)["desiredStatus"] == "RUNNING"
    assert delete_pod(SECRET, "p1", http, REST) is True
    assert [(m, u) for m, u, _ in seen] == [("POST", f"{REST}/pods"), ("GET", f"{REST}/pods"), ("GET", f"{REST}/pods/p1"),
                                            ("DELETE", f"{REST}/pods/p1")]
    assert all(auth == f"Bearer {SECRET}" for _, _, auth in seen) and not any(SECRET in u for _, u, _ in seen)


def test_a_pod_that_is_already_gone_is_not_an_error_but_other_failures_are():
    gone = rp(lambda r: httpx.Response(404, json={"error": "pod not found"}))
    assert get_pod(SECRET, "p1", gone, REST) is None and delete_pod(SECRET, "p1", gone, REST) is False

    with pytest.raises(RunPodError, match="HTTP 400: no GPUs available") as e:
        create_pod(SECRET, {}, rp(lambda r: httpx.Response(400, json={"error": "no GPUs available"})), REST)
    assert SECRET not in str(e.value)
    with pytest.raises(RunPodError, match="HTTP 401: no details"):
        delete_pod(SECRET, "p1", rp(lambda r: httpx.Response(401, text="nope")), REST)

    def boom(request):
        raise httpx.ReadTimeout(f"timed out talking to {request.url}")

    with pytest.raises(RunPodError) as e:
        create_pod(SECRET, {}, rp(boom), REST)
    assert SECRET not in str(e.value) and SECRET not in repr(e.value.__cause__) and "ReadTimeout" in str(e.value)


def test_gpu_prices_are_zero_where_a_cloud_does_not_offer_the_gpu():
    rows = [{"id": "NVIDIA RTX A4000", "displayName": "RTX A4000", "memoryInGb": 16, "secureCloud": True, "communityCloud": True,
             "securePrice": 0.25, "communityPrice": 0.17, "lowestPrice": {"stockStatus": "Low"}},
            {"id": "Tesla V100-PCIE-16GB", "displayName": "Tesla V100", "memoryInGb": 16, "secureCloud": False, "communityCloud": True,
             "securePrice": 0, "communityPrice": 0.19, "lowestPrice": {"stockStatus": None}}]
    out = fetch_gpus(SECRET, rp(lambda r: httpx.Response(200, json={"data": {"gpuTypes": rows}})), "https://rp.test/graphql")
    assert out[0] == {"id": "NVIDIA RTX A4000", "name": "RTX A4000", "memory_gb": 16, "secure_price": 0.25, "community_price": 0.17, "stock": "Low"}
    assert out[1]["secure_price"] == 0 and out[1]["community_price"] == 0.19 and out[1]["stock"] is None
