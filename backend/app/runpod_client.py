import httpx

from .config import settings


class RunPodError(Exception):
    pass


ACCOUNT_QUERY = "query { myself { clientBalance currentSpendPerHr spendLimit pods { id } } }"


def graphql(api_key: str, query: str, http: httpx.Client | None = None, url: str | None = None) -> dict:
    """POST a GraphQL query to RunPod and return its `data`. The key goes in a header, never a URL."""
    http = http or httpx.Client(timeout=30)
    try:
        resp = http.post(url or settings.runpod_graphql_url, json={"query": query},
                         headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as e:
        raise RunPodError(f"RunPod request failed ({type(e).__name__})") from None
    try:
        body = resp.json()
    except ValueError:
        body = {}
    errors = body.get("errors") if isinstance(body, dict) else None
    if resp.status_code != 200 or errors or not isinstance(body, dict):
        first = errors[0] if isinstance(errors, list) and errors else None
        detail = first.get("message") if isinstance(first, dict) else None
        raise RunPodError(f"RunPod HTTP {resp.status_code}: {detail or 'no details'}")
    return body.get("data") or {}


def fetch_account(api_key: str, http: httpx.Client | None = None, url: str | None = None) -> dict:
    """Credit balance and spend. Read-only and free; nothing here starts a pod or an endpoint."""
    me = graphql(api_key, ACCOUNT_QUERY, http, url).get("myself") or {}
    return {"balance": me.get("clientBalance"), "spend_per_hr": me.get("currentSpendPerHr"),
            "spend_limit_per_hr": me.get("spendLimit"), "pod_count": len(me.get("pods") or [])}


# RunPod blocks GraphQL introspection, so these field names come from its documentation (and were checked against the live API).
GPU_QUERY = ("query { gpuTypes { id displayName memoryInGb secureCloud communityCloud securePrice communityPrice "
             "lowestPrice(input: {gpuCount: 1}) { stockStatus } } }")


def fetch_gpus(api_key: str, http: httpx.Client | None = None, url: str | None = None) -> list[dict]:
    """Every GPU type with its hourly price in each cloud. Read-only and free. A price of 0 means "not offered there"."""
    rows = graphql(api_key, GPU_QUERY, http, url).get("gpuTypes") or []
    return [{"id": r["id"], "name": r.get("displayName") or r["id"], "memory_gb": r.get("memoryInGb") or 0,
             "secure_price": (r.get("securePrice") or 0) if r.get("secureCloud") else 0,
             "community_price": (r.get("communityPrice") or 0) if r.get("communityCloud") else 0,
             "stock": (r.get("lowestPrice") or {}).get("stockStatus")} for r in rows if r.get("id")]


# ---- pods, through RunPod's REST API (https://rest.runpod.io/v1) ----

def rest(api_key: str, method: str, path: str, body: dict | None = None, http: httpx.Client | None = None,
         url: str | None = None):
    """Call the REST API and return its JSON (None for a 204). The key goes in a header only, and never into an error."""
    http = http or httpx.Client(timeout=30)
    try:
        resp = http.request(method, (url or settings.runpod_rest_url) + path, json=body,
                            headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError as e:
        raise RunPodError(f"RunPod request failed ({type(e).__name__})") from None
    if resp.status_code == 204:
        return None
    try:
        data = resp.json()
    except ValueError:
        data = None
    if not 200 <= resp.status_code < 300:
        detail = None
        if isinstance(data, dict):
            detail = data.get("error") or data.get("message")
        raise RunPodError(f"RunPod HTTP {resp.status_code}: {str(detail)[:300] if detail else 'no details'}")
    return data


def create_pod(api_key: str, spec: dict, http: httpx.Client | None = None, url: str | None = None) -> dict:
    """Start a pod. THIS COSTS MONEY from the moment it runs; callers must have made sure it will be deleted."""
    return rest(api_key, "POST", "/pods", spec, http, url)


def get_pod(api_key: str, pod_id: str, http: httpx.Client | None = None, url: str | None = None) -> dict | None:
    """The pod, or None once RunPod no longer has it."""
    try:
        return rest(api_key, "GET", f"/pods/{pod_id}", None, http, url)
    except RunPodError as e:
        if "HTTP 404" in str(e):
            return None
        raise


def list_pods(api_key: str, http: httpx.Client | None = None, url: str | None = None) -> list[dict]:
    return rest(api_key, "GET", "/pods", None, http, url) or []


def delete_pod(api_key: str, pod_id: str, http: httpx.Client | None = None, url: str | None = None) -> bool:
    """Terminate a pod for good. False if it was already gone, so calling this twice is harmless."""
    try:
        rest(api_key, "DELETE", f"/pods/{pod_id}", None, http, url)
        return True
    except RunPodError as e:
        if "HTTP 404" in str(e):
            return False
        raise
