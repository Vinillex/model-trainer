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
