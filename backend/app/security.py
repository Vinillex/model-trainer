"""Who may call what once the backend listens on the LAN so workers can reach it.

The API has no user accounts, so: requests from this machine (the browser, Next.js) are trusted as before;
any other client may only reach /workers/*, and there it must present the worker token.
"""

import ipaddress
import secrets

from fastapi import Header, HTTPException
from starlette.responses import JSONResponse

from .config import settings


def is_loopback(host: str | None) -> bool:
    try:
        ip = ipaddress.ip_address(host or "")
    except ValueError:
        return False
    return (ip.ipv4_mapped or ip).is_loopback if ip.version == 6 else ip.is_loopback


# A tunnel client on this machine (cloudflared, an ssh -R) connects to us from 127.0.0.1, so an internet request through it
# would look local. These headers are added by proxies and tunnels, never by a browser talking to localhost, so their
# presence alone means "not this machine". Only the presence is checked, never the value: a caller can write any value.
PROXY_HEADERS = frozenset({b"x-forwarded-for", b"x-forwarded-host", b"x-forwarded-proto", b"forwarded", b"x-real-ip",
                           b"cf-connecting-ip", b"cf-ray", b"cf-visitor", b"cdn-loop", b"true-client-ip"})


def is_local_request(scope) -> bool:
    if not is_loopback((scope.get("client") or (None,))[0]):
        return False
    return not any(name.lower() in PROXY_HEADERS for name, _ in scope.get("headers") or [])


class LoopbackOnly:
    """ASGI middleware: clients that aren't this machine (or that arrive through a proxy or tunnel) get 403 on everything
    except /workers/*."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not is_local_request(scope) and not scope["path"].startswith("/workers/"):
            response = JSONResponse({"detail": "Only this machine may use this endpoint; workers use /workers/*."}, status_code=403)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def require_token(x_worker_token: str | None = Header(None)) -> None:
    if not settings.worker_token:
        raise HTTPException(503, "The worker API is disabled: set MT_WORKER_TOKEN in backend/.env and restart the backend")
    if not x_worker_token or not secrets.compare_digest(x_worker_token, settings.worker_token):
        raise HTTPException(401, "Missing or invalid worker token")


def worker_name(x_worker_name: str = Header(..., min_length=1, max_length=120)) -> str:
    return x_worker_name
