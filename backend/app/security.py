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


class LoopbackOnly:
    """ASGI middleware: non-loopback clients get 403 on everything except /workers/*."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and not is_loopback((scope.get("client") or (None,))[0]) \
                and not scope["path"].startswith("/workers/"):
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
