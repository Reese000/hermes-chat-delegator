"""Hermes Chat Delegator — MCP server for interacting with GUI chats.

Connects to the running hermes serve backend via HTTP REST and WebSocket.
Read operations (list, get, messages) work via HTTP.
Mutations (rename, delete) work via HTTP PATCH/DELETE.
Send prompt works via WebSocket JSON-RPC.

Config (env vars):
  HERMES_GUI_PORT         — Port of the running backend (default: auto-discover)
  HERMES_GUI_HOST         — Host (default: 127.0.0.1)
  HERMES_DASHBOARD_SESSION_TOKEN — Session token for authenticating with the backend.
                           Read automatically if set by the desktop shell.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from functools import partial
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlencode

import httpx
import psutil
import websockets
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)

# Kanban columns that only ever grow; compact board summaries count them
# instead of listing them unless a status filter asks for one explicitly.
_TERMINAL_COLUMNS = frozenset({"done", "archived"})

# Hermes' GET /api/sessions rejects limit>100 with a bodyless HTTP 422.
# Verified 2026-09-04 against a live backend by bisecting the query string:
# limit<=100 -> 200, limit>=101 -> 422, while min_messages/order are always
# accepted.  The 422 carries no body, so an unclamped request fails with a
# status code and nothing else to explain it.  Every caller must clamp
# rather than send its own constant, which is how fork_chat's hardcoded
# limit=500 came to fail 100% of the time.
MAX_SESSIONS_LIMIT = 100


# ---------------------------------------------------------------------------
# Tier 1: Persistent Draft Sessions
# ---------------------------------------------------------------------------

@dataclass
class DraftSessionRecord:
    """Persistent draft session metadata."""
    session_id: str
    live_session_id: str
    created_at: float
    title: Optional[str] = None

    def __getitem__(self, key: str):
        """Support dict-like access for backwards compatibility with tests."""
        return getattr(self, key)


class PersistentDraftSessions:
    """Manage draft sessions with disk persistence.

    Draft sessions (created without initial_message) must survive MCP server
    restarts so agents can bootstrap them on the first send_prompt. Stored
    in ~/.hermes/mcp_draft_sessions.json as JSON for easy inspection/recovery.

    Thread-safe (via asyncio.Lock) but not process-safe. Reads/writes are
    atomic via temp-file + rename.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or Path.home() / ".hermes" / "mcp_draft_sessions.json"
        self._lock = asyncio.Lock()
        self._cache: dict[str, DraftSessionRecord] = {}
        self._load_sync()

    def _load_sync(self) -> None:
        """Load from disk at startup (synchronous, called from __init__)."""
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self._cache = {
                    k: DraftSessionRecord(**v) for k, v in data.items()
                }
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(
                    "Failed to load draft sessions from %s: %s — starting fresh",
                    self.path, e,
                )
                self._cache = {}

    def __contains__(self, session_id: str) -> bool:
        """Check if a draft session exists (synchronous for test compatibility)."""
        return session_id in self._cache

    def __getitem__(self, session_id: str) -> DraftSessionRecord:
        """Get a draft session record synchronously (for test compatibility)."""
        return self._cache[session_id]

    def __setitem__(self, session_id: str, record: Union[DraftSessionRecord, dict]) -> None:
        """Set a draft session record synchronously (for test compatibility)."""
        if isinstance(record, dict):
            record = DraftSessionRecord(session_id=session_id, **record)
        self._cache[session_id] = record

    async def get(self, session_id: str) -> Optional[DraftSessionRecord]:
        """Get a draft session record."""
        async with self._lock:
            return self._cache.get(session_id)

    async def add(self, record: DraftSessionRecord) -> None:
        """Add and persist a draft session record."""
        async with self._lock:
            self._cache[record.session_id] = record
            logger.debug("Added draft session %s (live_id=%s)", record.session_id, record.live_session_id)
            await self._save()

    async def remove(self, session_id: str) -> None:
        """Remove and persist a draft session record."""
        async with self._lock:
            self._cache.pop(session_id, None)
            await self._save()

    async def _save(self) -> None:
        """Atomically persist cache to disk."""
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: asdict(v) for k, v in self._cache.items()}
            temp_path = self.path.with_suffix(".tmp")
            temp_path.write_text(json.dumps(data, indent=2))
            temp_path.replace(self.path)
            logger.debug("Saved %d draft sessions to %s", len(self._cache), self.path)
        except Exception as e:
            logger.error("Failed to save draft sessions to %s: %s", self.path, e, exc_info=True)


def _read_cached_port(path: Path) -> Optional[tuple[str, int]]:
    """Read the last-known-good (host, port) from disk, or None if absent/bad.

    The full ``auto_discover_backend`` sweep costs ~15s (netstat + two
    concurrent probe rounds, one of them after a token refresh) — measured
    against Hermes's own launcher, this routinely exceeds the 15s
    ``connect_timeout`` Hermes applies when spawning this server, so the
    process gets killed just as (or just before) it finishes discovering.
    The backend's port is stable for the life of a Hermes session, and this
    server is relaunched by Hermes many times a day, so caching the last
    successful port and re-verifying it with the existing single-port
    ``_port_answers`` probe (bounded at ~2s) turns the common case into a
    sub-2-second startup instead of a 15s one.
    """
    try:
        data = json.loads(path.read_text())
        host, port = data["host"], int(data["port"])
        return (host, port)
    except Exception:
        return None


def _write_cached_port(path: Path, host: str, port: int) -> None:
    """Atomically persist the last-known-good (host, port) to disk."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(json.dumps({"host": host, "port": port}))
        temp_path.replace(path)
    except Exception as e:
        logger.warning("Failed to cache backend port to %s: %s", path, e)


# ---------------------------------------------------------------------------
# Tier 4: Metrics & Structured Logging
# ---------------------------------------------------------------------------

@dataclass
class MetricsSnapshot:
    """Point-in-time metrics snapshot."""
    timestamp: float = field(default_factory=time.time)
    sessions_created: int = 0
    sessions_deleted: int = 0
    prompts_sent: int = 0
    prompts_succeeded: int = 0
    prompts_timed_out: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    ws_reconnects: int = 0
    ws_auth_failures: int = 0
    http_errors: int = 0


class MetricsCollector:
    """Collect and report operational metrics.

    Tracks request volumes, error rates, and latency. Snapshots are
    point-in-time; metrics are reset on snapshot() call.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._counts = {
            "sessions_created": 0,
            "sessions_deleted": 0,
            "prompts_sent": 0,
            "prompts_succeeded": 0,
            "prompts_timed_out": 0,
            "bytes_sent": 0,
            "bytes_received": 0,
            "ws_reconnects": 0,
            "ws_auth_failures": 0,
            "http_errors": 0,
        }

    async def increment(self, metric: str, by: int = 1) -> None:
        """Increment a metric counter."""
        async with self._lock:
            if metric in self._counts:
                self._counts[metric] += by
            else:
                logger.warning("Unknown metric: %s", metric)

    async def snapshot(self) -> MetricsSnapshot:
        """Return current metrics and reset counters."""
        async with self._lock:
            snap = MetricsSnapshot(**self._counts)
            self._counts = {k: 0 for k in self._counts}
            return snap

# The `websockets` client library defaults `max_size` to 2**20 (1,048,576
# bytes) — any single incoming frame larger than that kills the connection
# with close code 1009 ("message too big") and no further recovery. Hermes'
# session.resume response ships the ENTIRE session history in one frame, so
# any session whose stored transcript exceeds ~1MB becomes permanently
# unpromptable: every future send_prompt's resume step dies the same way,
# forever (this was defect #2 — confirmed via hermes' own server-side log:
# "ws closed ... reason=client_disconnect(code=1009,reason=frame exceeds
# limit of 1048576 bytes)"). Raise the client-side ceiling well past any
# realistic session transcript. Configurable via env var for sites with
# unusually large sessions or tighter memory constraints; 0/negative means
# "no limit" (matches `websockets`' own convention for max_size=None).
def _resolve_ws_max_size() -> Optional[int]:
    raw = os.environ.get("HERMES_WS_MAX_SIZE", "")
    if raw.strip():
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Ignoring non-integer HERMES_WS_MAX_SIZE=%r", raw)
        else:
            return value if value > 0 else None
    return 64 * 1024 * 1024  # 64 MiB default — generous headroom, still bounded


_WS_MAX_SIZE: Optional[int] = _resolve_ws_max_size()


async def _recv_response(ws, expected_id: int, timeout: float) -> dict:
    """Read WS frames until the JSON-RPC reply with id == expected_id.

    The same socket also carries unsolicited `event` broadcasts for any live
    session (e.g. another GUI tab's traffic), which don't carry our request
    id. A bare `ws.recv()` can return one of those instead of our reply,
    causing a KeyError on "result" downstream — skip anything that isn't
    our numbered response instead of assuming the first frame is it.
    """
    async def _loop():
        while True:
            resp = json.loads(await ws.recv())
            if resp.get("id") == expected_id:
                return resp
    return await asyncio.wait_for(_loop(), timeout=timeout)


# ---------------------------------------------------------------------------
# Backend connection
# ---------------------------------------------------------------------------

_SESSION_HEADER_NAME = "X-Hermes-Session-Token"



# Transport-level hiccups that a bare immediate retry reliably clears: a
# one-off refused/reset connection right after the backend restarts (the
# listening socket exists but the ASGI app isn't fully up yet), or a read
# reset on a connection the OS half-closed. Deliberately narrow — this must
# never swallow real application errors (4xx/5xx, timeouts long enough to
# indicate a genuinely stuck backend get their own handling upstream).
_TRANSIENT_HTTP_ERRORS = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.RemoteProtocolError,
)


class WsAuthFailure(RuntimeError):
    """The WebSocket upgrade was rejected for authentication reasons.

    This is the single most common way the delegator "goes down" while
    looking perfectly healthy, and it is worth stating precisely because the
    symptom points away from the cause.

    Both the backend port and its session token are *per-boot* values: the
    desktop shell launches the backend with ``--port 0`` and
    ``hermes_cli.web_server`` mints ``_SESSION_TOKEN`` with
    ``secrets.token_urlsafe(32)`` on every start.  The delegator captures the
    token once, at process start, and pins it into ``os.environ``.  After the
    backend restarts, that pinned token is stale — and the two halves of the
    API disagree about whether that matters:

      * ``/api/sessions`` on a headless ``hermes serve`` backend is
        **unauthenticated**, so it returns 200 for a stale token exactly as
        it would for a valid one.  ``auto_discover_backend`` therefore
        confirms the backend, and ``backend_status`` reports
        ``connected: true``.
      * ``/api/ws`` **does** validate, and rejects a stale token with 403.

    Verified directly against the live backend: no token -> connects;
    bogus token -> ``HTTP 403``.  So a stale token is strictly worse than no
    token at all, which is why the recovery path below drops it.
    """

    def __init__(self, port: int, status: Optional[int], detail: str = ""):
        self.port = port
        self.status = status
        self.detail = detail
        super().__init__(
            f"WebSocket upgrade to Hermes backend on port {port} was rejected "
            f"(HTTP {status if status is not None else '?'}) after re-discovering "
            "the session token and retrying without one. The backend is "
            "answering REST, so this is an auth/token problem, not an outage. "
            "If it persists, restart Hermes so a fresh token is minted."
            + (f" [{detail}]" if detail else "")
        )


class BackendAmbiguous(RuntimeError):
    """Multiple live backend instances detected with indistinguishable recency.

    This is NOT an outage — Hermes is running, but we cannot determine
    which instance is the "live" one.  The caller must NOT silently pick
    one, because a wrong choice invites destructive recovery against
    healthy work.

    ``ports`` lists the candidate ports so the user can shut down the
    extras.
    """

    def __init__(self, ports: list[int]):
        self.ports = sorted(ports)
        super().__init__(
            f"Multiple Hermes backend instances detected on ports "
            f"{', '.join(str(p) for p in self.ports)} with indistinguishable "
            "recency. Cannot determine which is live. Shut down the extra "
            "instance(s) and retry."
        )


# Upgrade rejections that indicate an auth problem rather than a transient
# transport blip. These are the ones worth re-discovering a token for.
_WS_AUTH_STATUSES = (401, 403)


def _ws_status_of(exc: BaseException) -> Optional[int]:
    """Best-effort HTTP status from a websockets upgrade failure.

    The `websockets` library has moved this attribute across versions
    (``InvalidStatusCode.status_code`` in <=11, ``InvalidStatus.response.status_code``
    in >=12), so probe both shapes and fall back to scraping the message
    rather than raising a different error while trying to describe this one.
    """
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    if isinstance(status, int):
        return status
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    m = re.search(r"HTTP (\d{3})", str(exc))
    return int(m.group(1)) if m else None


def _ws_url_for(backend: "GuiBackend", token: Optional[str]) -> str:
    """Build the /api/ws URL using an explicit token (or none)."""
    base = f"ws://{backend.host}:{backend.port}/api/ws"
    if token:
        base += "?" + urlencode({"token": token})
    return base


def _ws_connect(backend: "GuiBackend"):
    """``websockets.connect`` for a backend, with token self-healing.

    On an auth rejection this does what a human operator would do, in order,
    without needing the MCP client process to be restarted:

      1. Re-discover the session token from the live backend process and
         retry.  This repairs the common case — the backend restarted and
         minted a new token — which previously required restarting Claude
         Code itself, because nothing in-process ever refreshed the value.
      2. Failing that, retry with **no** token at all.  A stale token is
         actively rejected while an absent one is simply not checked on
         backends that do not require auth, so dropping it strictly widens
         the set of backends we can talk to.
      3. Only if all three attempts fail, raise :class:`WsAuthFailure`,
         which the handler layer reports as retryable and which resets the
         cached backend so the next call re-discovers from scratch.

    A successful repair is written back onto the backend object, so the very
    next WS call goes straight through instead of paying for discovery again.
    """

    class _Ctx:
        async def __aenter__(self):
            attempts: list[Optional[str]] = [backend.token]

            last_exc: Optional[BaseException] = None
            last_status: Optional[int] = None
            tried_refresh = False

            while attempts:
                token = attempts.pop(0)
                try:
                    self._cm = websockets.connect(
                        _ws_url_for(backend, token), max_size=_WS_MAX_SIZE
                    )
                    conn = await self._cm.__aenter__()
                    if token != backend.token:
                        # Persist the repair for subsequent calls.
                        logger.info(
                            "WS auth repaired for %s:%s (token %s)",
                            backend.host, backend.port,
                            "refreshed" if token else "dropped",
                        )
                        backend.token = token or ""
                    return conn
                except Exception as exc:
                    status = _ws_status_of(exc)
                    if status not in _WS_AUTH_STATUSES:
                        raise
                    last_exc, last_status = exc, status

                    if not tried_refresh:
                        tried_refresh = True
                        try:
                            from .token_discovery import refresh_session_token

                            fresh = refresh_session_token()
                        except Exception:
                            fresh = None
                        # Only worth another round-trip if it differs from
                        # what we just proved does not work.
                        if fresh and fresh != token:
                            attempts.append(fresh)
                        attempts.append(None)  # tokenless last resort

            raise WsAuthFailure(
                backend.port, last_status,
                type(last_exc).__name__ if last_exc else "",
            ) from last_exc

        async def __aexit__(self, *exc_info):
            return await self._cm.__aexit__(*exc_info)

    return _Ctx()


class KanbanApiError(RuntimeError):
    """Kanban plugin HTTP error (404 = disabled, 4xx/5xx = API error)."""

    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(detail)


class GuiBackend:
    """Client for the Hermes GUI backend.

    Manages multiple concurrent chat sessions via HTTP REST (reads/deletes)
    and WebSocket JSON-RPC (sends). Automatically discovers the backend
    port via netstat scanning and recovers from token staleness on restarts.

    Features:
      - Persistent draft session storage (survives MCP server restarts)
      - Connection pooling for HTTP requests
      - Metrics collection for monitoring and debugging
      - Structured error logging with recovery hints
      - Event-based prompt streaming (keep WebSocket open)
    """

    def __init__(self, host: str, port: int, token: str = ""):
        self.host = host
        self.port = port
        self.token = token
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=15, write=5, pool=5),
            headers=self._auth_headers(),
        )
        # Persistent draft sessions — survive MCP server restarts
        self._draft_sessions = PersistentDraftSessions()
        # Metrics collection for observability
        self._metrics = MetricsCollector()

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.token:
            headers[_SESSION_HEADER_NAME] = self.token
        return headers

    def _refresh_client_headers(self) -> None:
        """Rebuild the HTTP client's default headers from the current token.

        Called after a token refresh so REST calls immediately use the new
        credential without recreating the entire client (which would drop
        the connection pool).
        """
        self._client.headers.update(self._auth_headers())

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        base = f"ws://{self.host}:{self.port}/api/ws"
        # /api/ws authenticates via a ?token= query param, not the
        # X-Hermes-Session-Token header the REST client uses — the header
        # is silently ignored on the WS upgrade and the handshake 403s.
        if self.token:
            base += "?" + urlencode({"token": self.token})
        return base

    async def aclose(self):
        await self._client.aclose()

    async def _request(self, method: str, url: str, *, retries: int = 2,
                       retry_delay: float = 0.4, **kwargs) -> httpx.Response:
        """Issue an HTTP request, retrying transient transport failures.

        Observed in a 12h orchestration session: get_messages polls
        occasionally hit a one-off "not_connected" (httpx.ConnectError) that
        an immediate retry always cleared — the backend was up throughout,
        it was a momentary accept-queue/socket blip, not a real outage
        (defect #4). Retrying HERE, inside the backend client, means every
        caller (list/get/messages/rename/delete, and the send_prompt polling
        loop) gets the resilience automatically instead of each call site
        needing its own retry loop. Does NOT retry HTTP error responses
        (4xx/5xx) — those are real answers from a reachable server, not
        transport failures — nor does it retry timeouts (a slow backend
        should surface as slow, not silently multiply the wait).
        """
        last_exc: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                return await self._client.request(method, url, **kwargs)
            except _TRANSIENT_HTTP_ERRORS as e:
                last_exc = e
                if attempt < retries:
                    logger.debug(
                        "Transient %s on %s %s (attempt %d/%d) — retrying",
                        type(e).__name__, method, url, attempt + 1, retries,
                    )
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    continue
        assert last_exc is not None
        raise last_exc

    async def health(self) -> dict:
        r = await self._request("GET", f"{self.base_url}/health")
        return r.json()

    async def is_alive(self) -> bool:
        """Authenticated liveness probe — the ONLY reliable check.

        Probes ``/api/sessions`` with the backend's own token.  Returns
        True only if the response is HTTP 200 with a ``"sessions"`` key.

        This answers three questions in one request:
          1. Is something listening on this port?
          2. Is it a Hermes backend (not Foundry Manager, etc.)?
          3. Does our token authenticate against *this* instance?

        ``/health`` cannot do this: some Hermes instances (e.g. headless
        ``hermes serve``) do not expose ``/health`` at all, while unrelated
        services can expose Hermes-shaped ``/health`` banners.  The
        authenticated operation itself is the only probe that cannot be
        fooled by an unimplemented or over-implemented endpoint.
        """
        try:
            r = await self._request(
                "GET", f"{self.base_url}/api/sessions",
                params={"limit": 1, "min_messages": 0, "order": "recent"},
            )
            if r.status_code != 200:
                return False
            body = r.json()
            return isinstance(body, dict) and "sessions" in body
        except Exception:
            return False

    async def list_sessions(self, limit: int = 50) -> list[dict]:
        """List recent sessions, newest first.

        ``limit`` is clamped to ``MAX_SESSIONS_LIMIT``: the backend answers
        anything above it with a bodyless 422, so passing a caller's number
        through unchecked turns "you asked for more than we serve" into an
        opaque internal error.  Clamping degrades to fewer rows, which every
        caller here can tolerate, instead of returning nothing at all.
        """
        r = await self._request(
            "GET", f"{self.base_url}/api/sessions",
            params={
                "limit": min(max(int(limit), 1), MAX_SESSIONS_LIMIT),
                "min_messages": 0,
                "order": "recent",
            },
        )
        r.raise_for_status()
        return r.json().get("sessions", [])

    async def _list_all(self) -> list[dict]:
        """List the most recent sessions the backend will serve in one page.

        Formerly requested ``limit=500``, which the backend rejects with a
        bodyless 422 — that made ``fork_chat`` fail every single call *after*
        its WebSocket ``session.branch`` had already succeeded, so the fork
        was created and then orphaned behind an error.

        The only caller resolves a session created moments earlier, and the
        listing is ordered ``recent``, so the row being looked for is at the
        very front of the first page; one clamped page is sufficient.
        """
        return await self.list_sessions(limit=MAX_SESSIONS_LIMIT)

    async def get_session(self, session_id: str) -> dict:
        r = await self._request("GET", f"{self.base_url}/api/sessions/{session_id}")
        r.raise_for_status()
        return r.json()

    async def get_messages(self, session_id: str, last_n: int = 20) -> list[dict]:
        r = await self._request("GET", f"{self.base_url}/api/sessions/{session_id}/messages")
        r.raise_for_status()
        messages = r.json().get("messages", [])
        if last_n > 0:
            messages = messages[-last_n:]
        return messages

    async def list_interrupted_sessions(self) -> list[dict]:
        """Sessions that look like a backend restart/crash killed them
        mid-turn (see hermes' GET /api/sessions/interrupted). Empty list on
        any failure — including talking to a pre-fix backend that doesn't
        have this endpoint yet (404), which must never break callers that
        merely want to know "any interrupted sessions?" as a side note.
        """
        try:
            r = await self._request("GET", f"{self.base_url}/api/sessions/interrupted")
            r.raise_for_status()
            return r.json().get("sessions", [])
        except Exception:
            return []

    async def rename_session(self, session_id: str, title: str) -> dict:
        r = await self._request(
            "PATCH", f"{self.base_url}/api/sessions/{session_id}",
            json={"title": title},
        )
        r.raise_for_status()
        return r.json()

    async def delete_session(self, session_id: str) -> dict:
        r = await self._request("DELETE", f"{self.base_url}/api/sessions/{session_id}")
        r.raise_for_status()
        return r.json()

    async def _poll_for_reply(self, durable_session_id: str, last_assistant_id,
                              timeout: float) -> dict:
        """Poll the REST messages endpoint (durable id only — the live id
        404s, see send_prompt) until a NEW assistant message appears."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            await asyncio.sleep(2)
            try:
                msgs = await self.get_messages(durable_session_id, last_n=5)
                for m in reversed(msgs):
                    if m.get("role") == "assistant":
                        if last_assistant_id is None or m.get("id") != last_assistant_id:
                            return {
                                "response": m.get("content", ""),
                                "session_id": durable_session_id,
                                "message_count": len(msgs),
                                "status": "completed",
                            }
                        break  # Still the old one, keep waiting
            except Exception:
                pass

        # Timed out. Do one last check in case the reply landed in the
        # instant between the final poll and here — but NEVER fall back to
        # echoing last_assistant_id's own (pre-submit) content as if it were
        # the answer to THIS prompt. Doing that used to make a slow turn look
        # like it silently answered with stale/wrong text instead of visibly
        # timing out (defect #5) — a caller that doesn't check `timed_out`
        # would be misled. `response` is only ever populated here when a
        # genuinely NEW assistant message was found; otherwise it is
        # explicitly None with status="pending" so the ambiguity is
        # impossible to miss.
        try:
            msgs = await self.get_messages(durable_session_id, last_n=5)
        except Exception:
            msgs = []
        new_reply = None
        for m in reversed(msgs):
            if m.get("role") == "assistant":
                if last_assistant_id is None or m.get("id") != last_assistant_id:
                    new_reply = m
                break  # Only the most recent message counts either way

        if new_reply is not None:
            return {
                "response": new_reply.get("content", ""),
                "session_id": durable_session_id,
                "message_count": len(msgs),
                "timed_out": True,
                "status": "completed_late",
                "note": (
                    "The reply arrived right at the timeout boundary — "
                    "included here so it isn't lost, but the call did exceed "
                    "the requested timeout."
                ),
            }
        return {
            "response": None,
            "session_id": durable_session_id,
            "message_count": len(msgs),
            "timed_out": True,
            "status": "pending",
            "note": (
                "No new assistant reply yet — the turn may still be running "
                "on the backend. This is NOT a stale echo of a previous "
                "reply: `response` is null on purpose. Poll get_messages() "
                "on this session_id, or call send_prompt again with a "
                "longer timeout, to pick up the eventual answer."
            ),
        }

    async def create_session(self, title: str = "", cwd: str = "", model: str = "",
                             initial_message: str = "", timeout: float = 15,
                             prompt_timeout: float = 300) -> dict:
        """Create a new live session via session.create.

        If initial_message is given, submits it on the SAME connection using
        the live id session.create just returned. This must not be done as a
        separate send_prompt(durable_id, ...) call afterward — a brand-new
        session has no durable row yet (Hermes persists it on first prompt),
        so session.resume-by-durable-id 404s (backend error 4007) until
        after that first submit has happened.

        After sending the prompt, briefly waits for at least one event from the
        backend to keep the session alive. Closing the WS immediately after send
        causes the backend to reap the session as orphaned before the agent starts.
        """
        async with _ws_connect(self) as ws:
            params: dict = {}
            if title:
                params["title"] = title
            if cwd:
                params["cwd"] = cwd
            # Default to mimo-v2.5 (non-pro) if no model specified
            params["model"] = model or "xiaomi/mimo-v2.5"
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "session.create", "params": params,
            }))
            resp = await _recv_response(ws, 1, timeout)
            if "error" in resp:
                raise RuntimeError(f"session.create failed: {resp['error']}")
            result = resp["result"]
            created = {
                "session_id": result.get("stored_session_id", ""),
                "live_session_id": result.get("session_id", ""),
                "title": title or "untitled",
                "cwd": result.get("info", {}).get("cwd", ""),
            }

            if initial_message:
                await ws.send(json.dumps({
                    "jsonrpc": "2.0", "id": 2, "method": "prompt.submit",
                    "params": {"session_id": created["live_session_id"], "text": initial_message},
                }))
                # Keep the connection open briefly so the backend doesn't reap the
                # session as orphaned. Read for up to 500ms to wait for the agent
                # to acknowledge and start processing (at least one event).
                try:
                    await asyncio.wait_for(ws.recv(), timeout=0.5)
                except (asyncio.TimeoutError, Exception):
                    pass  # Connection closing is fine; we're just keeping it alive briefly

        if initial_message:
            reply = await self._poll_for_reply(
                created["session_id"], last_assistant_id=None, timeout=prompt_timeout,
            )
            created["initial_response"] = reply["response"]
            if reply.get("timed_out"):
                created["initial_response_timed_out"] = True
            await self._metrics.increment("sessions_created")
        else:
            # Register as draft: send_prompt will bootstrap the durable row.
            # Store in memory and attempt disk persistence (with error recovery).
            record = DraftSessionRecord(
                session_id=created["session_id"],
                live_session_id=created["live_session_id"],
                created_at=time.time(),
                title=created.get("title"),
            )
            try:
                await self._draft_sessions.add(record)
            except Exception as e:
                logger.warning(
                    "Draft session %s created but disk persistence failed: %s",
                    created["session_id"], e,
                )
            await self._metrics.increment("sessions_created")

        return created

    async def branch_session(self, session_id: str, title: str = "",
                             timeout: float = 15) -> dict:
        """Branch an existing session, copying its full history."""
        async with _ws_connect(self) as ws:
            # Resume the source session to get a live id
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "session.resume",
                "params": {"session_id": session_id},
            }))
            resp = await _recv_response(ws, 1, timeout)
            if "error" in resp:
                raise RuntimeError(f"resume failed: {resp['error']}")
            live_id = resp["result"].get("session_id", session_id)

            # Branch
            params: dict = {"session_id": live_id}
            if title:
                params["name"] = title
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 2, "method": "session.branch", "params": params,
            }))
            resp2 = await _recv_response(ws, 2, timeout)
            if "error" in resp2:
                err = resp2["error"]
                if err.get("code") == 4008:
                    raise RuntimeError(
                        "Cannot fork an empty chat — send at least one message "
                        "in the source session first."
                    )
                raise RuntimeError(f"session.branch failed: {err}")
            result = resp2["result"]

        # Resolve the new durable id by matching parent_session_id
        sessions = await self._list_all()
        for s in sessions:
            if s.get("parent_session_id") == session_id:
                return {
                    "session_id": s["id"],
                    "title": result.get("title"),
                    "forked_from": session_id,
                }
        raise RuntimeError(
            f"session.branch reported success (title={result.get('title')!r}) but no "
            f"session with parent_session_id={session_id!r} found. Backend may have changed."
        )

    async def set_model(self, session_id: str, model: str,
                        timeout: float = 30) -> dict:
        """Switch the model on an existing session via slash.exec /model.

        Resumes the session to get a live id, then sends ``/model <model>``
        through the backend's slash command handler. The backend pins the
        chosen model to this session (model_override) so subsequent prompts
        use the new model.

        If the session is busy (a turn is in-flight), sends /interrupt first
        then retries — the gateway's ``_mirror_slash_side_effects`` refuses
        to mutate a running session.

        On TimeoutError the attempt is retried ONCE with a doubled timeout
        (minimum 30s). Resume + slash.exec on a session that just had a turn
        interrupted frequently exceeds a tight budget; a single retry absorbs
        that without hiding a genuinely stuck backend. A second TimeoutError
        propagates so the tool handler can surface a structured retryable
        ``model_switch_timeout`` error.
        """
        try:
            return await self._set_model_once(session_id, model, timeout)
        except TimeoutError:
            retry_timeout = max(float(timeout) * 2.0, 30.0)
            logger.info(
                "set_model timed out after %.0fs for %s — retrying once with %.0fs",
                timeout, session_id, retry_timeout,
            )
            return await self._set_model_once(session_id, model, retry_timeout)

    async def _set_model_once(self, session_id: str, model: str,
                              timeout: float) -> dict:
        """One set_model attempt over a single WebSocket connection.

        Raises TimeoutError if resume or slash.exec does not reply within
        ``timeout`` seconds. Kept separate from set_model so the automatic
        timeout retry does not re-enter the busy-session interrupt path.
        """
        async with _ws_connect(self) as ws:
            # Resume to get a live session id
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "session.resume",
                "params": {"session_id": session_id},
            }))
            resp = await _recv_response(ws, 1, timeout)
            if "error" in resp:
                raise RuntimeError(f"resume failed: {resp['error']}")
            live_id = resp["result"].get("session_id", session_id)

            # Execute /model command
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 2, "method": "slash.exec",
                "params": {"session_id": live_id, "command": f"/model {model}"},
            }))
            resp2 = await _recv_response(ws, 2, timeout)
            if "error" in resp2:
                err = resp2["error"]
                raise RuntimeError(
                    f"slash.exec /model failed: {err.get('message', err)}"
                )

        result = resp2.get("result", {})
        output = result.get("output", "")
        warning = result.get("warning", "")

        # The slash worker validates and outputs "✓ Model switched" regardless,
        # but the gateway's _mirror_slash_side_effects may refuse to apply the
        # change if the session is busy. Check the warning field for this.
        if "session busy" in warning.lower():
            # Send /interrupt to clear the running flag, then retry
            interrupt_result = await self._interrupt_and_switch(
                session_id, model, timeout
            )
            interrupt_result["interrupted_first"] = True
            return interrupt_result

        # Check for common error patterns in the slash output
        lower_out = output.lower()
        if "could not switch" in lower_out or "not found" in lower_out:
            raise RuntimeError(f"Model switch failed: {output}")
        # Also treat gateway-side sync failures as warnings (non-fatal — the
        # worker DID save to config.yaml, so the model is globally switched;
        # the per-session override just didn't stick).
        if warning and "sync failed" in warning.lower():
            return {
                "session_id": session_id,
                "model": model,
                "output": output,
                "warning": warning,
                "status": "partial",
                "note": (
                    "Model saved to config.yaml (global default changed), but "
                    "the per-session override could not be applied. The new "
                    "model will take effect on the next session/rebuild."
                ),
            }

        return {
            "session_id": session_id,
            "model": model,
            "output": output,
            "warning": warning or None,
            "status": "ok",
        }

    async def _interrupt_and_switch(self, session_id: str, model: str,
                                    timeout: float = 30) -> dict:
        """Interrupt a busy session then retry the model switch.

        Best-effort: the interrupt reply is not required (the turn may already
        be finished by the time we send it), so a timeout on interrupt is
        swallowed. The subsequent slash.exec still has the full ``timeout``
        budget.
        """
        async with _ws_connect(self) as ws:
            # Resume
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "session.resume",
                "params": {"session_id": session_id},
            }))
            resp = await _recv_response(ws, 1, timeout)
            if "error" in resp:
                raise RuntimeError(f"resume failed: {resp['error']}")
            live_id = resp["result"].get("session_id", session_id)

            # Interrupt
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 2, "method": "session.interrupt",
                "params": {"session_id": live_id},
            }))
            # Don't wait long for interrupt — it may already be done
            try:
                await _recv_response(ws, 2, timeout=min(timeout, 5))
            except (TimeoutError, Exception):
                pass  # interrupt is best-effort

            # Brief pause for the running flag to clear
            await asyncio.sleep(1)

            # Retry /model
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 3, "method": "slash.exec",
                "params": {"session_id": live_id, "command": f"/model {model}"},
            }))
            resp3 = await _recv_response(ws, 3, timeout)
            if "error" in resp3:
                err = resp3["error"]
                raise RuntimeError(
                    f"slash.exec /model failed after interrupt: "
                    f"{err.get('message', err)}"
                )

        result = resp3.get("result", {})
        output = result.get("output", "")
        warning = result.get("warning", "")

        lower_out = output.lower()
        if "could not switch" in lower_out or "not found" in lower_out:
            raise RuntimeError(f"Model switch failed: {output}")

        return {
            "session_id": session_id,
            "model": model,
            "output": output,
            "warning": warning or None,
            "status": "ok",
        }

    # ---- Kanban plugin HTTP API (mounted at /api/plugins/kanban/) ----------

    async def _kanban_request(self, method: str, path: str, *, retries: int = 1, **kwargs) -> httpx.Response:
        """Issue a request to the kanban plugin API with Bearer token auth.

        The kanban plugin uses the same session token as the rest of the
        backend, but mounted at /api/plugins/kanban/ rather than /api/.
        404 on the kanban path means the plugin is disabled (not a missing route).
        """
        url = f"{self.base_url}/api/plugins/kanban{path}"
        headers = kwargs.pop("headers", {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            resp = await self._request(method, url, headers=headers, retries=retries, **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise KanbanApiError(404, "Kanban plugin is not enabled on this Hermes backend. Enable it via config or restart Hermes with the kanban plugin loaded.")
            raise
        # Check 404 internally too — callers call raise_for_status() which
        # would bypass our catch block when the response is returned normally
        if resp.status_code == 404:
            raise KanbanApiError(404, "Kanban plugin is not enabled on this Hermes backend. Enable it via config or restart Hermes with the kanban plugin loaded.")
        return resp

    async def kanban_get_board(self, *, status: Optional[str] = None, assignee: Optional[str] = None,
                               board: Optional[str] = None, compact: bool = True) -> dict:
        """GET /board — compact column summary (default) or full payload.

        ``status`` and ``assignee`` are applied client-side (the backend's
        /board endpoint ignores them), in both compact and full mode: a task
        is kept when its ``status`` — or, if absent, its column name — equals
        ``status`` (case-insensitive), and its ``assignee`` equals
        ``assignee``.  Previously both parameters were accepted and silently
        dropped, so ``status="running"`` returned the whole board (309 tasks,
        74k characters, 2026-09-29).

        In compact mode with no ``status`` filter, terminal columns
        (``_TERMINAL_COLUMNS``) are summarised as a count only; they grow
        without bound and are almost never what a board query is after.
        Ask for them explicitly (``status="done"``) to list them.
        """
        params: dict = {}
        if board:
            params["board"] = board
        r = await self._kanban_request("GET", "/board", params=params)
        r.raise_for_status()
        data = r.json()

        want_status = status.strip().lower() if isinstance(status, str) and status.strip() else None
        want_assignee = assignee.strip() if isinstance(assignee, str) and assignee.strip() else None

        def _keep(task: dict, col_name: str) -> bool:
            task_status = str(task.get("status") or col_name or "").lower()
            if want_status is not None and task_status != want_status:
                return False
            if want_assignee is not None and task.get("assignee") != want_assignee:
                return False
            return True

        columns = data.get("columns", []) or []

        if not compact:
            if want_status is None and want_assignee is None:
                return data
            filtered = dict(data)
            filtered["columns"] = [
                {**col, "tasks": [t for t in col.get("tasks", []) or []
                                  if _keep(t, col.get("name", ""))]}
                for col in columns
            ]
            return filtered

        result: dict = {"columns": [], "assignees": data.get("assignees", [])}
        summarised: dict = {}
        for col in columns:
            col_name = col.get("name", "")
            kept = [t for t in col.get("tasks", []) or [] if _keep(t, col_name)]
            if not kept:
                continue
            if want_status is None and str(col_name).lower() in _TERMINAL_COLUMNS:
                summarised[col_name] = len(kept)
                continue
            tasks_out = [{
                "id": t.get("id"),
                "title": (t.get("title") or "")[:80],
                "status": t.get("status"),
                "assignee": t.get("assignee"),
                "priority": t.get("priority", 0),
            } for t in kept]
            result["columns"].append({"name": col_name, "tasks": tasks_out, "count": len(tasks_out)})
        if summarised:
            result["summarised_columns"] = summarised
            result["note"] = ("Terminal columns shown as counts only; pass "
                              "status='<column>' to list them.")
        result["total"] = sum(c["count"] for c in result["columns"]) + sum(summarised.values())
        return result

    async def kanban_show_task(self, task_id: str, *, board: Optional[str] = None) -> dict:
        """GET /tasks/{id} — full task detail with comments, runs, events, log tail."""
        params: dict = {}
        if board:
            params["board"] = board
        r = await self._kanban_request("GET", f"/tasks/{task_id}", params=params)
        r.raise_for_status()
        return r.json()

    async def kanban_create_task(self, *, title: str, body: Optional[str] = None,
                                 assignee: Optional[str] = None, priority: int = 0,
                                 parents: Optional[list] = None, skills: Optional[list] = None,
                                 workspace_kind: Optional[str] = None,
                                 workspace_path: Optional[str] = None,
                                 goal_mode: bool = True, goal_max_turns: Optional[int] = None,
                                 model_override: Optional[str] = None,
                                 provider_override: Optional[str] = None,
                                 reasoning_effort: Optional[str] = None,
                                 max_runtime_seconds: Optional[int] = None,
                                 triage: Optional[bool] = None,
                                 hold: bool = False,
                                 idempotency_key: Optional[str] = None,
                                 completion_contract: Optional[str] = None,
                                 acceptance_criteria: Optional[list] = None,
                                 board: Optional[str] = None) -> dict:
        """POST /tasks — create a new kanban task.

        Validates assignee against GET /assignees before creating (the
        dispatcher silently drops unknown assignees). Auto-generates a
        deterministic idempotency_key from title+body+assignee if absent
        to prevent duplicate cards on retries.

        Safety rails (client-side only; the dispatcher and kanban.db are
        unchanged):

          * ``hold=True`` forces ``triage=True`` and clears ``assignee``
            so the dispatcher cannot auto-claim the card. Use for drafts,
            tests, or cards you will assign later.
          * When ``triage`` is left unspecified (None), ``goal_mode=False``,
            and the assignee is empty or ``default``, triage defaults to
            True — the common "create a card I don't want running yet"
            path must not auto-spawn a worker. Pass ``triage=False``
            explicitly to override and allow auto-dispatch.
          * When the create WILL auto-dispatch (known assignee, not
            triage, not hold), the response includes a ``warning`` field.

        completion_contract: GitHub PR-acceptance gate. Accepted values:
          "local-only" (default), "OWNER/REPO", or an exact
          https://github.com/OWNER/REPO/pull/N URL. Absent from the
          CreateTaskBody API — when non-default, uses the hermes CLI.
        acceptance_criteria: List of evidence strings appended as a
          "## Required handoff evidence" section to the body.
        """
        import re as _re

        # --- Client-side safety rails (see docstring) -----------------------
        if hold:
            # Explicit hold: land in triage, unassigned, so nothing can spawn.
            triage = True
            assignee = None
        elif triage is None:
            # Auto default: a non-goal card with no/weak assignee is almost
            # always a draft, not a spawn request.
            if not goal_mode and (not assignee or assignee.strip().lower() == "default"):
                triage = True
            else:
                triage = False
        else:
            triage = bool(triage)

        # Validate completion_contract
        valid_contracts = ("local-only",)
        contract_is_default = (not completion_contract) or completion_contract in valid_contracts
        if completion_contract and not contract_is_default:
            # Must be OWNER/REPO or exact PR URL
            if not (_re.match(r'^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$', completion_contract) or
                    _re.match(r'^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/pull/\d+$', completion_contract)):
                raise ValueError(
                    f"Invalid completion_contract: {completion_contract!r}. "
                    "Must be 'local-only', 'OWNER/REPO', or "
                    "'https://github.com/OWNER/REPO/pull/N'."
                )

        # Build body with acceptance_criteria
        effective_body = body or ""
        if acceptance_criteria:
            criteria_block = "\n\n## Required handoff evidence\n\n"
            for i, criterion in enumerate(acceptance_criteria, 1):
                criteria_block += f"{i}. {criterion}\n"
            effective_body += criteria_block

        # Validate assignee against known profiles
        if assignee:
            try:
                r = await self._kanban_request("GET", "/assignees",
                                               params={"board": board} if board else {})
                r.raise_for_status()
                valid = [a.strip().lower() for a in r.json().get("assignees", [])]
                if assignee.strip().lower() not in valid:
                    raise KanbanApiError(
                        400,
                        f"Unknown assignee '{assignee}'. Valid assignees: {', '.join(valid or ['(none)'])}. "
                        "The dispatcher silently drops tasks with unknown assignees."
                    )
            except KanbanApiError:
                raise
            except Exception:
                pass  # Network/validation failure: let the API decide

        # Auto-generate deterministic idempotency key if not provided
        if not idempotency_key:
            raw = f"{title}|{effective_body or ''}|{assignee or ''}"
            idempotency_key = hashlib.sha256(raw.encode()).hexdigest()[:32]

        # True when the dispatcher can pick this card up as soon as it lands.
        will_auto_dispatch = bool(assignee) and not triage and not hold

        # When completion_contract is non-default, use the CLI path
        if not contract_is_default:
            result = await self._kanban_create_via_cli(
                title=title, body=effective_body, assignee=assignee,
                priority=priority, parents=parents, skills=skills,
                workspace_kind=workspace_kind, workspace_path=workspace_path,
                goal_mode=goal_mode, goal_max_turns=goal_max_turns,
                model_override=model_override, provider_override=provider_override,
                reasoning_effort=reasoning_effort, max_runtime_seconds=max_runtime_seconds,
                triage=triage, idempotency_key=idempotency_key,
                completion_contract=completion_contract, board=board,
            )
        else:
            payload: dict = {
                "title": title, "body": effective_body, "assignee": assignee,
                "priority": priority, "parents": parents or [], "skills": skills,
                "workspace_kind": workspace_kind, "workspace_path": workspace_path,
                "goal_mode": goal_mode, "goal_max_turns": goal_max_turns,
                "model_override": model_override, "provider_override": provider_override,
                "reasoning_effort": reasoning_effort, "max_runtime_seconds": max_runtime_seconds,
                "triage": triage, "idempotency_key": idempotency_key,
            }
            params_q = {"board": board} if board else {}
            r = await self._kanban_request("POST", "/tasks", json=payload, params=params_q)
            r.raise_for_status()
            result = r.json()

        if will_auto_dispatch and isinstance(result, dict) and "warning" not in result:
            # Make the spawn side-effect visible to callers who only meant
            # to create a card. hold=true is the opt-out.
            result = {
                **result,
                "warning": (
                    f"Card will be auto-dispatched to assignee {assignee!r}. "
                    f"Pass hold=true to create without spawning a worker."
                ),
            }
        return result

    async def _kanban_create_via_cli(self, **kwargs) -> dict:
        """Create a kanban task via the hermes CLI for completion_contract support.

        Uses the full path to hermes.exe (resolved via %LOCALAPPDATA%) because
        'hermes' is not on PATH in most environments. Runs asynchronously so
        the event loop is not blocked.
        """
        import os
        hermes_path = os.path.expandvars(r"%LOCALAPPDATA%\hermes\bin\hermes.exe")
        if not os.path.isfile(hermes_path):
            raise RuntimeError(
                f"hermes CLI not found at {hermes_path}. "
                "Cannot create task via CLI for completion_contract."
            )

        # Build argv: hermes kanban [--board SLUG] create TITLE [flags] --json
        cmd = [hermes_path, "kanban"]
        if kwargs.get("board"):
            cmd += ["--board", kwargs["board"]]
        cmd += ["create", kwargs["title"], "--json"]

        if kwargs.get("body"):
            cmd += ["--body", kwargs["body"]]
        if kwargs.get("assignee"):
            cmd += ["--assignee", kwargs["assignee"]]
        if kwargs.get("priority") is not None:
            cmd += ["--priority", str(kwargs["priority"])]
        for p in (kwargs.get("parents") or []):
            cmd += ["--parent", p]
        for s in (kwargs.get("skills") or []):
            cmd += ["--skill", s]
        if kwargs.get("workspace_kind"):
            wk = kwargs["workspace_kind"]
            wp = kwargs.get("workspace_path")
            if wp and wk in ("dir", "worktree"):
                cmd += ["--workspace", f"{wk}:{wp}"]
            else:
                cmd += ["--workspace", wk]
        if kwargs.get("goal_mode"):
            cmd += ["--goal"]
        if kwargs.get("goal_max_turns") is not None:
            cmd += ["--goal-max-turns", str(kwargs["goal_max_turns"])]
        if kwargs.get("model_override"):
            cmd += ["--model", kwargs["model_override"]]
        if kwargs.get("provider_override"):
            cmd += ["--provider", kwargs["provider_override"]]
        if kwargs.get("max_runtime_seconds") is not None:
            cmd += ["--max-runtime", str(kwargs["max_runtime_seconds"])]
        if kwargs.get("triage"):
            cmd += ["--triage"]
        if kwargs.get("idempotency_key"):
            cmd += ["--idempotency-key", kwargs["idempotency_key"]]
        if kwargs.get("completion_contract"):
            cmd += ["--completion-contract", kwargs["completion_contract"]]

        # Run asynchronously so we don't block the event loop
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"hermes kanban create timed out after 30s")

        stdout_str = stdout.decode("utf-8", errors="replace")
        stderr_str = stderr.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"hermes kanban create failed (exit {proc.returncode}): "
                f"{stderr_str.strip() or stdout_str.strip()}"
            )
        try:
            return json.loads(stdout_str)
        except json.JSONDecodeError:
            return {"raw_output": stdout_str.strip()}

    async def kanban_update_task(self, task_id: str, *, board: Optional[str] = None, **fields) -> dict:
        """PATCH /tasks/{id} — update task fields.

        Supported fields: status, assignee, priority, title, body, block_reason,
        archive, summary, metadata, model_override, provider_override.
        """
        params_q = {"board": board} if board else {}
        r = await self._kanban_request("PATCH", f"/tasks/{task_id}", json=fields, params=params_q)
        r.raise_for_status()
        return r.json()

    async def kanban_comment(self, task_id: str, body: str, *,
                             author: str = "delegator", board: Optional[str] = None) -> dict:
        """POST /tasks/{id}/comments — add a comment to a task."""
        payload = {"body": body, "author": author}
        params_q = {"board": board} if board else {}
        r = await self._kanban_request("POST", f"/tasks/{task_id}/comments",
                                       json=payload, params=params_q)
        r.raise_for_status()
        return r.json()

    async def kanban_link(self, parent_id: str, child_id: str, *,
                          board: Optional[str] = None) -> dict:
        """POST /links — create parent→child dependency."""
        payload = {"parent_id": parent_id, "child_id": child_id}
        params_q = {"board": board} if board else {}
        r = await self._kanban_request("POST", "/links", json=payload, params=params_q)
        r.raise_for_status()
        return r.json()

    async def kanban_dispatch(self, *, board: Optional[str] = None,
                              dry_run: bool = False, max_n: int = 8) -> dict:
        """POST /dispatch — nudge the dispatcher to process ready tasks."""
        params_q: dict = {"dry_run": str(dry_run).lower(), "max": str(max_n)}
        if board:
            params_q["board"] = board
        r = await self._kanban_request("POST", "/dispatch", params=params_q)
        r.raise_for_status()
        return r.json()

    async def kanban_stats(self, *, board: Optional[str] = None) -> dict:
        """GET /stats + /workers/active + /diagnostics — combined stats summary."""
        params_q = {"board": board} if board else {}
        results = {}
        for label, path in [("stats", "/stats"), ("workers", "/workers/active"),
                            ("diagnostics", "/diagnostics")]:
            try:
                r = await self._kanban_request("GET", path, params=params_q)
                r.raise_for_status()
                results[label] = r.json()
            except Exception as e:
                results[label] = {"error": _exc_detail(e)}
        return results

    async def kanban_decompose(self, task_id: str, *, author: Optional[str] = None,
                               board: Optional[str] = None) -> dict:
        """POST /tasks/{id}/decompose — fan out a triage task into children."""
        payload = {"author": author}
        params_q = {"board": board} if board else {}
        r = await self._kanban_request("POST", f"/tasks/{task_id}/decompose",
                                       json=payload, params=params_q)
        r.raise_for_status()
        return r.json()

    # ---- Session control: slash commands, /goal, continuous work ----------

    async def run_slash(self, session_id: str, command: str,
                        timeout: float = 30) -> dict:
        """Execute a slash command via WebSocket JSON-RPC.

        Uses "command.dispatch" first (covers quick/plugin/bundle/skill/builtin
        stages). Falls back to "slash.exec" if command.dispatch returns 4018
        (not a dispatchable command — e.g. a plugin command that only works
        through the slash worker path).

        Returns {"output": "...", "ok": true} on success, or error dict.
        """
        async with _ws_connect(self) as ws:
            # Resume session to get live id
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "session.resume",
                "params": {"session_id": session_id},
            }))
            resp = await _recv_response(ws, 1, timeout=15)
            if "error" in resp:
                raise RuntimeError(f"resume failed: {resp['error']}")
            live_id = resp["result"].get("session_id", session_id)

            # Parse command to get name + arg
            cmd_stripped = command.strip().lstrip("/")
            parts = cmd_stripped.split(None, 1)
            name = parts[0] if parts else ""
            arg = parts[1] if len(parts) > 1 else ""

            # Try command.dispatch first
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 2, "method": "command.dispatch",
                "params": {"session_id": live_id, "name": name, "arg": arg},
            }))
            resp2 = await _recv_response(ws, 2, timeout=timeout)
            if "error" not in resp2:
                result = resp2.get("result", {})
                output = result.get("output", "")
                # command.dispatch returns {type: "send", message: ...} for goal-like commands
                if result.get("type") == "send":
                    output = result.get("notice") or output
                return {"output": output or "(no output)", "ok": True, "raw": result}

            # 4018 = not dispatchable via command.dispatch; fall back to slash.exec
            err_code = resp2.get("error", {}).get("code")
            err_msg = resp2.get("error", {}).get("message", "")
            if err_code == 4018:
                await ws.send(json.dumps({
                    "jsonrpc": "2.0", "id": 3, "method": "slash.exec",
                    "params": {"session_id": live_id, "command": f"/{cmd_stripped}"},
                }))
                resp3 = await _recv_response(ws, 3, timeout=timeout)
                if "error" in resp3:
                    slash_err = resp3["error"]
                    slash_code = slash_err.get("code", 0)
                    slash_msg = slash_err.get("message", "")
                    if slash_code == 5030:
                        raise RuntimeError(
                            f"Backend error executing /{cmd_stripped}: {slash_msg}. "
                            "Check the backend logs at "
                            "%LOCALAPPDATA%\\hermes\\logs\\errors.log for details."
                        )
                    raise RuntimeError(f"slash.exec failed: {slash_err}")
                result = resp3.get("result", {})
                return {"output": result.get("output", "") or "(no output)", "ok": True, "raw": result}

            # Other error — including -32603 (backend internal crash)
            if err_code == -32603:
                raise RuntimeError(
                    f"Backend crashed while processing the command (internal error -32603: {err_msg}). "
                    "Check the backend logs at %LOCALAPPDATA%\\hermes\\logs\\errors.log for the root cause."
                )
            raise RuntimeError(f"command.dispatch failed: {resp2['error']}")

    async def set_goal(self, session_id: str, goal: str, *,
                       gates: Optional[list] = None) -> dict:
        """Set a goal on a session via /goal slash commands.

        Sends '/goal <text>' then '/goal gate add <cmd>' for each gate,
        THEN submits the kickoff prompt if one was returned.

        Goal actions (status/pause/resume/clear) are available via
        goal_action().

        The kickoff prompt is submitted non-blocking: we return as soon as
        the turn has started, not after the multi-turn goal loop completes.
        The goal loop runs autonomously on the backend.
        """
        results = []

        # 1. Set the goal
        r = await self.run_slash(session_id, f"/goal {goal}")
        results.append(r)
        raw = r.get("raw", {})

        # 2. Add gates BEFORE kickoff so the loop starts with gates active
        for gate_cmd in (gates or []):
            r = await self.run_slash(session_id, f"/goal gate add {gate_cmd}")
            results.append(r)

        # 3. Submit kickoff prompt non-blocking (don't wait for goal loop)
        if raw.get("type") == "send" and raw.get("message"):
            cw = _cw_state.get(session_id, False)
            # Fire-and-forget: submit the kickoff prompt but don't block on
            # the full goal loop. Use a short timeout just to confirm the
            # turn started. A goal turn normally exceeds 10s, so a timeout
            # means the turn IS running — not that it failed.
            try:
                kickoff_result = await self.send_prompt(
                    session_id, raw["message"], timeout=10, continuous_work=cw,
                )
                # send_prompt returns {"timed_out": True} when the turn is
                # still running — this is the EXPECTED case for goal loops.
                if kickoff_result.get("timed_out"):
                    results.append({
                        "kickoff_submitted": True,
                        "status": "running",
                        "note": "Goal loop started (turn still running server-side). Use goal_action('status') to check progress.",
                    })
                else:
                    results.append({
                        "kickoff_submitted": True,
                        "status": "running",
                        "response_preview": (kickoff_result.get("response") or "")[:200],
                        "note": "Goal loop started. Use goal_action('status') to check progress.",
                    })
            except Exception as e:
                results.append({
                    "kickoff_submitted": False,
                    "error": str(e),
                    "note": "Goal was set but kickoff could not be submitted.",
                })

        return {"ok": all(r.get("ok", True) for r in results), "results": results}

    async def goal_action(self, session_id: str, action: str) -> dict:
        """Execute a goal control action: status, show, pause, resume, clear."""
        action = action.strip().lower()
        if action not in ("status", "show", "pause", "resume", "clear"):
            raise ValueError(f"Invalid goal action: {action!r}. Use: status, show, pause, resume, clear")
        return await self.run_slash(session_id, f"/goal {action}")

    async def set_continuous_work(self, session_id: str, enabled: bool) -> dict:
        """Toggle continuous-work mode via session.set_continuous_work RPC.

        Also updates the in-process CW registry so subsequent sends
        preserve the state across prompt.submit calls (which reset CW
        to False when the flag is absent).
        """
        async with _ws_connect(self) as ws:
            # Resume to get live id
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "session.resume",
                "params": {"session_id": session_id},
            }))
            resp = await _recv_response(ws, 1, timeout=15)
            if "error" in resp:
                raise RuntimeError(f"resume failed: {resp['error']}")
            live_id = resp["result"].get("session_id", session_id)

            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 2, "method": "session.set_continuous_work",
                "params": {"session_id": live_id, "enabled": enabled},
            }))
            resp2 = await _recv_response(ws, 2, timeout=10)
            if "error" in resp2:
                raise RuntimeError(f"set_continuous_work failed: {resp2['error']}")
            # Update in-process registry
            _cw_state[session_id] = enabled
            return {"ok": True, "enabled": enabled, "session_id": session_id}

    async def send_prompt(self, session_id: str, message: str,
                          timeout: float = 300,
                          continuous_work: Optional[bool] = None) -> dict:
        """Send a prompt via WebSocket. Returns when agent finishes or timeout.

        For draft sessions (created without initial_message), submits the
        first prompt against the live session id to bootstrap the durable
        row, then cleans up the draft mapping.

        Internally uses the streaming approach to keep the WebSocket alive
        during message polling. Closing the WS immediately after send causes
        the backend to reap the session as orphaned before the agent starts.

        Args:
            continuous_work: None (default) resolves from the in-process
                CW registry (set by set_continuous_work). True/False force
                the flag. prompt.submit RESETS CW to False when this param
                is absent, so callers must preserve the state.
        """
        if continuous_work is None:
            continuous_work = _cw_state.get(session_id, False)
        # Use send_prompt_stream internally to keep WS alive throughout polling,
        # then return just the final response in send_prompt's traditional format.
        final_result = None
        async for etype, payload in self.send_prompt_stream(
            session_id, message, timeout=timeout, continuous_work=continuous_work
        ):
            if etype == "__done__":
                final_result = payload
                break

        if final_result is None:
            # Should never happen but handle gracefully
            await self._metrics.increment("prompts_timed_out")
            return {
                "response": None,
                "session_id": session_id,
                "timed_out": True,
                "status": "error",
            }

        # Track success metrics
        await self._metrics.increment("prompts_sent")
        if not final_result.get("timed_out"):
            await self._metrics.increment("prompts_succeeded")

        return final_result

    async def send_prompt_stream(self, session_id: str, message: str,
                                 timeout: float = 300,
                                 continuous_work: Optional[bool] = None):
        """Yields (event_type, payload) tuples; last yield is ('__done__', final_dict).

        For draft sessions, submits against the live id directly (same as
        send_prompt's draft path). Removes the draft record after first prompt.

        A setup-phase TimeoutError (session.resume did not reply) yields
        ``__done__`` with the same structured pending shape as the mid-turn
        timeout — the caller should poll get_messages rather than treat it
        as a hard MCP failure. The draft mapping is preserved because the
        turn may not have started.

        Args:
            continuous_work: None (default) resolves from the in-process
                CW registry. True/False force the flag. prompt.submit
                RESETS CW to False when absent — callers must preserve state.
        """
        if continuous_work is None:
            continuous_work = _cw_state.get(session_id, False)
        draft_record = await self._draft_sessions.get(session_id)

        async with _ws_connect(self) as ws:
            if draft_record:
                live_id = draft_record.live_session_id
            else:
                # Resume durable session, or fallback to bootstrap on 4007
                await ws.send(json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "session.resume",
                    "params": {"session_id": session_id},
                }))
                try:
                    resp = await _recv_response(ws, 1, timeout=15)
                except TimeoutError:
                    # Setup timed out before the turn started. Do NOT remove
                    # the draft mapping — the session may still be usable.
                    # Surface as pending, not an error, so a short client
                    # timeout looks like "still running" rather than a hard
                    # MCP failure.
                    yield ("__done__", {
                        "response": None,
                        "session_id": session_id,
                        "timed_out": True,
                        "status": "pending",
                        "note": (
                            "Setup timed out waiting for session.resume. "
                            "The session may be busy; the turn may not "
                            "have started. Poll get_messages, then retry."
                        ),
                    })
                    return
                if "error" in resp:
                    error = resp["error"]
                    if error.get("code") == 4007:
                        # Session not found: fall back to bootstrap (submit directly)
                        logger.info(
                            "Session %s not found on resume; bootstrapping via direct submit "
                            "(likely recovered from MCP server restart)",
                            session_id,
                        )
                        live_id = session_id
                    else:
                        raise RuntimeError(f"resume failed: {error}")
                else:
                    live_id = resp["result"].get("session_id", session_id)

            # Submit — pass continuous_work to prevent prompt.submit from
            # resetting it to False (see methods_prompt.py:453/457)
            submit_params: dict = {"session_id": live_id, "text": message}
            if continuous_work:
                submit_params["continuous_work"] = True
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 2, "method": "prompt.submit",
                "params": submit_params,
            }))

            deadline = time.time() + timeout
            final_text_parts: list[str] = []
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=max(1.0, deadline - time.time()),
                    )
                except TimeoutError:
                    break
                evt = json.loads(raw)
                if evt.get("method") != "event":
                    continue
                params = evt.get("params", {})
                if params.get("session_id") != live_id:
                    continue  # another session's traffic
                etype = params.get("type")
                payload = params.get("payload", {})
                yield (etype, payload)
                if etype == "message.delta":
                    final_text_parts.append(
                        payload.get("delta") or payload.get("text") or ""
                    )
                if etype == "message.complete":
                    content = payload.get("content", "")
                    # Clean up draft record after successful completion
                    if draft_record:
                        await self._draft_sessions.remove(session_id)
                    yield ("__done__", {
                        "response": "".join(final_text_parts) or content,
                        "session_id": session_id,
                    })
                    return
                if etype == "error":
                    raise RuntimeError(f"agent error mid-turn: {payload}")

            # Timed out waiting for the reply — the turn may still be
            # running. Align with _poll_for_reply's pending shape so callers
            # (and h_send_prompt) treat this as "poll later", not failure.
            if draft_record:
                await self._draft_sessions.remove(session_id)
            yield ("__done__", {
                "response": "".join(final_text_parts),
                "session_id": session_id,
                "timed_out": True,
                "status": "pending",
                "note": (
                    "Timed out waiting for the agent reply; the turn may "
                    "still be running. Poll get_messages for the final "
                    "answer. send_prompt_stream is preferred when you need "
                    "visibility into a long turn."
                ),
            })


# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

def _safe_epoch(v) -> float:
    """Coerce a value to float epoch, returning 0.0 on failure."""
    if v is None:
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


async def auto_discover_backend(
    host: str = "127.0.0.1",
    require_session_id: Optional[str] = None,
) -> Optional[int]:
    """Find the Hermes GUI backend by probing localhost ports concurrently.

    ``require_session_id`` turns discovery from a heuristic into a decision
    procedure.  The recency-based ranking below assumes one live backend plus
    some stale ones; that assumption is false on a machine running a backend
    per Hermes *profile* (``hermes_cli.main --profile <name> serve``), because
    every one of them is genuinely live, answers ``/api/sessions`` with 200,
    and has recent activity.  Ranking those by recency picks whichever profile
    happened to move last, which has nothing to do with which backend owns the
    session the caller actually asked for — the observed failure being a
    permanent ``session_not_found on re-discovered backend`` while the real
    backend sat healthy on another port.

    When the caller knows the session id, ownership is directly testable, so
    we test it instead of guessing: a backend that serves the session scores
    above every recency match, and one that explicitly disowns it (404) is
    removed from consideration entirely rather than left to win a tiebreak.
    """
    try:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=5,
        )
        ports = set()
        for line in result.stdout.splitlines():
            if "LISTEN" in line and "127.0.0.1" in line:
                m = re.search(r'127\.0\.0\.1:(\d+)', line)
                if m:
                    ports.add(int(m.group(1)))
    except Exception:
        return None

    if not ports:
        return None

    session_token = os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN", "")

    async def _probe(port: int, session_token: str) -> tuple[int, int, float] | None:
        """Returns (port, strength, recency) — strength 3 for a backend that
        demonstrably owns ``require_session_id``, 2 for a confirmed usable
        backend (``/api/sessions`` actually answers with our auth), 1 for a
        same-process heuristic match.

        ``recency`` is the ``last_active`` epoch of the most recent session
        on this instance (0 if no sessions).  This is the primary
        discriminator when two instances both score strength 2: the live
        instance has recent activity; a stale one does not.

        Distinct Hermes components can share a host: e.g. the messaging
        gateway's OpenAI-compatible API server also serves a Hermes-shaped
        ``/health`` (including a ``version`` key) but gates ``/api/sessions``
        behind ``Authorization: Bearer <API_S...Y>`` — a scheme this
        client never sends. Matching on ``/health`` alone previously locked
        onto that gateway instead of the actual dashboard backend, and every
        session/chat call then 401'd with no way to recover short of manually
        overriding ``HERMES_GUI_PORT``.
        """
        try:
            async with httpx.AsyncClient(timeout=2) as c:
                headers = {}
                if session_token:
                    headers[_SESSION_HEADER_NAME] = session_token
                r2 = await c.get(
                    f"http://{host}:{port}/api/sessions",
                    headers=headers,
                    params={"limit": 1, "min_messages": 0, "order": "recent"},
                )
                if r2.status_code == 200:
                    body2 = r2.json()
                    if isinstance(body2, dict) and "sessions" in body2:
                        sessions = body2["sessions"]
                        recency = _safe_epoch(
                            sessions[0].get("last_active") if sessions else None
                        )
                        if not require_session_id:
                            return (port, 2, recency)

                        # Ownership is decidable, so decide it.  A 200 means
                        # this backend holds the session; a 404 means it
                        # provably does not, which is stronger information
                        # than any recency signal and must exclude the
                        # candidate rather than merely rank it lower.
                        # Anything else (5xx, timeout, malformed) is an
                        # inconclusive answer, not a denial: fall back to the
                        # recency ranking so an unrelated backend hiccup can
                        # never strand a caller that would otherwise have
                        # found its session.
                        try:
                            r3 = await c.get(
                                f"http://{host}:{port}"
                                f"/api/sessions/{require_session_id}/messages",
                                headers=headers,
                                params={"limit": 1},
                            )
                        except Exception:
                            return (port, 2, recency)
                        if r3.status_code == 200:
                            return (port, 3, recency)
                        if r3.status_code == 404:
                            return None
                        return (port, 2, recency)
                if r2.status_code == 401:
                    # Right shape, wrong/missing credential for this port —
                    # never treat as usable, even as a fallback.
                    return None

                r = await c.get(f"http://{host}:{port}/health")
                if r.status_code == 200:
                    body = r.json()
                    if any(k in body for k in ("version", "service", "hermes")):
                        return (port, 1, 0.0)
        except Exception:
            pass
        return None

    async def _sweep(token: str) -> Optional[int]:
        """Probe all candidate ports concurrently and pick the best match.

        Prefers a confirmed (``/api/sessions``-verified) match over a
        ``/health``-only heuristic one.  Among confirmed matches, prefers
        the most recently active instance (highest ``last_active`` epoch),
        which distinguishes the live backend from a stale one that also
        answers ``/api/sessions`` with 200.

        When two confirmed instances have indistinguishable recency (within
        60 seconds), that is a genuine ambiguity — report it rather than
        silently picking one, because a wrong choice invites destructive
        recovery against healthy work.
        """
        results = await asyncio.gather(
            *[_probe(p, token) for p in sorted(ports)]
        )
        found = [r for r in results if r is not None]
        if not found:
            return None

        # Sort: highest strength first, then most recent activity, then
        # lowest port as final tiebreaker (deterministic, not identity).
        found.sort(key=lambda pr: (-pr[1], -pr[2], pr[0]))

        best = found[0]

        # A backend that owns the requested session is not ambiguous with
        # backends that merely answer /api/sessions, however recently they
        # were active — ownership was measured, not inferred.  Only a second
        # backend claiming the *same* session id is a real conflict, and that
        # is what the check below is narrowed to in that case.
        owners = [r for r in found if r[1] == 3]
        if owners:
            if len(owners) == 1:
                return owners[0][0]
            logger.warning(
                "Ambiguous session ownership: %d backends claim session %s "
                "(ports %s).",
                len(owners), require_session_id,
                ", ".join(str(r[0]) for r in sorted(owners)),
            )
            raise BackendAmbiguous([r[0] for r in owners])

        # Ambiguity detection: if there are multiple strength-2 candidates
        # with similar recency, we cannot tell which is the live instance.
        # Report this rather than silently picking one.
        strength2 = [r for r in found if r[1] == 2]
        if len(strength2) >= 2:
            top_recency = strength2[0][2]
            runners_up = [
                r for r in strength2[1:]
                if r[2] > 0 and abs(r[2] - top_recency) < 60
            ]
            if runners_up:
                ambiguous_ports = [best[0]] + [r[0] for r in runners_up]
                logger.warning(
                    "Ambiguous backend discovery: %d instances with "
                    "indistinguishable recency (ports %s). "
                    "Cannot determine which is live.",
                    len(ambiguous_ports),
                    ", ".join(str(p) for p in sorted(ambiguous_ports)),
                )
                raise BackendAmbiguous(ambiguous_ports)

        return best[0]

    port = await _sweep(session_token)
    if port is not None:
        return port

    # Nothing answered. The dominant cause is a STALE TOKEN, not a dead
    # backend: this machine routinely runs more than one
    # ``hermes_cli.main serve`` process (a desktop-shell backend on
    # ``--port 0`` plus a pinned one on 8765), each minting its own
    # ``secrets.token_urlsafe(32)`` at start and rejecting the other's with
    # 401. Whichever token was captured at import is therefore correct only
    # until the process it came from exits — after which _probe's
    # "401 means never usable" rule (correct in itself) turns every
    # candidate into None and discovery reports the same message as a
    # genuinely absent Hermes.
    #
    # Re-reading the token from the live processes is the only thing that
    # distinguishes those two states, and it is exactly what
    # ``refresh_session_token`` exists for. Without this retry the failure is
    # unrecoverable in-process: nothing else invalidates the pinned token, so
    # every later call fails identically until the MCP client is restarted,
    # and restarting Hermes cannot help because the stale value lives here.
    try:
        from .token_discovery import refresh_session_token
    except Exception:
        logger.warning("Token refresh unavailable; cannot retry discovery")
        return None

    try:
        fresh = await asyncio.to_thread(refresh_session_token)
    except Exception as exc:
        logger.warning("Token refresh failed: %s: %s", type(exc).__name__, exc)
        return None

    if not fresh or fresh == session_token:
        # Either discovery could not run, or it returned what we already
        # tried. Re-probing would be identical work for an identical result.
        logger.info("No fresh session token available; backend not found")
        return None

    logger.info("Retrying backend discovery with a refreshed session token")
    return await _sweep(fresh)


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_backend: Optional[GuiBackend] = None

# Per-session continuous-work state.  prompt.submit RESETS CW to False when
# the flag is absent (methods_prompt.py:453/457), so any send without it
# silently disables CW.  This in-process registry lets the delegator
# preserve CW across sends: set_continuous_work updates it; send_prompt
# reads it when continuous_work=None (the default).  Explicit False clears.
_cw_state: dict[str, bool] = {}


async def _port_answers(host: str, port: int, timeout: float = 2.0) -> bool:
    """Liveness probe for a pinned port with Hermes identity verification.

    Returns True only if the port answers ``/api/sessions`` with HTTP 200
    AND the response body contains a ``"sessions"`` key — the shape that
    Hermes uniquely exposes.  This prevents adopting a non-Hermes service
    (e.g. Foundry Manager on port 3000) that happens to answer HTTP on
    the same path.

    Sends ``HERMES_DASHBOARD_SESSION_TOKEN`` (if set) the same way the rest
    of this module authenticates — without it, a token-gated backend 401s
    and this probe reports "not answering" even when the port is exactly
    right, silently defeating both the ``HERMES_GUI_PORT`` override and the
    cached-port fast path in ``_require_backend``.
    """
    try:
        headers = {}
        token = os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN", "")
        if token:
            headers[_SESSION_HEADER_NAME] = token
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"http://{host}:{port}/api/sessions", headers=headers)
            if r.status_code != 200:
                return False
            body = r.json()
            return isinstance(body, dict) and "sessions" in body
    except Exception:
        return False


async def _probe_port_status(host: str, port: int, timeout: float = 2.0) -> Optional[int]:
    """Like ``_port_answers`` but distinguishes *why* a port didn't answer.

    Returns the HTTP status code (200 only for a verified Hermes-shaped
    response with a ``"sessions"`` key; a 200 from some other service on
    that port is reported as ``None``, same as a hard connection failure),
    or ``None`` if the connection failed outright (refused/timeout/DNS/etc).

    The caller uses this to skip the expensive PowerShell-based token
    refresh in ``_require_backend`` when a cached port is simply dead — a
    fresh token cannot make a closed socket answer, so spending 3-8s
    shelling out to PowerShell for one is pure waste. That refresh is only
    worth attempting on a genuine 401 (port alive, token stale).
    """
    try:
        headers = {}
        token = os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN", "")
        if token:
            headers[_SESSION_HEADER_NAME] = token
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"http://{host}:{port}/api/sessions", headers=headers)
            if r.status_code == 200:
                try:
                    body = r.json()
                except Exception:
                    return None
                if not (isinstance(body, dict) and "sessions" in body):
                    return None
            return r.status_code
    except Exception:
        return None


def _find_backend_port_via_ancestry(max_depth: int = 8) -> Optional[int]:
    """Find the port of the Hermes backend that actually spawned this process.

    This machine commonly runs several Hermes backends at once — one per
    profile (default, agent-cam3, agent-deepnest, ...), each on its own
    ephemeral port. Each backend spawns its own ``chat-delegator``
    subprocess as a direct process-tree descendant. Port-scanning or
    caching "last known good port" is unsound in that environment: a
    cached/scanned port frequently belongs to a *different* profile's
    backend than the one that launched this particular process — live,
    reachable, but wrong, so it answers 401 and every "fast path" degrades
    into the full ~20-port sweep (observed: 24s end to end). Walking our
    own ancestry to the ``hermes_cli.main ... serve`` process that is our
    actual parent and reading the port IT listens on is deterministic and
    fast, and immune to other profiles' backends being alive on other ports.
    """
    try:
        proc = psutil.Process(os.getpid())
    except Exception:
        return None
    for _ in range(max_depth):
        try:
            proc = proc.parent()
        except Exception:
            return None
        if proc is None:
            return None
        try:
            cmdline = " ".join(proc.cmdline())
        except Exception:
            continue
        if "hermes_cli.main" not in cmdline:
            continue
        try:
            conns = proc.connections(kind="tcp")
        except Exception:
            continue
        for c in conns:
            if (
                c.status == psutil.CONN_LISTEN
                and c.laddr
                and c.laddr.ip in ("127.0.0.1", "0.0.0.0", "::1")
            ):
                return c.laddr.port
    return None


async def _require_backend() -> GuiBackend:
    """Get the backend, lazily discovering if needed.

    If a cached backend exists but is unreachable, attempts a fresh
    discovery before raising — a backend restart (new port) is the
    dominant cause of "Not connected" errors.
    """
    global _backend
    if _backend is not None:
        # Verify cached backend can actually serve requests before trusting it.
        # Uses the authenticated /api/sessions probe — NOT /health, which can
        # be absent on working backends (59681-style) or perfect on useless
        # ones (8642-style with wrong token).  See TestHeadlessHealthBody.
        try:
            if await _backend.is_alive():
                return _backend
        except Exception:
            logger.info("Cached backend %s:%s unreachable — re-discovering",
                        _backend.host, _backend.port)
            _backend = None

    host = os.environ.get("HERMES_GUI_HOST", "127.0.0.1")
    port_str = os.environ.get("HERMES_GUI_PORT", "")
    port = None
    if port_str:
        try:
            candidate = int(port_str)
        except ValueError:
            logger.warning("Ignoring non-integer HERMES_GUI_PORT=%r", port_str)
        else:
            if await _port_answers(host, candidate):
                port = candidate
            else:
                logger.warning(
                    "HERMES_GUI_PORT=%s is not answering — falling back to "
                    "auto-discovery", candidate,
                )
    if port is None:
        ancestry_port = _find_backend_port_via_ancestry()
        if ancestry_port is not None and await _port_answers(host, ancestry_port):
            port = ancestry_port
            logger.info("Resolved backend port via process ancestry: %s:%s", host, port)

    port_cache_path = Path.home() / ".hermes" / "mcp_backend_port.json"
    if port is None:
        cached = _read_cached_port(port_cache_path)
        if cached is not None and cached[0] == host:
            cached_port = cached[1]
            status = await _probe_port_status(host, cached_port)
            if status == 200:
                port = cached_port
            elif status == 401:
                # Port is alive and IS Hermes — the token is just stale.
                # Only worth the expensive PowerShell-based refresh when we
                # know a fresh token can actually fix it; see
                # _probe_port_status docstring for why an unreachable port
                # (status is None) skips this entirely instead of wasting
                # 3-8s shelling out to PowerShell for nothing.
                try:
                    from .token_discovery import refresh_session_token
                    fresh = await asyncio.to_thread(refresh_session_token)
                except Exception:
                    fresh = None
                if fresh and await _port_answers(host, cached_port):
                    port = cached_port
            if port is not None:
                logger.info("Reusing cached backend port %s:%s", host, port)
    if port is None:
        logger.info("Auto-discovering GUI backend...")
        try:
            port = await auto_discover_backend(host)
        except BackendAmbiguous as e:
            # Multiple live instances — surface the ambiguity to the caller
            # rather than silently picking one.  A wrong choice invites
            # destructive recovery against healthy work.
            raise RuntimeError(
                str(e)
            ) from e
        if port is None:
            raise RuntimeError(
                "Not connected to GUI backend — is Hermes running? "
                "Set HERMES_GUI_PORT to connect manually."
            )
    _write_cached_port(port_cache_path, host, port)
    token = os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN", "")
    _backend = GuiBackend(host, port, token=token)
    logger.info("Connected to backend on %s:%s", host, port)
    return _backend


async def _find_or_discover_backend(
    require_session_id: Optional[str] = None,
) -> Optional[GuiBackend]:
    """Discover or find a live backend without raising.

    Returns a GuiBackend if reachable, None if genuinely down.
    Unlike ``_require_backend``, never raises — callers use this
    to retry after a connection failure.

    ``require_session_id`` restricts discovery to the backend that owns that
    session.  It also suppresses the pinned-port shortcut below: when we are
    recovering from a session 404, the pinned port is precisely the backend
    that just disowned the session, so accepting it because it "answers"
    would re-select the known-wrong instance and make the retry a no-op.
    """
    global _backend
    host = os.environ.get("HERMES_GUI_HOST", "127.0.0.1")

    # Try the currently pinned port first
    port_str = os.environ.get("HERMES_GUI_PORT", "")
    if port_str and not require_session_id:
        try:
            candidate = int(port_str)
            if await _port_answers(host, candidate):
                token = os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN", "")
                _backend = GuiBackend(host, candidate, token=token)
                logger.info("Re-discovered backend on pinned port %s", candidate)
                return _backend
        except ValueError:
            pass

    # Full discovery
    port = await auto_discover_backend(host, require_session_id=require_session_id)
    if port is None and require_session_id:
        # No backend claimed the session.  That is not proof the session is
        # gone: a backend we cannot authenticate to (each Hermes profile
        # mints its own token) drops out of the aimed sweep on 401, and a
        # freshly-created draft may not expose a messages endpoint yet.
        # Affinity is therefore a preference that decides the outcome when it
        # fires, never a new way to fail — fall back to the unaimed sweep so
        # this path is strictly no worse than it was before affinity existed.
        logger.info(
            "No backend claimed session %s — falling back to unaimed discovery",
            require_session_id,
        )
        port = await auto_discover_backend(host)
    if port is None:
        return None

    token = os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN", "")
    _backend = GuiBackend(host, port, token=token)
    logger.info("Re-discovered backend on auto-discovered port %s", port)
    return _backend


async def _reconnect_backend(
    require_session_id: Optional[str] = None,
) -> GuiBackend:
    """Re-discover and replace the cached backend, or raise RuntimeError."""
    b = await _find_or_discover_backend(require_session_id=require_session_id)
    if b is None:
        raise RuntimeError(
            "Backend is unreachable after re-discovery — is Hermes running? "
            "Set HERMES_GUI_PORT to connect manually."
        )
    return b


def _reset_backend():
    """Mark backend as disconnected so next call re-discovers."""
    global _backend
    _backend = None


async def _require_backend_for_session(session_id: str) -> GuiBackend:
    """Resolve a backend AND confirm it actually owns ``session_id``.

    The WS prompt path (``send_prompt``/``send_prompt_stream``) has no
    equivalent of the HTTP 404-triggered re-discovery in ``_retry_handler``:
    ``prompt.submit`` against a session id the connected backend has never
    heard of does not error — the server just never emits a matching
    ``message.complete`` event, so the call silently burns its full timeout
    (up to 300s) and returns an empty/``timed_out`` result. On a machine
    running multiple Hermes backends at once (one per profile), landing on
    the wrong-but-live one is a real, observed failure mode, not a
    hypothetical.

    This does one cheap HTTP ownership probe (``get_session``, backed by
    the same 404 that already drives the HTTP retry path) BEFORE opening
    the WebSocket, so a wrong backend is caught and corrected in a
    sub-second round trip instead of a multi-minute silent hang.

    Skipped entirely for a known draft session: a draft is legitimately
    invisible to ``get_session`` on its OWN correct backend (see
    ``create_session`` — "NOT yet visible to list_chats/get_chat"), so the
    same 404 that signals "wrong backend" for a durable session is expected
    and harmless there. Re-discovering on it would chase a false positive
    and could break the existing bootstrap-on-4007 recovery in
    ``send_prompt_stream``.
    """
    b = await _require_backend()
    draft_sessions = getattr(b, "_draft_sessions", None)
    if draft_sessions is not None and await draft_sessions.get(session_id):
        return b
    try:
        await b.get_session(session_id)
        return b
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 404:
            raise
        logger.info(
            "Backend %s:%s does not own session %s — re-discovering before "
            "WS submit instead of risking a silent timeout",
            getattr(b, "host", "?"), getattr(b, "port", "?"), session_id,
        )
        return await _reconnect_backend(require_session_id=session_id)
    except (AttributeError, TypeError):
        # Backend double (tests) or a future backend variant without
        # get_session — nothing to verify against, trust the resolved
        # backend rather than hard-failing the whole prompt path.
        return b





# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="list_chats",
        description=(
            "List all chat sessions visible in the Hermes GUI, sorted by most "
            "recent activity. Shows active status. Sessions a backend "
            "restart/crash interrupted mid-turn are flagged with "
            "interrupted=true (see backend_status for the full list with "
            "reasoning) — a resume-and-continue candidate, since those "
            "sessions will not reply on their own."
        ),
        inputSchema={"type": "object", "properties": {
            "limit": {"type": "integer", "description": "Max sessions (default 50)"},
        }},
    ),
    Tool(
        name="get_chat",
        description=(
            "Get detailed info about a specific chat session from the GUI."
        ),
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session ID"},
        }, "required": ["session_id"]},
    ),
    Tool(
        name="get_messages",
        description=(
            "Get message history from a GUI chat session. Returns the last N "
            "messages by default (20). Pass last_n=0 for full history. A "
            "session's full history can run 60K+ chars in one call — use "
            "max_chars and/or compact to keep the response manageable."
        ),
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session ID"},
            "last_n": {"type": "integer", "description": "Only last N messages (default 20, 0=all)"},
            "max_chars": {
                "type": "integer",
                "description": (
                    "Truncate each message's content/reasoning text to this "
                    "many characters (0=no limit, default 0). A truncated "
                    "field gets a matching '<field>_truncated: true' marker "
                    "and '<field>_original_chars' so truncation is never "
                    "silent."
                ),
            },
            "compact": {
                "type": "boolean",
                "description": (
                    "Omit verbose fields (reasoning, raw tool-call/result "
                    "payloads) that aren't needed for a quick skim of the "
                    "conversation. Default false."
                ),
            },
        }, "required": ["session_id"]},
    ),
    Tool(
        name="send_prompt",
        description=(
            "Send a user prompt into a GUI chat session and wait for the agent "
            "response. The prompt appears in the GUI. Use this for short "
            "interactions where you just need the final answer. For turns that "
            "involve multiple tool calls and you want visibility into what the "
            "agent is doing, use send_prompt_stream instead. "
            "Continuous work state is automatically preserved across sends "
            "(resolved from the in-process registry set by set_continuous_work)."
            " On timeout the result is {timed_out: true, status: \"pending\"} "
            "(not an error) — the turn may still be running; poll get_messages "
            "for the reply. send_prompt_stream is preferred when you need "
            "visibility into a long turn. "
            "Note: if a human has this session's GUI tab open, using this tool "
            "will freeze that tab's live view until they reopen/refocus it (known "
            "backend limitation, not a bug in this server)."
        ),
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session to prompt into"},
            "message": {"type": "string", "description": "The user message"},
            "timeout": {"type": "number", "description": "Max seconds to wait (default 300). Most turns finish in under 60s."},
        }, "required": ["session_id", "message"]},
    ),
    Tool(
        name="create_chat",
        description=(
            "Create a new chat session in the Hermes GUI. Returns immediately "
            "with the new session's IDs. "
            "If initial_message is provided, the prompt is sent immediately, "
            "the agent's reply is returned, and the session is fully visible "
            "to all other tools right away. "
            "If omitted, an empty draft session is created that is NOT yet "
            "visible to list_chats/get_chat — but the first send_prompt "
            "against it will automatically bootstrap the durable row "
            "(no error). A bad model name will surface on the first "
            "send_prompt, not on create_chat. "
            "When goal is set, the session is created, then CW is enabled, "
            "then the goal is set with gates, then the kickoff prompt is "
            "submitted. When both goal and initial_message are given, the "
            "goal is set first and initial_message is sent second. "
            "Continuous work state is stored in an in-process registry and "
            "persisted across sends within this delegator process only "
            "(not across MCP restarts). "
            "Note: if initial_message is provided and a human has this "
            "session's GUI tab open, using this tool will freeze that tab's "
            "live view until they reopen/refocus it (known backend limitation, "
            "not a bug in this server)."
        ),
        inputSchema={"type": "object", "properties": {
            "title": {"type": "string", "description": "Optional title. Untitled if omitted."},
            "cwd": {"type": "string", "description": "Optional working directory for the new session."},
            "model": {"type": "string", "description": "Optional model override, e.g. 'xiaomi/mimo-v2.5'."},
            "initial_message": {
                "type": "string",
                "description": (
                    "Optional first message to send immediately. If provided, "
                    "this tool waits for the agent's reply (like send_prompt) "
                    "and the returned session is immediately visible to all "
                    "other tools. If omitted, an empty draft session is created "
                    "that other tools cannot see until you call send_prompt on it."
                ),
            },
            "timeout": {"type": "number", "description": "Max seconds to wait for initial_message's reply (default 300). Ignored if initial_message is omitted."},
            "goal": {"type": "string", "description": "Optional goal text. Sets a persistent goal on the new session."},
            "goal_gates": {"type": "array", "items": {"type": "string"}, "description": "Optional quality gate commands for the goal."},
            "continuous_work": {"type": "boolean", "description": "Optional. Enable continuous work on the new session. Stored in in-process registry."},
        }},
    ),
    Tool(
        name="fork_chat",
        description=(
            "Branch a new chat session from an existing one, copying its full "
            "message history. Equivalent to clicking 'Branch' in the GUI. The "
            "source session must already have at least one message — forking an "
            "empty/draft session fails with a clear error."
            " Note: if a human has the source session's GUI tab open, using this "
            "tool will freeze that tab's live view until they reopen/refocus it "
            "(known backend limitation, not a bug in this server)."
        ),
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Durable session ID to fork from."},
            "title": {"type": "string", "description": "Optional title for the new branch (defaults to '<source title> (branch)')."},
        }, "required": ["session_id"]},
    ),
    Tool(
        name="set_model",
        description=(
            "Switch the AI model for an existing chat session mid-conversation. "
            "The model is pinned to this session (persists across turns) until "
            "changed again. Pass the model id exactly as the provider lists "
            "it, e.g. 'deepseek/deepseek-v4-flash-0731' or "
            "'anthropic/claude-sonnet-4' — do NOT prefix the gateway "
            "('openrouter/...' is rejected; the gateway is already configured "
            "per session). A rejection lists near-miss candidates, so the "
            "error text tells you the correct spelling. The output field "
            "contains the backend's /model command output confirming the "
            "switch. On timeout the tool retries once automatically, then "
            "returns a retryable model_switch_timeout error."
        ),
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session to change model on"},
            "model": {"type": "string", "description": "Model id as the provider lists it, e.g. 'deepseek/deepseek-v4-flash-0731'. Do not prefix the gateway ('openrouter/...' is rejected)."},
            "timeout": {"type": "number", "description": "Max seconds to wait (default 30)"},
        }, "required": ["session_id", "model"]},
    ),
    Tool(
        name="send_prompt_stream",
        description=(
            "Send a prompt into a GUI chat session and stream turn progress "
            "(assistant text as it's generated, tool calls as they start/finish, "
            "reasoning if enabled) as MCP progress notifications while the agent "
            "works, then return the full transcript of the turn plus the final "
            "answer. If your MCP client doesn't display progress notifications, "
            "this behaves like send_prompt but the final result additionally "
            "includes every intermediate event (tool calls made, reasoning steps) "
            "for post-hoc inspection — useful for debugging what the agent "
            "actually did, not just its final answer. Use this instead of "
            "send_prompt when the turn might involve multiple tool calls and you "
            "want visibility into what the agent is doing while you wait."
            " Note: if a human has this session's GUI tab open, using this tool "
            "will freeze that tab's live view until they reopen/refocus it (known "
            "backend limitation, not a bug in this server)."
        ),
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session to prompt into"},
            "message": {"type": "string", "description": "The user message"},
            "timeout": {"type": "number", "description": "Max seconds to wait (default 300)"},
        }, "required": ["session_id", "message"]},
    ),
    Tool(
        name="rename_chat",
        description="Rename a chat session's title in the GUI.",
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session to rename"},
            "title": {"type": "string", "description": "New title"},
        }, "required": ["session_id", "title"]},
    ),
    Tool(
        name="delete_chat",
        description="Delete a chat session from the GUI.",
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session to delete"},
        }, "required": ["session_id"]},
    ),
    Tool(
        name="backend_status",
        description=(
            "Check the connection to the GUI backend. Reports whether the "
            "backend is reachable, the port, and basic health info."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="get_metrics",
        description=(
            "Get operational metrics (sessions, prompts, errors, reconnects). "
            "Metrics are cumulative since the last snapshot and reset after "
            "this call. Use to monitor MCP server health and performance."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    # ---- Kanban tools -------------------------------------------------------
    Tool(
        name="kanban_board",
        description=(
            "Get a compact kanban board summary: columns with their tasks, total "
            "count, and active assignees. Filters: status (column name) and "
            "assignee. Compact by default — never dumps the full payload."
        ),
        inputSchema={"type": "object", "properties": {
            "status": {"type": "string", "description": "Filter to one column (e.g. 'ready', 'running')"},
            "assignee": {"type": "string", "description": "Filter tasks by assignee"},
            "board": {"type": "string", "description": "Board slug (omit for current)"},
            "compact": {"type": "boolean", "description": "Default true = compact rows (done/archived shown as counts unless status asks for them); false = full payload, still filtered by status/assignee"},
        }},
    ),
    Tool(
        name="kanban_show",
        description=(
            "Show a single kanban task with full detail: title, body, status, "
            "assignee, priority, comments, run history, events, links, and a "
            "tail of the worker log. Returns 404 if the task doesn't exist."
        ),
        inputSchema={"type": "object", "properties": {
            "task_id": {"type": "string", "description": "Task ID (e.g. 't_123')"},
            "board": {"type": "string", "description": "Board slug (omit for current)"},
        }, "required": ["task_id"]},
    ),
    Tool(
        name="kanban_create",
        description=(
            "Create a new kanban task. Validates the assignee against the "
            "known profile list (the dispatcher silently drops unknown "
            "assignees). Auto-generates a deterministic idempotency key from "
            "title+body+assignee to prevent duplicate cards on retries. "
            "goal_mode defaults to TRUE. "
            "SAFETY RAILS (client-side only): "
            "(1) hold=true creates the card without dispatching a worker — "
            "lands in triage, unassigned. Use for drafts, tests, or when you "
            "will assign later. "
            "(2) When goal_mode=false AND the assignee is empty or 'default' "
            "AND triage is not specified, triage defaults to True so the "
            "common 'create a card I don't want running yet' path cannot "
            "auto-spawn a worker. Pass triage=false explicitly to let "
            "'default' spawn. "
            "(3) When the create WILL auto-dispatch (known assignee, not "
            "triage, not hold), the response includes a warning field. "
            "completion_contract is a GitHub "
            "PR-acceptance gate ('local-only', 'OWNER/REPO', or exact PR URL); "
            "when non-default, uses the hermes CLI. acceptance_criteria appends "
            "evidence requirements to the body."
        ),
        inputSchema={"type": "object", "properties": {
            "title": {"type": "string", "description": "Task title (required)"},
            "body": {"type": "string", "description": "Task description / body text"},
            "assignee": {"type": "string", "description": "Profile name to assign (validated against known assignees)"},
            "priority": {"type": "integer", "description": "Priority (higher = sooner; default 0)"},
            "parents": {"type": "array", "items": {"type": "string"}, "description": "Parent task IDs for dependency gating"},
            "skills": {"type": "array", "items": {"type": "string"}, "description": "Skill names to force-load into the worker"},
            "workspace_kind": {"type": "string", "description": "'scratch', 'dir', or 'worktree'"},
            "workspace_path": {"type": "string", "description": "Absolute path for dir/worktree workspace"},
            "goal_mode": {"type": "boolean", "description": "Run worker in goal loop (default TRUE)"},
            "goal_max_turns": {"type": "integer", "description": "Turn budget for goal_mode workers"},
            "model_override": {"type": "string", "description": "Pin worker to this model"},
            "provider_override": {"type": "string", "description": "Provider for model_override"},
            "reasoning_effort": {"type": "string", "description": "none|minimal|…|ultra"},
            "max_runtime_seconds": {"type": "integer", "description": "Per-task runtime cap"},
            "triage": {
                "type": "boolean",
                "description": (
                    "Land in triage instead of ready/todo. When omitted and "
                    "goal_mode=false with an empty/'default' assignee, "
                    "defaults to true (safe: no auto-dispatch). Pass false "
                    "explicitly to allow 'default' to spawn."
                ),
            },
            "hold": {
                "type": "boolean",
                "description": (
                    "Create the card without dispatching a worker (lands in "
                    "triage, unassigned). Use for drafts, tests, or when you "
                    "will assign later. Default false."
                ),
            },
            "idempotency_key": {"type": "string", "description": "Deterministic key to prevent duplicate cards on retries"},
            "completion_contract": {"type": "string", "description": "GitHub PR-acceptance gate: 'local-only' (default), 'OWNER/REPO', or 'https://github.com/OWNER/REPO/pull/N'. When non-default, uses the hermes CLI."},
            "acceptance_criteria": {"type": "array", "items": {"type": "string"}, "description": "Evidence requirements appended as '## Required handoff evidence' section to body"},
            "board": {"type": "string", "description": "Board slug (omit for current)"},
        }, "required": ["title"]},
    ),
    Tool(
        name="kanban_update",
        description=(
            "Update a kanban task: reassign, change priority, edit title/body, "
            "or transition status (block with reason, unblock, archive). "
            "The 'status' field supports: blocked, ready, todo, triage, review, "
            "done, archived. Returns the updated task on success."
        ),
        inputSchema={"type": "object", "properties": {
            "task_id": {"type": "string", "description": "Task ID to update"},
            "assignee": {"type": "string", "description": "New assignee (empty string to unassign)"},
            "priority": {"type": "integer", "description": "New priority"},
            "title": {"type": "string", "description": "New title"},
            "body": {"type": "string", "description": "New body text"},
            "status": {"type": "string", "description": "Status transition: blocked, ready, todo, triage, review, done, archived"},
            "block_reason": {"type": "string", "description": "Reason when status=blocked"},
            "board": {"type": "string", "description": "Board slug (omit for current)"},
        }, "required": ["task_id"]},
    ),
    Tool(
        name="kanban_comment",
        description="Add a comment to a kanban task.",
        inputSchema={"type": "object", "properties": {
            "task_id": {"type": "string", "description": "Task ID"},
            "body": {"type": "string", "description": "Comment text (markdown supported)"},
            "author": {"type": "string", "description": "Comment author (default 'delegator')"},
            "board": {"type": "string", "description": "Board slug (omit for current)"},
        }, "required": ["task_id", "body"]},
    ),
    Tool(
        name="kanban_link",
        description="Create a parent→child dependency between two kanban tasks.",
        inputSchema={"type": "object", "properties": {
            "parent_id": {"type": "string", "description": "Parent task ID"},
            "child_id": {"type": "string", "description": "Child task ID"},
            "board": {"type": "string", "description": "Board slug (omit for current)"},
        }, "required": ["parent_id", "child_id"]},
    ),
    Tool(
        name="kanban_dispatch",
        description=(
            "Nudge the dispatcher to process ready tasks immediately instead "
            "of waiting for the next tick. Use dry_run=true to preview what "
            "would be dispatched without actually spawning workers."
        ),
        inputSchema={"type": "object", "properties": {
            "dry_run": {"type": "boolean", "description": "Preview only (default false)"},
            "max_n": {"type": "integer", "description": "Max tasks to dispatch (default 8)"},
            "board": {"type": "string", "description": "Board slug (omit for current)"},
        }},
    ),
    Tool(
        name="kanban_stats",
        description=(
            "Combined kanban statistics: per-status counts, active workers, "
            "and diagnostics summary. Returns {stats, workers, diagnostics}."
        ),
        inputSchema={"type": "object", "properties": {
            "board": {"type": "string", "description": "Board slug (omit for current)"},
        }},
    ),
    Tool(
        name="kanban_decompose",
        description=(
            "Fan out a triage task into child tasks via the auxiliary LLM. "
            "Non-OK is not an error — check the 'ok' field in the response."
        ),
        inputSchema={"type": "object", "properties": {
            "task_id": {"type": "string", "description": "Task ID to decompose"},
            "author": {"type": "string", "description": "Author name for the decomposition"},
            "board": {"type": "string", "description": "Board slug (omit for current)"},
        }, "required": ["task_id"]},
    ),
    # ---- Session control tools -----------------------------------------------
    Tool(
        name="run_slash",
        description=(
            "Execute a slash command on a session via WebSocket JSON-RPC. "
            "Uses command.dispatch (covers quick/plugin/bundle/skill/builtin "
            "stages) with fallback to slash.exec. Returns the command's output. "
            "Examples: '/goal status', '/model deepseek/deepseek-v4-flash-0731', "
            "'/compress'"
        ),
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session to run the command on"},
            "command": {"type": "string", "description": "The slash command including '/' prefix (e.g. '/goal status')"},
        }, "required": ["session_id", "command"]},
    ),
    Tool(
        name="set_goal",
        description=(
            "Set a persistent goal on a session. Goals stay active across "
            "turns; after each turn an auxiliary-model judge checks if the goal "
            "is satisfied. Optionally add quality gates (shell commands that "
            "must pass before the goal is declared done). The goal is set via "
            "/goal <text> then gates via /goal gate add <cmd>. "
            "When the backend returns a kickoff prompt, it is automatically "
            "submitted as a turn so the goal loop starts immediately."
        ),
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session to set the goal on"},
            "goal": {"type": "string", "description": "The goal objective text"},
            "gates": {"type": "array", "items": {"type": "string"}, "description": "Shell commands that must pass before goal is done"},
        }, "required": ["session_id", "goal"]},
    ),
    Tool(
        name="goal_action",
        description=(
            "Control an active goal: 'status' (current state), 'show' "
            "(with contract), 'pause', 'resume', or 'clear' (delete)."
        ),
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session with the active goal"},
            "action": {"type": "string", "description": "Action: status, show, pause, resume, clear"},
        }, "required": ["session_id", "action"]},
    ),
    Tool(
        name="set_continuous_work",
        description=(
            "Toggle continuous-work mode on a session. When enabled, the "
            "agent automatically continues working after each turn until "
            "the goal is satisfied or the budget is exhausted. Uses the "
            "session.set_continuous_work RPC. CW state is stored in an "
            "in-process registry (not persisted across MCP server restarts). "
            "prompt.submit resets CW to False when the flag is absent; the "
            "registry prevents silent disabling on subsequent sends. "
            "Explicit False clears the registry. "
            "NOTE: the backend's session.resume does NOT expose "
            "continuous_work in its info dict, so CW state is remembered "
            "only for this delegator process."
        ),
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session to toggle CW on"},
            "enabled": {"type": "boolean", "description": "Enable (true) or disable (false) continuous work"},
        }, "required": ["session_id", "enabled"]},
    ),
]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _text(text: str) -> list[TextContent]:
    return [TextContent(type="text", text=text)]


def _err(code: str, msg: str, retryable: bool = False) -> list[TextContent]:
    return _text(json.dumps({"error": {"code": code, "message": msg, "retryable": retryable}}))


async def _retry_handler(fn, *args, **kwargs):
    """Execute a handler, re-discovering the backend on connection/auth failure.

    Pattern: call the handler. If it fails with ConnectError (backend
    restarted to new port), WsAuthFailure (stale token), or a session-
    scoped 404 (split instance — the backend is reachable but doesn't own
    the session we asked about), re-discover the backend and retry ONCE.
    If the retry also fails, return the error from the retry (which has
    more diagnostic information).

    BackendAmbiguous is never retried — it means multiple live instances
    exist and we cannot tell which is the real one.  Surfacing the
    candidate ports lets the user shut down the extras.

    This replaces the per-handler ``except httpx.ConnectError: _reset_backend()``
    pattern with a centralized retry that actually re-discovers instead of
    just clearing the cache and hoping the next call will discover.
    """
    try:
        return await fn(*args, **kwargs)
    except BackendAmbiguous as e:
        # Never retry — multiple live instances is not transient.
        _reset_backend()
        return _err("ambiguous_backend", str(e), retryable=False)
    except (httpx.ConnectError, WsAuthFailure) as first_exc:
        # Re-discover: find the backend at its current port with a fresh token
        logger.info("Connection failed (%s) — re-discovering backend",
                     type(first_exc).__name__)
        try:
            await _reconnect_backend()
        except RuntimeError:
            # Backend genuinely unreachable after re-discovery
            _reset_backend()
            if isinstance(first_exc, WsAuthFailure):
                return _err("ws_auth_failed", str(first_exc), retryable=True)
            return _err("not_connected",
                        "Backend unreachable after re-discovery — is Hermes running?",
                        retryable=True)

        # Retry the operation with the newly discovered backend
        try:
            return await fn(*args, **kwargs)
        except BackendAmbiguous as e:
            _reset_backend()
            return _err("ambiguous_backend", str(e), retryable=False)
        except WsAuthFailure as second_exc:
            _reset_backend()
            return _err("ws_auth_failed", str(second_exc), retryable=True)
        except httpx.ConnectError:
            _reset_backend()
            return _err("not_connected",
                        "Backend unreachable after re-discovery — is Hermes running?",
                        retryable=True)
        except Exception as e:
            return _err("backend_error", f"Failed after re-discovery: {e}")
    except httpx.HTTPStatusError as first_exc:
        # A 404 on a session-scoped endpoint is self-contradictory when the
        # session was recently visible (e.g. list_chats just returned it).
        # This is the signature of a split instance: the cached backend is
        # reachable but doesn't own the session.  Re-discover and retry once.
        if first_exc.response.status_code != 404:
            raise
        # The session id is in the URL that just 404'd, so the retry can be
        # aimed rather than repeated.  Without this the re-discovery re-runs
        # the same recency ranking, re-selects the same backend that just
        # disowned the session, and the "retry once" path reliably returns
        # the identical 404 it was written to recover from.
        _sid_m = re.search(r"/api/sessions/([^/?#]+)", str(first_exc.request.url))
        _sid = _sid_m.group(1) if _sid_m else None
        logger.info("Session 404 on %s — re-discovering backend that owns %s",
                     first_exc.request.url, _sid or "(unknown session)")
        try:
            await _reconnect_backend(require_session_id=_sid)
        except RuntimeError:
            _reset_backend()
            return _err("session_not_found",
                        f"Session not found on any reachable backend "
                        f"(first attempt 404 on {first_exc.request.url})",
                        retryable=True)
        try:
            return await fn(*args, **kwargs)
        except httpx.HTTPStatusError as second_exc:
            if second_exc.response.status_code == 404:
                return _err("session_not_found",
                            f"Session not found on re-discovered backend "
                            f"({second_exc.request.url})")
            return _err("backend_error",
                        f"HTTP {second_exc.response.status_code}: {second_exc}")
        except Exception as e:
            return _err("backend_error", f"Failed after re-discovery: {e}")


def _summarize_event(etype: str, payload: dict) -> str:
    """One-line human-readable summary of a WS event for progress notifications."""
    if etype == "message.delta":
        return payload.get("delta") or payload.get("text") or ""
    if etype == "message.start":
        return "agent started thinking"
    if etype == "message.complete":
        return "agent finished"
    if etype == "tool.start":
        return f"tool: {payload.get('name', '?')} started"
    if etype == "tool.complete":
        return f"tool: {payload.get('name', '?')} finished"
    if etype == "tool.generating":
        return f"tool: {payload.get('name', '?')} running"
    if etype == "reasoning.delta":
        return "reasoning..."
    if etype == "thinking.delta":
        return "thinking..."
    if etype == "status.update":
        return f"status: {payload.get('status', '?')}"
    if etype == "error":
        return f"error: {payload}"
    if etype == "approval.request":
        return f"approval needed: {payload.get('tool', '?')}"
    return str(payload)[:80]


async def h_list_chats(args: dict):
    async def _do():
        b = await _require_backend()
        sessions = await b.list_sessions(limit=args.get("limit", 50))
        sessions.sort(key=lambda s: _safe_epoch(s.get("last_active")), reverse=True)
        interrupted_ids = {
            s.get("id") for s in await b.list_interrupted_sessions()
        }

        def _fmt_ts(ts):
            if not ts:
                return "unknown"
            from datetime import datetime, timezone
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            now = datetime.now(tz=timezone.utc)
            diff = now - dt
            if diff.total_seconds() < 60:
                return "just now"
            elif diff.total_seconds() < 3600:
                return f"{int(diff.total_seconds() // 60)}m ago"
            elif diff.total_seconds() < 86400:
                return f"{int(diff.total_seconds() // 3600)}h ago"
            else:
                return dt.strftime("%b %d %H:%M")

        result = []
        for s in sessions:
            is_active = s.get("is_active", False)
            last_active = s.get("last_active")
            entry = {
                "session_id": s.get("id", ""),
                "title": s.get("title") or "untitled",
                "status": "active" if is_active else _fmt_ts(last_active),
                "model": s.get("model"),
                "source": s.get("source"),
                "message_count": s.get("message_count", 0),
                "cwd": s.get("cwd"),
            }
            if s.get("id") in interrupted_ids:
                entry["interrupted"] = True
            result.append(entry)
        return _text(json.dumps({"sessions": result, "count": len(result)}, indent=2, default=str))
    return await _retry_handler(_do)


async def h_get_chat(args: dict):
    async def _do():
        b = await _require_backend()
        info = await b.get_session(args["session_id"])
        return _text(json.dumps(info, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return _err("session_not_found", f"Session {args['session_id']!r} not found")
        return _err("backend_error", f"HTTP {e.response.status_code}: {e}")


# Verbose fields stripped by compact=True: not needed for a quick skim of
# the conversation, but they're what routinely push a get_messages response
# past 60K chars (defect #6) — raw reasoning traces and full tool-call/result
# payloads.
_COMPACT_DROP_FIELDS = (
    "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items", "tool_calls",
)
# Text fields eligible for max_chars truncation.
_TRUNCATABLE_TEXT_FIELDS = ("content", "reasoning", "reasoning_content")


def _truncate_messages(messages: list[dict], max_chars: int = 0,
                       compact: bool = False) -> list[dict]:
    """Apply the get_messages truncation/compact options (defect #6).

    Never mutates the input messages (copies each dict first). Truncation is
    never silent: a clipped field gets a sibling '<field>_truncated: true'
    and '<field>_original_chars' so a caller can tell content was cut rather
    than that the message was actually short.
    """
    if max_chars <= 0 and not compact:
        return messages
    out = []
    for m in messages:
        m = dict(m)
        if compact:
            for field in _COMPACT_DROP_FIELDS:
                m.pop(field, None)
        if max_chars > 0:
            for field in _TRUNCATABLE_TEXT_FIELDS:
                value = m.get(field)
                if isinstance(value, str) and len(value) > max_chars:
                    m[field] = value[:max_chars]
                    m[f"{field}_truncated"] = True
                    m[f"{field}_original_chars"] = len(value)
        out.append(m)
    return out


async def h_get_messages(args: dict):
    async def _do():
        b = await _require_backend()
        last_n = args.get("last_n", 20)
        messages = await b.get_messages(args["session_id"], last_n=last_n)
        max_chars = int(args.get("max_chars", 0) or 0)
        compact = bool(args.get("compact", False))
        messages = _truncate_messages(messages, max_chars=max_chars, compact=compact)
        return _text(json.dumps({
            "session_id": args["session_id"],
            "message_count": len(messages),
            "messages": messages,
        }, indent=2, default=str))
    return await _retry_handler(_do)


async def h_send_prompt(args: dict):
    async def _do():
        b = await _require_backend_for_session(args["session_id"])
        result = await b.send_prompt(
            args["session_id"],
            args["message"],
            timeout=args.get("timeout", 300),
        )
        return _text(json.dumps(result, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except TimeoutError:
        # Last-resort: send_prompt_stream already converts its own setup
        # and poll timeouts into a structured pending payload. If a
        # TimeoutError still escapes, report pending rather than
        # internal_error — the turn may well be running.
        return _text(json.dumps({
            "response": None,
            "session_id": args.get("session_id"),
            "timed_out": True,
            "status": "pending",
            "note": (
                "Timed out waiting for the session; the turn may still "
                "be running. Poll get_messages for the reply."
            ),
        }, indent=2, default=str))
    except RuntimeError as e:
        msg = str(e)
        if "4007" in msg or "session not found" in msg.lower():
            return _err(
                "unpromptable_draft",
                "This session was created without an initial message and has "
                "no durable row yet. send_prompt normally bootstraps it "
                "automatically, but the draft mapping was lost (e.g. MCP "
                "server restarted). Fix: call create_chat with initial_message, "
                "or delete this session and start a new one.",
            )
        return _err("backend_error", msg)


async def h_create_chat(args: dict):
    async def _do():
        b = await _require_backend()
        created = await b.create_session(
            title=args.get("title", ""),
            cwd=args.get("cwd", ""),
            model=args.get("model", ""),
            initial_message=args.get("initial_message", ""),
            prompt_timeout=args.get("timeout", 300),
        )
        session_id = created.get("session_id", "")

        if args.get("cwd") and created.get("cwd") and created["cwd"] != args["cwd"]:
            created["cwd_warning"] = (
                f"Requested cwd={args['cwd']!r} but backend used {created['cwd']!r}. "
                "The path may not exist."
            )

        # G3: Apply CW, goal, and gates in order: create → set CW → set goal → kickoff
        cw_enabled = args.get("continuous_work")
        goal_text = args.get("goal")
        goal_gates = args.get("goal_gates")

        if cw_enabled is not None and session_id:
            try:
                await b.set_continuous_work(session_id, cw_enabled)
                created["continuous_work"] = cw_enabled
            except Exception as e:
                created["continuous_work_error"] = str(e)

        if goal_text and session_id:
            try:
                goal_result = await b.set_goal(session_id, goal_text, gates=goal_gates)
                created["goal"] = goal_result
            except Exception as e:
                created["goal_error"] = str(e)

        if args.get("initial_message"):
            created["persisted"] = True
            # If goal was set, initial_message is secondary — goal kickoff already ran
            if not goal_text:
                # No goal: send initial_message as the first turn
                pass  # create_session already sent it
        else:
            created["persisted"] = False
            created["note"] = (
                "Empty draft — not yet visible to list_chats/get_chat. "
                "The first send_prompt against this session_id will "
                "automatically bootstrap it."
            )
        return _text(json.dumps(created, indent=2, default=str))
    return await _retry_handler(_do)


async def h_fork_chat(args: dict):
    async def _do():
        b = await _require_backend()
        result = await b.branch_session(args["session_id"], title=args.get("title", ""))
        return _text(json.dumps(result, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except RuntimeError as e:
        msg = str(e)
        if "empty" in msg.lower():
            return _err("empty_session", msg)
        return _err("backend_error", msg)


async def h_set_model(args: dict):
    timeout = float(args.get("timeout", 30) or 30)

    async def _do():
        b = await _require_backend()
        result = await b.set_model(
            args["session_id"],
            args["model"],
            timeout=timeout,
        )
        return _text(json.dumps(result, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except RuntimeError as e:
        return _err("backend_error", str(e))
    except TimeoutError:
        # set_model already retried once with a doubled timeout before
        # re-raising; a second timeout means the session is still busy.
        # Surface as a structured retryable error so the caller knows to
        # wait/retry rather than treating it as an opaque internal failure.
        waited = max(timeout * 2.0, 30.0)
        return _err(
            "model_switch_timeout",
            (
                f"Timed out after {waited:.0f}s waiting for session.resume/"
                f"/model. The session may be busy. Retry with a larger "
                f"timeout, or interrupt the turn first via "
                f"goal_action/run_slash."
            ),
            retryable=True,
        )


async def h_send_prompt_stream(args: dict):
    async def _do():
        session_id = args["session_id"]
        b = await _require_backend_for_session(session_id)
        message = args["message"]
        timeout = args.get("timeout", 300)

        ctx = _current_server.request_context
        progress_token = ctx.meta.progressToken if ctx.meta else None

        events: list[dict] = []
        final = None
        progress = 0
        async for etype, payload in b.send_prompt_stream(session_id, message, timeout=timeout):
            if etype == "__done__":
                final = payload
                break
            events.append({"type": etype, "payload": payload})
            if progress_token is not None:
                progress += 1
                summary = _summarize_event(etype, payload)
                if summary:
                    await ctx.session.send_progress_notification(
                        progress_token=progress_token,
                        progress=progress,
                        total=None,
                        message=summary,
                    )

        return _text(json.dumps({
            **(final or {}),
            "event_count": len(events),
            "events": events,
        }, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except TimeoutError:
        # Last-resort: send_prompt_stream yields structured pending on its
        # own setup/poll timeouts. If one still escapes, keep the contract
        # (pending, not internal_error) so callers can poll get_messages.
        return _text(json.dumps({
            "response": None,
            "session_id": args.get("session_id"),
            "timed_out": True,
            "status": "pending",
            "note": (
                "Timed out waiting for the session; the turn may still "
                "be running. Poll get_messages for the reply."
            ),
            "event_count": 0,
            "events": [],
        }, indent=2, default=str))
    except RuntimeError as e:
        msg = str(e)
        if "4007" in msg or "session not found" in msg.lower():
            return _err(
                "unpromptable_draft",
                "This session was created without an initial message and "
                "has no durable row yet. send_prompt_stream normally "
                "bootstraps it automatically, but the draft mapping was "
                "lost (e.g. MCP server restarted). Fix: call create_chat "
                "with initial_message, or delete this session and start "
                "a new one.",
            )
        return _err("backend_error", msg)


async def h_rename_chat(args: dict):
    async def _do():
        b = await _require_backend()
        await b.rename_session(args["session_id"], args["title"])
        return _text(json.dumps({"ok": True, "title": args["title"]}))
    return await _retry_handler(_do)


async def h_delete_chat(args: dict):
    async def _do():
        b = await _require_backend()
        await b.delete_session(args["session_id"])
        return _text(json.dumps({"ok": True, "deleted": args["session_id"]}))
    return await _retry_handler(_do)


async def h_backend_status(args: dict):
    try:
        b = await _require_backend()
        # health() is informational only — the liveness gate is is_alive()
        # inside _require_backend.  health() may 404 on working backends
        # (59681-style) so catch and report as None rather than failing.
        try:
            health = await b.health()
        except Exception:
            health = {"note": "/health not available on this instance"}
        interrupted = await b.list_interrupted_sessions()
        ws_available = None
        try:
            async with _ws_connect(b):
                ws_available = True
        except WsAuthFailure:
            ws_available = False
        except Exception:
            pass
        result = {
            "connected": True,
            "host": b.host,
            "port": b.port,
            "token_configured": bool(b.token),
            "ws_available": ws_available,
            "can_create_or_prompt_chats": ws_available,
            "health": health,
            "interrupted_sessions": interrupted,
        }
        if ws_available is False:
            result["ws_note"] = (
                "WebSocket auth is failing: list_chats/get_chat/get_messages "
                "work, but create_chat, send_prompt, fork_chat and set_model "
                "cannot. Token re-discovery and a tokenless retry were both "
                "already attempted. Restart Hermes to mint a fresh token."
            )
        if interrupted:
            result["note"] = (
                f"{len(interrupted)} session(s) look interrupted mid-turn "
                "(likely a backend restart/crash) and will not reply on "
                "their own — see interrupted_sessions for ids. Use "
                "send_prompt on one to nudge it forward, or investigate "
                "with get_messages first."
            )
        return _text(json.dumps(result, indent=2, default=str))
    except BackendAmbiguous as e:
        return _text(json.dumps({
            "connected": False,
            "error": "ambiguous_backend",
            "message": str(e),
            "candidate_ports": e.ports,
            "retryable": False,
        }, indent=2))
    except Exception as e:
        try:
            b = await _find_or_discover_backend()
            if b is not None:
                try:
                    health = await b.health()
                except Exception:
                    health = {"note": "/health not available on this instance"}
                return _text(json.dumps({
                    "connected": True,
                    "host": b.host,
                    "port": b.port,
                    "token_configured": bool(b.token),
                    "health": health,
                    "note": "Re-discovered backend after initial connection failure.",
                }, indent=2, default=str))
        except Exception:
            pass
        return _text(json.dumps({
            "connected": False,
            "error": str(e),
            "retryable": True,
        }, indent=2))


async def h_get_metrics(args: dict):
    """Get operational metrics since last snapshot.

    Returns counters for sessions created/deleted, prompts sent/succeeded/timed_out,
    bytes transferred, WebSocket reconnects, auth failures, and HTTP errors.
    Resets counters after snapshot (cumulative metrics start fresh).
    """
    try:
        b = await _require_backend()
        snap = await b._metrics.snapshot()
        return _text(json.dumps({
            "timestamp": snap.timestamp,
            "sessions_created": snap.sessions_created,
            "sessions_deleted": snap.sessions_deleted,
            "prompts_sent": snap.prompts_sent,
            "prompts_succeeded": snap.prompts_succeeded,
            "prompts_timed_out": snap.prompts_timed_out,
            "bytes_sent": snap.bytes_sent,
            "bytes_received": snap.bytes_received,
            "ws_reconnects": snap.ws_reconnects,
            "ws_auth_failures": snap.ws_auth_failures,
            "http_errors": snap.http_errors,
        }, indent=2))
    except Exception as e:
        return _err("metrics_unavailable", f"Metrics unavailable: {e}")


# ---------------------------------------------------------------------------
# Kanban tool handlers
# ---------------------------------------------------------------------------

async def h_kanban_board(args: dict):
    async def _do():
        b = await _require_backend()
        data = await b.kanban_get_board(
            status=args.get("status"), assignee=args.get("assignee"),
            board=args.get("board"), compact=args.get("compact", True),
        )
        return _text(json.dumps(data, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except KanbanApiError as e:
        return _err("kanban_error", e.detail)
    except httpx.HTTPStatusError as e:
        return _err("kanban_error", f"HTTP {e.response.status_code}: {_exc_detail(e)}")
    except Exception as e:
        return _err("kanban_error", _exc_detail(e))


async def h_kanban_show(args: dict):
    async def _do():
        b = await _require_backend()
        data = await b.kanban_show_task(args["task_id"], board=args.get("board"))
        return _text(json.dumps(data, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except KanbanApiError as e:
        return _err("kanban_error", e.detail)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return _err("kanban_not_found", f"Task {args['task_id']!r} not found")
        return _err("kanban_error", f"HTTP {e.response.status_code}: {_exc_detail(e)}")
    except Exception as e:
        return _err("kanban_error", _exc_detail(e))


async def h_kanban_create(args: dict):
    async def _do():
        b = await _require_backend()
        # triage is Optional here: None means "let the backend method pick
        # the safe default" (see kanban_create_task's safety rails). An
        # explicit true/false from the caller is forwarded as-is.
        data = await b.kanban_create_task(
            title=args["title"], body=args.get("body"),
            assignee=args.get("assignee"), priority=args.get("priority", 0),
            parents=args.get("parents"), skills=args.get("skills"),
            workspace_kind=args.get("workspace_kind"),
            workspace_path=args.get("workspace_path"),
            goal_mode=args.get("goal_mode", True),
            goal_max_turns=args.get("goal_max_turns"),
            model_override=args.get("model_override"),
            provider_override=args.get("provider_override"),
            reasoning_effort=args.get("reasoning_effort"),
            max_runtime_seconds=args.get("max_runtime_seconds"),
            triage=args.get("triage"),
            hold=bool(args.get("hold", False)),
            idempotency_key=args.get("idempotency_key"),
            completion_contract=args.get("completion_contract"),
            acceptance_criteria=args.get("acceptance_criteria"),
            board=args.get("board"),
        )
        return _text(json.dumps(data, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except KanbanApiError as e:
        return _err("kanban_error", e.detail)
    except httpx.HTTPStatusError as e:
        return _err("kanban_error", f"HTTP {e.response.status_code}: {_exc_detail(e)}")
    except Exception as e:
        return _err("kanban_error", _exc_detail(e))


async def h_kanban_update(args: dict):
    task_id = args.pop("task_id")
    board = args.pop("board", None)
    # Remove None values so we don't overwrite with null
    fields = {k: v for k, v in args.items() if v is not None}
    async def _do():
        b = await _require_backend()
        data = await b.kanban_update_task(task_id, board=board, **fields)
        return _text(json.dumps(data, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except KanbanApiError as e:
        return _err("kanban_error", e.detail)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return _err("kanban_not_found", f"Task {task_id!r} not found")
        if e.response.status_code == 409:
            return _err("kanban_conflict", f"Conflict: {_exc_detail(e)}")
        return _err("kanban_error", f"HTTP {e.response.status_code}: {_exc_detail(e)}")
    except Exception as e:
        return _err("kanban_error", _exc_detail(e))


async def h_kanban_comment(args: dict):
    async def _do():
        b = await _require_backend()
        data = await b.kanban_comment(
            args["task_id"], args["body"],
            author=args.get("author", "delegator"),
            board=args.get("board"),
        )
        return _text(json.dumps(data, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except KanbanApiError as e:
        return _err("kanban_error", e.detail)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return _err("kanban_not_found", f"Task {args['task_id']!r} not found")
        return _err("kanban_error", f"HTTP {e.response.status_code}: {_exc_detail(e)}")
    except Exception as e:
        return _err("kanban_error", _exc_detail(e))


async def h_kanban_link(args: dict):
    async def _do():
        b = await _require_backend()
        data = await b.kanban_link(
            args["parent_id"], args["child_id"],
            board=args.get("board"),
        )
        return _text(json.dumps(data, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except KanbanApiError as e:
        return _err("kanban_error", e.detail)
    except httpx.HTTPStatusError as e:
        return _err("kanban_error", f"HTTP {e.response.status_code}: {_exc_detail(e)}")
    except Exception as e:
        return _err("kanban_error", _exc_detail(e))


async def h_kanban_dispatch(args: dict):
    async def _do():
        b = await _require_backend()
        data = await b.kanban_dispatch(
            board=args.get("board"),
            dry_run=args.get("dry_run", False),
            max_n=args.get("max_n", 8),
        )
        return _text(json.dumps(data, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except KanbanApiError as e:
        return _err("kanban_error", e.detail)
    except Exception as e:
        return _err("kanban_error", _exc_detail(e))


async def h_kanban_stats(args: dict):
    async def _do():
        b = await _require_backend()
        data = await b.kanban_stats(board=args.get("board"))
        return _text(json.dumps(data, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except KanbanApiError as e:
        return _err("kanban_error", e.detail)
    except Exception as e:
        return _err("kanban_error", _exc_detail(e))


async def h_kanban_decompose(args: dict):
    async def _do():
        b = await _require_backend()
        data = await b.kanban_decompose(
            args["task_id"],
            author=args.get("author"),
            board=args.get("board"),
        )
        return _text(json.dumps(data, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except KanbanApiError as e:
        return _err("kanban_error", e.detail)
    except Exception as e:
        return _err("kanban_error", _exc_detail(e))


# ---------------------------------------------------------------------------
# Session control tool handlers
# ---------------------------------------------------------------------------

async def h_run_slash(args: dict):
    async def _do():
        b = await _require_backend_for_session(args["session_id"])
        result = await b.run_slash(args["session_id"], args["command"])
        return _text(json.dumps(result, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except Exception as e:
        return _err("slash_error", _exc_detail(e))


async def h_set_goal(args: dict):
    async def _do():
        b = await _require_backend_for_session(args["session_id"])
        result = await b.set_goal(
            args["session_id"], args["goal"],
            gates=args.get("gates"),
        )
        return _text(json.dumps(result, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except Exception as e:
        return _err("goal_error", _exc_detail(e))


async def h_goal_action(args: dict):
    async def _do():
        b = await _require_backend_for_session(args["session_id"])
        result = await b.goal_action(args["session_id"], args["action"])
        return _text(json.dumps(result, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except ValueError as e:
        return _err("invalid_action", str(e))
    except Exception as e:
        return _err("goal_error", _exc_detail(e))


async def h_set_continuous_work(args: dict):
    async def _do():
        b = await _require_backend_for_session(args["session_id"])
        result = await b.set_continuous_work(args["session_id"], args["enabled"])
        return _text(json.dumps(result, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except Exception as e:
        return _err("cw_error", _exc_detail(e))


HANDLERS = {
    "list_chats": h_list_chats,
    "get_chat": h_get_chat,
    "get_messages": h_get_messages,
    "send_prompt": h_send_prompt,
    "create_chat": h_create_chat,
    "fork_chat": h_fork_chat,
    "set_model": h_set_model,
    "send_prompt_stream": h_send_prompt_stream,
    "rename_chat": h_rename_chat,
    "delete_chat": h_delete_chat,
    "backend_status": h_backend_status,
    "get_metrics": h_get_metrics,
    # Kanban tools
    "kanban_board": h_kanban_board,
    "kanban_show": h_kanban_show,
    "kanban_create": h_kanban_create,
    "kanban_update": h_kanban_update,
    "kanban_comment": h_kanban_comment,
    "kanban_link": h_kanban_link,
    "kanban_dispatch": h_kanban_dispatch,
    "kanban_stats": h_kanban_stats,
    "kanban_decompose": h_kanban_decompose,
    # Session control tools
    "run_slash": h_run_slash,
    "set_goal": h_set_goal,
    "goal_action": h_goal_action,
    "set_continuous_work": h_set_continuous_work,
}


def _exc_detail(e: BaseException) -> str:
    """Human-readable detail for an exception, never the empty string.

    Several exceptions that reach the tool dispatcher stringify to nothing —
    a bare ``TimeoutError()`` is the common one — which rendered as
    "Internal error in list_chats: " with nothing after the colon. A report
    that names no cause is indistinguishable from a truncated one, so fall
    back to the type name (and, for an HTTP error, the status) instead.
    TimeoutError additionally gets a short actionable hint so the caller
    knows the fix is "wait longer", not "this is broken".
    """
    detail = str(e).strip()
    if detail:
        return detail
    if isinstance(e, httpx.HTTPStatusError):
        return f"{type(e).__name__} (HTTP {e.response.status_code}, empty body)"
    # On Python 3.11+ asyncio.TimeoutError is an alias of TimeoutError; a
    # single isinstance covers both spellings.
    if isinstance(e, TimeoutError):
        return (
            f"{type(e).__name__} (no message) — operation exceeded its "
            f"timeout; retry with a larger 'timeout' argument"
        )
    return f"{type(e).__name__} (no message)"


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

_current_server: Optional[Server] = None


def create_server() -> Server:
    global _current_server
    server = Server("hermes-chat-delegator")
    _current_server = server

    @server.list_tools()
    async def list_tools():
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        handler = HANDLERS.get(name)
        if handler is None:
            return _err("unknown_tool", f"Unknown tool: {name}")
        try:
            return await handler(arguments)
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return _err("internal_error",
                        f"Internal error in {name}: {_exc_detail(e)}")

    return server


async def _run():
    server = create_server()
    try:
        # Try connecting at startup, but don't fail if backend isn't up yet
        try:
            await _require_backend()
        except RuntimeError:
            logger.warning("Backend not available at startup — will retry on first tool call")
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        if _backend is not None:
            await _backend.aclose()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hermes Chat Delegator MCP Server")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    asyncio.run(_run())


if __name__ == "__main__":
    main()
