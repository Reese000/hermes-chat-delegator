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
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from functools import partial
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx
import websockets
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)


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
        r = await self._request(
            "GET", f"{self.base_url}/api/sessions",
            params={"limit": limit, "min_messages": 0, "order": "recent"},
        )
        r.raise_for_status()
        return r.json().get("sessions", [])

    async def _list_all(self) -> list[dict]:
        """List all sessions (no limit)."""
        r = await self._request(
            "GET", f"{self.base_url}/api/sessions",
            params={"limit": 500, "min_messages": 0, "order": "recent"},
        )
        r.raise_for_status()
        return r.json().get("sessions", [])

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
            if model:
                params["model"] = model
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
            # Persist to disk so it survives MCP server restarts.
            record = DraftSessionRecord(
                session_id=created["session_id"],
                live_session_id=created["live_session_id"],
                created_at=time.time(),
                title=created.get("title"),
            )
            await self._draft_sessions.add(record)
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
                        timeout: float = 15) -> dict:
        """Switch the model on an existing session via slash.exec /model.

        Resumes the session to get a live id, then sends ``/model <model>``
        through the backend's slash command handler. The backend pins the
        chosen model to this session (model_override) so subsequent prompts
        use the new model.

        If the session is busy (a turn is in-flight), sends /interrupt first
        then retries — the gateway's ``_mirror_slash_side_effects`` refuses
        to mutate a running session.
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
                                    timeout: float = 15) -> dict:
        """Interrupt a busy session then retry the model switch."""
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
            except (asyncio.TimeoutError, Exception):
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

    async def send_prompt(self, session_id: str, message: str,
                          timeout: float = 300) -> dict:
        """Send a prompt via WebSocket. Returns when agent finishes or timeout.

        For draft sessions (created without initial_message), submits the
        first prompt against the live session id to bootstrap the durable
        row, then cleans up the draft mapping.

        Internally uses the streaming approach to keep the WebSocket alive
        during message polling. Closing the WS immediately after send causes
        the backend to reap the session as orphaned before the agent starts.
        """
        # Use send_prompt_stream internally to keep WS alive throughout polling,
        # then return just the final response in send_prompt's traditional format.
        final_result = None
        async for etype, payload in self.send_prompt_stream(session_id, message, timeout=timeout):
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
                                 timeout: float = 300):
        """Yields (event_type, payload) tuples; last yield is ('__done__', final_dict).

        For draft sessions, submits against the live id directly (same as
        send_prompt's draft path). Removes the draft record after first prompt.
        """
        draft_record = await self._draft_sessions.get(session_id)

        async with _ws_connect(self) as ws:
            if draft_record:
                live_id = draft_record.live_session_id
            else:
                # Resume
                await ws.send(json.dumps({
                    "jsonrpc": "2.0", "id": 1, "method": "session.resume",
                    "params": {"session_id": session_id},
                }))
                resp = await _recv_response(ws, 1, timeout=15)
                if "error" in resp:
                    raise RuntimeError(f"resume failed: {resp['error']}")
                live_id = resp["result"].get("session_id", session_id)

            # Submit
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 2, "method": "prompt.submit",
                "params": {"session_id": live_id, "text": message},
            }))

            deadline = time.time() + timeout
            final_text_parts: list[str] = []
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=max(1.0, deadline - time.time()),
                    )
                except asyncio.TimeoutError:
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
                    # Clean up draft record if this was a draft session
                    if draft_record:
                        await self._draft_sessions.remove(session_id)
                    content = payload.get("content", "")
                    yield ("__done__", {
                        "response": "".join(final_text_parts) or content,
                        "session_id": session_id,
                    })
                    return
                if etype == "error":
                    raise RuntimeError(f"agent error mid-turn: {payload}")

            # Timed out
            if draft_record:
                await self._draft_sessions.remove(session_id)
            yield ("__done__", {
                "response": "".join(final_text_parts),
                "session_id": session_id,
                "timed_out": True,
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


async def _port_answers(host: str, port: int, timeout: float = 2.0) -> bool:
    """Liveness probe for a pinned port with Hermes identity verification.

    Returns True only if the port answers ``/api/sessions`` with HTTP 200
    AND the response body contains a ``"sessions"`` key — the shape that
    Hermes uniquely exposes.  This prevents adopting a non-Hermes service
    (e.g. Foundry Manager on port 3000) that happens to answer HTTP on
    the same path.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"http://{host}:{port}/api/sessions")
            if r.status_code != 200:
                return False
            body = r.json()
            return isinstance(body, dict) and "sessions" in body
    except Exception:
        return False


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
            "agent is doing, use send_prompt_stream instead."
            " Note: if a human has this session's GUI tab open, using this tool "
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
            "changed again. Use the provider/model format, e.g. "
            "'openrouter/anthropic/claude-sonnet-4'. The output field contains "
            "the backend's /model command output confirming the switch."
        ),
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session to change model on"},
            "model": {"type": "string", "description": "Model in provider/model format, e.g. 'openrouter/anthropic/claude-sonnet-4'"},
            "timeout": {"type": "number", "description": "Max seconds to wait (default 15)"},
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
        b = await _require_backend()
        result = await b.send_prompt(
            args["session_id"],
            args["message"],
            timeout=args.get("timeout", 300),
        )
        return _text(json.dumps(result, indent=2, default=str))
    try:
        return await _retry_handler(_do)
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
        if args.get("cwd") and created.get("cwd") and created["cwd"] != args["cwd"]:
            created["cwd_warning"] = (
                f"Requested cwd={args['cwd']!r} but backend used {created['cwd']!r}. "
                "The path may not exist."
            )
        if args.get("initial_message"):
            created["persisted"] = True
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
    async def _do():
        b = await _require_backend()
        result = await b.set_model(
            args["session_id"],
            args["model"],
            timeout=args.get("timeout", 15),
        )
        return _text(json.dumps(result, indent=2, default=str))
    try:
        return await _retry_handler(_do)
    except RuntimeError as e:
        return _err("backend_error", str(e))


async def h_send_prompt_stream(args: dict):
    async def _do():
        b = await _require_backend()
        session_id = args["session_id"]
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
}


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
            return _err("internal_error", f"Internal error in {name}: {e}")

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
