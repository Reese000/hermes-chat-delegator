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
from functools import partial
from typing import Optional

import httpx
import websockets
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)


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


class GuiBackend:
    def __init__(self, host: str, port: int, token: str = ""):
        self.host = host
        self.port = port
        self.token = token
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5, read=15, write=5, pool=5),
            headers=self._auth_headers(),
        )

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.token:
            headers[_SESSION_HEADER_NAME] = self.token
        return headers

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/api/ws"

    async def aclose(self):
        await self._client.aclose()

    async def health(self) -> dict:
        r = await self._client.get(f"{self.base_url}/health")
        return r.json()

    async def list_sessions(self, limit: int = 50) -> list[dict]:
        r = await self._client.get(
            f"{self.base_url}/api/sessions",
            params={"limit": limit, "min_messages": 0, "order": "recent"},
        )
        r.raise_for_status()
        return r.json().get("sessions", [])

    async def _list_all(self) -> list[dict]:
        """List all sessions (no limit)."""
        r = await self._client.get(
            f"{self.base_url}/api/sessions",
            params={"limit": 500, "min_messages": 0, "order": "recent"},
        )
        r.raise_for_status()
        return r.json().get("sessions", [])

    async def get_session(self, session_id: str) -> dict:
        r = await self._client.get(f"{self.base_url}/api/sessions/{session_id}")
        r.raise_for_status()
        return r.json()

    async def get_messages(self, session_id: str, last_n: int = 20) -> list[dict]:
        r = await self._client.get(f"{self.base_url}/api/sessions/{session_id}/messages")
        r.raise_for_status()
        messages = r.json().get("messages", [])
        if last_n > 0:
            messages = messages[-last_n:]
        return messages

    async def rename_session(self, session_id: str, title: str) -> dict:
        r = await self._client.patch(
            f"{self.base_url}/api/sessions/{session_id}",
            json={"title": title},
        )
        r.raise_for_status()
        return r.json()

    async def delete_session(self, session_id: str) -> dict:
        r = await self._client.delete(f"{self.base_url}/api/sessions/{session_id}")
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
                            }
                        break  # Still the old one, keep waiting
            except Exception:
                pass

        # Timeout — return whatever we have
        msgs = await self.get_messages(durable_session_id, last_n=5)
        last = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
        return {
            "response": last.get("content", "") if last else "(timeout)",
            "session_id": durable_session_id,
            "message_count": len(msgs),
            "timed_out": True,
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
        """
        async with websockets.connect(self.ws_url) as ws:
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

        if initial_message:
            reply = await self._poll_for_reply(
                created["session_id"], last_assistant_id=None, timeout=prompt_timeout,
            )
            created["initial_response"] = reply["response"]
            if reply.get("timed_out"):
                created["initial_response_timed_out"] = True

        return created

    async def branch_session(self, session_id: str, title: str = "",
                             timeout: float = 15) -> dict:
        """Branch an existing session, copying its full history."""
        async with websockets.connect(self.ws_url) as ws:
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

    async def send_prompt(self, session_id: str, message: str,
                          timeout: float = 300) -> dict:
        """Send a prompt via WebSocket. Returns when agent finishes or timeout."""
        # Get the last assistant message id BEFORE submitting. A brand-new
        # draft session (from create_session with no prior prompt) has no
        # durable row yet, so the messages endpoint 404s — that just means
        # there's no prior baseline, not a real error.
        try:
            before_msgs = await self.get_messages(session_id, last_n=5)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                before_msgs = []
            else:
                raise
        last_assistant_id = None
        for m in reversed(before_msgs):
            if m.get("role") == "assistant":
                last_assistant_id = m.get("id")
                break

        async with websockets.connect(self.ws_url) as ws:
            # Resume
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 1, "method": "session.resume",
                "params": {"session_id": session_id},
            }))
            resp = await _recv_response(ws, 1, timeout=15)
            if "error" in resp:
                raise RuntimeError("resume failed: " + str(resp.get("error", "unknown")))
            live_id = resp["result"].get("session_id", session_id)

            # Submit
            await ws.send(json.dumps({
                "jsonrpc": "2.0", "id": 2, "method": "prompt.submit",
                "params": {"session_id": live_id, "text": message},
            }))

        # Poll via REST (durable id only — the live id 404s here, confirmed
        # live: 404 on live id, 200 on durable id for the same session).
        return await self._poll_for_reply(session_id, last_assistant_id, timeout)

    async def send_prompt_stream(self, session_id: str, message: str,
                                 timeout: float = 300):
        """Yields (event_type, payload) tuples; last yield is ('__done__', final_dict)."""
        async with websockets.connect(self.ws_url) as ws:
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
                    content = payload.get("content", "")
                    yield ("__done__", {
                        "response": "".join(final_text_parts) or content,
                        "session_id": session_id,
                    })
                    return
                if etype == "error":
                    raise RuntimeError(f"agent error mid-turn: {payload}")

            # Timed out
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


async def auto_discover_backend(host: str = "127.0.0.1") -> Optional[int]:
    """Find the Hermes GUI backend by probing localhost ports concurrently."""
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

    async def _probe(port: int) -> Optional[int]:
        try:
            async with httpx.AsyncClient(timeout=2) as c:
                r = await c.get(f"http://{host}:{port}/health")
                if r.status_code == 200:
                    body = r.json()
                    # Positive identification: look for Hermes-specific markers.
                    if any(k in body for k in ("version", "service", "hermes")):
                        return port
                # A headless Hermes backend's /health legitimately returns
                # {"error": "Headless backend ..."} — indistinguishable from
                # an unrelated service on /health alone. Confirm identity via
                # a Hermes-specific endpoint instead of guessing from the
                # shape of /health (matching "no error key" previously
                # accepted any unrelated service's bare {"status": "ok"}).
                r2 = await c.get(f"http://{host}:{port}/api/sessions")
                if r2.status_code == 200:
                    body2 = r2.json()
                    if isinstance(body2, dict) and "sessions" in body2:
                        return port
        except Exception:
            pass
        return None

    # Probe all candidate ports concurrently
    results = await asyncio.gather(*[_probe(p) for p in sorted(ports)])
    for r in results:
        if r is not None:
            return r
    return None


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_backend: Optional[GuiBackend] = None


async def _require_backend() -> GuiBackend:
    """Get the backend, lazily discovering if needed."""
    global _backend
    if _backend is None:
        host = os.environ.get("HERMES_GUI_HOST", "127.0.0.1")
        port_str = os.environ.get("HERMES_GUI_PORT", "")
        if port_str:
            port = int(port_str)
        else:
            logger.info("Auto-discovering GUI backend...")
            port = await auto_discover_backend(host)
            if port is None:
                raise RuntimeError(
                    "Not connected to GUI backend — is Hermes running? "
                    "Set HERMES_GUI_PORT to connect manually."
                )
        token = os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN", "")
        _backend = GuiBackend(host, port, token=token)
        logger.info("Connected to backend on %s:%s", host, port)
    return _backend


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
            "recent activity. Shows active status."
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
            "messages by default (20). Pass last_n=0 for full history."
        ),
        inputSchema={"type": "object", "properties": {
            "session_id": {"type": "string", "description": "Session ID"},
            "last_n": {"type": "integer", "description": "Only last N messages (default 20, 0=all)"},
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
            "with the new session's IDs. IMPORTANT: the new session is NOT yet "
            "visible to list_chats/get_chat/get_messages until a message has "
            "actually been sent in it — Hermes only persists the session row on "
            "first prompt. Pass initial_message to send a first prompt "
            "immediately and get a session that's fully visible right away; "
            "omit it to create an empty draft session (matches what the GUI "
            "does when you click 'New chat'). Note: a bad model name will "
            "surface as an error on the first send_prompt, not on create_chat. "
            "Note: if initial_message is provided and a human has this session's "
            "GUI tab open, using this tool will freeze that tab's live view until "
            "they reopen/refocus it (known backend limitation, not a bug in this server)."
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
]


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _text(text: str) -> list[TextContent]:
    return [TextContent(type="text", text=text)]


def _err(code: str, msg: str, retryable: bool = False) -> list[TextContent]:
    return _text(json.dumps({"error": {"code": code, "message": msg, "retryable": retryable}}))


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
    b = await _require_backend()
    try:
        sessions = await b.list_sessions(limit=args.get("limit", 50))
        sessions.sort(key=lambda s: _safe_epoch(s.get("last_active")), reverse=True)

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
            result.append({
                "session_id": s.get("id", ""),
                "title": s.get("title") or "untitled",
                "status": "active" if is_active else _fmt_ts(last_active),
                "model": s.get("model"),
                "source": s.get("source"),
                "message_count": s.get("message_count", 0),
                "cwd": s.get("cwd"),
            })
        return _text(json.dumps({"sessions": result, "count": len(result)}, indent=2, default=str))
    except httpx.ConnectError:
        _reset_backend()
        return _err("not_connected", "Backend unreachable — it may have restarted. Will retry on next call.", retryable=True)
    except Exception as e:
        return _err("backend_error", f"Failed: {e}")


async def h_get_chat(args: dict):
    b = await _require_backend()
    try:
        info = await b.get_session(args["session_id"])
        return _text(json.dumps(info, indent=2, default=str))
    except httpx.ConnectError:
        _reset_backend()
        return _err("not_connected", "Backend unreachable — it may have restarted.", retryable=True)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return _err("session_not_found", f"Session {args['session_id']!r} not found")
        return _err("backend_error", f"HTTP {e.response.status_code}: {e}")
    except Exception as e:
        return _err("backend_error", f"Failed: {e}")


async def h_get_messages(args: dict):
    b = await _require_backend()
    try:
        last_n = args.get("last_n", 20)
        messages = await b.get_messages(args["session_id"], last_n=last_n)
        return _text(json.dumps({
            "session_id": args["session_id"],
            "message_count": len(messages),
            "messages": messages,
        }, indent=2, default=str))
    except httpx.ConnectError:
        _reset_backend()
        return _err("not_connected", "Backend unreachable — it may have restarted.", retryable=True)
    except Exception as e:
        return _err("backend_error", f"Failed: {e}")


async def h_send_prompt(args: dict):
    b = await _require_backend()
    try:
        result = await b.send_prompt(
            args["session_id"],
            args["message"],
            timeout=args.get("timeout", 300),
        )
        return _text(json.dumps(result, indent=2, default=str))
    except httpx.ConnectError:
        _reset_backend()
        return _err("not_connected", "Backend unreachable — it may have restarted.", retryable=True)
    except Exception as e:
        return _err("backend_error", f"Failed: {e}")


async def h_create_chat(args: dict):
    b = await _require_backend()
    try:
        created = await b.create_session(
            title=args.get("title", ""),
            cwd=args.get("cwd", ""),
            model=args.get("model", ""),
            initial_message=args.get("initial_message", ""),
            prompt_timeout=args.get("timeout", 300),
        )
        # Warn if cwd was requested but didn't stick
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
                "Empty draft — invisible to list_chats/get_chat until a message is sent."
            )
        return _text(json.dumps(created, indent=2, default=str))
    except httpx.ConnectError:
        _reset_backend()
        return _err("not_connected", "Backend unreachable — it may have restarted.", retryable=True)
    except Exception as e:
        return _err("backend_error", f"Failed: {e}")


async def h_fork_chat(args: dict):
    b = await _require_backend()
    try:
        result = await b.branch_session(args["session_id"], title=args.get("title", ""))
        return _text(json.dumps(result, indent=2, default=str))
    except httpx.ConnectError:
        _reset_backend()
        return _err("not_connected", "Backend unreachable — it may have restarted.", retryable=True)
    except RuntimeError as e:
        msg = str(e)
        if "empty" in msg.lower():
            return _err("empty_session", msg)
        return _err("backend_error", msg)
    except Exception as e:
        return _err("backend_error", f"Failed: {e}")


async def h_send_prompt_stream(args: dict):
    b = await _require_backend()
    session_id = args["session_id"]
    message = args["message"]
    timeout = args.get("timeout", 300)

    # Access MCP server's request context for progress notifications
    ctx = _current_server.request_context
    progress_token = ctx.meta.progressToken if ctx.meta else None

    events: list[dict] = []
    final = None
    progress = 0
    try:
        async for etype, payload in b.send_prompt_stream(session_id, message, timeout=timeout):
            if etype == "__done__":
                final = payload
                break
            events.append({"type": etype, "payload": payload})
            if progress_token is not None:
                progress += 1
                summary = _summarize_event(etype, payload)
                if summary:  # skip empty deltas
                    await ctx.session.send_progress_notification(
                        progress_token=progress_token,
                        progress=progress,
                        total=None,
                        message=summary,
                    )
    except httpx.ConnectError:
        _reset_backend()
        return _err("not_connected", "Backend unreachable — it may have restarted.", retryable=True)
    except Exception as e:
        return _err("backend_error", f"Failed: {e}")

    return _text(json.dumps({
        **(final or {}),
        "event_count": len(events),
        "events": events,
    }, indent=2, default=str))


async def h_rename_chat(args: dict):
    b = await _require_backend()
    try:
        await b.rename_session(args["session_id"], args["title"])
        return _text(json.dumps({"ok": True, "title": args["title"]}))
    except httpx.ConnectError:
        _reset_backend()
        return _err("not_connected", "Backend unreachable — it may have restarted.", retryable=True)
    except Exception as e:
        return _err("backend_error", f"Failed: {e}")


async def h_delete_chat(args: dict):
    b = await _require_backend()
    try:
        await b.delete_session(args["session_id"])
        return _text(json.dumps({"ok": True, "deleted": args["session_id"]}))
    except httpx.ConnectError:
        _reset_backend()
        return _err("not_connected", "Backend unreachable — it may have restarted.", retryable=True)
    except Exception as e:
        return _err("backend_error", f"Failed: {e}")


async def h_backend_status(args: dict):
    try:
        b = await _require_backend()
        health = await b.health()
        return _text(json.dumps({
            "connected": True,
            "host": b.host,
            "port": b.port,
            "token_configured": bool(b.token),
            "health": health,
        }, indent=2))
    except Exception as e:
        return _text(json.dumps({
            "connected": False,
            "error": str(e),
            "retryable": True,
        }, indent=2))


HANDLERS = {
    "list_chats": h_list_chats,
    "get_chat": h_get_chat,
    "get_messages": h_get_messages,
    "send_prompt": h_send_prompt,
    "create_chat": h_create_chat,
    "fork_chat": h_fork_chat,
    "send_prompt_stream": h_send_prompt_stream,
    "rename_chat": h_rename_chat,
    "delete_chat": h_delete_chat,
    "backend_status": h_backend_status,
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
