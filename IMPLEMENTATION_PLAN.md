# Implementation Plan — hermes-chat-delegator v2

**Date:** 2026-07-16
**Author:** Claude Code (planning pass), grounded against the live `hermes-agent` install at
`C:\Users\reese\AppData\Local\hermes\hermes-agent\` (specifically `tui_gateway/server.py`,
the process the delegator's WebSocket actually talks to) and `AUDIT.md` (2026-07-15, this repo).
**Goal:** implement the 3 missing tools, then take the server from "7 of 10 tools work, one
of them was recently broken" to a small, boring, zero-surprise MCP server that a fresh agent
with no memory of this conversation can pick up and use correctly on the first try.

This plan supersedes nothing in `AUDIT.md` — it incorporates every finding there as Phase 0,
because none of the new work (Phase 1) is safe to build on top of a `send_prompt` that
returns stale answers (AUDIT C2) or a discovery routine that can bind to the wrong process
(AUDIT H2).

---

## 0. Ground truth: how the backend actually works

This is the part every prior doc (including the current README) got wrong or guessed at. Two
different identifier spaces exist and every new tool must bridge them correctly:

| | Durable session id | Live session id |
|---|---|---|
| Example | `20260715_122256_4deb1e` | `4deb1e2a` (8 hex chars) |
| Where it lives | SQLite (`state.db`), forever | In-process dict `_sessions{}`, until the GUI process evicts it |
| Who uses it | REST: `GET/PATCH/DELETE /api/sessions/{id}`, `/messages` | WebSocket JSON-RPC: `session.branch`, `session.title`, `prompt.submit`, etc. |
| How to get one from the other | — | `session.resume(session_id=<durable>)` → returns a live id (fast-path reuse if already live) |

`server.py`'s existing `send_prompt` already does this bridging correctly (resume → live id →
`prompt.submit` → poll durable id via REST). **Every new WS-based tool must follow the same
pattern**, confirmed against `tui_gateway/server.py`:

- `_sess_nowait()` / `_sess()` (line 1482-1493) resolve `params["session_id"]` by doing
  `_sessions.get(session_id)` — i.e. **only live ids work**, not durable ones.
- `session.resume` (line 5530) accepts a durable id (or title) and returns a live payload;
  fast-path reuses an already-live session under `_session_resume_lock`.

### Real WS methods (verified by line number, `tui_gateway/server.py`)

**`session.create`** (line 5148) — creates a new **live-only** session (no DB row yet).
Relevant params: `title`, `messages` (seed history), `cwd`, `model`, `provider`,
`reasoning_effort`, `fast`, `source`, `profile`, `cols`, `parent_session_id` (cosmetic
UI-nesting marker only — does **not** copy history, see below).
Returns `{"session_id": <live>, "stored_session_id": <durable>, "message_count", "messages", "info": {...}}`.

> **Gotcha (must be in the tool description):** per the comment at line 5246-5251, the DB row
> is created *lazily on the first prompt*. A session created via `session.create` and never
> prompted is invisible to `list_chats`/`get_chat`/`get_messages` (which read the DB via
> REST) until `send_prompt` is called on it at least once. This is not a bug to work around —
> it's real backend behavior the tool description must warn about.

**`session.branch`** (line 8017) — clones an **existing live session's full history** into a
new persisted session + new live agent. Params: `session_id` (must be a **live** id — resume
first), `name` (optional new title). Requires the source session to already have non-empty
history (line 8028-8029: `if not history: return _err(rid, 4008, "nothing to branch — send a
message first")`). Returns `{"session_id": <new live id>, "title", "parent": <old durable key>}`.

> **Gotcha:** unlike `session.create`, this response does **not** include the new durable id
> directly (no `stored_session_id` key). But `db.create_session(new_key, ...)` runs
> synchronously before the RPC returns (line 8049), so the row is immediately queryable via
> REST — no lazy-row problem here. Resolution strategy (Section 3.2): call
> `GET /api/sessions?limit=5&order=recent` right after and take the first row whose
> `parent_session_id` equals the source's durable id (confirmed present in `get_chat` output
> — see the `"parent_session_id"` field already returned today).

**`prompt.submit`** (line 8407) — returns immediately: `{"status": "streaming"}`. All actual
turn progress arrives asynchronously as WebSocket notifications sharing one envelope
(`_emit`, line 1143):

```json
{"jsonrpc": "2.0", "method": "event", "params": {"type": "<event-type>", "session_id": "<live id>", "payload": {...}}}
```

Event types seen in the turn-execution path (grep of `_emit(` call sites): `message.start`,
`message.delta`, `message.complete`, `reasoning.delta`, `thinking.delta`, `tool.start`,
`tool.generating`, `tool.complete`, `status.update`, `error`, `session.info`,
`approval.request`. This is the entire feed `send_prompt_stream` needs to consume.

> **Addendum, discovered 2026-07-16 mid-implementation (not in the original 2026-07-15
> `AUDIT.md` — added here as C4, see Phase 0 table):** `session.resume` and `prompt.submit`
> both hijack the target session's live transport. `write_json` (line 1122-1140) routes any
> `event`-method frame to whatever object is stored at `session["transport"]`, and that field
> gets overwritten to `current_transport()` — the connection the *current* RPC call arrived
> on — inside `prompt.submit`'s handler (line 8417-8418) and inside the payload builders used
> by `session.resume`/`session.create` (the same `"transport": current_transport() or
> _stdio_transport` pattern at lines 4664, 5242, 5494, 6034). Every WS-based tool in this repo
> opens its own short-lived `websockets.connect()`, issues its call, and closes — so it
> becomes the session's transport owner for a moment, then vanishes, leaving
> `session["transport"]` pointing at a dead socket. If a human has that session's tab open in
> the GUI, calling any of these tools on it silently kills that tab's live updates (message
> deltas, tool-call progress) for the rest of that browser/desktop session — **not a data-loss
> bug** (REST/DB still records everything; `get_messages` keeps working), purely a live-view
> staleness bug. It self-heals the moment the GUI's own frontend calls `session.resume` again
> (reopening/refocusing the tab), which reclaims the transport. `prompt.background` (line
> 9991) does **not** avoid this — it spawns an unrelated detached sub-agent task, not a
> continuation of the visible chat, and doesn't address the resume-step hijack anyway. No
> clean fix exists within the current backend's single-transport-per-session design; this repo
> can only avoid making it worse (see C4 in Phase 0) — a real fix would need the Hermes backend
> itself to support multiple subscribers per session, which is out of scope here.

---

## Phase 0 — Foundation fixes (blocking; land before any new tool)

Pulled directly from `AUDIT.md`. Every new tool in Phase 1 reuses `GuiBackend`, so an unfixed
`GuiBackend` bug (unbounded payloads, one-shot connect, wrong-id polling) gets inherited by
every tool built on top of it. Order matches the audit's "Recommended order of work."

| # | Fix | File / lines | Change |
|---|---|---|---|
| C1 | `websockets` undeclared dependency | `pyproject.toml:7-11`, `server.py:100` | Add `"websockets>=12.0"` to `dependencies`. Move `import websockets` to module top so a missing dep fails at process start, not mid-tool-call. |
| C2 | `send_prompt` returns the *previous* turn's answer | `server.py:98-136` | Before submitting, record `last_assistant_id` = id of the last `role=="assistant"` message (not just `before_count`). Poll until a message with `role=="assistant"` and `id > last_assistant_id` exists; return that one. Eliminates the race where the user-echo bumps `len(msgs)` before the agent has replied. |
| C3 | `apply_patch.sh` Patch 1 never writes; idempotency guard self-locks | `patches/apply_patch.sh:19-57` | Superseded entirely by Phase 4 (token auth) — delete this script rather than fix it. |
| C4 | `session.resume`/`prompt.submit` hijack the target session's live transport, silently freezing a human's open GUI tab on that session (see addendum above) | `server.py`'s every WS-based `GuiBackend` method | Cannot be fully fixed client-side (the backend has no "resume without claiming transport ownership" mode). Mitigate: (1) document this loudly in every affected tool's description — `send_prompt`, `send_prompt_stream`, `fork_chat`, `create_chat` — so a caller knows using them on a session someone is actively watching will freeze that tab's live view until reopened; (2) `backend_status` or a new lightweight check could warn when the target `session_id` looks actively viewed (best-effort — no reliable "tab is focused" signal exists over REST); (3) file the real fix as a Hermes backend issue (multi-subscriber transport) — out of scope for this repo. |
| H1 | Localhost bypass = any local process can drive the agent | patch script + `web_server.py` (external) | Phase 4 below. |
| H2 | Discovery accepts `401` + first-ascending-port | `server.py:151-172` | Require `200` only; probe ports concurrently via `asyncio.gather`; positively identify via `/health` payload shape (see Phase 3.2). |
| H3 | No reconnect — backend down at startup = permanently dead process | `server.py:166-171, 384-396` | `_require_backend()` becomes async and lazily re-runs discovery when `_backend is None`; wrap every `GuiBackend` call with one retry-after-rediscover on `httpx.ConnectError`. |
| M1 | `send_prompt` polls the caller's `session_id`, not the resumed `internal_sid` | `server.py:106-127` | Poll `internal_sid` throughout (same fix shape needed for every new WS-based tool — see Phase 1). |
| M2 | Untracked, zero history | whole repo | `git init && git add -A && git commit` once Phase 0-1 land (do this last, see Phase 5). |
| M3 | Zero tests | whole repo | Phase 5. |
| M4 | Stale `__pycache__` for deleted modules | `hermes_chat_delegator/__pycache__/` | Delete; add `.gitignore`. |
| M5 | `get_messages` / poll loop pull unbounded history | `server.py:77-81, 292-305` | Default `last_n` to `20` (was `0`=unlimited); `send_prompt`'s poll should request only a tail (e.g. last 5), not full history, each 2s. |
| L1 | Windows-only `netstat` discovery, undocumented | `server.py:153-154` | Either document `requires-python` + OS constraint in `pyproject.toml`, or switch to `psutil.net_connections()` for portability. Recommend: document now, portability is not in scope for this pass. |
| L2 | New `httpx.AsyncClient` per call | `server.py:47-86` | `GuiBackend.__init__` owns one `httpx.AsyncClient` (`self._client`), reused across calls; closed on process exit. |
| L3 | `last_active` sort assumes numeric epoch | `server.py:260` | `key=lambda s: _safe_epoch(s.get("last_active"))` where `_safe_epoch` coerces str/None to `0.0` instead of raising `TypeError`. |
| L4 | `__main__.py` has no `if __name__ == "__main__"` guard | `__main__.py` | Add the guard — importing the module currently boots the server as a side effect. |
| L5 | Dead code: unused `Any` import, unused `health` var, dead `'localhost'` string branch | `server.py:21, 411` + patch script | Delete. `request.client.host` is always an IP; drop the literal `'localhost'` string check everywhere it appears. |

---

## Phase 1 — The three missing tools

### 1.1 `create_chat`

**Tool schema** (append to `TOOLS` list in `server.py`):

```python
Tool(
    name="create_chat",
    description=(
        "Create a new chat session in the Hermes GUI. Returns immediately with the new "
        "session's IDs. IMPORTANT: the new session is NOT yet visible to list_chats/get_chat "
        "/get_messages until a message has actually been sent in it — Hermes only persists "
        "the session row on first prompt. Pass `initial_message` to send a first prompt "
        "immediately and get a session that's fully visible right away; omit it to create an "
        "empty draft session (matches what the GUI does when you click 'New chat')."
    ),
    inputSchema={"type": "object", "properties": {
        "title": {"type": "string", "description": "Optional title. Untitled if omitted."},
        "cwd": {"type": "string", "description": "Optional working directory for the new session."},
        "model": {"type": "string", "description": "Optional model override, e.g. 'xiaomi/mimo-v2.5'."},
        "initial_message": {
            "type": "string",
            "description": (
                "Optional first message to send immediately. If provided, this tool waits for "
                "the agent's reply (like send_prompt) and the returned session is immediately "
                "visible to all other tools. If omitted, an empty draft session is created "
                "that other tools cannot see until you call send_prompt on it."
            ),
        },
        "timeout": {"type": "number", "description": "Max seconds to wait for initial_message's reply (default 300). Ignored if initial_message is omitted."},
    }},
),
```

**`GuiBackend` method:**

```python
async def create_session(self, title="", cwd="", model="", timeout=15) -> dict:
    """Create a new live session via session.create. Does NOT send a first prompt."""
    async with websockets.connect(self.ws_url) as ws:
        params = {}
        if title:
            params["title"] = title
        if cwd:
            params["cwd"] = cwd
        if model:
            params["model"] = model
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session.create", "params": params}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if "error" in resp:
            raise RuntimeError(f"session.create failed: {resp['error']}")
        result = resp["result"]
        return {
            "session_id": result["stored_session_id"],   # durable id — this is what we expose
            "live_session_id": result["session_id"],       # internal, exposed for debugging only
            "title": title or "untitled",
        }
```

**Handler:**

```python
async def h_create_chat(args: dict):
    b = _require_backend()
    try:
        created = await b.create_session(
            title=args.get("title", ""),
            cwd=args.get("cwd", ""),
            model=args.get("model", ""),
        )
        if msg := args.get("initial_message"):
            result = await b.send_prompt(created["session_id"], msg, timeout=args.get("timeout", 300))
            created["initial_response"] = result["response"]
            created["persisted"] = True
        else:
            created["persisted"] = False
            created["note"] = "Empty draft — invisible to list_chats/get_chat until a message is sent."
        return _text(json.dumps(created, indent=2))
    except Exception as e:
        return _err(f"Failed: {e}")
```

**Edge cases to handle explicitly (not silently swallow):**
- `model` value not recognized by the backend → `session.create` still succeeds (model
  validation happens at agent-build time, which is deferred/async); document that a bad model
  name will surface as an error on the *first* `send_prompt`/`initial_message`, not on
  `create_chat` itself. Say this in the tool description.
- `cwd` that doesn't exist → per backend code (`explicit_cwd` check, line 5164-5167), a
  nonexistent path is silently *not* treated as an explicit cwd override and falls back to the
  launch directory. Surface this: after creation, if `cwd` was passed, read back
  `result["info"]["cwd"]` and warn in the response if it doesn't match what was requested.

### 1.2 `fork_chat`

**Tool schema:**

```python
Tool(
    name="fork_chat",
    description=(
        "Branch a new chat session from an existing one, copying its full message history. "
        "Equivalent to clicking 'Branch' in the GUI. The source session must already have at "
        "least one message — forking an empty/draft session fails with a clear error."
    ),
    inputSchema={"type": "object", "properties": {
        "session_id": {"type": "string", "description": "Durable session ID to fork from."},
        "title": {"type": "string", "description": "Optional title for the new branch (defaults to '<source title> (branch)')."},
    }, "required": ["session_id"]},
),
```

**`GuiBackend` method:**

```python
async def branch_session(self, session_id: str, title: str = "", timeout: float = 15) -> dict:
    async with websockets.connect(self.ws_url) as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session.resume", "params": {"session_id": session_id}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if "error" in resp:
            raise RuntimeError(f"resume failed: {resp['error']}")
        live_id = resp["result"].get("session_id", session_id)

        params = {"session_id": live_id}
        if title:
            params["name"] = title
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "session.branch", "params": params}))
        resp2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if "error" in resp2:
            err = resp2["error"]
            if err.get("code") == 4008:
                raise RuntimeError("Cannot fork an empty chat — send at least one message in the source session first.")
            raise RuntimeError(f"session.branch failed: {err}")
        result = resp2["result"]

    # session.branch's response omits the new durable id — resolve it by matching
    # parent_session_id against the source's durable id, most-recent first.
    sessions = await self._list_all()
    for s in sessions:
        if s.get("parent_session_id") == session_id:
            return {"session_id": s["id"], "title": result.get("title"), "forked_from": session_id}
    # Should not happen (branch persists synchronously) — fail loudly instead of guessing.
    raise RuntimeError(
        f"session.branch reported success (title={result.get('title')!r}) but no session with "
        f"parent_session_id={session_id!r} was found in /api/sessions. Backend behavior may have "
        f"changed — check tui_gateway/server.py's session.branch handler."
    )
```

**Handler:**

```python
async def h_fork_chat(args: dict):
    b = _require_backend()
    try:
        result = await b.branch_session(args["session_id"], title=args.get("title", ""))
        return _text(json.dumps(result, indent=2))
    except Exception as e:
        return _err(f"Failed: {e}")
```

### 1.3 `send_prompt_stream`

**Constraint that shapes the design:** MCP tool calls are request/response, not a native
streaming transport. But the installed `mcp` SDK (verified: `mcp/server/lowlevel/server.py`,
`mcp/server/session.py:291`) supports **progress notifications** — a tool handler can push
incremental `(progress, total, message)` updates tied to the *caller-supplied*
`_meta.progressToken` on the original request, before returning its final result. Not every
client sends a progress token; when it doesn't, we degrade gracefully to "collect everything
and return it at the end" — functionally close to `send_prompt`, just with more detail in the
final payload.

**Tool schema:**

```python
Tool(
    name="send_prompt_stream",
    description=(
        "Send a prompt into a GUI chat session and stream turn progress (assistant text as "
        "it's generated, tool calls as they start/finish, reasoning if enabled) as MCP "
        "progress notifications while the agent works, then return the full transcript of "
        "the turn plus the final answer. If your MCP client doesn't display progress "
        "notifications, this behaves like send_prompt but the final result additionally "
        "includes every intermediate event (tool calls made, reasoning steps) for post-hoc "
        "inspection — useful for debugging what the agent actually did, not just its final answer."
    ),
    inputSchema={"type": "object", "properties": {
        "session_id": {"type": "string", "description": "Session to prompt into"},
        "message": {"type": "string", "description": "The user message"},
        "timeout": {"type": "number", "description": "Max seconds to wait (default 300)"},
    }, "required": ["session_id", "message"]},
),
```

**`GuiBackend` method** (async generator; handler adapts it to progress notifications):

```python
async def send_prompt_stream(self, session_id: str, message: str, timeout: float = 300):
    """Yields (event_type, payload) tuples as they arrive; last yield is ('__done__', final_result_dict)."""
    async with websockets.connect(self.ws_url) as ws:
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session.resume", "params": {"session_id": session_id}}))
        resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
        if "error" in resp:
            raise RuntimeError(f"resume failed: {resp['error']}")
        live_id = resp["result"].get("session_id", session_id)

        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "prompt.submit", "params": {"session_id": live_id, "text": message}}))

        deadline = time.time() + timeout
        final_text_parts = []
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(1.0, deadline - time.time()))
            except asyncio.TimeoutError:
                break
            evt = json.loads(raw)
            if evt.get("method") != "event":
                continue
            params = evt.get("params", {})
            if params.get("session_id") != live_id:
                continue  # another session's traffic on the same socket — ignore
            etype = params.get("type")
            payload = params.get("payload", {})
            yield (etype, payload)
            if etype == "message.delta":
                final_text_parts.append(payload.get("delta") or payload.get("text") or "")
            if etype == "message.complete":
                yield ("__done__", {"response": "".join(final_text_parts) or payload.get("content", ""), "session_id": session_id})
                return
            if etype == "error":
                raise RuntimeError(f"agent error mid-turn: {payload}")
        # Timed out — return whatever text accumulated rather than nothing.
        yield ("__done__", {"response": "".join(final_text_parts), "session_id": session_id, "timed_out": True})
```

> Note: exact field name for delta text (`delta` vs `text`) inside `message.delta`'s payload
> is not yet confirmed from the audit alone — **verify against one live captured event during
> implementation** (log the raw payload once, adjust the key name, done). Flagging this now
> rather than guessing wrong in a plan that claims certainty it doesn't have.

**Handler** (uses the low-level `Server`'s `request_context` to get the progress token):

```python
async def h_send_prompt_stream(args: dict):
    b = _require_backend()
    session_id = args["session_id"]
    message = args["message"]
    timeout = args.get("timeout", 300)

    ctx = _current_server.request_context
    progress_token = ctx.meta.progressToken if ctx.meta else None

    events = []
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
                summary = _summarize_event(etype, payload)  # short human-readable line
                await ctx.session.send_progress_notification(
                    progress_token=progress_token, progress=progress, total=None, message=summary,
                )
    except Exception as e:
        return _err(f"Failed: {e}")

    return _text(json.dumps({**(final or {}), "event_count": len(events), "events": events}, indent=2, default=str))
```

`_summarize_event` is a small pure function (easy to unit test, per M3): turns
`("tool.start", {"name": "terminal", ...})` into `"tool: terminal started"`,
`("message.delta", {"delta": "Sure, "})` into the delta text itself, etc. — one line, one
`if/elif` per known event type, default to `str(payload)[:80]` for unknown types (forward
compatible with new event types the backend might add later).

`create_server()` needs to stash the `Server` instance somewhere `h_send_prompt_stream` can
reach `request_context` from — either a module-level `_current_server` set inside
`create_server()`, or (cleaner) have `call_tool` pass `server` into handlers that need it via a
partial/closure. Decide during implementation; either is a few lines.

---

## Phase 2 — Ergonomics: usable with zero prior context

The audit and this plan both assume a reader who's already spent an hour in this codebase.
The next agent that picks up this MCP server cold will not have that. Concrete changes:

1. **Rewrite `README.md`** to match reality:
   - Cut `create_chat`'s phantom usage example (it currently shows a call that doesn't exist);
     replace with the real 10-tool table once Phase 1 lands.
   - Delete the reference to `patches/web_server_localhost_bypass.patch` (doesn't exist —
     the real file is `apply_patch.sh`) — moot anyway once Phase 4 removes the patch step.
   - Fix the architecture section: sends go over WebSocket JSON-RPC (`prompt.submit`), not
     `/api/sessions/{id}/chat` as currently claimed.
   - Add a **"Gotchas"** section up front covering: durable vs. live session ids (readers
     never need to know this to use the tools, but need to know *that a session just created
     won't show up in list_chats yet*, and that forking an empty chat fails).

2. **Every tool description carries its own "when to use X vs Y."** Specifically:
   `send_prompt` vs `send_prompt_stream` — description should say outright: "Use send_prompt
   for short interactions; use send_prompt_stream when the turn might involve multiple tool
   calls and you want visibility into what the agent is doing while you wait, or a fuller
   record afterward." An agent with no context should never have to guess which of two
   similarly-named tools to reach for.

3. **Structured errors instead of bare strings.** Every handler currently returns
   `f"ERROR: Failed: {e}"` — a raw Python exception message, no error code, no indication of
   whether retrying makes sense. Replace `_err()`'s contract:

   ```python
   def _err(code: str, msg: str, retryable: bool = False) -> list[TextContent]:
       return _text(json.dumps({"error": {"code": code, "message": msg, "retryable": retryable}}))
   ```

   Codes: `"not_connected"` (backend unreachable — retryable after backend restart),
   `"session_not_found"`, `"empty_session"` (fork on empty chat), `"backend_error"` (unexpected
   REST/WS failure — includes the raw upstream message), `"timeout"`. This lets a calling
   agent branch on `error.code` instead of substring-matching prose, and `retryable` tells it
   whether to just try again.

4. **Sane defaults everywhere (ties to M5):** `get_messages.last_n` defaults to 20, not
   unbounded. `send_prompt`/`send_prompt_stream` timeout defaults documented as "300s, most
   turns finish in under 60s" so a caller isn't left guessing whether 300 is normal or a sign
   something's stuck.

5. **`backend_status` becomes a real self-check, not just "did `/health` respond."** Add: is
   the auth patch (pre-Phase-4) or token (post-Phase-4) actually working — i.e. does an
   authenticated-looking probe succeed. Post-Phase-4 this simplifies to "token present and
   accepted."

---

## Phase 3 — Resilience hardening

1. **Lazy reconnect (H3):** `_require_backend()` becomes:

   ```python
   async def _require_backend() -> GuiBackend:
       global _backend
       if _backend is None:
           port = await auto_discover_backend()
           if port is None:
               raise RuntimeError("Not connected to GUI backend — is Hermes running?")
           _backend = GuiBackend(os.environ.get("HERMES_GUI_HOST", "127.0.0.1"), port)
       return _backend
   ```

   Additionally, wrap each handler's top-level `try` to catch `httpx.ConnectError` /
   `ConnectionRefusedError` specifically, set `_backend = None`, and retry the operation once
   before surfacing an error — so a GUI restart self-heals the next tool call instead of
   requiring the MCP server process itself to be restarted.

2. **Positive backend identification (H2):** `/health`'s response shape (already seen live:
   `{"error": "Headless backend (hermes serve): web UI disabled..."}` when there's no GUI, and
   presumably a normal payload otherwise) should be checked for a Hermes-specific marker
   (e.g. a `version` or `service` key) rather than accepting any `200`/`401`. Probe candidate
   ports concurrently with `asyncio.gather(*[probe(p) for p in ports])` instead of sequential
   scanning — cuts worst-case discovery latency from `O(ports) × 2s` to `~2s`.

3. **Connection reuse (L2):** `GuiBackend.__init__` creates one `httpx.AsyncClient(timeout=...)`
   used by every method; close it in a `GuiBackend.aclose()` called from a `finally` in `_run()`.

4. **Defensive timestamp/type handling (L3):** small `_safe_epoch(v)` helper — `float(v)` in a
   `try/except (TypeError, ValueError)` returning `0.0` on failure — used in the `list_chats`
   sort key.

5. **Concurrency safety for the new WS-based tools:** each of `create_session`,
   `branch_session`, `send_prompt`, `send_prompt_stream` opens its **own** short-lived
   `websockets.connect()` (matches the existing `send_prompt` pattern) rather than sharing one
   long-lived socket — simplest correct option, avoids needing to demultiplex concurrent
   in-flight calls' events on a shared connection. Trade-off (worth stating, not hiding):
   under heavy concurrent tool use this opens many WS connections; acceptable at this
   server's actual call volume (a handful of concurrent chats, not hundreds).

---

## Phase 4 — Security: kill the localhost auth bypass (H1)

This is the one item in this plan that touches code **outside** this repo (the installed
Hermes CLI at `hermes_cli/web_server.py`), so it needs explicit sign-off before touching it —
flagging that now rather than burying it.

**Proposed fix:** instead of patching `web_server.py` to trust any `127.0.0.1` connection
(current state — grants *any local process* the ability to inject prompts into your running
agent sessions, per AUDIT H1), have `GuiBackend` read the real dashboard session token
(`HERMES_DASHBOARD_SESSION_TOKEN`, already referenced in your Hermes memory notes) from the
environment and send it via the `_SESSION_HEADER_NAME` header on every REST/WS call, exactly
as a normal authenticated dashboard client would. This:

- Closes the local-privilege-escalation hole entirely (AUDIT H1).
- Deletes `patches/apply_patch.sh` and the "re-apply after every `hermes update`" chore
  (obsoletes AUDIT C3 rather than fixing it).
- Removes a whole setup step from the README — directly serves "usable with zero prior
  context": there's no core-file patching, no GUI restart required, just install and set one
  env var (or read it automatically if Hermes exposes it in a predictable config location —
  worth checking `~/.hermes/` for where this token is stored before assuming it must be
  hand-copied).

**This phase needs your explicit go-ahead before implementation**, since it changes what
"localhost" access means for a real running agent, and because it touches a file outside this
repo (in the installed Hermes app, not `hermes-chat-delegator/`).

---

## Phase 5 — Testing & packaging

1. **Unit tests** (`tests/test_server.py`, new): `GuiBackend` exercised against
   `httpx.MockTransport` for REST calls and a small fake-websocket class (an async
   iterator yielding canned JSON-RPC frames) for `send_prompt`/`create_session`/
   `branch_session`/`send_prompt_stream`. Specifically test:
   - The C2 regression: submit a prompt, have the mock return the *old* assistant message
     first, assert the fixed poll logic does **not** return it.
   - `branch_session`'s empty-history error path returns the friendly message, not a raw
     `KeyError`/backend error code.
   - `_safe_epoch` and `_summarize_event` (pure functions — trivial, fast, no I/O).
2. **Live smoke test script** (`scripts/smoke_test.py`): the exact sequence already run
   manually in this conversation — `backend_status` → `list_chats` → `get_chat` →
   `get_messages` → `rename_chat` round-trip → `create_chat` (draft + with
   `initial_message`) → `fork_chat` → `send_prompt_stream` → `delete_chat` cleanup of
   everything this script created. Safe to run repeatedly since it cleans up after itself
   (unlike this session, which had to ask you to name disposable real sessions because
   `create_chat` didn't exist yet).
3. **Packaging (M2/M4):** delete `hermes_chat_delegator/__pycache__/`; add `.gitignore`
   (`__pycache__/`, `*.egg-info/`, `*.pyc`); `git init && git add -A && git commit -m
   "Initial commit: hermes-chat-delegator v2"` — do this **after** Phase 0-2 land so the
   first commit isn't already carrying known bugs.

---

## Rollout order

```
Phase 0 (foundation fixes)  ──┬──> Phase 1 (3 new tools)  ──> Phase 2 (ergonomics/README)
                               │                                       │
                               └──> Phase 3 (resilience)  ─────────────┤
                                                                        ▼
                               Phase 4 (security — needs your go-ahead)
                                                                        │
                                                                        ▼
                                                              Phase 5 (tests + packaging + commit)
```

Phase 0 blocks everything (new tools reuse the buggy `GuiBackend`). Phases 1-3 can proceed in
parallel once Phase 0 lands. Phase 4 is independent but should land before Phase 5's first
commit so the security fix isn't a follow-up patch to already-committed vulnerable code.
Phase 5 is last by construction (tests need the real behavior to test against; the commit
should capture the finished state).

## Acceptance criteria

- [ ] All 10 README-advertised tools exist, are registered in `TOOLS`/`HANDLERS`, and are
      individually exercised live (not just unit-tested) at least once, the same way the 7
      existing tools were verified in this conversation.
- [ ] `send_prompt` returns the *new* turn's answer, verified with a test that would have
      caught AUDIT C2 (assert against the previous-turn trap, not just the happy path).
  - [ ] `create_chat` documents and the smoke test confirms the lazy-persistence gotcha.
- [ ] `fork_chat` fails with a clear, typed error on an empty source session; succeeds and
      returns a durable id that immediately works with `get_chat`/`get_messages` on a
      non-empty one.
- [ ] `send_prompt_stream` emits progress notifications when a progress token is present and
      degrades cleanly (no crash, no hang) when it isn't.
- [ ] Backend restart (kill + relaunch Hermes GUI mid-session) no longer requires restarting
      the MCP server process (H3 fixed).
- [ ] `send_prompt`, `send_prompt_stream`, `create_chat`, and `fork_chat`'s tool descriptions
      each warn that using them on a session with an open GUI tab will freeze that tab's live
      view until reopened (C4) — this is a documentation requirement since no client-side code
      fix exists.
- [ ] `patches/apply_patch.sh` and the "re-apply after update" step are gone from the README
      (contingent on Phase 4 approval).
- [ ] `git log` shows a real history for this repo.
