# Adversarial Audit — hermes-chat-delegator

**Date:** 2026-07-15 (supersedes the earlier same-day audit, which was written against a
pre-`send_prompt` version of `server.py` and is now stale — its C2 "send_prompt not
implemented" no longer holds, and its line numbers are wrong.)
**Scope:** Full project — `hermes_chat_delegator/` (`server.py`, `__init__.py`,
`__main__.py`), `patches/apply_patch.sh`, `pyproject.toml`, `README.md`.
**Method:** Read as a skeptical reviewer — assume every happy-path claim is wrong until
the code proves it. Verified against the live repo state (untracked, has stale
`__pycache__`, no tests).
**Size:** ~440 LOC Python + 1 shell script.

## Summary

`send_prompt` — the headline feature — is now implemented, but **it is broken in two
independent ways**: it depends on an undeclared package (`ImportError` on a clean
install), and even when that package is present its poll loop returns the *previous*
turn's answer instead of the one it just requested. Separately, now that prompt
injection into live agent sessions actually works, the localhost auth bypass is no
longer "anyone can read/delete chats" — it's "any local process can drive your agent,"
which is a materially worse security posture than the prior audit rated. The patch
script still silently no-ops half its work, and the project remains untracked with zero
tests.

| Severity | Count |
|----------|-------|
| Critical | 3 |
| High     | 3 |
| Medium   | 5 |
| Low      | 5 |

---

## Critical

### C1 — `send_prompt` depends on `websockets`, which is not a declared dependency

`server.py:90` does `import websockets` *inside* `GuiBackend.send_prompt`.
`pyproject.toml:7-11` declares only `mcp`, `httpx`, `pydantic`. `websockets` is nowhere
in the dependency set. On a clean `pip install -e .`, the first `send_prompt` call raises
`ModuleNotFoundError: No module named 'websockets'`, surfaced to the agent as
`ERROR: Failed: ...`. It "works on this machine" only because some other tool pulled
`websockets` into the shared venv.

The import is hidden inside the function (not at module top), so nothing fails at import
time and nothing static catches it — it fails only when the feature is first exercised.

**Fix:** add `websockets>=12.0` to `dependencies`, and move the import to module scope so
a missing dep fails loudly at startup rather than mid-task.

### C2 — `send_prompt` returns a stale response (poll loop trips on the user echo)

`server.py:122-144`. The loop's completion condition is `len(msgs) > before_count`, then
it returns "the last assistant message." But submitting a prompt first appends the
**user** message to the session, so `len(msgs)` reaches `before_count + 1` *immediately*,
before the agent has produced anything. On the first 2-second poll the condition is
already true; the reversed scan for `role == "assistant"` then skips the fresh user
message and returns **the assistant reply from the previous turn**.

Failure scenario: session ends on an assistant message (count N). Caller sends
"What's the status?". Backend stores the user message (count N+1 > N). Within 2 s the new
agent reply usually is not ready, so the newest assistant message in the list is still
the *old* one — that old text is returned as if it answered the new prompt. The caller
gets a confidently-wrong, stale answer with no error.

This defeats the entire point of the tool (delegate a task, get *that task's* answer).

**Fix:** gate on assistant messages specifically. Record the last assistant message's id
(or the assistant-role count) before sending, and only return once a *newer* assistant
message exists. Ideally consume the WebSocket completion/stream event instead of polling
REST, which also removes the fixed 2 s latency floor.

### C3 — `apply_patch.sh` Patch 1 never writes to disk, and idempotency locks it out forever

`patches/apply_patch.sh:19-57`. The first `python3 -c` block reads the file, does
`content = content.replace(old, new)`, and prints `Patched _has_valid_session_token` —
but has **no `open(path, 'w')`**. The edit lives only in that process's memory and is
discarded when it exits. Patch 2 (`:60-85`) is the only block that writes.

Compounding it, the idempotency guard at `:13` (`grep -q "Trust localhost"`) matches text
that **Patch 2 also inserts**. So after any successful run, the guard trips and every
future invocation prints `Already patched — skipping` and exits 0 — meaning
`_has_valid_session_token` can never be patched by this script. After a `hermes update`
wipes the file, re-running restores only the middleware bypass, silently, while reporting
success.

**Fix:** merge both edits into one Python block with a single write-back; make the guard
verify *both* insertion sites are present; exit non-zero when a target pattern is neither
found nor already applied.

---

## High

### H1 — Localhost auth bypass now grants agent control, not just chat access

The patch makes the Hermes backend trust any `127.0.0.1` connection with no token. With
`send_prompt` now live (C2/C1 notwithstanding), the blast radius has grown:

- **Any local process** — not just this MCP server — can enumerate sessions and
  **inject prompts into running agent chats**, i.e. make the agent execute arbitrary
  instructions, plus rename/delete sessions. This is local privilege escalation into the
  agent's authority.
- **Browser reachability.** A malicious web page can POST cross-origin to
  `http://127.0.0.1:<port>`. CORS blocks the attacker from *reading* the response but does
  not block sending state-changing requests; combined with DNS-rebinding to defeat the
  origin check, and JS-side port discovery, auth removal leaves no defense in depth.

**Fix:** drop the bypass entirely. Have the MCP server read the dashboard session token
(from Hermes config/env) and send it in `_SESSION_HEADER_NAME`. That also eliminates the
re-patch-after-every-update chore and the C3 breakage.

### H2 — Auto-discovery accepts `401` and any first-matching port

`server.py:151-172`. Discovery scans every listening localhost port (via `netstat`) and
returns the first where `GET /api/sessions` yields **200 or 401**.

- Accepting `401` means "some auth-protected server rejected us" is treated as "found the
  Hermes backend" — it will bind to any unrelated local service that happens to 401.
- Ports are tried in ascending numeric order, so a lower-port impostor wins over the real
  backend on a higher port.
- Sequential probing at 2 s timeout each can add tens of seconds of startup latency, and
  blindly firing HTTP GETs at arbitrary local services can have side effects.

**Fix:** positively identify the backend — hit `/health` and check for a Hermes-specific
signature; drop `401` acceptance (a real backend answers 200 under the bypass or with a
token); probe concurrently with `asyncio.gather`.

### H3 — No reconnect: backend down at startup ⇒ server permanently dead

`server.py:181-184, 397-414`. `_connect()` runs once at process start. If discovery fails
(GUI not yet up), `_backend` stays `None` and every tool call returns
`ERROR: ... Not connected to GUI backend.` for the life of the process. Since the MCP
server is host-launched and long-lived, the user must manually restart it after the GUI
starts.

**Fix:** make `_require_backend()` lazily (re)discover when `_backend is None`, and
re-discover on connection-refused so a GUI restart self-heals.

---

## Medium

### M1 — `send_prompt` resumes into `internal_sid` but polls the caller's `session_id`

`server.py:106, 112-120, 127`. `session.resume` returns a possibly-different
`internal_sid`, which is correctly used for `prompt.submit`. But the completion poll calls
`self.get_messages(session_id)` with the *original* caller-supplied id. If the two ids
diverge (the code explicitly anticipates they might, hence the fallback at `:106`), the
poll watches the wrong session, never sees the new messages, and always times out. Poll on
the same id you submitted against.

### M2 — Not under version control

The directory is untracked in the parent `Projects` repo (`?? ./`) and has no `.git` of
its own — zero history, zero backup. The stale `__pycache__` (M4) proves at least one
refactor (`backend_client`, `process_manager`) has already been lost. Commit it with a
`.gitignore` covering `__pycache__/` and `*.egg-info`.

### M3 — Zero tests

No test files exist. The deterministic logic (timestamp formatting `:262-276`, port
parsing `:157-161`, the C2 poll-completion predicate, handler output shapes) is trivially
unit-testable, and `GuiBackend` can be exercised against `httpx.MockTransport` with a fake
WebSocket — no live GUI required. The C2 stale-response bug is exactly what one poll-loop
test would have caught.

### M4 — Stale `__pycache__` for deleted modules

`hermes_chat_delegator/__pycache__/` holds compiled `backend_client.cpython-311.pyc` and
`process_manager.cpython-311.pyc` whose source no longer exists. Harmless at runtime but
misleading. Delete and gitignore.

### M5 — `get_messages` returns unbounded payloads by default

`server.py:305-318`. With `last_n=0` (the default) the entire message history is JSON-
dumped into one tool result. `send_prompt` (`:92, 127`) also pulls full history on every
2 s poll. Long sessions blow out the caller's context and waste bandwidth. Default
`last_n` to a sane cap (e.g. 20) and have the poll fetch only a tail.

---

## Low

- **L1 — Windows-only discovery, undocumented.** `server.py:153-154` shells out to
  `netstat -ano` and parses Windows-format output. `pyproject.toml` claims only
  `requires-python >= 3.10` with no OS marker. Document the constraint or use
  `psutil.net_connections` for portability.
- **L2 — New `httpx.AsyncClient` per request** (`server.py:47-86`). Have `GuiBackend` own
  one client for connection reuse.
- **L3 — `last_active` sort assumes numeric epoch** (`server.py:260`). The key returns the
  raw value when truthy; if the backend ever emits ISO-8601 strings, the sort raises
  `TypeError` (str vs int) and the whole `list_chats` call fails. Normalize defensively.
- **L4 — `__main__.py` calls `main()` at import** with no `if __name__ == "__main__"`
  guard, so importing the module programmatically boots the server.
- **L5 — Dead/unused code.** `Any` imported but unused (`server.py:21`); `health` assigned
  and never read in `_connect()` (`server.py:411`); the `'localhost'` branch in both patch
  bypasses is dead — `request.client.host` is always an IP, never the literal string.

---

## README / doc drift

- README lists **10** tools; the server implements **7**. Phantom tools:
  `send_prompt_stream`, `create_chat`, `fork_chat` (`README.md:36, 37, 40`). The Usage
  block (`:56`) demonstrates `create_chat`, which does not exist — any agent following the
  README will call a nonexistent tool.
- README references `patches/web_server_localhost_bypass.patch` (`:83, 97`); no such file
  exists — only `apply_patch.sh`.
- README architecture note cites endpoint `/api/sessions/{id}/chat` (`:75`); the server
  uses WebSocket JSON-RPC (`prompt.submit`) for sends, not that REST path.

---

## What's good

- Clean, consistent shape: one handler per tool, a `HANDLERS` dispatch table, uniform
  `_text`/`_err` helpers.
- Every HTTP call sets an explicit timeout and uses `raise_for_status()`.
- Top-level `call_tool` catch-all (`server.py:388-392`) degrades a handler bug to an error
  string instead of killing the stdio server.
- Logging is directed to stderr (`server.py:429-433`), correct for a stdio MCP server —
  keeps stdout clean for the protocol.
- Env-var overrides (`HERMES_GUI_HOST` / `HERMES_GUI_PORT`) give an escape hatch when
  discovery misfires.

---

## Recommended order of work

1. **C1** — declare `websockets`; move the import to module scope. One line; unblocks the
   whole feature.
2. **C2 / M1** — fix the poll to gate on a *new assistant* message and poll the id you
   submitted against. Add the unit test that proves it (M3 starter).
3. **C3** — fix or (better) obsolete `apply_patch.sh` via the H1 token approach.
4. **H1** — replace the localhost bypass with session-token auth. Removes the largest
   security exposure and the re-patch chore in one move.
5. **H2 / H3** — positive backend identification + lazy reconnect.
6. **README** — cut phantom tools and the phantom patch file; fix the endpoint note.
7. **M2 / M4** — commit the project; delete and gitignore `__pycache__`.
8. **M5 / Low** — payload caps, client reuse, defensive timestamp handling.
