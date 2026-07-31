# hermes-chat-delegator

**MCP server** for delegating Hermes GUI chats — manage multiple concurrent chat sessions programmatically from any MCP-compatible client (Claude, Hermes, etc.).

This is the server that makes the following tools available to your agents:
`list_chats`, `get_chat`, `get_messages`, `send_prompt`, `send_prompt_stream`, `create_chat`, `fork_chat`, `set_model`, `rename_chat`, `delete_chat`, and `backend_status`.

---

## Quick Start

```bash
# Install (editable, from repo root)
pip install -e .

# Set the session token (same token the Hermes desktop shell uses).
# The token is set automatically if you're running inside the Hermes process tree.
# For standalone use, find it in your Hermes config or set it explicitly:
export HERMES_DASHBOARD_SESSION_TOKEN="your-token-here"

# Run (stdio transport — standard for MCP servers)
hermes-chat-delegator
```

No patching of Hermes core files required. Just install and go.

---

## Tech Stack

| Component | Library | Purpose |
|-----------|---------|---------|
| MCP server | `mcp >= 1.0.0` | stdio transport, tool registration, progress notifications |
| HTTP client | `httpx >= 0.27.0` | Async REST calls to Hermes backend |
| WebSocket | `websockets >= 12.0` | JSON-RPC session create/resume/branch/prompt |
| Package | `hatchling` (build), `pytest` (tests) | Build and test toolchain |

Python ≥ 3.10.

---

## Architecture

```
MCP Client (Claude, Hermes, etc.)
    │
    │  stdio (MCP protocol)
    ▼
hermes-chat-delegator (this server)
    │
    ├── HTTP REST ──► /api/sessions, /api/sessions/{id}, /api/sessions/{id}/messages, /health
    │                 (read ops: list, get, messages, health)
    │                 (write ops: rename via PATCH, delete via DELETE)
    │
    └── WebSocket JSON-RPC ──► /api/ws
                                session.create     — create new live session
                                session.resume     — resolve durable id → live id
                                session.branch     — fork session history
                                prompt.submit      — send a prompt
                                slash.exec         — execute /commands (e.g. /model)
                                event (unsolicited) — stream events (message.delta, tool.start, etc.)
    │
    ▼
Hermes GUI Backend (hermes serve / desktop app)
    └── tui_gateway/server.py  (e.g. at %LocalAppData%/hermes/hermes-agent/tui_gateway/server.py)
```

### Two identifier spaces

The backend uses **durable** session IDs (SQLite-persisted, e.g. `20260715_122256_4deb1e`) for REST APIs and **live** session IDs (in-process, e.g. `4deb1e2a`) for WebSocket RPCs. This server bridges them automatically — you only ever see durable IDs.

| | Durable session id | Live session id |
|---|---|---|
| Example | `20260715_122256_4deb1e` | `4deb1e2a` (8 hex chars) |
| Where it lives | SQLite (`state.db`), forever | In-process dict, evicted on GUI restart |
| Used by | REST endpoints | WebSocket JSON-RPC methods |
| How to get | From `session.create` result or REST lists | `session.resume(durable_id)` on WS |

### Auto-discovery

On startup, the server scans `netstat -ano` output for localhost listening ports, then probes each candidate concurrently:
1. **Strong match** (priority 2): Port responds to `GET /api/sessions` with a `{"sessions": [...]}` payload — confirmed as the Hermes dashboard backend.
2. **Weak match** (priority 1, fallback only): Port responds to `GET /health` with a Hermes-shaped response (has `version`, `service`, or `hermes` key).
3. **Explicit override**: Set `HERMES_GUI_PORT` to skip discovery entirely.

This two-tier probe prevents the server from locking onto the OpenAI-compatible API gateway (which also serves a Hermes-shaped `/health` but uses Bearer auth, not session tokens).

### Lazy reconnect

If the backend is unreachable at startup, the server logs a warning and retries on the first tool call. If a tool call hits `httpx.ConnectError` mid-session, the backend handle is reset and will be re-discovered on the next call — so a GUI restart self-heals without restarting the MCP server.

---

## Tools (11)

| Tool | Description | Uses |
|------|-------------|------|
| `list_chats` | List all chat sessions sorted by recent activity | REST |
| `get_chat` | Get detailed info about a specific session | REST |
| `get_messages` | Get message history (last 20 by default; supports `max_chars` and `compact`) | REST |
| `send_prompt` | Send a prompt and wait for the agent's reply (polls REST) | WS + REST |
| `send_prompt_stream` | Send a prompt with streaming progress notifications via MCP progress | WS (event-driven) |
| `create_chat` | Create a new session (optionally with a first message) | WS + REST |
| `fork_chat` | Branch a new session from an existing one (copies history) | WS + REST |
| `set_model` | Switch the AI model on an existing session mid-conversation | WS |
| `rename_chat` | Rename a session's title | REST |
| `delete_chat` | Delete a session | REST |
| `backend_status` | Check connection to the GUI backend, list interrupted sessions | REST |

### Tool details

#### `send_prompt` vs `send_prompt_stream`
- **`send_prompt`**: Use for short interactions where you just want the final answer. Sends via WS, then polls the REST messages endpoint every 2 seconds until a new assistant reply appears or the timeout expires.
- **`send_prompt_stream`**: Use when the turn might involve multiple tool calls and you want visibility into what the agent is doing, or a fuller record afterward. Consumes the WS event stream (`message.delta`, `tool.start`, `tool.complete`, `reasoning.delta`, etc.) and forwards them as MCP progress notifications. If the client doesn't support progress notifications, it behaves like `send_prompt` but returns all intermediate events in the final payload.

#### `create_chat`
Creates a new session. Two modes:
- **Draft** (no `initial_message`): Returns immediately with session IDs. The session is **not visible** to `list_chats`/`get_chat`/`get_messages` until a message is sent in it (Hermes persists session rows lazily on first prompt).
- **With `initial_message`**: Also sends the first prompt and waits for the reply. The returned session is immediately visible to all other tools.

#### `fork_chat`
Branches from an existing session, copying its full message history. The source session must have at least one message — forking an empty/draft session fails with `empty_session` error code.

#### `set_model`
Switches the model mid-conversation via `/model <provider/model>` slash command. If the session is busy (a turn in-flight), sends `/interrupt` first then retries.

### Known backend limitation (transport hijack)

Every WS-based tool (`send_prompt`, `send_prompt_stream`, `create_chat`, `fork_chat`, `set_model`) opens a short-lived WebSocket connection to the backend. The Hermes backend's `session.resume` handler overwrites the session's transport reference to the current connection. When the short-lived WS closes, the session's transport pointer becomes stale. **If a human has that session's tab open in the GUI, using any of these tools will freeze that tab's live view** (message deltas, tool-call progress no longer stream) until the user reopens or refocuses the tab, which triggers a new `session.resume` from the GUI's own frontend.

This is a known Hermes backend limitation (single-transport-per-session design), not a bug in this server. No data is lost — `get_messages` via REST always returns the full, correct history.

---

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `HERMES_GUI_HOST` | `127.0.0.1` | Backend host |
| `HERMES_GUI_PORT` | auto-discover | Backend port (set explicitly if auto-discovery fails or you want deterministic behavior) |
| `HERMES_DASHBOARD_SESSION_TOKEN` | auto from desktop shell | Auth token sent as `X-Hermes-Session-Token` header |
| `HERMES_WS_MAX_SIZE` | 64 MiB | Maximum WebSocket frame size in bytes. Set to `0` for unlimited. The `websockets` library's default (1 MiB) is too small for sessions with large transcripts — this was the root cause of permanent "frame too large" disconnects on long-running sessions. |

---

## Error Handling

All errors return structured JSON:

```json
{
  "error": {
    "code": "not_connected",
    "message": "Backend unreachable — it may have restarted. Will retry on next call.",
    "retryable": true
  }
}
```

### Error codes

| Code | Meaning | Retryable |
|------|---------|-----------|
| `not_connected` | Backend unreachable (connect refused, discovery failed) | Yes |
| `session_not_found` | Session ID not found (404 from REST) | No |
| `empty_session` | Fork attempted on a session with no messages | No |
| `timeout` | Agent turn did not complete within the requested timeout | Yes |
| `backend_error` | Unexpected REST/WS failure (includes upstream message) | Varies |
| `unknown_tool` | Tool name not in the HANDLERS dispatch table | No |
| `internal_error` | Unhandled exception in handler code | No |

---

## Gotchas

- **New sessions are invisible until prompted.** `create_chat` without `initial_message` creates a draft that won't appear in `list_chats` / `get_chat` / `get_messages` until you send a message in it. This is real backend behavior, not a bug.
- **Forking an empty chat fails.** `fork_chat` requires the source session to have at least one message. Forking a draft session returns a clear `empty_session` error.
- **Invalid model names surface late.** A bad model name in `create_chat(initial_message=...)` or `set_model` will error on `prompt.submit` / `slash.exec`, not on the WS `session.create` itself — model validation is deferred to agent-build time.
- **Session timeout.** Most agent turns finish in under 60s. The default timeout is 300s. If a turn times out, the response includes `"timed_out": true` with `"response": null` (never a stale echo of a previous reply).
- **Transport hijack on GUI tabs.** See [Known backend limitation](#known-backend-limitation-transport-hijack) above.

---

## Requirements

- Python ≥ 3.10
- Windows (auto-discovery uses `netstat -ano`; set `HERMES_GUI_PORT` on other platforms)
- A running Hermes GUI backend (desktop app or `hermes serve`), specifically its **dashboard** web server (not the OpenAI-compatible API gateway)

---

## Interconnectedness with Hermes Agent

This MCP server connects to the **Hermes Agent** project (located elsewhere in `C:\Users\reese\Projects\`) — specifically the `tui_gateway/server.py` module inside the Hermes desktop application. The relationship is:

1. **Hermes Agent** provides the GUI backend (REST + WebSocket server at `127.0.0.1:<port>`).
2. **hermes-chat-delegator** is an MCP bridge that translates between the MCP protocol (stdio JSON-RPC) and the Hermes backend's internal protocol (REST + WebSocket JSON-RPC).
3. Without a running Hermes backend, this server can start but all tools return `not_connected` errors.

The server does **not** depend on any files from the separate Hermes Agent directory — it communicates only over the network (localhost). The implementation plan and audit documents reference the Hermes Agent source at `%LocalAppData%/hermes/hermes-agent/tui_gateway/server.py` for protocol ground-truth verification, but this is a development-time reference, not a runtime dependency.

---

## Development

### Running tests

```bash
# From repo root
python -m pytest tests/ -v
```

Tests use `httpx.MockTransport` and patched `websockets` — no live GUI backend required.

### Project structure

```
hermes-chat-delegator/
├── hermes_chat_delegator/
│   ├── __init__.py          # Package marker
│   ├── __main__.py          # python -m entry point
│   └── server.py            # ~1350 lines — all of the logic
├── tests/
│   └── test_server.py       # Unit tests (mock transport, no live backend)
├── scripts/
│   └── smoke_test.py        # End-to-end smoke test (requires live backend)
├── AUDIT.md                 # Adversarial audit (findings, mostly resolved)
├── IMPLEMENTATION_PLAN.md   # v2 design plan + ground-truth analysis
├── pyproject.toml           # Project metadata and dependencies
└── README.md                # This file
```

The entire server is a single file (`server.py`, ~1350 lines). It contains:
- `GuiBackend` class — backend connection, REST + WebSocket methods (~520 lines)
- Auto-discovery logic (~80 lines)
- 11 tool definitions + 11 async handlers (~300 lines)
- MCP server setup and entry point (~60 lines)
- Helpers (`_safe_epoch`, `_summarize_event`, `_truncate_messages`, `_err`, `_text`)

---

## License

MIT
