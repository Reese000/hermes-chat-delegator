# hermes-chat-delegator

MCP server for delegating Hermes GUI chats — manage multiple concurrent chat sessions programmatically from any MCP-compatible client.

## Quick Start

```bash
# Install
pip install -e .

# Set the session token (same token the Hermes desktop shell uses)
# The token is set automatically if you're running inside the Hermes process tree.
# For standalone use, find it in your Hermes config or set it explicitly:
export HERMES_DASHBOARD_SESSION_TOKEN="your-token-here"

# Run (stdio transport — standard for MCP servers)
hermes-chat-delegator
```

No patching of Hermes core files required. No re-applying after `hermes update`. Just install and go.

## Gotchas

- **New sessions are invisible until prompted.** `create_chat` without `initial_message` creates a draft that won't appear in `list_chats` / `get_chat` / `get_messages` until you send a message in it. This is real backend behavior, not a bug.
- **Forking an empty chat fails.** `fork_chat` requires the source session to have at least one message. Forking a draft session returns a clear error.
- **`send_prompt` vs `send_prompt_stream`.** Use `send_prompt` for short interactions where you just need the final answer. Use `send_prompt_stream` when the turn might involve multiple tool calls and you want visibility into what the agent is doing, or a fuller record afterward.
- **Session timeout.** Most agent turns finish in under 60s. The default timeout is 300s. If a turn times out, the response will include `"timed_out": true`.

## Architecture

```
MCP Client (Claude, Hermes, etc.)
    │
    │  stdio (MCP protocol)
    ▼
hermes-chat-delegator (this server)
    │
    ├── HTTP REST ──► /api/sessions, /api/sessions/{id}, etc.
    │                 (read ops: list, get, messages, rename, delete)
    │
    └── WebSocket JSON-RPC ──► /api/ws
                                (session.create, session.resume,
                                 session.branch, prompt.submit)
    │
    ▼
Hermes GUI Backend (hermes serve / desktop app)
```

**Two identifier spaces:** The backend uses *durable* session IDs (SQLite-persisted, e.g. `20260715_122256_4deb1e`) for REST APIs and *live* session IDs (in-process, e.g. `4deb1e2a`) for WebSocket RPCs. This server bridges them automatically — you only ever see durable IDs.

## Tools (10)

| Tool | Description |
|------|-------------|
| `list_chats` | List all chat sessions sorted by recent activity |
| `get_chat` | Get detailed info about a specific session |
| `get_messages` | Get message history (last 20 by default) |
| `send_prompt` | Send a prompt and wait for the agent's reply |
| `send_prompt_stream` | Send a prompt with streaming progress notifications |
| `create_chat` | Create a new session (optionally with a first message) |
| `fork_chat` | Branch a new session from an existing one |
| `rename_chat` | Rename a session's title |
| `delete_chat` | Delete a session |
| `backend_status` | Check connection to the GUI backend |

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `HERMES_GUI_HOST` | `127.0.0.1` | Backend host |
| `HERMES_GUI_PORT` | auto-discover | Backend port (set explicitly if auto-discover fails) |
| `HERMES_DASHBOARD_SESSION_TOKEN` | auto from desktop shell | Auth token for the backend |

## Error Handling

All errors return structured JSON:

```json
{
  "error": {
    "code": "not_connected",
    "message": "Backend unreachable — it may have restarted.",
    "retryable": true
  }
}
```

Error codes: `not_connected`, `session_not_found`, `empty_session`, `backend_error`, `timeout`, `unknown_tool`, `internal_error`.

## Requirements

- Python ≥ 3.10
- Windows (auto-discovery uses `netstat -ano`; set `HERMES_GUI_PORT` on other platforms)
- A running Hermes GUI backend (desktop app or `hermes serve`)

## License

MIT
