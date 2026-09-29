"""Unit tests for new kanban and session control tools.

Tests GuiBackend kanban methods against httpx.MockTransport and
WebSocket-based session control tools against fake websockets.
No live GUI required.

Covers F1-F7, G1-G5, H1-H3 regressions and feature tests.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hermes_chat_delegator import server
from hermes_chat_delegator.server import (
    GuiBackend,
    KanbanApiError,
    _cw_state,
    _err,
    _text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _KanbanMockTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: dict[str, tuple[int, dict]]):
        self.responses = responses
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = str(request.url.raw_path, encoding="utf-8") if isinstance(request.url.raw_path, bytes) else request.url.raw_path
        for pattern, (status, body) in sorted(self.responses.items(), key=lambda x: -len(x[0])):
            if pattern in path:
                return httpx.Response(status, json=body)
        return httpx.Response(404, json={"error": "not found"})


def _make_kanban_backend(responses):
    backend = GuiBackend("127.0.0.1", 9999, token="test-token")
    transport = _KanbanMockTransport(responses)
    backend._client = httpx.AsyncClient(transport=transport, timeout=5, headers=backend._auth_headers())
    return backend, transport


class _FakeWebSocket:
    def __init__(self, frames: list[dict]):
        self._frames = iter(frames)
        self._sent: list[str] = []

    async def send(self, data: str):
        self._sent.append(data)

    async def recv(self) -> str:
        try:
            frame = next(self._frames)
        except StopIteration:
            raise asyncio.TimeoutError()
        return json.dumps(frame)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _ws_resume_ok(live_id="live_001"):
    return {"jsonrpc": "2.0", "id": 1, "result": {"session_id": live_id}}

def _ws_cmd_dispatch_ok(output="(no output)"):
    return {"jsonrpc": "2.0", "id": 2, "result": {"output": output, "ok": True}}

def _ws_cmd_dispatch_send(notice="Goal set", message="kickoff prompt"):
    return {"jsonrpc": "2.0", "id": 2, "result": {"type": "send", "notice": notice, "message": message}}

def _ws_cmd_dispatch_4018():
    return {"jsonrpc": "2.0", "id": 2, "error": {"code": 4018, "message": "not a command"}}

def _ws_cmd_dispatch_32603():
    return {"jsonrpc": "2.0", "id": 2, "error": {"code": -32603, "message": "internal error"}}

def _ws_cmd_dispatch_general_error(code=5000, msg="generic error"):
    return {"jsonrpc": "2.0", "id": 2, "error": {"code": code, "message": msg}}

def _ws_slash_exec_ok(output="done"):
    return {"jsonrpc": "2.0", "id": 3, "result": {"output": output}}

def _ws_set_cw_ok(enabled=True):
    return {"jsonrpc": "2.0", "id": 2, "result": {"continuous_work": enabled}}

def _ws_prompt_submit_ok():
    return {"method": "event", "params": {"session_id": "live_001", "type": "message.start", "payload": {}}}

def _ws_prompt_msg_complete(content="reply"):
    return {"method": "event", "params": {"session_id": "live_001", "type": "message.complete", "payload": {"content": content}}}

def _ws_resume_error(code=4007, msg="session not found"):
    return {"jsonrpc": "2.0", "id": 1, "error": {"code": code, "message": msg}}


# ---------------------------------------------------------------------------
# Kanban tool tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestKanbanBoard:
    async def test_compact_board(self):
        board_resp = {"columns": [{"name": "ready", "tasks": [
            {"id": "t_1", "title": "Short", "status": "ready", "assignee": "default", "priority": 5},
        ]}], "assignees": ["default"]}
        backend, _ = _make_kanban_backend({"/board": (200, board_resp)})
        result = await backend.kanban_get_board(compact=True)
        assert result["total"] == 1

    async def test_empty_columns_filtered(self):
        board_resp = {"columns": [{"name": "ready", "tasks": []}], "assignees": []}
        backend, _ = _make_kanban_backend({"/board": (200, board_resp)})
        result = await backend.kanban_get_board(compact=True)
        assert result["columns"] == []

    async def test_full_board(self):
        board_resp = {"columns": [{"name": "todo", "tasks": [{"id": "t_1"}]}], "assignees": []}
        backend, _ = _make_kanban_backend({"/board": (200, board_resp)})
        result = await backend.kanban_get_board(compact=False)
        assert result["columns"][0]["tasks"][0]["id"] == "t_1"

    # 2026-09-29: status/assignee were accepted and silently dropped, so
    # status="running" returned the whole 309-task board (74k characters).
    _MIXED_BOARD = {"columns": [
        {"name": "running", "tasks": [
            {"id": "t_r1", "title": "R", "status": "running", "assignee": "default", "priority": 1},
            {"id": "t_r2", "title": "R2", "status": "running", "assignee": "agent-x", "priority": 1},
        ]},
        {"name": "ready", "tasks": [
            {"id": "t_q1", "title": "Q", "status": "ready", "assignee": "default", "priority": 1},
        ]},
        {"name": "done", "tasks": [
            {"id": f"t_d{i}", "title": "D", "status": "done", "assignee": "default", "priority": 0}
            for i in range(50)
        ]},
    ], "assignees": ["default", "agent-x"]}

    async def test_status_filter_is_applied(self):
        backend, _ = _make_kanban_backend({"/board": (200, self._MIXED_BOARD)})
        result = await backend.kanban_get_board(status="running")
        ids = [t["id"] for c in result["columns"] for t in c["tasks"]]
        assert ids == ["t_r1", "t_r2"]
        assert result["total"] == 2
        assert "summarised_columns" not in result

    async def test_status_filter_is_case_insensitive(self):
        backend, _ = _make_kanban_backend({"/board": (200, self._MIXED_BOARD)})
        result = await backend.kanban_get_board(status="Running")
        assert result["total"] == 2

    async def test_assignee_filter_is_applied(self):
        backend, _ = _make_kanban_backend({"/board": (200, self._MIXED_BOARD)})
        result = await backend.kanban_get_board(status="running", assignee="agent-x")
        ids = [t["id"] for c in result["columns"] for t in c["tasks"]]
        assert ids == ["t_r2"]

    async def test_unfiltered_compact_counts_done_instead_of_listing(self):
        backend, _ = _make_kanban_backend({"/board": (200, self._MIXED_BOARD)})
        result = await backend.kanban_get_board()
        listed = {c["name"] for c in result["columns"]}
        assert listed == {"running", "ready"}
        assert result["summarised_columns"] == {"done": 50}
        assert result["total"] == 53

    async def test_done_listed_when_explicitly_requested(self):
        backend, _ = _make_kanban_backend({"/board": (200, self._MIXED_BOARD)})
        result = await backend.kanban_get_board(status="done")
        assert result["total"] == 50
        assert len(result["columns"][0]["tasks"]) == 50

    async def test_full_payload_honours_filters(self):
        backend, _ = _make_kanban_backend({"/board": (200, self._MIXED_BOARD)})
        result = await backend.kanban_get_board(status="ready", compact=False)
        kept = [t["id"] for c in result["columns"] for t in c["tasks"]]
        assert kept == ["t_q1"]

    async def test_status_falls_back_to_column_name(self):
        board = {"columns": [{"name": "running", "tasks": [{"id": "t_x"}]}], "assignees": []}
        backend, _ = _make_kanban_backend({"/board": (200, board)})
        result = await backend.kanban_get_board(status="running")
        assert result["total"] == 1


@pytest.mark.asyncio
class TestKanbanShow:
    async def test_returns_detail(self):
        task_resp = {"task": {"id": "t_1", "title": "Test"}, "comments": [], "runs": []}
        backend, _ = _make_kanban_backend({"/tasks/t_1": (200, task_resp)})
        result = await backend.kanban_show_task("t_1")
        assert result["task"]["id"] == "t_1"


@pytest.mark.asyncio
class TestKanbanCreate:
    async def test_valid_assignee(self):
        backend, _ = _make_kanban_backend({
            "/assignees": (200, {"assignees": ["default"]}),
            "/tasks": (200, {"task": {"id": "t_new"}}),
        })
        result = await backend.kanban_create_task(title="X", assignee="default")
        assert result["task"]["id"] == "t_new"

    async def test_unknown_assignee_rejected(self):
        backend, _ = _make_kanban_backend({"/assignees": (200, {"assignees": ["a"]})})
        with pytest.raises(KanbanApiError):
            await backend.kanban_create_task(title="X", assignee="bad")

    async def test_deterministic_idempotency_key(self):
        backend, _ = _make_kanban_backend({
            "/assignees": (200, {"assignees": ["a"]}),
            "/tasks": (200, {"task": {"id": "t"}}),
        })
        await backend.kanban_create_task(title="X", body="Y", assignee="a")
        k1 = json.loads(backend._client._transport.requests[-1].content)["idempotency_key"]
        await backend.kanban_create_task(title="X", body="Y", assignee="a")
        k2 = json.loads(backend._client._transport.requests[-1].content)["idempotency_key"]
        assert k1 == k2 == hashlib.sha256(b"X|Y|a").hexdigest()[:32]

    async def test_goal_mode_default_true(self):
        backend, _ = _make_kanban_backend({"/tasks": (200, {"task": {"id": "t"}})})
        await backend.kanban_create_task(title="X")
        assert json.loads(backend._client._transport.requests[-1].content)["goal_mode"] is True

    async def test_rejects_free_text_contract(self):
        backend, _ = _make_kanban_backend({})
        with pytest.raises(ValueError, match="Invalid completion_contract"):
            await backend.kanban_create_task(title="X", completion_contract="fix bugs")

    async def test_accepts_local_only(self):
        backend, _ = _make_kanban_backend({"/tasks": (200, {"task": {"id": "t"}})})
        await backend.kanban_create_task(title="X", completion_contract="local-only")
        assert json.loads(backend._client._transport.requests[-1].content)["title"] == "X"

    async def test_accepts_owner_repo(self):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(json.dumps({"task": {"id": "t"}}).encode(), b""))
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc
            backend, _ = _make_kanban_backend({})
            result = await backend.kanban_create_task(title="X", completion_contract="owner/repo")
            cmd = mock_exec.call_args[0]
            assert cmd[0].endswith("hermes.exe")
            assert "--completion-contract" in cmd

    async def test_accepts_pr_url(self):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(json.dumps({"task": {"id": "t"}}).encode(), b""))
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc
            backend, _ = _make_kanban_backend({})
            url = "https://github.com/owner/repo/pull/123"
            await backend.kanban_create_task(title="X", completion_contract=url)
            assert url in mock_exec.call_args[0]

    async def test_acceptance_criteria_appended(self):
        backend, _ = _make_kanban_backend({"/tasks": (200, {"task": {"id": "t"}})})
        await backend.kanban_create_task(title="X", body="Orig", acceptance_criteria=["Test passes"])
        body = json.loads(backend._client._transport.requests[-1].content)
        assert "## Required handoff evidence" in body["body"]
        assert "1. Test passes" in body["body"]

    async def test_cli_failure_raises(self):
        """H2: non-zero exit produces clear error."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b"error: board not found"))
            mock_proc.returncode = 1
            mock_exec.return_value = mock_proc
            backend, _ = _make_kanban_backend({})
            with pytest.raises(RuntimeError, match="hermes kanban create failed"):
                await backend.kanban_create_task(title="X", completion_contract="o/r")

    async def test_cli_timeout_kills(self):
        """H2: timeout kills process."""
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_proc.kill = MagicMock()
            mock_proc.wait = AsyncMock()
            mock_exec.return_value = mock_proc
            backend, _ = _make_kanban_backend({})
            with pytest.raises(RuntimeError, match="timed out"):
                await backend._kanban_create_via_cli(title="X", completion_contract="o/r")
            mock_proc.kill.assert_called_once()


# ---------------------------------------------------------------------------
# Defect 4: kanban_create safety rails (hold / default triage / warning)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestKanbanCreateSafetyRails:
    """A tester who only wants a card must be able to create one without
    a live worker being spawned."""

    async def test_hold_forces_triage_and_clears_assignee(self):
        backend, transport = _make_kanban_backend({
            "/assignees": (200, {"assignees": ["default"]}),
            "/tasks": (200, {"task": {"id": "t_hold"}}),
        })
        result = await backend.kanban_create_task(
            title="Draft card", assignee="default", goal_mode=False, hold=True,
        )
        assert result["task"]["id"] == "t_hold"
        payload = json.loads(transport.requests[-1].content)
        assert payload["triage"] is True
        assert not payload.get("assignee")

    async def test_hold_skips_assignee_validation(self):
        """hold=True clears assignee before validation, so an unknown
        assignee must not be rejected — the caller is not assigning yet."""
        backend, transport = _make_kanban_backend({
            "/tasks": (200, {"task": {"id": "t_hold"}}),
        })
        result = await backend.kanban_create_task(
            title="Draft", assignee="nonexistent-profile", hold=True,
        )
        assert result["task"]["id"] == "t_hold"
        # No /assignees validation call should have been made
        assert not any("/assignees" in str(r.url) for r in transport.requests)

    async def test_hold_adds_no_dispatch_warning(self):
        backend, _ = _make_kanban_backend({
            "/tasks": (200, {"task": {"id": "t_hold"}}),
        })
        result = await backend.kanban_create_task(title="Draft", hold=True)
        assert "warning" not in result

    async def test_default_assignee_no_goal_defaults_to_triage(self):
        """goal_mode=False + assignee='default' + triage unspecified must
        land in triage so the dispatcher cannot auto-claim it."""
        backend, transport = _make_kanban_backend({
            "/assignees": (200, {"assignees": ["default"]}),
            "/tasks": (200, {"task": {"id": "t_safe"}}),
        })
        await backend.kanban_create_task(
            title="Card", assignee="default", goal_mode=False,
        )
        payload = json.loads(transport.requests[-1].content)
        assert payload["triage"] is True

    async def test_empty_assignee_no_goal_defaults_to_triage(self):
        backend, transport = _make_kanban_backend({
            "/tasks": (200, {"task": {"id": "t_safe"}}),
        })
        await backend.kanban_create_task(title="Card", goal_mode=False)
        payload = json.loads(transport.requests[-1].content)
        assert payload["triage"] is True

    async def test_explicit_triage_false_overrides_safe_default(self):
        """Callers who really want 'default' to spawn can pass triage=false."""
        backend, transport = _make_kanban_backend({
            "/assignees": (200, {"assignees": ["default"]}),
            "/tasks": (200, {"task": {"id": "t_spawn"}}),
        })
        result = await backend.kanban_create_task(
            title="Card", assignee="default", goal_mode=False, triage=False,
        )
        payload = json.loads(transport.requests[-1].content)
        assert payload["triage"] is False
        # And it will auto-dispatch → warning must be present
        assert "warning" in result
        assert "auto-dispatched" in result["warning"]
        assert "hold=true" in result["warning"]

    async def test_goal_mode_true_does_not_force_triage(self):
        """goal_mode=True is the default path; it must keep its old
        triage=False behaviour when the caller did not opt into hold."""
        backend, transport = _make_kanban_backend({
            "/assignees": (200, {"assignees": ["default"]}),
            "/tasks": (200, {"task": {"id": "t_goal"}}),
        })
        await backend.kanban_create_task(title="Card", assignee="default")
        payload = json.loads(transport.requests[-1].content)
        assert payload["triage"] is False

    async def test_auto_dispatch_includes_warning(self):
        backend, _ = _make_kanban_backend({
            "/assignees": (200, {"assignees": ["default"]}),
            "/tasks": (200, {"task": {"id": "t_warn"}}),
        })
        result = await backend.kanban_create_task(
            title="Card", assignee="default",
        )
        assert result["task"]["id"] == "t_warn"
        assert "warning" in result
        assert "auto-dispatched" in result["warning"]
        assert "default" in result["warning"]

    async def test_no_warning_when_triage(self):
        backend, _ = _make_kanban_backend({
            "/assignees": (200, {"assignees": ["default"]}),
            "/tasks": (200, {"task": {"id": "t_tri"}}),
        })
        result = await backend.kanban_create_task(
            title="Card", assignee="default", triage=True,
        )
        assert "warning" not in result

    async def test_h_kanban_create_forwards_hold(self):
        """Handler must pass hold through so the tool schema is not a lie."""
        captured = {}

        class FakeBackend:
            async def kanban_create_task(self, **kwargs):
                captured.update(kwargs)
                return {"task": {"id": "t_x"}}

        with patch.object(server, "_require_backend", new_callable=AsyncMock) as mock_rb:
            mock_rb.return_value = FakeBackend()
            await server.h_kanban_create({
                "title": "Card", "hold": True, "assignee": "default",
                "goal_mode": False,
            })

        assert captured.get("hold") is True

    async def test_h_kanban_create_triage_absent_means_none(self):
        """Handler must NOT coerce a missing triage to False — None lets
        the backend method pick the safe default."""
        captured = {}

        class FakeBackend:
            async def kanban_create_task(self, **kwargs):
                captured.update(kwargs)
                return {"task": {"id": "t_x"}}

        with patch.object(server, "_require_backend", new_callable=AsyncMock) as mock_rb:
            mock_rb.return_value = FakeBackend()
            await server.h_kanban_create({
                "title": "Card", "assignee": "default", "goal_mode": False,
            })

        assert captured.get("triage") is None


class TestKanbanCreateToolSchemaSafety:
    def test_tool_schema_has_hold_and_documents_safety(self):
        t = next(x for x in server.TOOLS if x.name == "kanban_create")
        props = t.inputSchema["properties"]
        assert "hold" in props
        assert "hold" in t.description.lower()
        # Safe default documented
        assert "triage defaults to True" in t.description or "defaults to True" in t.description
        # Warning behaviour documented
        assert "warning" in t.description.lower()


# ---------------------------------------------------------------------------
# G1: CLI path tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestKanbanCreateCLIG1:
    def test_full_hermes_path(self):
        path = os.path.expandvars(r"%LOCALAPPDATA%\hermes\bin\hermes.exe")
        assert path.endswith("hermes.exe")

    async def test_missing_hermes_raises(self):
        with patch("os.path.isfile", return_value=False):
            backend = GuiBackend("127.0.0.1", 9999)
            with pytest.raises(RuntimeError, match="hermes CLI not found"):
                await backend._kanban_create_via_cli(title="X", completion_contract="o/r")

    async def test_board_before_create(self):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b'{"task":{"id":"t"}}', b""))
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc
            backend, _ = _make_kanban_backend({})
            await backend.kanban_create_task(title="X", completion_contract="o/r", board="myboard")
            cmd = mock_exec.call_args[0]
            board_idx = next(i for i, a in enumerate(cmd) if a == "--board")
            create_idx = cmd.index("create")
            assert board_idx < create_idx

    async def test_priority_zero_included(self):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b'{"task":{"id":"t"}}', b""))
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc
            backend, _ = _make_kanban_backend({})
            await backend.kanban_create_task(title="X", completion_contract="o/r", priority=0)
            cmd = mock_exec.call_args[0]
            assert "--priority" in cmd
            assert cmd[cmd.index("--priority") + 1] == "0"

    async def test_async_subprocess(self):
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b'{"task":{"id":"t"}}', b""))
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc
            backend, _ = _make_kanban_backend({})
            await backend.kanban_create_task(title="X", completion_contract="o/r")
            mock_exec.assert_called_once()


import os  # noqa: E402 — needed for G1 path test


# ---------------------------------------------------------------------------
# Kanban remaining
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestKanbanUpdate:
    async def test_update_priority(self):
        backend, _ = _make_kanban_backend({"/tasks/t_1": (200, {"task": {"id": "t_1"}})})
        await backend.kanban_update_task("t_1", priority=10)
        assert json.loads(backend._client._transport.requests[-1].content)["priority"] == 10

    async def test_archive(self):
        backend, _ = _make_kanban_backend({"/tasks/t_1": (200, {"task": {"id": "t_1"}})})
        await backend.kanban_update_task("t_1", status="archived")
        assert json.loads(backend._client._transport.requests[-1].content)["status"] == "archived"


@pytest.mark.asyncio
class TestKanbanComment:
    async def test_add(self):
        backend, _ = _make_kanban_backend({"/tasks/t_1/comments": (200, {"ok": True})})
        assert (await backend.kanban_comment("t_1", "test"))["ok"] is True


@pytest.mark.asyncio
class TestKanbanLink:
    async def test_create(self):
        backend, _ = _make_kanban_backend({"/links": (200, {"ok": True})})
        assert (await backend.kanban_link("t_p", "t_c"))["ok"] is True


@pytest.mark.asyncio
class TestKanbanDispatch:
    async def test_dispatch(self):
        backend, _ = _make_kanban_backend({"/dispatch": (200, {"spawned": 2})})
        assert (await backend.kanban_dispatch())["spawned"] == 2


@pytest.mark.asyncio
class TestKanbanStats:
    async def test_combined(self):
        backend, _ = _make_kanban_backend({
            "/stats": (200, {"total": 10}),
            "/workers/active": (200, {"workers": [], "count": 0}),
            "/diagnostics": (200, {"diagnostics": [], "count": 0}),
        })
        assert (await backend.kanban_stats())["stats"]["total"] == 10


@pytest.mark.asyncio
class TestKanbanDecompose:
    """H2: decompose routes correctly."""
    async def test_decompose(self):
        backend, _ = _make_kanban_backend({"/tasks/t_triage/decompose": (200, {"ok": True, "child_ids": ["t_c1"]})})
        result = await backend.kanban_decompose("t_triage")
        assert result["ok"] is True
        assert "t_c1" in result["child_ids"]


@pytest.mark.asyncio
class TestKanbanPluginDisabled:
    async def test_404_raises(self):
        backend, _ = _make_kanban_backend({})
        with pytest.raises(KanbanApiError) as exc_info:
            await backend.kanban_get_board()
        assert exc_info.value.status == 404


@pytest.mark.asyncio
class TestKanbanHttpErrors:
    """H2: HTTP error mapping."""
    async def test_500_mapped(self):
        backend, _ = _make_kanban_backend({"/tasks/t_1/comments": (500, {"detail": "boom"})})
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await backend.kanban_comment("t_1", "x")
        assert exc_info.value.response.status_code == 500

    async def test_409_conflict(self):
        backend, _ = _make_kanban_backend({"/tasks/t_1": (409, {"detail": "conflict"})})
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await backend.kanban_update_task("t_1", status="ready")
        assert exc_info.value.response.status_code == 409


# ---------------------------------------------------------------------------
# F6: Error message tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRunSlashErrors:
    async def test_4018_falls_back(self):
        ws = [_ws_resume_ok(), _ws_cmd_dispatch_4018(), _ws_slash_exec_ok("ok")]
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = lambda url, **kw: _FakeWebSocket(ws)
            result = await GuiBackend("127.0.0.1", 9999, token="t").run_slash("s", "/m")
        assert result["ok"] is True

    async def test_32603_clear_message(self):
        ws = [_ws_resume_ok(), _ws_cmd_dispatch_32603()]
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = lambda url, **kw: _FakeWebSocket(ws)
            with pytest.raises(RuntimeError, match="Backend crashed"):
                await GuiBackend("127.0.0.1", 9999, token="t").run_slash("s", "/goal x")

    async def test_32603_mentions_log(self):
        ws = [_ws_resume_ok(), _ws_cmd_dispatch_32603()]
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = lambda url, **kw: _FakeWebSocket(ws)
            with pytest.raises(RuntimeError, match="errors.log"):
                await GuiBackend("127.0.0.1", 9999, token="t").run_slash("s", "/goal x")

    async def test_resume_error_raises(self):
        """H2: resume failure propagates."""
        ws = [_ws_resume_error()]
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = lambda url, **kw: _FakeWebSocket(ws)
            with pytest.raises(RuntimeError, match="resume failed"):
                await GuiBackend("127.0.0.1", 9999, token="t").run_slash("bad", "/goal x")


# ---------------------------------------------------------------------------
# goal_action tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestGoalAction:
    async def test_valid_action_routes(self):
        """H2: valid action sends /goal <action>."""
        ws = [_ws_resume_ok(), _ws_cmd_dispatch_ok("Goal paused")]
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = lambda url, **kw: _FakeWebSocket(ws)
            result = await GuiBackend("127.0.0.1", 9999, token="t").goal_action("s", "pause")
        assert result["ok"] is True

    async def test_invalid_action_raises(self):
        backend = GuiBackend("127.0.0.1", 9999, token="t")
        with pytest.raises(ValueError, match="Invalid goal action"):
            await backend.goal_action("s", "bogus")

    async def test_all_valid_actions(self):
        """H2: status, show, pause, resume, clear all accepted."""
        for action in ["status", "show", "pause", "resume", "clear"]:
            ws = [_ws_resume_ok(), _ws_cmd_dispatch_ok(f"action: {action}")]
            with patch("hermes_chat_delegator.server.websockets") as m:
                m.connect = lambda url, **kw: _FakeWebSocket(ws)
                result = await GuiBackend("127.0.0.1", 9999, token="t").goal_action("s", action)
            assert result["ok"] is True


# ---------------------------------------------------------------------------
# G2: set_goal ordering and non-blocking kickoff
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSetGoalG2:
    async def test_gates_before_kickoff(self):
        ws_goal = [_ws_resume_ok(), _ws_cmd_dispatch_send("Goal set", "kickoff")]
        ws_gate = [_ws_resume_ok(), _ws_cmd_dispatch_ok("Gate added")]
        ws_kick = [_ws_resume_ok(), _ws_prompt_submit_ok(), _ws_prompt_msg_complete("working")]
        order = []
        def fake_ws(url, **kw):
            n = len(order); order.append(n)
            return _FakeWebSocket([ws_goal, ws_gate, ws_kick][n])
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = fake_ws
            result = await GuiBackend("127.0.0.1", 9999, token="t").set_goal("s", "fix", gates=["pytest"])
        assert order == [0, 1, 2]

    async def test_kickoff_non_blocking(self):
        ws_goal = [_ws_resume_ok(), _ws_cmd_dispatch_send("Goal set", "kickoff")]
        ws_kick = [_ws_resume_ok(), _ws_prompt_submit_ok(), _ws_prompt_msg_complete("started")]
        cnt = {"n": 0}
        def fake_ws(url, **kw):
            cnt["n"] += 1
            return _FakeWebSocket(ws_goal if cnt["n"] == 1 else ws_kick)
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = fake_ws
            result = await GuiBackend("127.0.0.1", 9999, token="t").set_goal("s", "fix")
        kickoff = [r for r in result["results"] if r.get("kickoff_submitted")]
        assert len(kickoff) == 1
        assert kickoff[0]["status"] == "running"

    async def test_no_kickoff_when_not_send(self):
        ws = [_ws_resume_ok(), _ws_cmd_dispatch_ok("Goal set")]
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = lambda url, **kw: _FakeWebSocket(ws)
            result = await GuiBackend("127.0.0.1", 9999, token="t").set_goal("s", "fix")
        kickoff = [r for r in result["results"] if r.get("kickoff_submitted")]
        assert len(kickoff) == 0

    async def test_gates_no_kickoff(self):
        """H2: gates added, no kickoff when goal returns plain output."""
        ws_goal = [_ws_resume_ok(), _ws_cmd_dispatch_ok("Goal set")]
        ws_gate = [_ws_resume_ok(), _ws_cmd_dispatch_ok("Gate added")]
        cnt = {"n": 0}
        def fake_ws(url, **kw):
            cnt["n"] += 1
            return _FakeWebSocket(ws_goal if cnt["n"] == 1 else ws_gate)
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = fake_ws
            result = await GuiBackend("127.0.0.1", 9999, token="t").set_goal("s", "fix", gates=["pytest"])
        assert len(result["results"]) == 2
        kickoff = [r for r in result["results"] if r.get("kickoff_submitted")]
        assert len(kickoff) == 0


# ---------------------------------------------------------------------------
# H3: set_goal kickoff timeout handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSetGoalKickoffTimeoutH3:
    async def test_timeout_reported_as_running(self):
        """H3: send_prompt timeout means turn started — report running, not failure."""
        ws_goal = [_ws_resume_ok(), _ws_cmd_dispatch_send("Goal set", "kickoff")]
        # send_prompt will timeout (ws recv raises TimeoutError after frames exhausted)
        ws_kick = [_ws_resume_ok(), _ws_prompt_submit_ok()]
        cnt = {"n": 0}
        def fake_ws(url, **kw):
            cnt["n"] += 1
            return _FakeWebSocket(ws_goal if cnt["n"] == 1 else ws_kick)
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = fake_ws
            result = await GuiBackend("127.0.0.1", 9999, token="t").set_goal("s", "fix")
        kickoff = [r for r in result["results"] if r.get("kickoff_submitted")]
        assert len(kickoff) == 1
        # Timeout should be reported as "running", not "failed"
        assert kickoff[0]["status"] == "running"
        assert kickoff[0].get("note", "").startswith("Goal loop started")

    async def test_real_failure_reported(self):
        """H3: connection error is a real failure."""
        ws_goal = [_ws_resume_ok(), _ws_cmd_dispatch_send("Goal set", "kickoff")]
        # send_prompt will raise RuntimeError (connection error)
        cnt = {"n": 0}
        def fake_ws(url, **kw):
            cnt["n"] += 1
            if cnt["n"] == 1:
                return _FakeWebSocket(ws_goal)
            raise ConnectionRefusedError("backend down")
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = fake_ws
            result = await GuiBackend("127.0.0.1", 9999, token="t").set_goal("s", "fix")
        kickoff = [r for r in result["results"] if r.get("kickoff_submitted") is False]
        assert len(kickoff) == 1
        assert "error" in kickoff[0]


# ---------------------------------------------------------------------------
# F2: CW persistence tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSetContinuousWork:
    async def test_enable_updates_registry(self):
        ws = [_ws_resume_ok(), _ws_set_cw_ok(True)]
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = lambda url, **kw: _FakeWebSocket(ws)
            await GuiBackend("127.0.0.1", 9999, token="t").set_continuous_work("s", True)
        assert _cw_state.get("s") is True
        _cw_state.pop("s", None)

    async def test_disable_clears(self):
        _cw_state["s"] = True
        ws = [_ws_resume_ok(), _ws_set_cw_ok(False)]
        with patch("hermes_chat_delegator.server.websockets") as m:
            m.connect = lambda url, **kw: _FakeWebSocket(ws)
            await GuiBackend("127.0.0.1", 9999, token="t").set_continuous_work("s", False)
        assert _cw_state.get("s") is False
        _cw_state.pop("s", None)


@pytest.mark.asyncio
class TestContinuousWorkPersistence:
    async def test_resolves_from_registry(self):
        _cw_state["s"] = True
        ws = [_ws_resume_ok(), _ws_prompt_submit_ok(), _ws_prompt_msg_complete("ok")]
        with patch("hermes_chat_delegator.server.websockets") as m:
            fws = _FakeWebSocket(ws); m.connect = lambda url, **kw: fws
            await GuiBackend("127.0.0.1", 9999, token="t").send_prompt("s", "hi", timeout=5)
        assert json.loads(fws._sent[1])["params"].get("continuous_work") is True
        _cw_state.pop("s", None)

    async def test_defaults_false(self):
        _cw_state.pop("s", None)
        ws = [_ws_resume_ok(), _ws_prompt_submit_ok(), _ws_prompt_msg_complete("ok")]
        with patch("hermes_chat_delegator.server.websockets") as m:
            fws = _FakeWebSocket(ws); m.connect = lambda url, **kw: fws
            await GuiBackend("127.0.0.1", 9999, token="t").send_prompt("s", "hi", timeout=5)
        assert "continuous_work" not in json.loads(fws._sent[1])["params"]

    async def test_explicit_overrides(self):
        _cw_state["s"] = False
        ws = [_ws_resume_ok(), _ws_prompt_submit_ok(), _ws_prompt_msg_complete("ok")]
        with patch("hermes_chat_delegator.server.websockets") as m:
            fws = _FakeWebSocket(ws); m.connect = lambda url, **kw: fws
            await GuiBackend("127.0.0.1", 9999, token="t").send_prompt("s", "hi", timeout=5, continuous_work=True)
        assert json.loads(fws._sent[1])["params"].get("continuous_work") is True
        _cw_state.pop("s", None)

    async def test_stream_resolves(self):
        _cw_state["s"] = True
        ws = [_ws_resume_ok(), _ws_prompt_submit_ok(), _ws_prompt_msg_complete("ok")]
        with patch("hermes_chat_delegator.server.websockets") as m:
            fws = _FakeWebSocket(ws); m.connect = lambda url, **kw: fws
            async for _, _ in GuiBackend("127.0.0.1", 9999, token="t").send_prompt_stream("s", "hi", timeout=5):
                pass
        assert json.loads(fws._sent[1])["params"].get("continuous_work") is True
        _cw_state.pop("s", None)


# ---------------------------------------------------------------------------
# G3: create_chat ordering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCreateChatG3:
    async def test_goal_and_cw_applied(self):
        with patch("hermes_chat_delegator.server._require_backend") as mock_rb:
            b = GuiBackend("127.0.0.1", 9999)
            b.create_session = AsyncMock(return_value={"session_id": "n1"})
            b.set_continuous_work = AsyncMock(return_value={"ok": True})
            b.set_goal = AsyncMock(return_value={"ok": True, "results": []})
            mock_rb.return_value = b
            result = await server.h_create_chat({"title": "t", "goal": "fix", "continuous_work": True})
            data = json.loads(result[0].text)
            assert data.get("continuous_work") is True
            assert "goal" in data

    async def test_cw_before_goal_ordering(self):
        order = []
        with patch("hermes_chat_delegator.server._require_backend") as mock_rb:
            b = GuiBackend("127.0.0.1", 9999)
            b.create_session = AsyncMock(return_value={"session_id": "n1"})
            async def track_cw(sid, e):
                order.append("cw"); return {"ok": True}
            async def track_goal(sid, g, gates=None):
                order.append("goal"); return {"ok": True, "results": []}
            b.set_continuous_work = track_cw; b.set_goal = track_goal
            mock_rb.return_value = b
            await server.h_create_chat({"goal": "fix", "continuous_work": True})
        assert order == ["cw", "goal"]


# ---------------------------------------------------------------------------
# G4: CW backend resolution
# ---------------------------------------------------------------------------

class TestCWBackendResolution:
    def test_cw_is_process_local(self):
        assert isinstance(server._cw_state, dict)

    def test_tool_mentions_process_local(self):
        tool = next(t for t in server.TOOLS if t.name == "set_continuous_work")
        assert "process" in tool.description.lower() or "not persisted" in tool.description.lower()


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

class TestToolDefinitions:
    def test_all_have_handlers(self):
        h = set(server.HANDLERS.keys())
        for t in server.TOOLS:
            assert t.name in h

    def test_kanban_create_has_new_params(self):
        t = next(x for x in server.TOOLS if x.name == "kanban_create")
        assert "completion_contract" in t.inputSchema["properties"]
        assert "acceptance_criteria" in t.inputSchema["properties"]

    def test_set_goal_no_max_turns(self):
        t = next(x for x in server.TOOLS if x.name == "set_goal")
        assert "max_turns" not in t.inputSchema["properties"]

    def test_create_chat_has_goal_cw(self):
        t = next(x for x in server.TOOLS if x.name == "create_chat")
        p = t.inputSchema["properties"]
        assert "goal" in p and "goal_gates" in p and "continuous_work" in p


# ---------------------------------------------------------------------------
# Idempotency key determinism
# ---------------------------------------------------------------------------

class TestIdempotencyKeyDeterminism:
    def test_same_same(self):
        assert hashlib.sha256(b"A|B|C").hexdigest()[:32] == hashlib.sha256(b"A|B|C").hexdigest()[:32]

    def test_diff_diff(self):
        assert hashlib.sha256(b"A|B|C").hexdigest()[:32] != hashlib.sha256(b"A|B|D").hexdigest()[:32]
