"""Unit tests for hermes_chat_delegator.server.

Tests GuiBackend against httpx.MockTransport and fake websockets,
plus pure-function tests for _safe_epoch, _summarize_event, etc.
No live GUI required.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hermes_chat_delegator import server
from hermes_chat_delegator.server import (
    BackendAmbiguous,
    DraftSessionRecord,
    GuiBackend,
    WsAuthFailure,
    _safe_epoch,
    _summarize_event,
    _err,
    _text,
    _truncate_messages,
    _resolve_ws_max_size,
    auto_discover_backend,
)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------

class TestSafeEpoch:
    def test_none(self):
        assert _safe_epoch(None) == 0.0

    def test_int(self):
        assert _safe_epoch(1700000000) == 1700000000.0

    def test_float(self):
        assert _safe_epoch(1700000000.5) == 1700000000.5

    def test_string_number(self):
        assert _safe_epoch("1700000000") == 1700000000.0

    def test_string_non_numeric(self):
        assert _safe_epoch("not-a-number") == 0.0

    def test_empty_string(self):
        assert _safe_epoch("") == 0.0

    def test_zero(self):
        assert _safe_epoch(0) == 0.0


class TestSummarizeEvent:
    def test_message_delta(self):
        result = _summarize_event("message.delta", {"delta": "Hello"})
        assert result == "Hello"

    def test_message_delta_text_key(self):
        result = _summarize_event("message.delta", {"text": "World"})
        assert result == "World"

    def test_message_start(self):
        result = _summarize_event("message.start", {})
        assert "started" in result

    def test_message_complete(self):
        result = _summarize_event("message.complete", {})
        assert "finished" in result

    def test_tool_start(self):
        result = _summarize_event("tool.start", {"name": "terminal"})
        assert "terminal" in result
        assert "started" in result

    def test_tool_complete(self):
        result = _summarize_event("tool.complete", {"name": "read_file"})
        assert "read_file" in result
        assert "finished" in result

    def test_unknown_event(self):
        result = _summarize_event("future.event", {"foo": "bar"})
        assert "foo" in result


class TestErr:
    def test_err_structure(self):
        result = _err("not_connected", "Backend down", retryable=True)
        assert len(result) == 1
        data = json.loads(result[0].text)
        assert data["error"]["code"] == "not_connected"
        assert data["error"]["message"] == "Backend down"
        assert data["error"]["retryable"] is True

    def test_err_default_not_retryable(self):
        result = _err("session_not_found", "Gone")
        data = json.loads(result[0].text)
        assert data["error"]["retryable"] is False


class TestText:
    def test_text_structure(self):
        result = _text('{"key": "value"}')
        assert len(result) == 1
        assert result[0].type == "text"
        assert result[0].text == '{"key": "value"}'


# ---------------------------------------------------------------------------
# GuiBackend tests with mock HTTP
# ---------------------------------------------------------------------------

class MockTransport(httpx.AsyncBaseTransport):
    """Records requests and returns pre-configured responses."""

    def __init__(self, responses: dict[str, tuple[int, dict]]):
        self.responses = responses
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = str(request.url.raw_path, encoding="utf-8") if isinstance(request.url.raw_path, bytes) else request.url.raw_path
        # Sort by length descending so more specific paths match first
        for pattern, (status, body) in sorted(self.responses.items(), key=lambda x: -len(x[0])):
            if pattern in path:
                return httpx.Response(status, json=body)
        return httpx.Response(404, json={"error": "not found"})


class FakeWebSocket:
    """Fake WebSocket that yields canned JSON-RPC frames."""

    def __init__(self, frames: list[dict]):
        self._frames = iter(frames)
        self._sent: list[str] = []

    async def send(self, data: str):
        self._sent.append(data)

    async def recv(self) -> str:
        frame = next(self._frames)
        return json.dumps(frame)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.fixture
def mock_backend():
    """Create a GuiBackend with a mock HTTP client."""
    responses = {
        "/health": (200, {"status": "ok", "version": "0.1.0"}),
        "/api/sessions": (200, {"sessions": [
            {"id": "sess_001", "title": "Test Chat", "is_active": False,
             "last_active": 1700000000, "message_count": 5, "model": "test"},
        ]}),
        "/api/sessions/sess_001": (200, {
            "id": "sess_001", "title": "Test Chat", "model": "test",
        }),
        "/api/sessions/sess_001/messages": (200, {"messages": [
            {"id": "m1", "role": "user", "content": "Hello"},
            {"id": "m2", "role": "assistant", "content": "Hi there!"},
            {"id": "m3", "role": "user", "content": "Status?"},
            {"id": "m4", "role": "assistant", "content": "All good."},
        ]}),
    }
    transport = MockTransport(responses)
    backend = GuiBackend("127.0.0.1", 9999, token="test-token")
    backend._client = httpx.AsyncClient(
        transport=transport, timeout=5,
        headers=backend._auth_headers(),
    )
    return backend, transport


class TestTruncateMessages:
    """Unit tests for the get_messages truncation/compact options (defect #6)."""

    def test_no_options_returns_input_unchanged(self):
        msgs = [{"id": "1", "role": "user", "content": "hi"}]
        assert _truncate_messages(msgs) == msgs

    def test_max_chars_truncates_and_marks_content(self):
        msgs = [{"id": "1", "role": "assistant", "content": "x" * 100}]
        out = _truncate_messages(msgs, max_chars=10)
        assert out[0]["content"] == "x" * 10
        assert out[0]["content_truncated"] is True
        assert out[0]["content_original_chars"] == 100
        # Input must not be mutated.
        assert len(msgs[0]["content"]) == 100

    def test_max_chars_does_not_flag_short_content(self):
        msgs = [{"id": "1", "role": "user", "content": "short"}]
        out = _truncate_messages(msgs, max_chars=100)
        assert out[0]["content"] == "short"
        assert "content_truncated" not in out[0]

    def test_compact_drops_verbose_fields(self):
        msgs = [{
            "id": "1", "role": "assistant", "content": "answer",
            "reasoning": "long chain of thought" * 50,
            "tool_calls": [{"name": "terminal", "arguments": "{}"}],
        }]
        out = _truncate_messages(msgs, compact=True)
        assert "reasoning" not in out[0]
        assert "tool_calls" not in out[0]
        assert out[0]["content"] == "answer"

    def test_combined_max_chars_and_compact(self):
        msgs = [{
            "id": "1", "role": "assistant", "content": "y" * 50,
            "reasoning": "z" * 50,
        }]
        out = _truncate_messages(msgs, max_chars=5, compact=True)
        assert out[0]["content"] == "y" * 5
        assert "reasoning" not in out[0]


class TestResolveWsMaxSize:
    """Unit tests for the WebSocket max_size resolution (defect #2)."""

    def test_default_is_64mib(self, monkeypatch):
        monkeypatch.delenv("HERMES_WS_MAX_SIZE", raising=False)
        assert _resolve_ws_max_size() == 64 * 1024 * 1024

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("HERMES_WS_MAX_SIZE", "1000")
        assert _resolve_ws_max_size() == 1000

    def test_env_zero_or_negative_means_unlimited(self, monkeypatch):
        monkeypatch.setenv("HERMES_WS_MAX_SIZE", "0")
        assert _resolve_ws_max_size() is None
        monkeypatch.setenv("HERMES_WS_MAX_SIZE", "-5")
        assert _resolve_ws_max_size() is None

    def test_default_exceeds_the_websockets_library_default(self, monkeypatch):
        """The whole point: our floor must be well above `websockets`'
        built-in 2**20 default that caused defect #2."""
        monkeypatch.delenv("HERMES_WS_MAX_SIZE", raising=False)
        assert _resolve_ws_max_size() > 2 ** 20


@pytest.mark.asyncio
class TestRequestRetry:
    """Unit tests for GuiBackend._request's transient-error retry (defect #4)."""

    async def test_retries_once_on_connect_error_then_succeeds(self):
        backend = GuiBackend("127.0.0.1", 9999)
        ok_response = httpx.Response(200, json={"status": "ok"})
        mock_request = AsyncMock(side_effect=[
            httpx.ConnectError("refused"),
            ok_response,
        ])
        backend._client.request = mock_request

        resp = await backend._request("GET", "http://x/health", retry_delay=0)

        assert resp is ok_response
        assert mock_request.call_count == 2

    async def test_gives_up_after_exhausting_retries(self):
        backend = GuiBackend("127.0.0.1", 9999)
        mock_request = AsyncMock(side_effect=httpx.ConnectError("refused"))
        backend._client.request = mock_request

        with pytest.raises(httpx.ConnectError):
            await backend._request("GET", "http://x/health", retries=2, retry_delay=0)

        assert mock_request.call_count == 3  # initial attempt + 2 retries

    async def test_does_not_retry_http_error_status(self):
        """A real 404/500 from a reachable server is not a transport
        failure — must not be retried (raise_for_status happens upstream,
        _request itself just needs to not eat time retrying a real answer)."""
        backend = GuiBackend("127.0.0.1", 9999)
        error_response = httpx.Response(404, json={"error": "not found"})
        mock_request = AsyncMock(return_value=error_response)
        backend._client.request = mock_request

        resp = await backend._request("GET", "http://x/api/sessions/missing")

        assert resp is error_response
        assert mock_request.call_count == 1


@pytest.mark.asyncio
class TestGuiBackendRest:
    async def test_health(self, mock_backend):
        b, _ = mock_backend
        result = await b.health()
        assert result["status"] == "ok"

    async def test_list_sessions(self, mock_backend):
        b, _ = mock_backend
        sessions = await b.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["id"] == "sess_001"

    async def test_get_session(self, mock_backend):
        b, _ = mock_backend
        info = await b.get_session("sess_001")
        assert info["title"] == "Test Chat"

    async def test_get_messages_with_limit(self, mock_backend):
        b, _ = mock_backend
        msgs = await b.get_messages("sess_001", last_n=2)
        assert len(msgs) == 2
        assert msgs[-1]["role"] == "assistant"

    async def test_auth_header_sent(self, mock_backend):
        b, transport = mock_backend
        await b.health()
        req = transport.requests[0]
        assert req.headers.get("X-Hermes-Session-Token") == "test-token"


# ---------------------------------------------------------------------------
# send_prompt regression test (C2: stale response trap)
# ---------------------------------------------------------------------------

class FakeWebSocketSendPrompt:
    """Simulates the send_prompt WebSocket flow.

    The key test: after prompt.submit, the messages API returns the OLD
    assistant message first, then the NEW one. The fixed poll logic must
    NOT return the old one.
    """

    def __init__(self, frames, timeout_on_exhaust=True):
        self._frames = iter(frames)
        self._sent: list[str] = []
        self._timeout_on_exhaust = timeout_on_exhaust

    async def send(self, data):
        self._sent.append(data)

    async def recv(self):
        try:
            return json.dumps(next(self._frames))
        except StopIteration:
            if self._timeout_on_exhaust:
                raise asyncio.TimeoutError()
            raise

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
class TestSendPromptStaleResponse:
    async def test_does_not_return_old_assistant_message(self):
        """Regression test for AUDIT C2: send_prompt must not return the
        previous turn's answer when the new reply hasn't arrived yet."""

        # Simulate: session has old messages, we submit a new prompt,
        # and the messages API returns the old assistant message on first poll,
        # then the new one on second poll.
        old_msgs = [
            {"id": "old1", "role": "user", "content": "Old question"},
            {"id": "old2", "role": "assistant", "content": "OLD ANSWER"},
        ]
        new_msgs = [
            {"id": "old1", "role": "user", "content": "Old question"},
            {"id": "old2", "role": "assistant", "content": "OLD ANSWER"},
            {"id": "new_user", "role": "user", "content": "New question"},
            {"id": "new_asst", "role": "assistant", "content": "NEW ANSWER"},
        ]

        responses = {
            "/health": (200, {"status": "ok"}),
            "/api/sessions": (200, {"sessions": []}),
            # First call: get old messages (before submit)
            "/api/sessions/test/messages": [
                (200, {"messages": old_msgs}),
                # After submit: first poll returns old msgs (user echo only)
                (200, {"messages": old_msgs + [{"id": "new_user", "role": "user", "content": "New question"}]}),
                # Second poll: new assistant message appears
                (200, {"messages": new_msgs}),
            ],
        }

        # We need a custom transport that returns different responses on same path
        call_count = {"messages": 0}

        class StaleTestTransport(httpx.AsyncBaseTransport):
            def __init__(self):
                self.requests = []

            async def handle_async_request(self, request):
                self.requests.append(request)
                path = str(request.url)
                if "/health" in path:
                    return httpx.Response(200, json={"status": "ok"})
                if "/api/sessions" in path and "/messages" in path:
                    call_count["messages"] += 1
                    if call_count["messages"] == 1:
                        # Before submit: old messages
                        return httpx.Response(200, json={"messages": old_msgs})
                    elif call_count["messages"] == 2:
                        # First poll: only user echo, no new assistant
                        return httpx.Response(200, json={
                            "messages": old_msgs + [
                                {"id": "new_user", "role": "user", "content": "New question"}
                            ]
                        })
                    else:
                        # Second poll: new assistant arrived
                        return httpx.Response(200, json={"messages": new_msgs})
                if "/api/sessions" in path:
                    return httpx.Response(200, json={"sessions": []})
                return httpx.Response(404, json={})

        transport = StaleTestTransport()
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=transport, timeout=5)

        # WebSocket events with the new answer
        ws_frames = [
            {"id": 1, "result": {"session_id": "live_abc"}},
            {"id": 2, "result": {"status": "streaming"}},
            # Event stream with NEW answer (not the old one)
            {"method": "event", "params": {"type": "message.start", "session_id": "live_abc", "payload": {}}},
            {"method": "event", "params": {"type": "message.delta", "session_id": "live_abc", "payload": {"delta": "NEW "}}},
            {"method": "event", "params": {"type": "message.delta", "session_id": "live_abc", "payload": {"delta": "ANSWER"}}},
            {"method": "event", "params": {"type": "message.complete", "session_id": "live_abc", "payload": {"content": "NEW ANSWER"}}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            result = await backend.send_prompt("test", "New question", timeout=10)

        # Must return the NEW answer, not the old one
        assert result["response"] == "NEW ANSWER", (
            f"Got stale response: {result['response']!r} — C2 regression!"
        )


@pytest.mark.asyncio
class TestPollForReplyTimeoutPending:
    """Regression tests for defect #5: a send_prompt timeout with no new
    assistant reply must return an unambiguous pending status, never a
    stale echo of the pre-submit assistant message."""

    async def test_no_new_reply_returns_explicit_pending(self):
        old_msgs = [
            {"id": "old1", "role": "user", "content": "Old question"},
            {"id": "old2", "role": "assistant", "content": "OLD ANSWER"},
        ]

        class NeverRepliesTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                path = str(request.url)
                if "/messages" in path:
                    # The new assistant reply NEVER shows up — simulates a
                    # turn that's still running (or got interrupted) when
                    # the caller's timeout expires.
                    return httpx.Response(200, json={"messages": old_msgs})
                return httpx.Response(404, json={})

        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=NeverRepliesTransport(), timeout=5)

        # timeout=0 skips the (real, 2s-per-iteration) poll loop entirely and
        # goes straight to the post-loop final check — deterministic and
        # fast, and still exercises the exact "no new reply" code path the
        # loop's own timeout would eventually reach.
        result = await backend._poll_for_reply(
            "test", last_assistant_id="old2", timeout=0,
        )

        assert result["response"] is None, (
            f"Timeout with no new reply must not echo stale content, got: "
            f"{result['response']!r}"
        )
        assert result["timed_out"] is True
        assert result["status"] == "pending"

    async def test_new_reply_landing_at_deadline_is_not_lost(self):
        """A reply that arrives exactly at the timeout boundary (the final
        post-loop check) should still be surfaced — just flagged as late,
        not silently dropped in favor of a null pending response."""
        new_msgs = [
            {"id": "old1", "role": "user", "content": "Old question"},
            {"id": "old2", "role": "assistant", "content": "OLD ANSWER"},
            {"id": "new_asst", "role": "assistant", "content": "JUST LANDED"},
        ]

        class LateReplyTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                path = str(request.url)
                if "/messages" in path:
                    return httpx.Response(200, json={"messages": new_msgs})
                return httpx.Response(404, json={})

        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=LateReplyTransport(), timeout=5)

        # timeout=0 skips the poll loop (see note above) so this exercises
        # the post-loop final check specifically, not the in-loop path
        # (which would return status="completed" instead of "completed_late").
        result = await backend._poll_for_reply(
            "test", last_assistant_id="old2", timeout=0,
        )

        assert result["response"] == "JUST LANDED"
        assert result["timed_out"] is True
        assert result["status"] == "completed_late"


@pytest.mark.asyncio
class TestSendPromptFreshDraftSession:
    async def test_404_on_baseline_fetch_is_not_fatal(self):
        """Regression test: reproduced live via create_chat(initial_message=...)
        against a real backend. A brand-new draft session (from
        create_session, no prompt sent yet) has no durable row, so GET
        .../messages 404s. send_prompt's baseline fetch must treat that as
        "no prior messages" instead of propagating the HTTPStatusError."""

        new_msgs = [
            {"id": "u1", "role": "user", "content": "hi"},
            {"id": "a1", "role": "assistant", "content": "PONG"},
        ]
        call_count = {"messages": 0}

        class FreshSessionTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                path = str(request.url)
                if "/health" in path:
                    return httpx.Response(200, json={"status": "ok"})
                if "/messages" in path:
                    call_count["messages"] += 1
                    if call_count["messages"] == 1:
                        # Baseline fetch, before the first prompt ever lands.
                        return httpx.Response(404, json={"error": "session not found"})
                    return httpx.Response(200, json={"messages": new_msgs})
                if "/api/sessions" in path:
                    return httpx.Response(200, json={"sessions": []})
                return httpx.Response(404, json={})

        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=FreshSessionTransport(), timeout=5)

        # For a draft (no prior durable row): create → live_id; submit against live_id
        # Then event stream delivers the response
        ws_frames = [
            {"id": 1, "result": {"session_id": "live_new"}},
            {"id": 2, "result": {"status": "streaming"}},
            # Event stream
            {"method": "event", "params": {"type": "message.start", "session_id": "live_new", "payload": {}}},
            {"method": "event", "params": {"type": "message.delta", "session_id": "live_new", "payload": {"delta": "PONG"}}},
            {"method": "event", "params": {"type": "message.complete", "session_id": "live_new", "payload": {"content": "PONG"}}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            result = await backend.send_prompt("brand_new_sess", "hi", timeout=10)

        assert result["response"] == "PONG"


# ---------------------------------------------------------------------------
# create_session: draft path + initial_message path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestCreateSession:
    async def test_draft_returns_durable_id_no_prompt(self):
        """Draft create_session calls session.create, returns durable id,
        does NOT call prompt.submit."""
        transport = MockTransport({"/health": (200, {"status": "ok"})})
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=transport, timeout=5)

        ws_frames = [
            {"id": 1, "result": {
                "session_id": "live_draft",
                "stored_session_id": "20260716_120000_abc",
                "info": {"cwd": "/some/path"},
            }},
        ]

        captured_ws = []
        def fake_ws(url, **kwargs):
            ws = FakeWebSocketSendPrompt(ws_frames)
            captured_ws.append(ws)
            return ws

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = fake_ws
            result = await backend.create_session(title="My Draft", cwd="/some/path")

        assert result["session_id"] == "20260716_120000_abc"
        assert result["live_session_id"] == "live_draft"
        assert result["title"] == "My Draft"
        assert result["cwd"] == "/some/path"
        # Verify only session.create was sent (no prompt.submit)
        assert len(captured_ws) == 1
        sent = json.loads(captured_ws[0]._sent[0])
        assert sent["method"] == "session.create"
        assert sent["params"]["title"] == "My Draft"

    async def test_initial_message_submits_on_same_connection(self):
        """Regression test: reproduced live against the real backend.
        create_session(initial_message=...) must submit the prompt on the
        SAME websocket connection using the live id session.create already
        returned — not via a separate send_prompt(durable_id, ...) call
        afterward. A brand-new session has no durable row yet (Hermes
        persists it on first prompt), so session.resume-by-durable-id 404s
        (backend error 4007 'session not found') until after that first
        submit lands."""

        new_msgs = [
            {"id": "n1", "role": "user", "content": "hi"},
            {"id": "n2", "role": "assistant", "content": "HELLO"},
        ]

        class InitTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, req):
                p = req.url.raw_path
                if isinstance(p, bytes): p = p.decode()
                if "/messages" in p:
                    return httpx.Response(200, json={"messages": new_msgs})
                if "/health" in p:
                    return httpx.Response(200, json={"status": "ok"})
                if "/sessions" in p:
                    return httpx.Response(200, json={"sessions": []})
                return httpx.Response(404, json={})

        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=InitTransport(), timeout=5)

        ws_frames = [
            {"id": 1, "result": {
                "session_id": "live_new",
                "stored_session_id": "new_sess",
                "info": {},
            }},
        ]

        captured_ws = []
        def fake_ws(url, **kwargs):
            ws = FakeWebSocketSendPrompt(ws_frames)
            captured_ws.append(ws)
            return ws

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = fake_ws
            result = await backend.create_session(
                title="Init", model="test", initial_message="hi",
            )

        assert result["session_id"] == "new_sess"
        assert result["initial_response"] == "HELLO"
        # Exactly one WS connection opened — create and submit share it.
        assert len(captured_ws) == 1
        sent = [json.loads(s) for s in captured_ws[0]._sent]
        assert sent[0]["method"] == "session.create"
        assert sent[1]["method"] == "prompt.submit"
        # Submitted against the live id, never a resume-by-durable-id call.
        assert sent[1]["params"]["session_id"] == "live_new"

    async def test_create_session_ws_error(self):
        """session.create returning an error should raise RuntimeError."""
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=MockTransport({}), timeout=5)

        ws_frames = [{"id": 1, "error": {"code": -1, "message": "nope"}}]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            with pytest.raises(RuntimeError, match="session.create failed"):
                await backend.create_session()

    async def test_ignores_stray_broadcast_event_before_response(self):
        """Regression test: reproduced live against the real backend. The
        shared WS connection also carries unsolicited `event` broadcasts for
        other live sessions (e.g. another GUI tab). If one of those arrives
        before our numbered session.create reply, the old code treated it as
        the response and crashed with KeyError('result'). It must be skipped
        instead."""
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=MockTransport({}), timeout=5)

        ws_frames = [
            {"jsonrpc": "2.0", "method": "event",
             "params": {"session_id": "some_other_live_session", "type": "message.delta"}},
            {"id": 1, "result": {
                "session_id": "live_new",
                "stored_session_id": "new_sess",
                "info": {},
            }},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            result = await backend.create_session(title="Real Work")

        assert result["session_id"] == "new_sess"


# ---------------------------------------------------------------------------
# send_prompt_stream: async generator yields events in order
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSendPromptStream:
    async def test_yields_events_in_order_with_full_transcript(self):
        """Stream a realistic event sequence and verify order + final answer."""
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=MockTransport({}), timeout=5)

        # Build WS frames: resume response + event stream
        ws_frames = [
            # Resume response
            {"id": 1, "result": {"session_id": "live_xyz"}},
            # prompt.submit ack (ignored by stream loop — it's not method="event")
            {"id": 2, "result": {"status": "streaming"}},
            # Event stream
            {"method": "event", "params": {"type": "message.start", "session_id": "live_xyz", "payload": {}}},
            {"method": "event", "params": {"type": "message.delta", "session_id": "live_xyz", "payload": {"delta": "Sure, "}}},
            {"method": "event", "params": {"type": "message.delta", "session_id": "live_xyz", "payload": {"delta": "I can "}}},
            {"method": "event", "params": {"type": "tool.start", "session_id": "live_xyz", "payload": {"name": "terminal"}}},
            {"method": "event", "params": {"type": "tool.complete", "session_id": "live_xyz", "payload": {"name": "terminal"}}},
            {"method": "event", "params": {"type": "message.delta", "session_id": "live_xyz", "payload": {"delta": "help."}}},
            {"method": "event", "params": {"type": "message.complete", "session_id": "live_xyz", "payload": {"content": "Sure, I can help."}}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            events = []
            final = None
            async for etype, payload in backend.send_prompt_stream("test", "help me", timeout=10):
                if etype == "__done__":
                    final = payload
                    break
                events.append((etype, payload.get("delta", "") or payload.get("name", "")))

        # Verify event order
        types = [e[0] for e in events]
        assert types == [
            "message.start",
            "message.delta", "message.delta",
            "tool.start", "tool.complete",
            "message.delta",
            "message.complete",
        ], f"Event order wrong: {types}"

        # Verify final answer assembled from deltas
        assert final is not None
        assert final["response"] == "Sure, I can help."
        assert final["session_id"] == "test"

    async def test_ignores_other_sessions_events(self):
        """Events for a different session_id should be silently skipped."""
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=MockTransport({}), timeout=5)

        ws_frames = [
            {"id": 1, "result": {"session_id": "live_ours"}},
            {"id": 2, "result": {"status": "streaming"}},
            # Another session's event (wrong session_id) — should be skipped
            {"method": "event", "params": {"type": "message.delta", "session_id": "live_other", "payload": {"delta": "ignore me"}}},
            # Our session's event
            {"method": "event", "params": {"type": "message.delta", "session_id": "live_ours", "payload": {"delta": "YES"}}},
            {"method": "event", "params": {"type": "message.complete", "session_id": "live_ours", "payload": {"content": "YES"}}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            events = []
            final = None
            async for etype, payload in backend.send_prompt_stream("s", "go", timeout=10):
                if etype == "__done__":
                    final = payload
                    break
                events.append(etype)

        assert events == ["message.delta", "message.complete"]  # only ours
        assert final["response"] == "YES"

    async def test_error_event_raises(self):
        """An 'error' event mid-turn should raise RuntimeError."""
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=MockTransport({}), timeout=5)

        ws_frames = [
            {"id": 1, "result": {"session_id": "live_err"}},
            {"id": 2, "result": {"status": "streaming"}},
            {"method": "event", "params": {"type": "error", "session_id": "live_err", "payload": {"message": "boom"}}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            with pytest.raises(RuntimeError, match="agent error mid-turn"):
                async for _ in backend.send_prompt_stream("s", "x", timeout=10):
                    pass

    async def test_timeout_returns_what_it_has(self):
        """If WS yields no complete event, stream should time out with
        partial text and timed_out=True."""
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=MockTransport({}), timeout=5)

        # Only resume + submit, then silence (no events at all)
        ws_frames = [
            {"id": 1, "result": {"session_id": "live_t"}},
            {"id": 2, "result": {"status": "streaming"}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            final = None
            async for etype, payload in backend.send_prompt_stream("s", "x", timeout=2):
                if etype == "__done__":
                    final = payload
                    break

        assert final is not None
        assert final["timed_out"] is True
        assert final["response"] == ""


# ---------------------------------------------------------------------------
# branch_session error path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestBranchSession:
    async def test_empty_session_error(self):
        """branch_session should return a friendly error on empty source."""
        transport = MockTransport({
            "/health": (200, {"status": "ok"}),
            "/api/sessions": (200, {"sessions": []}),
        })
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=transport, timeout=5)

        ws_frames = [
            {"id": 1, "result": {"session_id": "live_abc"}},
            {"id": 2, "error": {"code": 4008, "message": "nothing to branch"}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            with pytest.raises(RuntimeError, match="empty"):
                await backend.branch_session("empty_sess")


@pytest.mark.asyncio
class TestSetModel:
    """Unit tests for GuiBackend.set_model (slash.exec /model)."""

    async def test_successful_model_switch(self):
        """set_model resumes session, calls slash.exec /model, returns ok."""
        transport = MockTransport({
            "/health": (200, {"status": "ok"}),
            "/api/sessions": (200, {"sessions": []}),
        })
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=transport, timeout=5)

        ws_frames = [
            # session.resume response
            {"id": 1, "result": {"session_id": "live_xyz"}},
            # slash.exec /model response
            {"id": 2, "result": {
                "output": "Switched to openrouter/anthropic/claude-sonnet-4 (session only).",
            }},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            result = await backend.set_model("durable_123", "openrouter/anthropic/claude-sonnet-4")

        assert result["status"] == "ok"
        assert result["session_id"] == "durable_123"
        assert result["model"] == "openrouter/anthropic/claude-sonnet-4"
        assert "Switched" in result["output"]

    async def test_slash_exec_error_raises(self):
        """slash.exec returning an error raises RuntimeError."""
        transport = MockTransport({
            "/health": (200, {"status": "ok"}),
            "/api/sessions": (200, {"sessions": []}),
        })
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=transport, timeout=5)

        ws_frames = [
            {"id": 1, "result": {"session_id": "live_xyz"}},
            {"id": 2, "error": {"code": 4004, "message": "empty command"}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            with pytest.raises(RuntimeError, match="slash.exec"):
                await backend.set_model("durable_123", "bad/model")

    async def test_busy_session_interrupts_and_retries(self):
        """When slash.exec returns a 'session busy' warning, set_model
        should send session.interrupt and retry the model switch."""
        transport = MockTransport({
            "/health": (200, {"status": "ok"}),
            "/api/sessions": (200, {"sessions": []}),
        })
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=transport, timeout=5)

        # Track all WS frames across two connections
        call_count = {"n": 0}
        frames_per_call = [
            # First connection: resume + slash.exec (busy warning)
            [
                {"id": 1, "result": {"session_id": "live_abc"}},
                {"id": 2, "result": {
                    "output": "  \u2713 Model switched: xiaomi/mimo-v2.5",
                    "warning": "session busy \u2014 /interrupt the current turn before running /model",
                }},
            ],
            # Second connection: resume + interrupt + slash.exec (success)
            [
                {"id": 1, "result": {"session_id": "live_abc"}},
                {"id": 2, "result": {}},  # interrupt response
                {"id": 3, "result": {
                    "output": "  \u2713 Model switched: xiaomi/mimo-v2.5",
                }},
            ],
        ]

        def make_ws(url, **kwargs):
            idx = call_count["n"]
            call_count["n"] += 1
            return FakeWebSocketSendPrompt(frames_per_call[idx])

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = make_ws
            # Patch asyncio.sleep to avoid real delay
            with patch("hermes_chat_delegator.server.asyncio.sleep", new_callable=AsyncMock):
                result = await backend.set_model("durable_123", "xiaomi/mimo-v2.5")

        assert result["status"] == "ok"
        assert result["interrupted_first"] is True
        assert call_count["n"] == 2  # two WS connections

    async def test_resume_failure_raises(self):
        """A failed session.resume raises RuntimeError."""
        transport = MockTransport({
            "/health": (200, {"status": "ok"}),
            "/api/sessions": (200, {"sessions": []}),
        })
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=transport, timeout=5)

        ws_frames = [
            {"id": 1, "error": {"code": 4001, "message": "session not found"}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            with pytest.raises(RuntimeError, match="resume failed"):
                await backend.set_model("nonexistent", "some/model")


# ---------------------------------------------------------------------------
# auto_discover_backend
# ---------------------------------------------------------------------------

class PortRoutingTransport(httpx.AsyncBaseTransport):
    """Routes requests to per-port canned responses, keyed by request port."""

    def __init__(self, by_port: dict[int, dict[str, tuple[int, dict]]]):
        self.by_port = by_port

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        routes = self.by_port.get(request.url.port, {})
        for pattern, (status, body) in routes.items():
            if pattern in request.url.path:
                return httpx.Response(status, json=body)
        return httpx.Response(404, json={"error": "not found"})


_RealAsyncClient = httpx.AsyncClient


def _patched_async_client(transport):
    """Returns a real httpx.AsyncClient wired to `transport`, dropping the
    caller's `transport` kwarg (if any) so ours always wins. Must close over
    the pre-patch class — calling `httpx.AsyncClient` here would recurse into
    whatever mock currently sits at that name."""
    def _make(**kw):
        kw.pop("transport", None)
        return _RealAsyncClient(transport=transport, **kw)
    return _make


@pytest.mark.asyncio
class TestAutoDiscoverBackend:
    async def test_picks_real_backend_not_impostor_on_lower_port(self):
        """Regression test for the H2 discovery bug: a headless Hermes
        backend's /health legitimately returns a bare {"error": ...} body
        (no distinguishing markers), which must not lose to an unrelated
        local service whose /health merely lacks an "error" key — even when
        that impostor sits on a numerically lower port and is probed first.
        """
        transport = PortRoutingTransport({
            8001: {"/health": (200, {"status": "ok"})},  # impostor, lower port
            9999: {
                "/health": (200, {"error": "Headless backend (hermes serve): web UI disabled"}),
                "/api/sessions": (200, {"sessions": []}),
            },
        })
        netstat_output = (
            "  TCP    127.0.0.1:8001    0.0.0.0:0    LISTENING    111\n"
            "  TCP    127.0.0.1:9999    0.0.0.0:0    LISTENING    222\n"
        )

        with patch("hermes_chat_delegator.server.subprocess.run") as mock_run, \
             patch("hermes_chat_delegator.server.httpx.AsyncClient", side_effect=_patched_async_client(transport)):
            mock_run.return_value = MagicMock(stdout=netstat_output)
            port = await auto_discover_backend("127.0.0.1")

        assert port == 9999

    async def test_returns_none_when_no_port_matches(self):
        transport = PortRoutingTransport({})
        netstat_output = "  TCP    127.0.0.1:8001    0.0.0.0:0    LISTENING    111\n"

        with patch("hermes_chat_delegator.server.subprocess.run") as mock_run, \
             patch("hermes_chat_delegator.server.httpx.AsyncClient", side_effect=_patched_async_client(transport)):
            mock_run.return_value = MagicMock(stdout=netstat_output)
            port = await auto_discover_backend("127.0.0.1")

        assert port is None

    async def test_positive_version_marker_still_accepted(self):
        """Non-headless backends that expose version/service markers on
        /health should still be identified without needing /api/sessions."""
        transport = PortRoutingTransport({
            7000: {"/health": (200, {"version": "1.2.3", "service": "hermes"})},
        })
        netstat_output = "  TCP    127.0.0.1:7000    0.0.0.0:0    LISTENING    333\n"

        with patch("hermes_chat_delegator.server.subprocess.run") as mock_run, \
             patch("hermes_chat_delegator.server.httpx.AsyncClient", side_effect=_patched_async_client(transport)):
            mock_run.return_value = MagicMock(stdout=netstat_output)
            port = await auto_discover_backend("127.0.0.1")

        assert port == 7000

    async def test_gated_gateway_impostor_loses_to_unauthenticated_dashboard(self):
        """Regression test: a Hermes messaging-gateway API server can serve a
        Hermes-shaped /health (version key present) on a lower port while
        gating /api/sessions behind Authorization: Bearer <API_SERVER_KEY> —
        a scheme this client never sends. It must not win over the actual
        headless dashboard backend (unauthenticated /api/sessions) sitting on
        a higher port, even though it's probed first and matches on /health.
        """
        transport = PortRoutingTransport({
            8642: {
                "/health": (200, {"status": "ok", "platform": "hermes-agent", "version": "0.18.2"}),
                "/api/sessions": (401, {"error": "unauthorized"}),
            },
            41689: {
                "/health": (404, {"error": "not found"}),
                "/api/sessions": (200, {"sessions": [], "total": 0}),
            },
        })
        netstat_output = (
            "  TCP    127.0.0.1:8642    0.0.0.0:0    LISTENING    111\n"
            "  TCP    127.0.0.1:41689   0.0.0.0:0    LISTENING    222\n"
        )

        with patch("hermes_chat_delegator.server.subprocess.run") as mock_run, \
             patch("hermes_chat_delegator.server.httpx.AsyncClient", side_effect=_patched_async_client(transport)):
            mock_run.return_value = MagicMock(stdout=netstat_output)
            port = await auto_discover_backend("127.0.0.1")

        assert port == 41689

    async def test_prefers_live_instance_over_stale_on_lower_port(self):
        """Regression test for the exact incident topology:
        Two backends alive simultaneously:
          - stale on port 8765, sessions from 2026-07-31 (3 weeks old),
            auth_required=false (answers /api/sessions with 200)
          - live on port 32556, sessions from minutes ago,
            auth_required=false

        The stale instance on the LOWER port must NOT win.  The live
        instance has more recent sessions and must be selected.
        """
        now = time.time()
        transport = PortRoutingTransport({
            8765: {  # stale instance — old sessions, lower port
                "/health": (200, {"ok": True, "version": "0.20.2", "auth_required": False}),
                "/api/sessions": (200, {"sessions": [
                    {"id": "old_1", "last_active": now - 2_000_000},  # ~23 days ago
                    {"id": "old_2", "last_active": now - 3_000_000},
                ]}),
            },
            32556: {  # live instance — recent sessions, higher port
                "/health": (200, {"ok": True, "version": "0.20.2", "auth_required": False}),
                "/api/sessions": (200, {"sessions": [
                    {"id": "live_1", "last_active": now - 300},  # 5 min ago
                    {"id": "live_2", "last_active": now - 60},   # 1 min ago
                ]}),
            },
        })
        netstat_output = (
            "  TCP    127.0.0.1:8765     0.0.0.0:0    LISTENING    111\n"
            "  TCP    127.0.0.1:32556    0.0.0.0:0    LISTENING    222\n"
        )

        with patch("hermes_chat_delegator.server.subprocess.run") as mock_run, \
             patch("hermes_chat_delegator.server.httpx.AsyncClient",
                   side_effect=_patched_async_client(transport)):
            mock_run.return_value = MagicMock(stdout=netstat_output)
            port = await auto_discover_backend("127.0.0.1")

        assert port == 32556, (
            f"Live instance on 32556 (recent sessions) must be preferred "
            f"over stale instance on 8765 (old sessions), got port={port}"
        )

    async def test_ambiguous_recency_returns_none(self):
        """When two confirmed instances have indistinguishable recency
        (within 60s), auto_discover must raise BackendAmbiguous rather
        than silently picking one — a wrong choice invites destructive
        recovery."""
        now = time.time()
        transport = PortRoutingTransport({
            8000: {
                "/health": (200, {"ok": True, "version": "0.20.2"}),
                "/api/sessions": (200, {"sessions": [
                    {"id": "s1", "last_active": now - 10},  # 10s ago
                ]}),
            },
            9000: {
                "/health": (200, {"ok": True, "version": "0.20.2"}),
                "/api/sessions": (200, {"sessions": [
                    {"id": "s2", "last_active": now - 30},  # 30s ago — within 60s
                ]}),
            },
        })
        netstat_output = (
            "  TCP    127.0.0.1:8000    0.0.0.0:0    LISTENING    111\n"
            "  TCP    127.0.0.1:9000    0.0.0.0:0    LISTENING    222\n"
        )

        with patch("hermes_chat_delegator.server.subprocess.run") as mock_run, \
             patch("hermes_chat_delegator.server.httpx.AsyncClient",
                   side_effect=_patched_async_client(transport)):
            mock_run.return_value = MagicMock(stdout=netstat_output)
            with pytest.raises(BackendAmbiguous) as exc_info:
                await auto_discover_backend("127.0.0.1")

        assert set(exc_info.value.ports) == {8000, 9000}
        assert "8000" in str(exc_info.value)
        assert "9000" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Resilience: re-discovery on backend restart / stale token
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestRequireBackendReDiscovery:
    """_require_backend must re-discover when the cached backend dies."""

    async def test_cached_backend_replaced_on_health_failure(self):
        """When the cached backend fails health check, _require_backend
        re-discovers and returns a new backend."""
        import hermes_chat_delegator.server as mod
        # Clear any cached backend
        mod._backend = None

        # Simulate auto_discover_backend finding port 22222
        with patch.object(mod, "auto_discover_backend", new_callable=AsyncMock) as mock_disc:
            mock_disc.return_value = 22222
            with patch.object(mod, "_port_answers", new_callable=AsyncMock) as mock_port:
                mock_port.return_value = False  # pinned port dead
                b = await mod._require_backend()

        assert b.port == 22222
        mock_disc.assert_called_once()


@pytest.mark.asyncio
class TestRetryHandler:
    """_retry_handler must re-discover and retry on connection/auth failures."""

    async def test_connect_error_triggers_rediscovery_and_retry(self):
        """A ConnectError on first call triggers re-discovery; second call succeeds."""
        import hermes_chat_delegator.server as mod

        call_count = {"n": 0}

        async def flaky_operation():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise httpx.ConnectError("Connection refused")
            return _text('{"ok": true}')

        with patch.object(mod, "_reconnect_backend", new_callable=AsyncMock) as mock_reconnect:
            mock_reconnect.return_value = GuiBackend("127.0.0.1", 9999)
            result = await mod._retry_handler(flaky_operation)

        data = json.loads(result[0].text)
        assert data == {"ok": True}
        mock_reconnect.assert_called_once()
        assert call_count["n"] == 2

    async def test_ws_auth_failure_triggers_rediscovery_and_retry(self):
        """A WsAuthFailure triggers re-discovery; retry with new token succeeds."""
        import hermes_chat_delegator.server as mod

        call_count = {"n": 0}

        async def auth_fails_first():
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise WsAuthFailure(9999, 403, "stale token")
            return _text('{"ok": true}')

        with patch.object(mod, "_reconnect_backend", new_callable=AsyncMock) as mock_reconnect:
            mock_reconnect.return_value = GuiBackend("127.0.0.1", 9999)
            result = await mod._retry_handler(auth_fails_first)

        data = json.loads(result[0].text)
        assert data == {"ok": True}
        assert call_count["n"] == 2

    async def test_genuinely_down_returns_honest_error(self):
        """When re-discovery finds nothing, returns not_connected error."""
        import hermes_chat_delegator.server as mod

        async def always_fails():
            raise httpx.ConnectError("Connection refused")

        with patch.object(mod, "_reconnect_backend", new_callable=AsyncMock) as mock_reconnect:
            mock_reconnect.side_effect = RuntimeError("unreachable")
            result = await mod._retry_handler(always_fails)

        data = json.loads(result[0].text)
        assert data["error"]["code"] == "not_connected"
        assert data["error"]["retryable"] is True
        assert "re-discovery" in data["error"]["message"].lower()

    async def test_retry_fails_returns_retry_error(self):
        """When re-discovery succeeds but retry also fails, returns the retry error."""
        import hermes_chat_delegator.server as mod

        async def always_auth_fails():
            raise WsAuthFailure(9999, 403, "still stale")

        with patch.object(mod, "_reconnect_backend", new_callable=AsyncMock) as mock_reconnect:
            mock_reconnect.return_value = GuiBackend("127.0.0.1", 9999)
            result = await mod._retry_handler(always_auth_fails)

        data = json.loads(result[0].text)
        assert data["error"]["code"] == "ws_auth_failed"
        assert data["error"]["retryable"] is True


@pytest.mark.asyncio
class TestFindOrDiscoverBackend:
    """_find_or_discover_backend returns None when genuinely down."""

    async def test_returns_none_when_nothing_found(self):
        import hermes_chat_delegator.server as mod
        mod._backend = None

        with patch.object(mod, "auto_discover_backend", new_callable=AsyncMock) as mock_disc:
            mock_disc.return_value = None
            with patch.object(mod, "_port_answers", new_callable=AsyncMock) as mock_port:
                mock_port.return_value = False
                result = await mod._find_or_discover_backend()

        assert result is None

    async def test_returns_backend_when_discovered(self):
        import hermes_chat_delegator.server as mod
        mod._backend = None

        with patch.object(mod, "auto_discover_backend", new_callable=AsyncMock) as mock_disc:
            mock_disc.return_value = 55555
            result = await mod._find_or_discover_backend()

        assert result is not None
        assert result.port == 55555


@pytest.mark.asyncio
class TestBackendStatusReDiscovery:
    """h_backend_status must re-discover when initial connection fails."""

    async def test_re_discovers_on_connection_failure(self):
        """backend_status re-discovers and returns connected on fresh backend."""
        import hermes_chat_delegator.server as mod

        # Pre-populate with a dead backend that raises on health check
        dead = GuiBackend("127.0.0.1", 11111)
        dead._client = httpx.AsyncClient(
            transport=MockTransport({}), timeout=2,
        )
        mod._backend = dead

        # The fresh backend has a good transport
        fresh_transport = MockTransport({
            "/health": (200, {"status": "ok", "version": "0.20.2"}),
            "/api/sessions": (200, {"sessions": []}),
            "/api/sessions/interrupted": (200, {"sessions": []}),
        })

        async def fake_find():
            fresh = GuiBackend("127.0.0.1", 22222)
            fresh._client = httpx.AsyncClient(transport=fresh_transport, timeout=5)
            mod._backend = fresh
            return fresh

        # Force _require_backend to fail by making health raise
        async def failing_health():
            raise httpx.ConnectError("dead")

        # Patch _require_backend to raise, triggering the except path in h_backend_status
        with patch.object(mod, "_require_backend", side_effect=RuntimeError("not connected")), \
             patch.object(mod, "_find_or_discover_backend", side_effect=fake_find):
            result = await mod.h_backend_status({})

        data = json.loads(result[0].text)
        assert data["connected"] is True
        assert data["port"] == 22222


# ---------------------------------------------------------------------------
# Unpromptable draft: create_chat(no initial_message) -> send_prompt flow
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestUnpromptableDraft:
    """Verify that a draft session (created without initial_message) can be
    successfully prompted via send_prompt — the draft mapping bootstraps
    the durable row by submitting against the live id directly."""

    async def test_send_prompt_boots_draft_durable_row(self):
        """Draft session → send_prompt succeeds: submits against live_id,
        skips session.resume, polls durable id for the reply."""
        new_msgs = [
            {"id": "u1", "role": "user", "content": "hello"},
            {"id": "a1", "role": "assistant", "content": "HI THERE"},
        ]

        # Call tracking: first GET /messages (baseline) returns empty (draft
        # has no prior messages), second GET returns the new reply.
        call_count = {"n": 0}

        class DraftTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, req):
                p = req.url.raw_path
                if isinstance(p, bytes):
                    p = p.decode()
                if "/messages" in p:
                    call_count["n"] += 1
                    if call_count["n"] == 1:
                        # Baseline: no prior messages for a draft
                        return httpx.Response(200, json={"messages": []})
                    return httpx.Response(200, json={"messages": new_msgs})
                if "/health" in p:
                    return httpx.Response(200, json={"status": "ok"})
                if "/sessions" in p:
                    return httpx.Response(200, json={"sessions": []})
                return httpx.Response(404, json={})

        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=DraftTransport(), timeout=5)

        # Step 1: create_session without initial_message (draft)
        ws_frames_create = [
            {"id": 1, "result": {
                "session_id": "live_draft_abc",
                "stored_session_id": "durable_draft_001",
                "info": {},
            }},
        ]

        captured_ws = []
        def fake_ws_create(url, **kwargs):
            ws = FakeWebSocketSendPrompt(ws_frames_create)
            captured_ws.append(ws)
            return ws

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = fake_ws_create
            result = await backend.create_session(title="Draft Test")

        assert result["session_id"] == "durable_draft_001"
        assert result["live_session_id"] == "live_draft_abc"
        # Draft was registered
        assert "durable_draft_001" in backend._draft_sessions
        assert backend._draft_sessions["durable_draft_001"]["live_session_id"] == "live_draft_abc"

        # Step 2: send_prompt against the draft session
        ws_frames_prompt = [
            # prompt.submit ack (NO session.resume — draft path skips it)
            {"id": 1, "result": {"status": "streaming"}},
            # Event stream with response
            {"method": "event", "params": {"type": "message.start", "session_id": "live_draft_abc", "payload": {}}},
            {"method": "event", "params": {"type": "message.delta", "session_id": "live_draft_abc", "payload": {"delta": "HI "}}},
            {"method": "event", "params": {"type": "message.delta", "session_id": "live_draft_abc", "payload": {"delta": "THERE"}}},
            {"method": "event", "params": {"type": "message.complete", "session_id": "live_draft_abc", "payload": {"content": "HI THERE"}}},
        ]

        def fake_ws_prompt(url, **kwargs):
            ws = FakeWebSocketSendPrompt(ws_frames_prompt)
            captured_ws.append(ws)
            return ws

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = fake_ws_prompt
            result = await backend.send_prompt("durable_draft_001", "hello", timeout=10)

        assert result["response"] == "HI THERE"
        # Draft mapping was consumed (popped)
        assert "durable_draft_001" not in backend._draft_sessions

        # Verify the WS calls: first was session.create, second was prompt.submit
        # with NO session.resume in between
        sent_create = json.loads(captured_ws[0]._sent[0])
        assert sent_create["method"] == "session.create"
        sent_prompt = json.loads(captured_ws[1]._sent[0])
        assert sent_prompt["method"] == "prompt.submit"
        # Submitted against the live id, not the durable id
        assert sent_prompt["params"]["session_id"] == "live_draft_abc"

    async def test_draft_mapping_survives_create_then_prompt(self):
        """After create_session (draft), send_prompt pops the draft entry
        and uses it. A second send_prompt on the same id falls through to
        the normal resume path."""
        class SimpleTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, req):
                p = req.url.raw_path
                if isinstance(p, bytes):
                    p = p.decode()
                if "/messages" in p:
                    return httpx.Response(200, json={"messages": []})
                if "/health" in p:
                    return httpx.Response(200, json={"status": "ok"})
                if "/sessions" in p:
                    return httpx.Response(200, json={"sessions": []})
                return httpx.Response(404, json={})

        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=SimpleTransport(), timeout=5)

        # Create draft
        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kw: FakeWebSocketSendPrompt([
                {"id": 1, "result": {"session_id": "live_x", "stored_session_id": "durable_x", "info": {}}},
            ])
            await backend.create_session()

        assert "durable_x" in backend._draft_sessions

        # First send_prompt: consumes draft, submits against live_id (no resume)
        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kw: FakeWebSocketSendPrompt([
                {"id": 1, "result": {"status": "streaming"}},
                # Event stream
                {"method": "event", "params": {"type": "message.start", "session_id": "live_x", "payload": {}}},
                {"method": "event", "params": {"type": "message.delta", "session_id": "live_x", "payload": {"delta": "First response"}}},
                {"method": "event", "params": {"type": "message.complete", "session_id": "live_x", "payload": {"content": "First response"}}},
            ])
            await backend.send_prompt("durable_x", "first msg", timeout=5)

        assert "durable_x" not in backend._draft_sessions  # consumed

        # Second send_prompt: falls through to normal resume path
        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kw: FakeWebSocketSendPrompt([
                {"id": 1, "result": {"session_id": "live_x"}},
                {"id": 2, "result": {"status": "streaming"}},
                # Event stream
                {"method": "event", "params": {"type": "message.start", "session_id": "live_x", "payload": {}}},
                {"method": "event", "params": {"type": "message.delta", "session_id": "live_x", "payload": {"delta": "Second response"}}},
                {"method": "event", "params": {"type": "message.complete", "session_id": "live_x", "payload": {"content": "Second response"}}},
            ])
            await backend.send_prompt("durable_x", "second msg", timeout=5)

        # Both calls succeeded — no crash

    async def test_send_prompt_stream_handles_draft(self):
        """Draft session → send_prompt_stream also uses the draft path."""
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=MockTransport({}), timeout=5)

        # Register a draft manually
        backend._draft_sessions["durable_s"] = {
            "live_session_id": "live_s",
            "created_at": time.time(),
        }

        ws_frames = [
            # prompt.submit ack (no session.resume — draft path)
            {"id": 1, "result": {"status": "streaming"}},
            # Event stream
            {"method": "event", "params": {"type": "message.start", "session_id": "live_s", "payload": {}}},
            {"method": "event", "params": {"type": "message.delta", "session_id": "live_s", "payload": {"delta": "Draft reply"}}},
            {"method": "event", "params": {"type": "message.complete", "session_id": "live_s", "payload": {"content": "Draft reply"}}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            final = None
            async for etype, payload in backend.send_prompt_stream("durable_s", "go", timeout=10):
                if etype == "__done__":
                    final = payload
                    break

        assert final is not None
        assert final["response"] == "Draft reply"
        assert "durable_s" not in backend._draft_sessions  # consumed

    async def test_residual_4007_translates_to_unpromptable_draft(self):
        """When send_prompt hits a 4007 on a session with no draft mapping
        (e.g. MCP server restarted and lost the mapping), the error is
        translated into a clear unpromptable_draft message."""
        import hermes_chat_delegator.server as mod

        # Patch _require_backend to return a backend whose send_prompt raises 4007
        class FailingBackend:
            port = 9999
            async def send_prompt(self, sid, msg, timeout=300):
                raise RuntimeError("resume failed: {'code': 4007, 'message': 'session not found'}")
            async def send_prompt_stream(self, sid, msg, timeout=300):
                raise RuntimeError("resume failed: {'code': 4007, 'message': 'session not found'}")

        with patch.object(mod, "_require_backend", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = FailingBackend()
            result = await mod.h_send_prompt({"session_id": "dead_draft", "message": "test"})

        data = json.loads(result[0].text)
        assert data["error"]["code"] == "unpromptable_draft"
        assert "no durable row" in data["error"]["message"]
        assert "create_chat with initial_message" in data["error"]["message"]

    async def test_residual_4007_stream_translates_to_unpromptable_draft(self):
        """Same translation for send_prompt_stream."""
        import hermes_chat_delegator.server as mod

        class FailingBackend:
            port = 9999
            async def send_prompt_stream(self, sid, msg, timeout=300):
                # Must be an async generator (yield), not just raise
                raise RuntimeError("resume failed: {'code': 4007, 'message': 'session not found'}")
                yield  # pragma: no cover — makes this an async generator

        with patch.object(mod, "_require_backend", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = FailingBackend()
            # Need a request context for progress notifications
            mock_ctx = MagicMock()
            mock_ctx.meta = None
            mock_server = MagicMock()
            mock_server.request_context = mock_ctx
            with patch.object(mod, "_current_server", mock_server):
                result = await mod.h_send_prompt_stream({
                    "session_id": "dead_draft", "message": "test"
                })

        data = json.loads(result[0].text)
        assert data["error"]["code"] == "unpromptable_draft"
        assert "no durable row" in data["error"]["message"]

    async def test_non_4007_runtime_error_passes_through(self):
        """A RuntimeError that is NOT a 4007/session-not-found must still
        surface as a backend_error, not an unpromptable_draft."""
        import hermes_chat_delegator.server as mod

        class OtherErrorBackend:
            port = 9999
            async def send_prompt(self, sid, msg, timeout=300):
                raise RuntimeError("something else went wrong")

        with patch.object(mod, "_require_backend", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = OtherErrorBackend()
            result = await mod.h_send_prompt({"session_id": "x", "message": "y"})

        data = json.loads(result[0].text)
        assert data["error"]["code"] == "backend_error"
        assert "something else went wrong" in data["error"]["message"]


# ===========================================================================
# Identity-aware backend resolution: RED tests
# These must FAIL against the current code, then GREEN after fixes.
# ===========================================================================


@pytest.mark.asyncio
class TestHeadlessHealthBody:
    """The liveness gate must NOT be /health. Two real backends prove this:

    - 59681: no /health endpoint at all (404), but serves /api/sessions
      with the correct token. This IS the working backend.
    - 8642: perfect /health with {"platform":"hermes-agent","version":"..."},
      but 401 on every real operation. This is a SECOND instance with a
      different token — useless.

    The authenticated /api/sessions probe is the only liveness test that
    answers all three questions: is something listening, is it Hermes,
    does my token work here.
    """

    async def test_no_health_endpoint_but_valid_sessions_is_alive(self):
        """59681-style — backend has no /health (404) but serves
        /api/sessions with auth. is_alive() must succeed here.

        NOTE: old code (health() as gate) ALSO passes this test because
        health() returns a truthy dict on 404 — so this is NOT a
        regression test for old code. It is a guard that prevents future
        regressions from breaking the 59681-style backend. The paired
        test_perfect_health_but_401_sessions_is_not_alive IS the
        regression test — it fails with old code.
        """
        import hermes_chat_delegator.server as mod
        mod._backend = None

        class NoHealthButSessionsTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                path = str(request.url.raw_path, encoding="utf-8") if isinstance(request.url.raw_path, bytes) else request.url.raw_path
                if "/health" in path:
                    return httpx.Response(404, json={"error": "not found"})
                if "/api/sessions" in path:
                    return httpx.Response(200, json={"sessions": [
                        {"id": "real_session", "last_active": 1000},
                    ]})
                return httpx.Response(404, json={})

        working = GuiBackend("127.0.0.1", 59681, token="real-token")
        working._client = httpx.AsyncClient(
            transport=NoHealthButSessionsTransport(), timeout=5
        )
        mod._backend = working

        # Mock auto_discover to prevent real netstat from finding 59681
        async def fake_discover(host="127.0.0.1"):
            return None  # nothing found

        with patch.object(mod, "auto_discover_backend", side_effect=fake_discover), \
             patch.object(mod, "_port_answers", new_callable=AsyncMock, return_value=False), \
             patch.dict(os.environ, {"HERMES_GUI_PORT": "", "HERMES_DASHBOARD_SESSION_TOKEN": ""}, clear=False):
            # _require_backend should keep this backend — is_alive() should
            # succeed via /api/sessions even though /health returns 404
            result = await mod._require_backend()

        assert result.port == 59681, (
            f"Backend on 59681 serves /api/sessions — must not be evicted; "
            f"got port={result.port}"
        )

    async def test_perfect_health_but_401_sessions_is_not_alive(self):
        """RED: 8642-style — backend has perfect /health with hermes-agent
        banner, but 401 on /api/sessions (wrong token). _require_backend
        must NOT treat this as alive. Current code: health() returns truthy
        → cached → every real operation fails."""
        import hermes_chat_delegator.server as mod
        mod._backend = None

        class PerfectHealthButUnauthorizedTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                path = str(request.url.raw_path, encoding="utf-8") if isinstance(request.url.raw_path, bytes) else request.url.raw_path
                if "/health" in path:
                    return httpx.Response(200, json={
                        "status": "ok", "platform": "hermes-agent",
                        "version": "0.20.2",
                    })
                if "/api/sessions" in path:
                    return httpx.Response(401, json={"error": "unauthorized"})
                return httpx.Response(404, json={})

        wrong_token = GuiBackend("127.0.0.1", 8642, token="wrong-token")
        wrong_token._client = httpx.AsyncClient(
            transport=PerfectHealthButUnauthorizedTransport(), timeout=5
        )
        mod._backend = wrong_token

        # _require_backend should reject this — is_alive() fails on 401
        # Even though health() succeeds, the authenticated probe fails.
        discover_calls = []

        async def fake_discover(host="127.0.0.1"):
            discover_calls.append(host)
            return 99999

        with patch.object(mod, "auto_discover_backend", side_effect=fake_discover), \
             patch.object(mod, "_port_answers", new_callable=AsyncMock, return_value=False), \
             patch.dict(os.environ, {"HERMES_GUI_PORT": "", "HERMES_DASHBOARD_SESSION_TOKEN": ""}, clear=False):
            result = await mod._require_backend()

        assert result.port == 99999, (
            f"Backend on 8642 returns 401 on /api/sessions — must be evicted; "
            f"got port={result.port}"
        )


@pytest.mark.asyncio
class TestSplitPortTokenResolution:
    """Bug (b): port and token resolved independently by two first-match scans.
    Two backends on different ports with different tokens — the resolver must
    find the (port, token) pair that authenticates, not mix them."""

    async def test_correct_token_selected_for_port(self):
        """Two backends on different ports: discovery must pick the one that
        has the most recent sessions, not the one on the lower port."""
        now = time.time()
        transport = PortRoutingTransport({
            8001: {
                "/health": (200, {"version": "1.0"}),
                "/api/sessions": (200, {"sessions": [
                    {"id": "sess_X", "last_active": now - 2000000},  # ~23 days ago
                ]}),
            },
            9002: {
                "/health": (200, {"version": "1.0"}),
                "/api/sessions": (200, {"sessions": [
                    {"id": "sess_Z", "last_active": now - 50},  # 50s ago
                ]}),
            },
        })
        netstat_output = (
            "  TCP    127.0.0.1:8001    0.0.0.0:0    LISTENING    111\n"
            "  TCP    127.0.0.1:9002    0.0.0.0:0    LISTENING    222\n"
        )

        with patch("hermes_chat_delegator.server.subprocess.run") as mock_run, \
             patch("hermes_chat_delegator.server.httpx.AsyncClient",
                   side_effect=_patched_async_client(transport)):
            mock_run.return_value = MagicMock(stdout=netstat_output)
            port = await auto_discover_backend("127.0.0.1")

        assert port == 9002, (
            f"Expected port 9002 (more recent sessions), got {port}"
        )

    async def test_401_on_port_demotes_it(self):
        """When a port returns 401 on /api/sessions (auth required),
        it must not be selected even if /health answers."""
        transport = PortRoutingTransport({
            8001: {
                "/health": (200, {"version": "1.0", "service": "hermes"}),
                "/api/sessions": (401, {"error": "unauthorized"}),
            },
            9002: {
                "/health": (200, {"version": "1.0"}),
                "/api/sessions": (200, {"sessions": []}),
            },
        })
        netstat_output = (
            "  TCP    127.0.0.1:8001    0.0.0.0:0    LISTENING    111\n"
            "  TCP    127.0.0.1:9002    0.0.0.0:0    LISTENING    222\n"
        )

        with patch("hermes_chat_delegator.server.subprocess.run") as mock_run, \
             patch("hermes_chat_delegator.server.httpx.AsyncClient",
                   side_effect=_patched_async_client(transport)):
            mock_run.return_value = MagicMock(stdout=netstat_output)
            port = await auto_discover_backend("127.0.0.1")

        assert port == 9002, (
            f"Port 8001 returned 401 — must not be selected, got {port}"
        )

    async def test_non_hermes_service_rejected_by_port_answers(self):
        """_port_answers must require positive Hermes identity, not just HTTP.

        Foundry Manager on port 3000 returns 200 with unrelated JSON on
        /api/sessions.  A service returning 200 with an unrelated body
        must NOT be accepted as a Hermes backend.
        """
        import hermes_chat_delegator.server as mod

        class NonHermesTransport(httpx.AsyncBaseTransport):
            """Simulates a non-Hermes service (e.g. Foundry Manager) that
            returns 200 with unrelated JSON on /api/sessions."""
            async def handle_async_request(self, request):
                return httpx.Response(200, json={
                    "tasks": [], "total": 0,  # FM shape, not Hermes
                })

        non_hermes_client = httpx.AsyncClient(
            transport=NonHermesTransport(), timeout=2
        )

        # _port_answers creates its own AsyncClient — mock it
        with patch("hermes_chat_delegator.server.httpx.AsyncClient",
                   return_value=non_hermes_client):
            result = await mod._port_answers("127.0.0.1", 3000)

        assert result is False, (
            "_port_answers must reject non-Hermes service returning "
            "200 with unrelated JSON (no 'sessions' key)"
        )


@pytest.mark.asyncio
class TestSessionScoped404TriggersRediscovery:
    """Bug (c): a 404 on a session-scoped endpoint when the session was
    recently visible is a routing failure (split instance), not a missing
    session. Must trigger re-discovery and retry."""

    async def test_404_on_known_session_triggers_retry_via_handler(self):
        """get_messages for a session that list_chats just returned, going
        through _retry_handler: a 404 must trigger re-discovery and retry
        on the correct backend.

        GREEN: _retry_handler catches HTTPStatusError(404), calls
        _reconnect_backend, retries, and returns success.
        """
        import hermes_chat_delegator.server as mod

        # Backend A (cached): list_chats returns session X, but get_messages
        # for X returns 404 — this is the split-instance signature.
        class SplitTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                path = str(request.url.raw_path, encoding="utf-8") if isinstance(request.url.raw_path, bytes) else request.url.raw_path
                if "/health" in path:
                    return httpx.Response(200, json={"status": "ok"})
                if "/api/sessions/X/messages" in path:
                    # Split instance: 404 for session X
                    return httpx.Response(404, json={"error": "session not found"})
                if "/api/sessions" in path and "/messages" not in path:
                    return httpx.Response(200, json={"sessions": [{"id": "X"}]})
                return httpx.Response(404, json={})

        fresh_transport = MockTransport({
            "/health": (200, {"status": "ok", "version": "0.20.2"}),
            "/api/sessions/X/messages": (200, {"messages": [
                {"id": "m1", "role": "user", "content": "hi"},
                {"id": "m2", "role": "assistant", "content": "hello"},
            ]}),
        })

        # Pre-cache backend A (the split instance)
        split_backend = GuiBackend("127.0.0.1", 11111)
        split_backend._client = httpx.AsyncClient(transport=SplitTransport(), timeout=5)
        mod._backend = split_backend

        fresh_backend = GuiBackend("127.0.0.1", 22222)
        fresh_backend._client = httpx.AsyncClient(transport=fresh_transport, timeout=5)

        # Patch _require_backend so the retry path returns the fresh backend
        # instead of running real auto_discover_backend (which finds live ports)
        call_count = {"n": 0}
        async def fake_require():
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: return the split backend (will 404)
                return split_backend
            # Retry: return the fresh backend
            return fresh_backend

        with patch.object(mod, "_require_backend", side_effect=fake_require), \
             patch.object(mod, "_reconnect_backend", new_callable=AsyncMock):
            result = await mod.h_get_messages({"session_id": "X", "last_n": 5})

        data = json.loads(result[0].text)
        # Must NOT be an error — the retry should have found the right backend
        assert "error" not in data, (
            f"404 on known session must trigger re-discovery, got: {data}"
        )
        assert data["message_count"] == 2
        assert data["messages"][-1]["content"] == "hello"


@pytest.mark.asyncio
class TestDraftSessionRecovery:
    """Bug (d): create_chat returns a draft with persisted=false, and if the
    bootstrapping send_prompt times out, the session is silently lost —
    get_chat reports session_not_found. A draft that cannot be recovered
    after a timeout is a data-loss bug."""

    async def test_draft_timeout_preserves_session_id(self):
        """When send_prompt on a draft times out, the response must include
        the session_id so the caller can retry — not silently lose it."""
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=MockTransport({}), timeout=5)

        # Register a draft session in the persistent store
        record = DraftSessionRecord(
            session_id="durable_draft",
            live_session_id="live_draft",
            created_at=time.time(),
        )
        # Directly add to the in-memory cache for testing
        backend._draft_sessions._cache["durable_draft"] = record

        # WS: prompt.submit succeeds, but poll times out
        ws_frames = [
            {"id": 1, "result": {"status": "streaming"}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            # timeout=0 skips the poll loop, simulating a timeout
            result = await backend.send_prompt("durable_draft", "hello", timeout=0)

        # The result must indicate timeout AND preserve the session_id
        assert result["timed_out"] is True
        assert result["session_id"] == "durable_draft"

    async def test_4007_fallback_to_direct_submit(self):
        """When session.resume fails with 4007 (durable row not yet created),
        send_prompt must fall back to direct prompt.submit against the durable
        id — bootstrapping the durable row, same as the draft path.

        RED: current code raises RuntimeError on 4007 with no fallback.
        This is the data-loss path: both create_chat(with initial_message)
        and create_chat(no message) + send_prompt hit this when the MCP
        server restarts and loses the in-memory draft mapping.
        """
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=MockTransport({}), timeout=5)

        # Session.resume → 4007 error (session lost), then fallback to direct submit
        # Event stream with agent response
        ws_frames = [
            {"id": 1, "error": {"code": 4007, "message": "session not found"}},
            {"id": 2, "result": {"status": "streaming"}},
            # Event stream after fallback to direct submit
            {"method": "event", "params": {"type": "message.start", "session_id": "durable_sess", "payload": {}}},
            {"method": "event", "params": {"type": "message.delta", "session_id": "durable_sess", "payload": {"delta": "HI "}}},
            {"method": "event", "params": {"type": "message.delta", "session_id": "durable_sess", "payload": {"delta": "THERE"}}},
            {"method": "event", "params": {"type": "message.complete", "session_id": "durable_sess", "payload": {"content": "HI THERE"}}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            # No draft mapping — simulating MCP server restart
            result = await backend.send_prompt("durable_sess", "hello", timeout=10)

        # Must succeed via fallback, not raise RuntimeError
        assert result["response"] == "HI THERE", (
            f"Expected fallback to succeed, got: {result}"
        )

    async def test_4007_fallback_also_works_for_initial_message_path(self):
        """create_chat with initial_message returns a session_id, but if the
        MCP server restarts and the durable row was never created (initial
        prompt was submitted against live_id, not durable_id), a subsequent
        send_prompt must still work via the 4007 fallback."""
        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=MockTransport({}), timeout=5)

        ws_frames = [
            {"id": 1, "error": {"code": 4007, "message": "session not found"}},
            {"id": 2, "result": {"status": "streaming"}},
            # Event stream with response
            {"method": "event", "params": {"type": "message.start", "session_id": "initial_msg_sess", "payload": {}}},
            {"method": "event", "params": {"type": "message.delta", "session_id": "initial_msg_sess", "payload": {"delta": "SURE"}}},
            {"method": "event", "params": {"type": "message.complete", "session_id": "initial_msg_sess", "payload": {"content": "SURE"}}},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kwargs: FakeWebSocketSendPrompt(ws_frames)
            result = await backend.send_prompt("initial_msg_sess", "follow up", timeout=10)

        assert result["response"] == "SURE"

    async def test_draft_timeout_get_chat_recovers(self):
        """After a draft times out on send_prompt, get_chat should still
        find the session — the backend persisted it on first prompt.submit,
        even if we didn't get the reply."""
        import hermes_chat_delegator.server as mod

        # Backend: session exists after the prompt was submitted
        class PostSubmitTransport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                path = str(request.url.raw_path, encoding="utf-8") if isinstance(request.url.raw_path, bytes) else request.url.raw_path
                if "/health" in path:
                    return httpx.Response(200, json={"status": "ok"})
                if path.rstrip("/") == "/api/sessions/durable_draft":
                    # Session IS persisted (backend saved it on prompt.submit)
                    return httpx.Response(200, json={
                        "id": "durable_draft", "title": "Draft",
                        "model": "test", "message_count": 1,
                    })
                if "/api/sessions" in path:
                    return httpx.Response(200, json={"sessions": []})
                return httpx.Response(404, json={})

        fresh_backend = GuiBackend("127.0.0.1", 22222)
        fresh_backend._client = httpx.AsyncClient(transport=PostSubmitTransport(), timeout=5)
        mod._backend = fresh_backend

        # After the timeout, get_chat should find the session
        info = await fresh_backend.get_session("durable_draft")
        assert info["id"] == "durable_draft"
        assert info["title"] == "Draft"


@pytest.mark.asyncio
class TestSessionAffinityDiscovery:
    """Discovery must pick the backend that OWNS the session, not the one
    that was merely active most recently.

    Regression: this machine runs one backend per Hermes profile
    (``hermes_cli.main --profile <name> serve``).  All of them are live, all
    answer ``/api/sessions`` with 200, and all have recent activity, so the
    recency ranking picked whichever profile happened to move last.  When
    that was not the profile holding the caller's session, the session-404
    recovery path re-ran the same ranking, re-selected the same wrong
    backend, and returned a permanent ``session_not_found on re-discovered
    backend`` while the real backend sat healthy on another port.
    """

    SID = "20260822_184745_cbe1a5"

    def _netstat(self, *ports):
        return "".join(
            f"  TCP    127.0.0.1:{p}    0.0.0.0:0    LISTENING    {i}\n"
            for i, p in enumerate(ports, start=100)
        )

    async def _discover(self, transport, *ports, **kw):
        with patch("hermes_chat_delegator.server.subprocess.run") as mock_run, \
             patch("hermes_chat_delegator.server.httpx.AsyncClient",
                   side_effect=_patched_async_client(transport)):
            mock_run.return_value = MagicMock(stdout=self._netstat(*ports))
            return await auto_discover_backend("127.0.0.1", **kw)

    async def test_owner_beats_more_recent_nonowner(self):
        """The owning backend wins even when another backend is far more
        recently active.  Without session affinity this returns 9002."""
        now = time.time()
        transport = PortRoutingTransport({
            8001: {  # owns the session, but quiet for a day
                f"/api/sessions/{self.SID}/messages": (200, {"messages": []}),
                "/api/sessions": (200, {"sessions": [
                    {"id": self.SID, "last_active": now - 86400},
                ]}),
            },
            9002: {  # busy right now, but does not have the session
                f"/api/sessions/{self.SID}/messages": (404, {"error": "not found"}),
                "/api/sessions": (200, {"sessions": [
                    {"id": "other", "last_active": now - 5},
                ]}),
            },
        })
        port = await self._discover(transport, 8001, 9002,
                                    require_session_id=self.SID)
        assert port == 8001, (
            f"Expected 8001 (owns {self.SID}), got {port} — recency beat ownership"
        )

    async def test_unaimed_discovery_picks_the_wrong_one(self):
        """Companion RED proof: the same topology without affinity selects the
        recently-active non-owner.  This is the bug being fixed, pinned so the
        affinity test above cannot silently pass for the wrong reason."""
        now = time.time()
        transport = PortRoutingTransport({
            8001: {"/api/sessions": (200, {"sessions": [
                {"id": self.SID, "last_active": now - 86400}]})},
            9002: {"/api/sessions": (200, {"sessions": [
                {"id": "other", "last_active": now - 5}]})},
        })
        port = await self._discover(transport, 8001, 9002)
        assert port == 9002

    async def test_disowning_backend_is_excluded_not_merely_ranked(self):
        """A backend that 404s the session is removed from consideration, so
        it cannot win by being the only other candidate."""
        now = time.time()
        transport = PortRoutingTransport({
            9002: {
                f"/api/sessions/{self.SID}/messages": (404, {"error": "not found"}),
                "/api/sessions": (200, {"sessions": [
                    {"id": "other", "last_active": now - 5},
                ]}),
            },
        })
        port = await self._discover(transport, 9002, require_session_id=self.SID)
        assert port is None, f"Disowning backend should be excluded, got {port}"

    async def test_inconclusive_answer_does_not_exclude(self):
        """A 5xx on the ownership probe is not a denial — the candidate stays
        eligible via the recency path so a transient hiccup cannot strand us."""
        now = time.time()
        transport = PortRoutingTransport({
            8001: {
                f"/api/sessions/{self.SID}/messages": (503, {"error": "busy"}),
                "/api/sessions": (200, {"sessions": [
                    {"id": self.SID, "last_active": now - 5},
                ]}),
            },
        })
        port = await self._discover(transport, 8001, require_session_id=self.SID)
        assert port == 8001, f"5xx must not exclude the candidate, got {port}"

    async def test_falls_back_to_unaimed_when_nobody_claims(self):
        """If no backend claims the session (e.g. an unauthenticable profile,
        or a draft with no messages endpoint yet), behaviour must degrade to
        exactly what it was before affinity existed — never to a new failure."""
        now = time.time()
        transport = PortRoutingTransport({
            9002: {
                f"/api/sessions/{self.SID}/messages": (404, {"error": "not found"}),
                "/api/sessions": (200, {"sessions": [
                    {"id": "other", "last_active": now - 5},
                ]}),
            },
        })
        with patch("hermes_chat_delegator.server.subprocess.run") as mock_run, \
             patch("hermes_chat_delegator.server.httpx.AsyncClient",
                   side_effect=_patched_async_client(transport)), \
             patch.dict(os.environ, {"HERMES_GUI_PORT": ""}, clear=False):
            mock_run.return_value = MagicMock(stdout=self._netstat(9002))
            server._backend = None
            backend = await server._find_or_discover_backend(
                require_session_id=self.SID
            )
        server._backend = None
        assert backend is not None, "must fall back rather than return None"
        assert backend.port == 9002


@pytest.mark.asyncio
class TestDraftSessionFilePersistence:
    """Smoke test: verify draft sessions are persisted to disk."""

    async def test_draft_session_file_created_on_create(self):
        """Creating a draft session should write to ~/.hermes/mcp_draft_sessions.json."""
        import tempfile
        from pathlib import Path

        # Use a temporary file for this test
        temp_file = Path(tempfile.gettempdir()) / "test_draft_sessions.json"
        if temp_file.exists():
            temp_file.unlink()

        backend = GuiBackend("127.0.0.1", 9999)
        backend._client = httpx.AsyncClient(transport=MockTransport({}), timeout=5)
        # Override the draft sessions path for testing
        backend._draft_sessions.path = temp_file

        # Create a draft session
        ws_frames = [
            {"id": 1, "result": {
                "session_id": "live_test",
                "stored_session_id": "durable_test_001",
                "info": {},
            }},
        ]

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kw: FakeWebSocketSendPrompt(ws_frames)
            result = await backend.create_session(title="Persistence Test")

        assert result["session_id"] == "durable_test_001"
        # File should exist and contain the draft session
        assert temp_file.exists(), f"Draft session file {temp_file} was not created"
        data = json.loads(temp_file.read_text())
        assert "durable_test_001" in data
        assert data["durable_test_001"]["live_session_id"] == "live_test"

        # Clean up
        if temp_file.exists():
            temp_file.unlink()


# ---------------------------------------------------------------------------
# Regression: GET /api/sessions rejects limit>100 with a bodyless 422
# ---------------------------------------------------------------------------


class CapEnforcingTransport(httpx.AsyncBaseTransport):
    """Mimics the real backend's cap on ``/api/sessions?limit=``.

    The live Hermes backend answers ``limit>=101`` with an HTTP 422 that
    carries **no body** (verified 2026-09-04 by bisecting the query string
    against a running instance). Reproducing the empty body matters: it is
    why the original failure surfaced as a bare status code with nothing to
    explain it.
    """

    def __init__(self, cap: int = server.MAX_SESSIONS_LIMIT):
        self.cap = cap
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        limit = request.url.params.get("limit")
        if limit is not None and int(limit) > self.cap:
            return httpx.Response(422, content=b"")
        return httpx.Response(200, json={"sessions": [
            {"id": "sess_new", "title": "forked", "parent_session_id": "sess_src"},
        ]})

    @property
    def limits(self) -> list[int]:
        return [int(r.url.params["limit"]) for r in self.requests
                if "limit" in r.url.params]


@pytest.fixture
def capped_backend():
    transport = CapEnforcingTransport()
    backend = GuiBackend("127.0.0.1", 9999, token="test-token")
    backend._client = httpx.AsyncClient(
        transport=transport, timeout=5, headers=backend._auth_headers(),
    )
    return backend, transport


@pytest.mark.asyncio
class TestSessionsLimitCap:
    async def test_list_sessions_clamps_above_cap(self, capped_backend):
        """A caller asking for more than the backend serves gets rows, not a 422."""
        b, t = capped_backend
        sessions = await b.list_sessions(limit=500)
        assert sessions, "expected rows back rather than an exception"
        assert t.limits == [server.MAX_SESSIONS_LIMIT]

    async def test_list_sessions_preserves_limit_under_cap(self, capped_backend):
        """Clamping must not inflate a small request into a large one."""
        b, t = capped_backend
        await b.list_sessions(limit=8)
        assert t.limits == [8]

    async def test_list_sessions_floors_at_one(self, capped_backend):
        """limit=0 would be rejected as out of range at the other end."""
        b, t = capped_backend
        await b.list_sessions(limit=0)
        assert t.limits == [1]

    async def test_list_all_stays_within_cap(self, capped_backend):
        """``_list_all`` previously hardcoded limit=500.

        That made ``fork_chat`` fail on every call, and it failed *after*
        the WebSocket branch had already created the session — so the fork
        was orphaned behind an error message.
        """
        b, t = capped_backend
        sessions = await b._list_all()
        assert [s["id"] for s in sessions] == ["sess_new"]
        assert t.limits and max(t.limits) <= server.MAX_SESSIONS_LIMIT

    async def test_cap_transport_actually_rejects(self, capped_backend):
        """Guards the guard: prove the fixture 422s an unclamped request.

        Without this, every assertion above would still pass if the
        transport silently accepted any limit.
        """
        b, t = capped_backend
        r = await b._request(
            "GET", f"{b.base_url}/api/sessions", params={"limit": 500},
        )
        assert r.status_code == 422
        assert r.content == b""


class TestExcDetail:
    """``_exc_detail`` must never return an empty string.

    The original dispatcher interpolated ``str(e)`` directly, so a bare
    ``TimeoutError()`` reported "Internal error in list_chats: " — a message
    naming no cause at all.
    """

    def test_empty_exception_str_falls_back_to_type_name(self):
        # TimeoutError also gets an actionable hint (defect 5): the bare
        # fallback alone ("TimeoutError (no message)") told the caller
        # nothing about what to do next.
        detail = server._exc_detail(TimeoutError())
        assert detail.startswith("TimeoutError (no message)")
        assert "timeout" in detail.lower()
        assert "retry" in detail.lower()

    def test_timeout_hint_is_actionable(self):
        """A bare TimeoutError must name the fix (larger timeout), not just
        the type — this was the live 'TimeoutError (no message)' complaint."""
        detail = server._exc_detail(asyncio.TimeoutError())
        assert "retry with a larger 'timeout' argument" in detail

    def test_non_timeout_bare_exception_keeps_plain_fallback(self):
        """The hint is TimeoutError-specific; other bare exceptions keep
        the plain type-name fallback so we do not invent stack traces."""
        assert server._exc_detail(RuntimeError()) == "RuntimeError (no message)"

    def test_whitespace_only_message_is_treated_as_empty(self):
        assert server._exc_detail(RuntimeError("   ")) == "RuntimeError (no message)"

    def test_real_message_is_preserved_verbatim(self):
        assert server._exc_detail(RuntimeError("boom")) == "boom"

    def test_bodyless_http_error_reports_its_status(self):
        """The backend's limit>100 rejection is a 422 with an empty body."""
        request = httpx.Request("GET", "http://127.0.0.1:9999/api/sessions")
        response = httpx.Response(422, content=b"", request=request)
        exc = httpx.HTTPStatusError("", request=request, response=response)
        detail = server._exc_detail(exc)
        assert "422" in detail
        assert detail.strip() != ""

    def test_never_returns_empty_for_any_bare_builtin(self):
        for exc_type in (TimeoutError, RuntimeError, ValueError, OSError,
                         ConnectionError, KeyError):
            assert server._exc_detail(exc_type()).strip() != ""


class TestSetModelDocumentsCorrectFormat:
    def test_description_does_not_advertise_gateway_prefix(self):
        """The docstring used to show 'openrouter/anthropic/claude-sonnet-4',
        which the backend rejects outright."""
        tool = next(t for t in server.TOOLS if t.name == "set_model")
        blob = tool.description + json.dumps(tool.inputSchema)
        assert "openrouter/anthropic" not in blob
        assert "openrouter/..." in blob or "Do not prefix" in blob or "do NOT prefix" in blob

    def test_default_timeout_is_30(self):
        """Defect 2: 15s was too tight for resume + slash.exec on a session
        that just had a turn interrupted."""
        tool = next(t for t in server.TOOLS if t.name == "set_model")
        assert "default 30" in tool.inputSchema["properties"]["timeout"]["description"]


# ---------------------------------------------------------------------------
# Defect 1: dead duplicate methods must be gone
# ---------------------------------------------------------------------------

class TestNoDuplicateGuiBackendMethods:
    """Python keeps only the last definition, so a second set_model /
    _interrupt_and_switch silently shadows the first — someone edits the
    dead one and nothing changes. Each name must appear exactly once."""

    def _method_def_lines(self, name: str) -> list[int]:
        import inspect
        src = inspect.getsource(GuiBackend)
        needle = f"async def {name}("
        return [
            i for i, line in enumerate(src.splitlines(), 1)
            if line.lstrip().startswith(needle)
        ]

    def test_set_model_defined_once(self):
        lines = self._method_def_lines("set_model")
        assert len(lines) == 1, f"set_model defined {len(lines)} times at {lines}"

    def test_interrupt_and_switch_defined_once(self):
        lines = self._method_def_lines("_interrupt_and_switch")
        assert len(lines) == 1, (
            f"_interrupt_and_switch defined {len(lines)} times at {lines}"
        )

    def test_set_model_helper_not_shadowed(self):
        """_set_model_once is the retry unit; it must also be unique."""
        lines = self._method_def_lines("_set_model_once")
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# Defect 2: set_model timeout → structured retryable + auto-retry
# ---------------------------------------------------------------------------

def _make_ws_backend():
    transport = MockTransport({
        "/health": (200, {"status": "ok"}),
        "/api/sessions": (200, {"sessions": []}),
    })
    backend = GuiBackend("127.0.0.1", 9999)
    backend._client = httpx.AsyncClient(transport=transport, timeout=5)
    return backend


class NeverRepliesWebSocket:
    """WS whose recv() raises TimeoutError immediately (frames exhausted).

    Simulates a session that never answers resume/slash.exec without making
    the test wait for a real wall-clock timeout.
    """

    def __init__(self, frames=None):
        self._frames = iter(frames or [])
        self._sent: list[str] = []

    async def send(self, data):
        self._sent.append(data)

    async def recv(self):
        try:
            return json.dumps(next(self._frames))
        except StopIteration:
            raise asyncio.TimeoutError()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


@pytest.mark.asyncio
class TestSetModelTimeout:
    """Defect 2: a TimeoutError must become a structured retryable error,
    and set_model must auto-retry once with a doubled timeout."""

    async def test_double_timeout_raises_timeout_error(self):
        """Both attempts time out → asyncio.TimeoutError propagates so
        h_set_model can map it to model_switch_timeout."""
        backend = _make_ws_backend()
        connect_count = {"n": 0}

        def make_ws(url, **kwargs):
            connect_count["n"] += 1
            return NeverRepliesWebSocket()

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = make_ws
            with pytest.raises(TimeoutError):
                await backend.set_model("durable_123", "some/model", timeout=0.01)

        # First attempt + one retry = two WS connections
        assert connect_count["n"] == 2

    async def test_auto_retry_succeeds_on_second_attempt(self):
        """First attempt times out; the doubled-timeout retry succeeds."""
        backend = _make_ws_backend()
        connect_count = {"n": 0}
        success_frames = [
            {"id": 1, "result": {"session_id": "live_ok"}},
            {"id": 2, "result": {"output": "Switched to some/model"}},
        ]

        def make_ws(url, **kwargs):
            connect_count["n"] += 1
            if connect_count["n"] == 1:
                return NeverRepliesWebSocket()  # first attempt times out
            return FakeWebSocketSendPrompt(success_frames)

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = make_ws
            result = await backend.set_model("durable_123", "some/model", timeout=0.01)

        assert result["status"] == "ok"
        assert result["model"] == "some/model"
        assert connect_count["n"] == 2

    async def test_h_set_model_maps_timeout_to_structured_error(self):
        """Handler-level contract: TimeoutError is NOT internal_error."""
        import hermes_chat_delegator.server as mod

        backend = _make_ws_backend()

        def make_ws(url, **kwargs):
            return NeverRepliesWebSocket()

        with patch.object(mod, "_require_backend", new_callable=AsyncMock) as mock_rb:
            mock_rb.return_value = backend
            with patch("hermes_chat_delegator.server.websockets") as mock_ws:
                mock_ws.connect = make_ws
                result = await mod.h_set_model({
                    "session_id": "durable_123",
                    "model": "some/model",
                    "timeout": 0.01,
                })

        data = json.loads(result[0].text)
        assert data["error"]["code"] == "model_switch_timeout"
        assert data["error"]["retryable"] is True
        # Message must be actionable, not "TimeoutError (no message)"
        msg = data["error"]["message"]
        assert "session.resume" in msg or "/model" in msg
        assert "larger" in msg.lower() or "retry" in msg.lower()

    async def test_h_set_model_does_not_return_internal_error_on_timeout(self):
        """Regression: the old path produced
        {"error":{"code":"internal_error","message":"Internal error in
        set_model: TimeoutError (no message)","retryable":false}}."""
        import hermes_chat_delegator.server as mod

        class AlwaysTimesOut:
            async def set_model(self, session_id, model, timeout=30):
                raise asyncio.TimeoutError()

        with patch.object(mod, "_require_backend", new_callable=AsyncMock) as mock_rb:
            mock_rb.return_value = AlwaysTimesOut()
            result = await mod.h_set_model({
                "session_id": "s", "model": "m", "timeout": 5,
            })

        data = json.loads(result[0].text)
        assert data["error"]["code"] != "internal_error"
        assert data["error"]["code"] == "model_switch_timeout"
        assert data["error"]["retryable"] is True


@pytest.mark.asyncio
class TestSendPromptStreamSetupTimeout:
    """Defect 3: a resume timeout during setup must yield structured
    pending (__done__), not raise into internal_error."""

    async def test_resume_timeout_yields_structured_pending(self):
        backend = _make_ws_backend()

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kw: NeverRepliesWebSocket()
            events = []
            final = None
            async for etype, payload in backend.send_prompt_stream(
                "s", "hi", timeout=5
            ):
                if etype == "__done__":
                    final = payload
                    break
                events.append(etype)

        assert final is not None, "setup timeout must yield __done__, not raise"
        assert final["timed_out"] is True
        assert final["status"] == "pending"
        assert final["response"] is None
        assert final["session_id"] == "s"
        assert "get_messages" in final.get("note", "")
        # Setup timeout means the turn may not have started — no mid-turn
        # events should have been emitted.
        assert events == []

    async def test_setup_timeout_does_not_remove_draft(self):
        """A setup-timeout happens BEFORE prompt.submit, so the turn may
        not have started. Deleting the draft mapping would strand the
        caller — the resume path must not call _draft_sessions.remove."""
        backend = _make_ws_backend()
        remove_calls = []
        backend._draft_sessions.remove = AsyncMock(
            side_effect=lambda sid: remove_calls.append(sid)
        )

        with patch("hermes_chat_delegator.server.websockets") as mock_ws:
            mock_ws.connect = lambda url, **kw: NeverRepliesWebSocket()
            final = None
            async for etype, payload in backend.send_prompt_stream(
                "durable_x", "hi", timeout=5
            ):
                if etype == "__done__":
                    final = payload
                    break

        assert final is not None
        assert final["status"] == "pending"
        assert remove_calls == [], (
            "setup-timeout must not delete the draft mapping — the turn "
            "may not have started"
        )

    async def test_h_send_prompt_never_lets_timeout_escape_as_internal_error(self):
        """Even if the backend method raises TimeoutError directly, the
        handler must convert it to structured pending."""
        import hermes_chat_delegator.server as mod

        class AlwaysTimesOut:
            async def send_prompt(self, sid, msg, timeout=300):
                raise asyncio.TimeoutError()

        with patch.object(mod, "_require_backend_for_session",
                          new_callable=AsyncMock) as mock_rb:
            mock_rb.return_value = AlwaysTimesOut()
            result = await mod.h_send_prompt({
                "session_id": "s", "message": "hi", "timeout": 5,
            })

        data = json.loads(result[0].text)
        assert "error" not in data
        assert data["timed_out"] is True
        assert data["status"] == "pending"

    async def test_send_prompt_tool_documents_timeout_as_pending(self):
        tool = next(t for t in server.TOOLS if t.name == "send_prompt")
        blob = tool.description
        assert "timed_out" in blob
        assert "pending" in blob
        assert "send_prompt_stream" in blob