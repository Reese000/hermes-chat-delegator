"""Unit tests for hermes_chat_delegator.server.

Tests GuiBackend against httpx.MockTransport and fake websockets,
plus pure-function tests for _safe_epoch, _summarize_event, etc.
No live GUI required.
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from hermes_chat_delegator.server import (
    GuiBackend,
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

        ws_frames = [
            {"id": 1, "result": {"session_id": "live_abc"}},
            {"id": 2, "result": {"status": "streaming"}},
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

        ws_frames = [
            {"id": 1, "result": {"session_id": "live_new"}},
            {"id": 2, "result": {"status": "streaming"}},
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
