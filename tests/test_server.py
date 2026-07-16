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

    def __init__(self, frames):
        self._frames = iter(frames)
        self._sent = []

    async def send(self, data):
        self._sent.append(data)

    async def recv(self):
        return json.dumps(next(self._frames))

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
            mock_ws.connect = lambda url: FakeWebSocketSendPrompt(ws_frames)
            result = await backend.send_prompt("test", "New question", timeout=10)

        # Must return the NEW answer, not the old one
        assert result["response"] == "NEW ANSWER", (
            f"Got stale response: {result['response']!r} — C2 regression!"
        )


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
            mock_ws.connect = lambda url: FakeWebSocketSendPrompt(ws_frames)
            with pytest.raises(RuntimeError, match="empty"):
                await backend.branch_session("empty_sess")
