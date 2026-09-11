"""Shared pytest fixtures.

``GuiBackend`` and the module-level ``_require_backend`` helper both derive
persistence paths from ``Path.home() / ".hermes" / ...`` at call time
(``PersistentDraftSessions`` for draft session records, the backend-port
cache for fast reconnects). Every test in this suite constructs a real
``GuiBackend`` directly (see ``GuiBackend("127.0.0.1", 9999, ...)`` calls
throughout test_server.py) rather than mocking it, so without isolation
every test run silently reads and overwrites the *real* user's
``~/.hermes/mcp_draft_sessions.json`` and ``~/.hermes/mcp_backend_port.json``
— up to and including clobbering a live, correct backend-port cache with a
test's mock port (observed: a test using port 99999 as a stand-in value
left that port cached as the real one, adding a doomed probe + token
refresh to every subsequent real launch).  Redirecting ``Path.home()`` to
a per-test tmp directory keeps the whole suite from touching real user state.
"""
import pathlib
import os
import sys
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: tmp_path))


@pytest.fixture(autouse=True)
def _block_real_hermes_cli():
    """Guard: prevent any test from spawning the real hermes.exe CLI.

    Intercepts asyncio.create_subprocess_exec so that any unmocked call
    raises immediately instead of hitting the real CLI and creating cards
    on the live kanban board. Tests that legitimately need the CLI path
    must patch asyncio.create_subprocess_exec themselves.
    """
    async def _blocked(*args, **kwargs):
        raise RuntimeError(
            "TEST SAFETY: asyncio.create_subprocess_exec was called without "
            f"being mocked. Args: {args}. This would have spawned a real "
            "process. Patch asyncio.create_subprocess_exec in your test."
        )

    with patch("asyncio.create_subprocess_exec", side_effect=_blocked):
        yield
