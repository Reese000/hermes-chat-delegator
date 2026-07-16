#!/usr/bin/env python3
"""Smoke test for hermes-chat-delegator.

Exercises every tool against a live Hermes backend. Cleans up after itself.
Requires: HERMES_DASHBOARD_SESSION_TOKEN env var (or backend running with
localhost bypass still active).

Usage:
    python scripts/smoke_test.py [--port PORT] [--host HOST]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_chat_delegator.server import GuiBackend


async def run_smoke_test(host: str, port: int):
    token = os.environ.get("HERMES_DASHBOARD_SESSION_TOKEN", "")
    b = GuiBackend(host, port, token=token)
    created_ids: list[str] = []

    print("=" * 60)
    print("HERMES CHAT DELEGATOR — SMOKE TEST")
    print("=" * 60)

    # 1. Backend status
    print("\n[1/10] backend_status...")
    try:
        health = await b.health()
        print(f"  ✓ Connected: {health}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        await b.aclose()
        return False

    # 2. List chats
    print("\n[2/10] list_chats...")
    try:
        sessions = await b.list_sessions(limit=5)
        print(f"  ✓ Found {len(sessions)} sessions")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")

    # 3. Get chat (use first session if available)
    target_session = None
    if sessions:
        target_session = sessions[0]["id"]
        print(f"\n[3/10] get_chat({target_session})...")
        try:
            info = await b.get_session(target_session)
            print(f"  ✓ Title: {info.get('title', 'untitled')}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
    else:
        print("\n[3/10] get_chat — SKIPPED (no sessions)")

    # 4. Get messages
    if target_session:
        print(f"\n[4/10] get_messages({target_session})...")
        try:
            msgs = await b.get_messages(target_session, last_n=3)
            print(f"  ✓ Got {len(msgs)} messages")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
    else:
        print("\n[4/10] get_messages — SKIPPED")

    # 5. Create chat (draft)
    print("\n[5/10] create_chat (draft)...")
    try:
        created = await b.create_session(title="smoke-test-draft")
        draft_id = created["session_id"]
        created_ids.append(draft_id)
        print(f"  ✓ Created draft: {draft_id}")
        print(f"  ⚠ Note: draft is invisible to list_chats until prompted")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        draft_id = None

    # 6. Create chat with initial_message
    print("\n[6/10] create_chat (with initial_message)...")
    try:
        created2 = await b.create_session(
            title="smoke-test-active",
        )
        active_id = created2["session_id"]
        created_ids.append(active_id)
        print(f"  ✓ Created session: {active_id}")
        # Send initial message
        result = await b.send_prompt(active_id, "Reply with just: smoke test OK", timeout=60)
        response = result.get("response", "")
        print(f"  ✓ Got response: {response[:100]}...")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        active_id = None

    # 7. Rename chat
    if active_id:
        print(f"\n[7/10] rename_chat({active_id})...")
        try:
            await b.rename_session(active_id, "smoke-test-renamed")
            print(f"  ✓ Renamed to 'smoke-test-renamed'")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    # 8. Fork chat
    if active_id:
        print(f"\n[8/10] fork_chat({active_id})...")
        try:
            forked = await b.branch_session(active_id, title="smoke-test-fork")
            fork_id = forked["session_id"]
            created_ids.append(fork_id)
            print(f"  ✓ Forked to: {fork_id}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    # 9. send_prompt_stream (on the active session)
    if active_id:
        print(f"\n[9/10] send_prompt_stream({active_id})...")
        try:
            events = []
            final = None
            async for etype, payload in b.send_prompt_stream(
                active_id, "Reply with just: stream OK", timeout=60
            ):
                if etype == "__done__":
                    final = payload
                    break
                events.append(etype)
            print(f"  ✓ Got {len(events)} events, final: {(final or {}).get('response', '')[:80]}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")

    # 10. Cleanup: delete all created sessions
    print(f"\n[10/10] Cleanup: deleting {len(created_ids)} created sessions...")
    for sid in created_ids:
        try:
            await b.delete_session(sid)
            print(f"  ✓ Deleted {sid}")
        except Exception as e:
            print(f"  ✗ Failed to delete {sid}: {e}")

    await b.aclose()

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Smoke test for hermes-chat-delegator")
    parser.add_argument("--host", default=os.environ.get("HERMES_GUI_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HERMES_GUI_PORT", "0")))
    args = parser.parse_args()

    if args.port == 0:
        print("ERROR: Set HERMES_GUI_PORT or pass --port")
        sys.exit(1)

    success = asyncio.run(run_smoke_test(args.host, args.port))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
