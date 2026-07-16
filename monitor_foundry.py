#!/usr/bin/env python3
"""Monitor a Hermes session and prompt it when idle for 6 hours."""

import asyncio
import json
import time
import sys
import httpx
import websockets

SESSION_ID = "20260709_211850_1438e4"
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = "46914"
DURATION = 6 * 3600  # 6 hours
CHECK_INTERVAL = 30  # seconds
BASE_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
WS_URL = f"ws://{BACKEND_HOST}:{BACKEND_PORT}/api/ws"


async def get_sessions():
    async with httpx.AsyncClient(timeout=5) as c:
        r = await c.get(f"{BASE_URL}/api/sessions", params={"limit": 200, "order": "recent"})
        r.raise_for_status()
        return r.json().get("sessions", [])


async def send_prompt(message: str, timeout: float = 30) -> str:
    """Send a prompt via WebSocket and wait for response."""
    async with httpx.AsyncClient(timeout=5) as c:
        r = await c.get(f"{BASE_URL}/api/sessions/{SESSION_ID}/messages")
        before_count = len(r.json().get("messages", []))

    async with websockets.connect(WS_URL) as ws:
        # Resume
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session.resume", "params": {"session_id": SESSION_ID}}))
        while True:
            resp = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            if resp.get("id") == 1 and "result" in resp:
                internal_sid = resp["result"].get("session_id", SESSION_ID)
                break
            if "error" in resp:
                raise RuntimeError(f"resume failed: {resp['error']}")

        # Submit
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "prompt.submit", "params": {"session_id": internal_sid, "text": message}}))

    # Poll is_active until False
    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(2)
        sessions = await get_sessions()
        foundry = next((s for s in sessions if SESSION_ID in s.get("id", "")), None)
        if foundry and not foundry.get("is_active", False):
            async with httpx.AsyncClient(timeout=5) as c:
                r = await c.get(f"{BASE_URL}/api/sessions/{SESSION_ID}/messages")
                msgs = r.json().get("messages", [])
            last = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
            return last.get("content", "") if last else ""

    return "(timeout)"


async def main():
    start = time.time()
    prompt_count = 0
    last_msg_count = 0

    while time.time() - start < DURATION:
        elapsed = int(time.time() - start)
        try:
            sessions = await get_sessions()
            foundry = next((s for s in sessions if SESSION_ID in s.get("id", "")), None)
            
            if not foundry:
                print(f"[{time.strftime('%H:%M:%S')}] Foundry not found", flush=True)
                await asyncio.sleep(CHECK_INTERVAL)
                continue

            is_active = foundry.get("is_active", False)
            msg_count = foundry.get("message_count", 0)

            if is_active:
                h, m = divmod(elapsed, 3600)
                print(f"[{time.strftime('%H:%M:%S')}] Active, {msg_count} msgs, {h}h{m//60}m elapsed", flush=True)
            else:
                # Idle - send prompt
                prompt_count += 1
                print(f"[{time.strftime('%H:%M:%S')}] Prompt #{prompt_count}", flush=True)
                response = await send_prompt("Continue your current work.")
                print(f"  Response: {response[:150]}", flush=True)

        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] Error: {e}", flush=True)

        await asyncio.sleep(CHECK_INTERVAL)

    print(f"\n=== 6 hours done, {prompt_count} prompts sent ===", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
