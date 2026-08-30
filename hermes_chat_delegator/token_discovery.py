"""Discover the session token the *live* Hermes backend is actually using.

Why this lives in the package rather than in the launcher shim
-------------------------------------------------------------
This logic started life inside ``launch_delegator.py``, where it ran exactly
once, at import, and pinned its result into ``os.environ``.  That is correct
only until the backend restarts.

``hermes_cli.web_server`` mints ``_SESSION_TOKEN`` with
``secrets.token_urlsafe(32)`` on every start, and the desktop shell launches
the backend with ``--port 0``, so BOTH the port and the token are per-boot
values.  After a mid-session backend restart the delegator process keeps the
old token forever, and the failure is silent in the worst possible way:

  * ``/api/sessions`` on a headless ``hermes serve`` backend is
    unauthenticated, so it answers 200 for *any* token — including a stale
    one.  ``auto_discover_backend`` therefore "confirms" the backend and
    ``backend_status`` reports ``connected: true``.
  * ``/api/ws`` *does* validate the token, and rejects a stale one with
    ``HTTP 403``.

Net effect: list_chats/get_chat/get_messages keep working while create_chat,
send_prompt, fork_chat and set_model all fail with an opaque
``server rejected WebSocket connection: HTTP 403``, and nothing in the
process can recover — restarting Hermes does not help, because the stale
value lives in the delegator, not the backend.

Exposing discovery as a callable here lets the WS layer re-run it on a 403
and self-heal, and gives the launcher a single source of truth to import.

The token is never written to disk: the desktop shell mints it in memory and
passes it to the backend as an environment variable, so the only place it
exists is that process's environment block.  We read it back from there.
"""
import datetime
import os
import subprocess
import sys
from typing import Optional

_SERVE_MARKER = "hermes_cli.main"
_SERVE_ARG = "serve"
TOKEN_VAR = "HERMES_DASHBOARD_SESSION_TOKEN"

_DIAG_LOG = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "hermes-mcp-bridge",
    "token_discovery.log",
)


def _diag(line: str) -> None:
    """Append one diagnostic line, best-effort.

    Every failure branch below was previously silent (bare
    ``except: continue``), which is exactly why token staleness took so long
    to localize. Leaving evidence is cheap; guessing is not.
    """
    try:
        os.makedirs(os.path.dirname(_DIAG_LOG), exist_ok=True)
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        with open(_DIAG_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] pid={os.getpid()} {line}\n")
    except OSError:
        pass


# Reads one variable out of a live same-user process via
# PEB -> RTL_USER_PROCESS_PARAMETERS -> Environment (x64 offsets).  Read-only;
# no writes to the target and no privilege escalation.
_READ_ENV_PS = r"""
$ErrorActionPreference = 'Stop'
$sig = @"
using System;
using System.Runtime.InteropServices;
public static class PebReader {
  [StructLayout(LayoutKind.Sequential)]
  public struct PBI { public IntPtr R1; public IntPtr Peb; public IntPtr R2a;
    public IntPtr R2b; public IntPtr Pid; public IntPtr R3; }
  [DllImport("ntdll.dll")] public static extern int NtQueryInformationProcess(
      IntPtr h, int cls, ref PBI pbi, int len, out int ret);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern IntPtr OpenProcess(
      int access, bool inherit, int pid);
  [DllImport("kernel32.dll", SetLastError=true)] public static extern bool ReadProcessMemory(
      IntPtr h, IntPtr addr, byte[] buf, int size, out IntPtr read);
  [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr h);
  static IntPtr ReadPtr(IntPtr h, IntPtr addr) {
    byte[] b = new byte[IntPtr.Size]; IntPtr r;
    if (!ReadProcessMemory(h, addr, b, b.Length, out r)) throw new Exception("rpm ptr");
    return IntPtr.Size == 8 ? (IntPtr)BitConverter.ToInt64(b,0) : (IntPtr)BitConverter.ToInt32(b,0);
  }
  public static string Env(int pid) {
    IntPtr h = OpenProcess(0x0400 | 0x0010, false, pid);
    if (h == IntPtr.Zero) throw new Exception("openprocess");
    try {
      var pbi = new PBI(); int ret;
      if (NtQueryInformationProcess(h, 0, ref pbi, Marshal.SizeOf(pbi), out ret) != 0)
        throw new Exception("ntqip");
      IntPtr p = ReadPtr(h, (IntPtr)(pbi.Peb.ToInt64() + 0x20));
      IntPtr envPtr = ReadPtr(h, (IntPtr)(p.ToInt64() + 0x80));
      byte[] szb = new byte[8]; IntPtr rd;
      if (!ReadProcessMemory(h, (IntPtr)(p.ToInt64() + 0x3F0), szb, 8, out rd))
        throw new Exception("rpm size");
      long size = BitConverter.ToInt64(szb, 0);
      if (size <= 0 || size > 4194304) throw new Exception("bad size");
      byte[] buf = new byte[size];
      if (!ReadProcessMemory(h, envPtr, buf, buf.Length, out rd)) throw new Exception("rpm env");
      return System.Text.Encoding.Unicode.GetString(buf);
    } finally { CloseHandle(h); }
  }
}
"@
Add-Type -TypeDefinition $sig -Language CSharp | Out-Null
$proc = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
  Where-Object { $_.CommandLine -like '*__MARKER__*' -and $_.CommandLine -like '*__SERVE__*' }
# The desktop shell spawns a thin wrapper which re-execs the real backend as
# its own child.  BOTH match the marker above, and the wrapper inherits the
# shell's env pin -- which is the stale token this module exists to avoid.
# The child mints its own token in memory and is the only one that binds a
# socket, so rank actual listeners first; without this the loop below breaks
# on the wrapper's stale token and every authenticated probe fails.
$listening = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Select-Object -ExpandProperty OwningProcess -Unique)
$proc = @($proc | Sort-Object `
  @{ Expression = { if ($listening -contains [int]$_.ProcessId) { 0 } else { 1 } } }, `
  ProcessId)
foreach ($p in $proc) {
  try {
    $block = [PebReader]::Env([int]$p.ProcessId)
    $hit = ($block -split "`0") | Where-Object { $_ -like '__VAR__=*' } | Select-Object -First 1
    if ($hit) { $hit.Substring('__VAR__'.Length + 1); break }
  } catch { }
}
"""


def _powershell_env() -> dict:
    """Environment for the discovery subprocess, with a usable TEMP.

    Windows PowerShell 5.1's ``Add-Type`` compiles C# out-of-process: it
    writes a ``.cs`` file into ``%TEMP%`` and shells out to ``csc.exe``.
    When this runs under a host whose ``TEMP`` points at the *system* temp
    (observed: Claude Code exports ``TEMP=C:\\WINDOWS\\TEMP``), that compile
    fails and discovery silently returns "" — which looks exactly like a dead
    backend even though Hermes is healthy the whole time.
    """
    env = dict(os.environ)
    candidates = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Temp"),
        os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "Temp"),
    ]
    for cand in candidates:
        if cand and os.path.isdir(cand) and os.access(cand, os.W_OK):
            env["TEMP"] = cand
            env["TMP"] = cand
            break
    return env


def discover_session_token() -> str:
    """Return the token the live backend is using, or "" on any failure.

    A missing token is not fatal: some backends (headless ``hermes serve``)
    accept unauthenticated WebSocket connections outright, and an empty token
    is strictly better than a stale one — a stale token is actively rejected
    with 403, whereas no token at all is simply not checked.

    Tries ``pwsh`` (PowerShell 7+) before ``powershell`` (5.1); pwsh hosts
    Roslyn in-process and never touches ``%TEMP%`` for ``Add-Type``.
    """
    if sys.platform != "win32":
        return ""
    import shutil

    script = (
        _READ_ENV_PS.replace("__MARKER__", _SERVE_MARKER)
        .replace("__SERVE__", _SERVE_ARG)
        .replace("__VAR__", TOKEN_VAR)
    )
    env = _powershell_env()
    for exe in ("pwsh", "powershell"):
        resolved = shutil.which(exe, path=env.get("PATH"))
        if resolved is None:
            continue
        try:
            out = subprocess.run(
                [exe, "-NoProfile", "-NonInteractive", "-Command", script],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=30, env=env,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _diag(f"discover_session_token: {exe!r} raised {type(exc).__name__}: {exc}")
            continue
        token = (out.stdout or "").strip()
        _diag(
            f"discover_session_token: {exe!r} rc={out.returncode} "
            f"token={'<empty>' if not token else f'<{len(token)} chars>'}"
        )
        if token:
            return token
    _diag("discover_session_token: exhausted pwsh and powershell, returning empty")
    return ""


def refresh_session_token() -> Optional[str]:
    """Re-discover the live token and publish it to ``os.environ``.

    Returns the new token, "" if discovery ran but found nothing, or None if
    discovery could not run at all. Callers use the return value to decide
    whether a retry is worth attempting.
    """
    token = discover_session_token()
    if token:
        os.environ[TOKEN_VAR] = token
    return token
