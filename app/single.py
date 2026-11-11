"""One copy of the program per data folder.

Every copy picks a free port by itself, so nothing stopped a second window from starting over the same
settings, queue and library. Both copies then write the same files: the queue was saved twice, the first-run
flag was restored by whichever copy remembered the older state, and each tried to start its own ComfyUI on
the same folder and database.

The lock is per data folder, so a second copy of the program installed on another disk still starts.
"""

import ctypes
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ERROR_ALREADY_EXISTS = 183
LOCK_FILE = "app.lock"
STILL_ACTIVE = 259
PROCESS_QUERY = 0x1000
_zombie_after = 45     # seconds: a copy that holds the folder this long without answering any request
_handle = None  # the mutex has to stay open while the program runs


def _name(data_dir):
    """A mutex name per data folder (the same folder always gives the same name)."""
    key = hashlib.sha1(str(Path(data_dir)).lower().encode("utf-8")).hexdigest()[:16]
    return f"Local\\MinimaxH3Workspace-{key}"


def _read(data_dir):
    try:
        d = json.loads((Path(data_dir) / LOCK_FILE).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def port_of(data_dir):
    port = _read(data_dir).get("port")
    return int(port) if isinstance(port, int) and 0 < port < 65536 else 0


def running(data_dir):
    """Address of the copy that already works with this folder, or "" when it is unknown or does not answer.

    The file may be left from a copy that was killed and never cleaned up, so the port is checked by a
    connection attempt rather than trusted.
    """
    port = port_of(data_dir)
    if not port:
        return ""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return f"http://127.0.0.1:{port}"
    except OSError:
        return ""


def alive(pid):
    """Is that process still running? (Our own pid counts as alive.)"""
    if not pid or pid == os.getpid():
        return bool(pid)
    if not hasattr(ctypes, "windll"):
        return False
    k = ctypes.windll.kernel32
    h = k.OpenProcess(PROCESS_QUERY, 0, int(pid))
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(k.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == STILL_ACTIVE
    finally:
        k.CloseHandle(h)


def has_window(pid):
    """Does that process own a window a person can see?

    This is the difference between a second copy (which has an open window somewhere) and the case we
    have to deal with: the window was closed, but the process stayed behind - no window, no answer on
    its port, and yet the folder and the ComfyUI core are still held by it.
    """
    if not pid or not hasattr(ctypes, "windll"):
        return False
    u = ctypes.windll.user32
    u.GetWindowTextLengthW.restype = ctypes.c_int
    found = []
    ProcEnum = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)

    def cb(hwnd, _lparam):
        # a window a person can actually see: visible and with a caption. The zombie left behind by
        # closing the window has none of those, and that is exactly what tells the two cases apart.
        wp = ctypes.c_ulong()
        if u.IsWindowVisible(hwnd) and u.GetWindowThreadProcessId(hwnd, ctypes.byref(wp)) and wp.value == int(pid):
            if u.GetWindowTextLengthW(hwnd) > 0:
                found.append(True)
        return 1

    u.EnumWindows(ProcEnum(cb), 0)
    return bool(found)


def sweep_zombie(data_dir):
    """Close a copy that holds this folder without a window, so a new start is possible.

    Only our own executable is touched, only when it answers nothing on its port, only after it has held
    the folder for a while - a copy that is still starting answers late by nature, and killing it would
    be worse than the dialog. Returns the pid that was closed.
    """
    d = _read(data_dir)
    pid = int(d.get("pid") or 0)
    if not pid or pid == os.getpid() or not alive(pid) or has_window(pid):
        return 0
    if running(data_dir):
        return 0                                    # it answers: somebody's window is just hidden
    try:
        age = time.time() - (Path(data_dir) / LOCK_FILE).stat().st_mtime
    except OSError:
        return 0
    if age < _zombie_after:
        return 0
    me = Path(sys.argv[0] if not getattr(sys, "frozen", False) else sys.executable).name.lower()
    exe = _process_exe(pid)
    if exe and Path(exe).name.lower() != me:
        return 0                                    # the pid belongs to something else entirely
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=25,
                       creationflags=0x08000000)   # CREATE_NO_WINDOW: whole tree, its ComfyUI included
        time.sleep(1.2)
        return pid
    except (OSError, subprocess.SubprocessError):
        return 0


def _process_exe(pid):
    ps = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').ExecutablePath"
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                           text=True, timeout=20, creationflags=0x08000000)
        return (r.stdout or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def hold(data_dir, port=None):
    """Claim this data folder for the running copy.

    Returns None when the folder is free (and remembers the port for the next copy), or the address of the
    already running copy (an empty string when the program holds it but its address is not written yet).
    """
    global _handle
    if not hasattr(ctypes, "windll"):  # not Windows: no named mutex here, do not block anything
        return None
    k = ctypes.windll.kernel32
    k.CreateMutexW.restype = ctypes.c_void_p
    k.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]

    def claim():
        h = k.CreateMutexW(None, 0, _name(data_dir))
        if not h or k.GetLastError() == ERROR_ALREADY_EXISTS:
            if h:
                k.CloseHandle(h)
            return None
        return h

    h = claim()
    if h is None:
        sweep_zombie(data_dir)         # a copy with no window and no answer must not lock the folder away
        h = claim()
    if h is None:
        return running(data_dir)
    _handle = h
    if port:
        try:
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            (Path(data_dir) / LOCK_FILE).write_text(json.dumps({"port": int(port), "pid": os.getpid()}),
                                                    encoding="utf-8")
        except OSError:
            pass
    return None


def release(data_dir):
    """Give the folder up when the program closes."""
    global _handle
    if _handle and hasattr(ctypes, "windll"):
        ctypes.windll.kernel32.CloseHandle(_handle)
    _handle = None
    try:
        (Path(data_dir) / LOCK_FILE).unlink()
    except OSError:
        pass
