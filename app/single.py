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
import socket
from pathlib import Path

ERROR_ALREADY_EXISTS = 183
LOCK_FILE = "app.lock"
_handle = None  # the mutex has to stay open while the program runs


def _name(data_dir):
    """A mutex name per data folder (the same folder always gives the same name)."""
    key = hashlib.sha1(str(Path(data_dir)).lower().encode("utf-8")).hexdigest()[:16]
    return f"Local\\MinimaxH3Workspace-{key}"


def running(data_dir):
    """Address of the copy that already works with this folder, or "" when it is unknown or does not answer.

    The file may be left from a copy that was killed and never cleaned up, so the port is checked by a
    connection attempt rather than trusted.
    """
    try:
        port = int(json.loads((Path(data_dir) / LOCK_FILE).read_text(encoding="utf-8"))["port"])
    except (OSError, ValueError, KeyError):
        return ""
    if not 0 < port < 65536:
        return ""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return f"http://127.0.0.1:{port}"
    except OSError:
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
    h = k.CreateMutexW(None, 0, _name(data_dir))
    if not h or k.GetLastError() == ERROR_ALREADY_EXISTS:
        if h:
            k.CloseHandle(h)
        return running(data_dir)
    _handle = h
    if port:
        try:
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            (Path(data_dir) / LOCK_FILE).write_text(json.dumps({"port": int(port)}), encoding="utf-8")
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
