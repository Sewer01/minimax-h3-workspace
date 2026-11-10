"""Pictures on the Windows clipboard.

The web view is not allowed to read or write the picture clipboard itself (Chromium asks the
host for a permission the window never grants), so the program does it: a quick format check
with ctypes, and the conversion to and from PNG through a short-lived .NET helper.

data_dir is set by the server: the helper script and the temporary PNG live next to the
program's other files, never in the Windows folder.
"""

import ctypes
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import engine

data_dir = Path(".")     # set by server at start-up

CLIP_HELPER = r"""param([string]$Out, [string]$In, [string]$From)
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

# put a PNG file on the clipboard: SetImage publishes CF_BITMAP/CF_DIB, which every program understands
if ($In) {
    $img = [System.Drawing.Image]::FromFile($In)
    try {
        $ok = $false
        for ($i = 0; $i -lt 25 -and -not $ok; $i++) {
            try { [System.Windows.Forms.Clipboard]::SetImage($img); $ok = $true } catch { Start-Sleep -Milliseconds 80 }
        }
        if (-not $ok) { exit 5 }
    } finally { $img.Dispose() }
    exit 0
}

# a picture file of a format the web view cannot show (TIFF, HEIC…): Windows decodes it, the PNG goes back
if ($From) {
    $img = [System.Drawing.Image]::FromFile($From)
    try { $img.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png) } finally { $img.Dispose() }
    exit 0
}

# take the picture off the clipboard and save it as PNG
$img = $null
for ($i = 0; $i -lt 25 -and $null -eq $img; $i++) {
    try { if ([System.Windows.Forms.Clipboard]::ContainsImage()) { $img = [System.Windows.Forms.Clipboard]::GetImage() } } catch { }
    if ($null -eq $img) { Start-Sleep -Milliseconds 80 }
}
if ($null -eq $img) { exit 3 }
try { $img.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png) } finally { $img.Dispose() }
"""

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CF_HDROP = 15

# a picture file copied in Explorer: the web view shows these itself, so the bytes go through untouched
RAW_IMAGE_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
                  ".bmp": "image/bmp", ".gif": "image/gif", ".avif": "image/avif"}
# ...and these Windows can decode but the page cannot show, so they are converted to PNG first
CONVERT_IMAGE_EXT = {".tif", ".tiff", ".heic", ".heif", ".jxl", ".dib", ".jp2", ".j2k", ".ico", ".wdp", ".hdp"}
IMAGE_EXT = set(RAW_IMAGE_MIME) | CONVERT_IMAGE_EXT


def _png_format():
    """The clipboard format literally called "PNG", if some program has registered it."""
    try:
        u = ctypes.windll.user32
        u.RegisterClipboardFormatW.restype = ctypes.c_uint
        return u.RegisterClipboardFormatW("PNG") or 0
    except Exception:
        return 0


def _clipboard_files():
    """Paths of the files copied in Explorer; no file contents are read here."""
    try:
        u, shell = ctypes.windll.user32, ctypes.windll.shell32
        u.GetClipboardData.restype = ctypes.c_void_p
        shell.DragQueryFileW.restype = ctypes.c_uint
        shell.DragQueryFileW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_uint]
        if not u.IsClipboardFormatAvailable(CF_HDROP) or not u.OpenClipboard(0):
            return []
        try:
            drop = u.GetClipboardData(CF_HDROP)
            if not drop:
                return []
            paths = []
            for i in range(min(shell.DragQueryFileW(drop, 0xFFFFFFFF, None, 0), 16)):
                length = shell.DragQueryFileW(drop, i, None, 0)
                buf = ctypes.create_unicode_buffer(length + 1)
                if shell.DragQueryFileW(drop, i, buf, length + 1):
                    paths.append(buf.value)
            return paths
        finally:
            u.CloseClipboard()
    except Exception:
        return []


def has_picture():
    """Is a picture on the clipboard? Cheap: asks Windows about formats and file names only."""
    try:
        u = ctypes.windll.user32
        formats = [2, 8, 17]  # CF_BITMAP, CF_DIB, CF_DIBV5
        png = _png_format()
        if png:
            formats.append(png)
        if any(u.IsClipboardFormatAvailable(f) for f in formats):
            return True
    except Exception:
        return False
    return any(Path(p).suffix.lower() in IMAGE_EXT for p in _clipboard_files())


def _ready_png():
    """PNG bytes already on the clipboard (some programs put a finished PNG there), else None."""
    fmt = _png_format()
    if not fmt:
        return None
    try:
        u, k = ctypes.windll.user32, ctypes.windll.kernel32
        u.GetClipboardData.restype = ctypes.c_void_p
        k.GlobalLock.restype = ctypes.c_void_p
        k.GlobalLock.argtypes = [ctypes.c_void_p]
        k.GlobalUnlock.argtypes = [ctypes.c_void_p]
        k.GlobalSize.restype = ctypes.c_size_t
        k.GlobalSize.argtypes = [ctypes.c_void_p]
        if not u.IsClipboardFormatAvailable(fmt) or not u.OpenClipboard(0):
            return None
        try:
            handle = u.GetClipboardData(fmt)
            if not handle:
                return None
            pointer = k.GlobalLock(handle)
            if not pointer:
                return None
            try:
                size = k.GlobalSize(handle)
                data = ctypes.string_at(pointer, size) if size > len(PNG_SIGNATURE) else b""
            finally:
                k.GlobalUnlock(handle)
        finally:
            u.CloseClipboard()
    except Exception:
        return None
    cut = data.find(PNG_SIGNATURE)  # a raw PNG starts with it; a .NET DataObject puts a small header before
    return data[cut:] if cut >= 0 else None


def _helper():
    """The .NET helper script, written once next to the app's other files."""
    path = data_dir / "tools" / "clipboard.ps1"
    try:
        current = path.read_text(encoding="utf-8-sig") if path.exists() else ""
    except OSError:
        current = ""
    if current != CLIP_HELPER:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(CLIP_HELPER, encoding="utf-8-sig")
    return path


def _run(*args):
    exe = shutil.which("powershell") or str(Path(os.environ.get("SystemRoot", r"C:\Windows")) /
                                           "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe")
    return subprocess.run([exe, "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-File", str(_helper()), *args],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180,
                          creationflags=engine.CREATE_NO_WINDOW)


def picture():
    """(bytes, mime) of the picture on the clipboard, or None when there is no picture."""
    ready = _ready_png()   # a finished PNG another program put there: the bytes go through as they are
    if ready:
        return ready, "image/png"
    dropped = [Path(p) for p in _clipboard_files()]
    for file in dropped:   # a picture file copied in Explorer
        mime = RAW_IMAGE_MIME.get(file.suffix.lower())
        if mime:
            try:
                return file.read_bytes(), mime
            except OSError:
                continue
    out = data_dir / "tmp" / f"clip-{uuid.uuid4().hex}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for file in dropped:  # TIFF, HEIC…: Windows decodes them, the page gets a PNG
            if file.suffix.lower() in CONVERT_IMAGE_EXT and file.exists():
                _run("-Out", str(out), "-From", str(file))
                break
        else:  # a bare bitmap on the clipboard (a screenshot, a copy inside a viewer)
            _run("-Out", str(out))
        return (out.read_bytes(), "image/png") if out.exists() and out.stat().st_size > len(PNG_SIGNATURE) else None
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        out.unlink(missing_ok=True)


def set_picture(data):
    """Copy PNG bytes to the clipboard so any other program can paste them."""
    if not data.startswith(PNG_SIGNATURE):
        return False
    path = data_dir / "tmp" / f"clip-{uuid.uuid4().hex}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    try:
        return _run("-In", str(path)).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        path.unlink(missing_ok=True)
