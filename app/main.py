"""Minimax H3 Workspace desktop entry point: runs the local server and shows it in its own window.

Falls back to the default browser if the WebView2 window cannot be created.
"""

import json
import asyncio
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

if getattr(sys, "frozen", False):
    # installed app: settings live in the install folder the user chose (see datadir.py), never in the Windows profile
    import datadir
    os.environ.setdefault("H3STUDIO_DATA", str(datadir.locate(Path(sys.executable).parent, os.environ.get("LOCALAPPDATA", "."))))

from aiohttp import web  # noqa: E402

import server  # noqa: E402
import single  # noqa: E402


def free_port(start=8765):
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("no free port")


class ServerThread(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port
        self.ready = threading.Event()
        self.loop = None
        self.runner = None

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.runner = web.AppRunner(server.build_app())
        self.loop.run_until_complete(self.runner.setup())
        self.loop.run_until_complete(web.TCPSite(self.runner, "127.0.0.1", self.port).start())
        self.ready.set()
        self.loop.run_forever()

    def shutdown(self):
        # cleanup also stops the managed ComfyUI process
        fut = asyncio.run_coroutine_threadsafe(self.runner.cleanup(), self.loop)
        try:
            fut.result(timeout=30)
        finally:
            self.loop.call_soon_threadsafe(self.loop.stop)


class JsApi:
    # pywebview exposes every public attribute to JS and walks it recursively,
    # so everything except the methods JS may call stays private (underscore)
    def __init__(self, port):
        self._window = None
        self._port = port

    def restart_app(self):
        """Close the program and open it again: a hidden helper waits until this process is really gone
        (ComfyUI is stopped on the way out) and only then starts it. Needed after the switches that are
         read once, at the start - the window drawn without the video card, for example."""
        import subprocess
        try:
            frozen = getattr(sys, "frozen", False)
            exe = sys.executable if frozen else sys.executable
            args = [] if frozen else [os.path.abspath(sys.argv[0])]
            quote = lambda s: "'" + s.replace("'", "''") + "'"
            workdir = os.path.dirname(exe) if frozen else os.path.dirname(os.path.abspath(sys.argv[0]))
            ps = (f"Wait-Process -Id {os.getpid()} -ErrorAction SilentlyContinue; Start-Sleep -Seconds 2; "
                  f"Start-Process -FilePath {quote(exe)}"
                  + ("".join(f" -ArgumentList {quote(a)}" for a in args))
                  + f" -WorkingDirectory {quote(workdir)}")
            # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP: DETACHED_PROCESS leaves PowerShell without a
            # console and it dies before it can wait
            subprocess.Popen(["powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps],
                             creationflags=0x08000000 | 0x00000200, close_fds=True)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        threading.Timer(0.5, lambda: self._window.destroy()).start()   # let this answer first
        return {"ok": True}

    def pick_folder(self, start=""):
        import webview
        res = self._window.create_file_dialog(webview.FOLDER_DIALOG, directory=start if os.path.isdir(start) else "")
        return res[0] if res else None

    @staticmethod
    def _last_dir():
        """The folder the user saved into last time (Windows "Save as" starts there next time)."""
        try:
            d = (server.DATA_DIR / "last_save_dir.txt").read_text(encoding="utf-8").strip()
            return d if os.path.isdir(d) else ""
        except OSError:
            return ""

    @staticmethod
    def _remember_dir(folder):
        try:
            if folder and os.path.isdir(folder):
                (server.DATA_DIR / "last_save_dir.txt").write_text(folder, encoding="utf-8")
        except OSError:
            pass

    def _copy(self, item_id, dest):
        """Stream a library video from the local server into the file the user chose."""
        import urllib.request
        with urllib.request.urlopen(f"http://127.0.0.1:{self._port}/api/file/{item_id}", timeout=60) as r, \
                open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)

    @staticmethod
    def _free_name(folder, name):
        stem, ext = os.path.splitext(name)
        path, n = os.path.join(folder, name), 1
        while os.path.exists(path):  # never overwrite files the user already has
            path, n = os.path.join(folder, f"{stem} ({n}){ext}"), n + 1
        return path

    def save_video(self, item_id, filename):
        """The usual Windows "Save as" window, then the video is copied there."""
        import webview
        ext = os.path.splitext(filename)[1].lower() or ".mp4"
        kinds = (f"Картинка {ext[1:].upper()} (*{ext})",) if ext != ".mp4" else ("Видео MP4 (*.mp4)",)
        res = self._window.create_file_dialog(webview.SAVE_DIALOG, directory=self._last_dir(), save_filename=filename, file_types=kinds)
        if not res:
            return {"ok": False, "cancelled": True}
        dest = res if isinstance(res, str) else res[0]
        if not dest.lower().endswith(ext):
            dest += ext
        try:
            self._copy(item_id, dest)
        except Exception as e:
            return {"ok": False, "error": str(e)}
        self._remember_dir(os.path.dirname(dest))
        return {"ok": True, "path": dest}

    def save_videos(self, items):
        """Several videos: one folder is chosen, every video is saved into it."""
        import webview
        res = self._window.create_file_dialog(webview.FOLDER_DIALOG, directory=self._last_dir())
        if not res:
            return {"ok": False, "cancelled": True}
        folder = res if isinstance(res, str) else res[0]
        self._remember_dir(folder)
        saved, failed = [], 0
        for it in items:
            try:
                path = self._free_name(folder, it["filename"])
                self._copy(it["id"], path)
                saved.append(path)
            except Exception:
                failed += 1
        return {"ok": bool(saved), "folder": folder, "count": len(saved), "failed": failed}


def window_mode():
    """How the window opens. It used to always open maximized, which cannot be dragged to a second
       monitor or made smaller without a fight; the choice is in Настройки → Интерфейс."""
    try:
        import json
        v = json.loads((server.DATA_DIR / "config.json").read_text(encoding="utf-8")).get("window_mode")
        return v if v in ("geo", "max", "window", "fullscreen") else "geo"
    except Exception:
        return "geo"


def window_geo():
    """The size the window was closed at. The tab strip needs width to keep the names on its buttons,
    and starting at 1440 every time threw away the width you had chosen."""
    try:
        g = json.loads((server.DATA_DIR / "window.json").read_text(encoding="utf-8"))
        w, h = int(g.get("width") or 0), int(g.get("height") or 0)
        if 1000 <= w <= 7680 and 620 <= h <= 4320:
            return w, h
    except Exception:
        pass
    # nothing remembered yet: wide enough that the tab strip keeps every name. The strip needs about
    # 1.5 thousand pixels plus the side column, so 2040 with room to spare, but never over the screen.
    try:
        import ctypes
        screen = ctypes.windll.user32.GetSystemMetrics(0) or 1920
        return int(max(1440, min(2040, screen - 60))), 1040
    except Exception:
        return 2040, 1040


def show_running_window(base):
    """Ask the copy that already works with these files to bring its own window forward.

    The old behaviour opened http://127.0.0.1:port in a browser, which gave the user a web page instead of
    the program - without the desktop window, and with two places to work in without noticing.

    base is empty when the copy that holds the folder does not answer at all: that is the zombie left
    behind on some machines after the window was closed, and single.py has already tried to clear it.
    """
    try:
        import json
        import urllib.request
        with urllib.request.urlopen(base.rstrip("/") + "/api/focus", timeout=5) as r:
            return bool(json.loads(r.read().decode("utf-8", "ignore")).get("ok"))
    except Exception:
        return False


def main():
    import ctypes
    if sys.platform == "win32":
        # no Windows error dialogs (e.g. nvidia-smi.exe 0xc0000142 under memory pressure); child processes inherit this
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)  # SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX
    port = free_port()
    busy = single.hold(server.DATA_DIR, port)  # a second copy over the same folder spoils settings, queue and ComfyUI
    if busy is not None:
        # Two copies never run over one folder - that much stays. But a person should not be sent to
        # the task manager either: if another copy is alive, its window is brought forward, and a
        # windowless ghost is closed by single.py before this line is ever reached. When we still
        # cannot show a window, this copy simply steps aside without a word.
        show_running_window(busy)
        return
    srv = ServerThread(port)
    srv.start()
    srv.ready.wait(30)
    url = f"http://127.0.0.1:{port}"

    def explain_no_webview():
        try:
            ctypes.windll.user32.MessageBoxW(
                0, "Не найден компонент Windows «Microsoft Edge WebView2», он рисует окно программы.\n\n"
                   "Скачайте и установите «WebView2 Runtime» (бесплатно, с сайта Microsoft: microsoft.com/edge/webview2), "
                   "затем запустите программу снова.\n\nПока программа откроется в обычном браузере.",
                "Minimax H3 Workspace", 0x40)
        except Exception:
            pass

    try:
        import webview
        if sys.platform == "win32":
            try:
                import report
                if report.webview2_version().startswith("не найден"):
                    explain_no_webview()
                    raise RuntimeError("no WebView2")
            except ImportError:
                pass
        try:
            webview.settings["ALLOW_DOWNLOADS"] = True  # a plain <a download> also works as a fallback
        except Exception:
            pass
        try:   # "Рисовать окно программы без видеокарты" (Настройки): on computers where the window itself
               # eats gigabytes of video memory that the models then have to do without
            if json.loads((server.DATA_DIR / "config.json").read_text(encoding="utf-8")).get("software_ui"):
                os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = "--disable-gpu --disable-gpu-compositing"
        except Exception:
            pass
        api = JsApi(port)
        mode = window_mode()
        # pywebview blocks dragging of every picture by default; the app drags pictures into slots
        w0, h0 = window_geo()
        api._window = webview.create_window(
            "Minimax H3 Workspace", url, width=w0, height=h0, min_size=(1000, 680),
            maximized=(mode == "max"), fullscreen=(mode == "fullscreen"),
            background_color="#12070a", js_api=api, draggable=True)

        def remember_geo(_=None):   # the next start opens the window the way you left it
            try:
                s = api._window.snapshot
                if s and s[2] and s[3]:
                    (server.DATA_DIR / "window.json").write_text(
                        json.dumps({"width": int(s[2]), "height": int(s[3])}), encoding="utf-8")
            except Exception:
                pass
        try:
            api._window.events.closed += remember_geo
        except Exception:
            pass
        webview.start(gui="edgechromium", private_mode=False,
                      storage_path=str(server.DATA_DIR / "webview"))
    except Exception:
        # no WebView2: use the browser and keep running until the console/process is closed
        webbrowser.open(url)
        threading.Event().wait()
    finally:
        srv.shutdown()
        single.release(server.DATA_DIR)
    # The window is closed and there is nothing left to do. WebView2 leaves a thread behind on some
    # machines: the process stays without a window, holds the data folder, the queue and the ComfyUI
    # core it started - and every next launch answers "the program is already running" while no window
    # exists anywhere. Cleanup above is done, so the exit is made sure of rather than hoped for.
    time.sleep(0.6)
    os._exit(0)


if __name__ == "__main__":
    main()
