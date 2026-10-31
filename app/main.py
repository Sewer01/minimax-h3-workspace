"""Minimax H3 Workspace desktop entry point: runs the local server and shows it in its own window.

Falls back to the default browser if the WebView2 window cannot be created.
"""

import asyncio
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

if getattr(sys, "frozen", False):
    # installed app: keep user data outside the program folder so updates do not touch it
    os.environ.setdefault("H3STUDIO_DATA", str(Path(os.environ["LOCALAPPDATA"]) / "MinimaxH3Workspace" / "data"))

from aiohttp import web  # noqa: E402

import server  # noqa: E402


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
    # so the window reference must stay private
    def __init__(self):
        self._window = None

    def pick_folder(self, start=""):
        import webview
        res = self._window.create_file_dialog(webview.FOLDER_DIALOG, directory=start if os.path.isdir(start) else "")
        return res[0] if res else None


def main():
    port = free_port()
    srv = ServerThread(port)
    srv.start()
    srv.ready.wait(30)
    url = f"http://127.0.0.1:{port}"

    try:
        import webview
        api = JsApi()
        api._window = webview.create_window("Minimax H3 Workspace", url, width=1440, height=920, min_size=(1000, 680),
                                           background_color="#12070a", js_api=api)
        webview.start(gui="edgechromium", private_mode=False,
                      storage_path=os.path.join(os.environ.get("LOCALAPPDATA", "."), "MinimaxH3Workspace", "webview"))
    except Exception:
        # no WebView2: use the browser and keep running until the console/process is closed
        webbrowser.open(url)
        threading.Event().wait()
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
