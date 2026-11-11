"""Hardware detection, first-run install and the managed ComfyUI process.

Two engine modes are stored in data/config.json:
  external - the user runs their own ComfyUI; we only connect to its URL
  managed  - we installed ComfyUI ourselves and start/stop it with the app
"""

import asyncio
import ctypes
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import aiohttp

import catalog

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


# ---------- config ----------

class Config:
    def __init__(self, path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (OSError, ValueError):
            self.data = {}  # a damaged settings file must not stop the program; the next write fixes it

    def get(self, key, default=None):
        return self.data.get(key, default)

    def update(self, **kw):
        """Put only the given keys on top of the file as it is on disk right now, and replace the file atomically.

        Reading the file first matters: a second copy of the program may work with the same data folder (the port
        is picked automatically, so nothing stopped it from starting), and writing our whole in-memory copy back
        quietly restored the keys the other copy had changed. That once reset `setup_done`: the program stopped
        starting its own ComfyUI and asked to download models that are lying on the disk.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            on_disk = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(on_disk, dict):
                self.data = {**self.data, **on_disk}
        except (OSError, ValueError):
            pass  # missing or half-written file: keep what we have in memory
        self.data.update(kw)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)  # a reader sees either the old file or the new one, never a part of it

    def forget(self, *keys):
        """Take keys out of the settings for good. update() merges the file on disk over memory on
           purpose, so popping from self.data and calling update() would write the key straight back —
           that is how a saved copy of the old set-up survived its own restore."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            on_disk = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(on_disk, dict):
                self.data = {**self.data, **on_disk}
        except (OSError, ValueError):
            pass
        for key in keys:
            self.data.pop(key, None)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    @property
    def configured(self):
        return self.data.get("engine") in ("external", "managed", "cloud") and self.data.get("setup_done")


# ---------- hardware ----------

def detect_hardware(install_dir=None):
    hw = {"nvidia": False, "gpu": None, "vram_gb": 0, "compute_cap": 0, "driver": 0, "ram_gb": 0, "disk_free_gb": None}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=10, creationflags=CREATE_NO_WINDOW).stdout
        gpus = [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]
        if gpus:
            best = max(gpus, key=lambda g: float(g[1]))  # pick the card with most memory
            hw.update(nvidia=True, gpu=best[0].replace("NVIDIA GeForce ", ""),
                      vram_gb=round(float(best[1]) / 1024), compute_cap=float(best[2]))
            try:
                hw["driver"] = float(best[3])
            except (IndexError, ValueError):
                pass
    except Exception:
        pass

    class MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    try:
        ms = MemStatus(dwLength=ctypes.sizeof(MemStatus))
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        hw["ram_gb"] = round(ms.ullTotalPhys / 1024 ** 3)
    except Exception:
        pass

    if install_dir:
        hw["disk_free_gb"] = disk_free_gb(install_dir)
    return hw


def disk_free_gb(path):
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    try:
        return round(shutil.disk_usage(p).free / 1e9)
    except OSError:
        return None


def default_install_dir():
    # biggest fixed drive other than C: if there is one, else the user's local app data
    best, best_free = None, 0
    for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:\\"
        if os.path.exists(root):
            try:
                free = shutil.disk_usage(root).free
            except OSError:
                continue
            if free > best_free:
                best, best_free = root, free
    if best and best_free > 120e9:
        return str(Path(best) / "MinimaxH3Workspace")
    return str(Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MinimaxH3Workspace")


CORE_PORTS = (8188, 8237)   # the range a core of this install may take: free_port and running_core must agree,
                            # otherwise a core on 8193+ is "not seen" and gets killed while another copy is using it


def free_port(start=CORE_PORTS[0], last=None):
    # the caller may ask for a range of its own (the agent's llama-server starts higher),
    # so the upper end widens instead of going empty
    last = last if last is not None else max(CORE_PORTS[1], start + 49)
    for port in range(start, last + 1):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError("Нет свободного порта для ComfyUI")


def find_7z(app_root):
    for p in (app_root / "bin" / "7zr.exe", Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "7-Zip" / "7z.exe"):
        if p.exists():
            return p
    return shutil.which("7z") or shutil.which("7zr")


# ---------- install ----------

class Cancelled(Exception):
    pass


HF_MIRROR = ""   # host of a Hugging Face mirror when the user switched it on (for slow connections)


def mirrored(url):
    """The same file on the mirror, if the user chose one."""
    if HF_MIRROR and url.startswith("https://huggingface.co/"):
        return f"https://{HF_MIRROR}/" + url[len("https://huggingface.co/"):]
    return url


def sha256_of(path):
    """Checksum of a downloaded file, read in megabyte blocks (the ComfyUI archive is ~2 GB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def system_proxy():
    """The proxy a VPN/proxy program set in Windows (or HTTPS_PROXY), if any: our own requests would otherwise bypass it."""
    try:
        import urllib.request
        px = urllib.request.getproxies()
        return px.get("https") or px.get("http") or None
    except Exception:
        return None


class Installer:
    """Runs the install steps and reports progress through `emit(dict)`."""

    def __init__(self, app_root, session, emit):
        self.app_root = app_root
        self.session = session
        self.emit = emit
        self.cancelled = False
        self.task = None

    def check(self):
        if self.cancelled:
            raise Cancelled()

    async def step(self, sid, state, **kw):
        await self.emit({"type": "setup", "step": sid, "state": state, **kw})

    async def download(self, url, dest, sid, label, size_hint=0, attempts=40, sha256=""):
        """Resumable download to dest (via dest.part).

        `sha256` verifies what arrived. Without it a mirror, a proxy or a broken CDN answer silently becomes
        the program's ComfyUI, and a file that is on the disk is never looked at again.
        """
        url = mirrored(url)
        dest = Path(dest)
        if dest.exists():
            if not sha256 or await asyncio.to_thread(sha256_of, dest) == sha256.lower():
                return
            await self.step(sid, "progress", label=f"{label} — файл не совпал с контрольной суммой, скачиваем заново…")
            dest.unlink()
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        for attempt in range(attempts):  # a long outage: wait and continue from the saved .part file
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                async with self.session.get(url, headers=headers, proxy=system_proxy(), timeout=aiohttp.ClientTimeout(sock_read=120)) as r:
                    if r.status == 416:  # already complete
                        break
                    if r.status not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status} при скачивании {label}")
                    if r.status == 200:
                        have = 0
                    total = have + int(r.headers.get("Content-Length", 0)) or size_hint
                    done, last = have, 0.0
                    with open(part, "ab" if have else "wb") as f:
                        async for chunk in r.content.iter_chunked(1 << 20):
                            self.check()
                            f.write(chunk)
                            done += len(chunk)
                            now = time.monotonic()
                            if now - last > 0.5:
                                last = now
                                await self.step(sid, "progress", label=label, done=done, total=total)
                    break
            except (Cancelled, asyncio.CancelledError):
                raise
            except Exception as e:
                if attempt == attempts - 1:
                    raise RuntimeError(f"Не удалось скачать {label}: {e}")
                why = (str(e) or type(e).__name__)[:90]
                await self.step(sid, "progress", label=f"{label} — обрыв связи ({why}), пробуем снова…")
                await asyncio.sleep(min(30, 3 + attempt * 3))
        part.replace(dest)
        if sha256 and await asyncio.to_thread(sha256_of, dest) != sha256.lower():
            dest.unlink(missing_ok=True)  # never unpack or load a file that does not match
            raise RuntimeError(f"{label} скачался повреждённым — контрольная сумма не совпала. Попробуйте ещё раз.")

    async def install_comfy(self, install_dir):
        root = Path(install_dir) / catalog.COMFY_DIRNAME
        if (root / "python_embeded" / "python.exe").exists():
            await self.step("comfy", "done", label="ComfyUI уже установлен")
            return root
        archive = Path(install_dir) / "downloads" / "ComfyUI_windows_portable_nvidia.7z"
        await self.step("comfy", "progress", label=f"Скачиваем ComfyUI {catalog.COMFY_VERSION}")
        await self.download(catalog.COMFY_URL, archive, "comfy", f"ComfyUI {catalog.COMFY_VERSION}", catalog.COMFY_SIZE,
                            sha256=catalog.COMFY_SHA256)
        seven = find_7z(self.app_root)
        if not seven:
            # official standalone 7-Zip extractor (~600 KB)
            seven = Path(install_dir) / "tools" / "7zr.exe"
            for url in ("https://github.com/ip7z/7zip/releases/latest/download/7zr.exe", "https://www.7-zip.org/a/7zr.exe"):
                try:
                    await self.download(url, seven, "comfy", "распаковщик 7-Zip")
                    break
                except RuntimeError:
                    if url.startswith("https://www."):
                        raise
        await self.step("comfy", "progress", label="Распаковываем ComfyUI… (пара минут)", done=0, total=100)
        proc = await asyncio.create_subprocess_exec(
            str(seven), "x", "-y", "-bsp1", f"-o{install_dir}", str(archive),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, creationflags=CREATE_NO_WINDOW)
        buf = b""
        while True:
            chunk = await proc.stdout.read(256)
            if not chunk:
                break
            buf = (buf + chunk)[-64:]
            m = re.findall(rb"(\d+)%", buf)
            if m:
                await self.step("comfy", "progress", label="Распаковываем ComfyUI…", done=int(m[-1]), total=100)
            if self.cancelled:
                proc.kill()
                raise Cancelled()
        if await proc.wait() != 0 or not (root / "python_embeded" / "python.exe").exists():
            raise RuntimeError("Не удалось распаковать ComfyUI")
        archive.unlink(missing_ok=True)
        await self.step("comfy", "done", label="ComfyUI установлен")
        return root

    async def install_nodes(self, comfy_root):
        nodes_dir = comfy_root / "ComfyUI" / "custom_nodes"
        for i, pack in enumerate(catalog.NODE_PACKS):
            target = nodes_dir / pack["name"]
            label = f'Ноды: {pack["name"]}'
            await self.step("nodes", "progress", label=label, done=i, total=len(catalog.NODE_PACKS))
            if target.exists():
                continue
            urls = (f'https://github.com/{pack["repo"]}/archive/{pack["commit"]}.zip',
                    f'https://codeload.github.com/{pack["repo"]}/zip/{pack["commit"]}')  # second address: some networks block only one
            zpath = comfy_root.parent / "downloads" / f'{pack["name"]}.zip'
            for rnd in range(12):
                try:
                    await self.download(urls[rnd % 2], zpath, "nodes", label, attempts=2)
                    break
                except RuntimeError as e:
                    if rnd == 11:
                        raise
                    await self.step("nodes", "progress", label=f"{label} — {str(e)[-120:]}; пробуем другой адрес…")
                    await asyncio.sleep(5)
            with zipfile.ZipFile(zpath) as z:
                top = z.namelist()[0].split("/")[0]
                z.extractall(nodes_dir)
            (nodes_dir / top).rename(target)
            zpath.unlink(missing_ok=True)
            req = target / "requirements.txt"
            if req.exists() and req.read_text(encoding="utf-8", errors="ignore").strip():
                await self.step("nodes", "progress", label=f'{label}: ставим зависимости')
                py = comfy_root / "python_embeded" / "python.exe"
                proc = await asyncio.create_subprocess_exec(
                    str(py), "-s", "-m", "pip", "install", "-r", str(req),
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    creationflags=CREATE_NO_WINDOW, env=local_env(comfy_root))
                await proc.wait()
        await self.step("nodes", "done", label="Ноды установлены")

    @staticmethod
    def write_extra_paths(comfy_root, models_dir):
        """Let ComfyUI also see an existing models folder (nothing is copied)."""
        # older setups keep diffusion models in "unet" and text encoders in "clip"
        cats = {"diffusion_models": ["diffusion_models", "unet"], "text_encoders": ["text_encoders", "clip"],
                "vae": ["vae"], "vae_approx": ["vae_approx"], "loras": ["loras"],
                "latent_upscale_models": ["latent_upscale_models"]}
        # one common folder for several programs: the files of the catalog may lie anywhere inside it (any depth, any subfolder),
        # so every folder that holds a wanted file is added to its category; ComfyUI reads subfolders of a listed folder by itself
        wanted = {}
        for variants in catalog.VARIANTS.values():
            for v in variants.values():
                wanted[Path(v["name"].replace("\\", "/")).name.lower()] = v["category"]
        found = {c: [] for c in cats}
        base = Path(models_dir)
        for dirpath, dirnames, filenames in os.walk(base):
            if len(Path(dirpath).relative_to(base).parts) >= 8:
                dirnames[:] = []
            for fn in filenames:
                cat = wanted.get(fn.lower())
                if cat in found and Path(dirpath).as_posix() not in found[cat]:
                    found[cat].append(Path(dirpath).as_posix())
        lines = ["h3studio_user_models:", f"    base_path: {base.as_posix()}"]
        for cat, folders in cats.items():
            lines += [f"    {cat}: |"] + [f"        {f}" for f in folders + found[cat]]
        (comfy_root / "ComfyUI" / "extra_model_paths.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    async def download_models(self, models_root, missing):
        total = sum(m["size"] for m in missing)
        done_before = 0
        for i, m in enumerate(missing):
            dest = Path(models_root) / m["category"] / Path(m["name"].replace("\\", "/"))
            label = f'{m["title"]} — {m["file"]}'
            base = done_before

            original_emit = self.emit  # emit_overall must call this one: self.emit is replaced below

            async def emit_overall(ev, base=base, idx=i, original_emit=original_emit):
                if ev.get("done") is not None:
                    ev = dict(ev, overall_done=base + ev["done"], overall_total=total, index=idx + 1, count=len(missing))
                await original_emit(ev)

            self.emit = emit_overall
            try:
                await self.download(m["url"], dest, "models", label, m["size"], sha256=catalog.file_sha256(m["url"]))
            finally:
                self.emit = original_emit
            done_before += m["size"]
        await self.step("models", "done", label="Все модели на месте")


# ---------- managed ComfyUI process ----------

def local_env(root):
    """Environment that keeps caches and temp files inside the install folder instead of the Windows profile."""
    cache = Path(root) / "cache"
    dirs = {"PIP_CACHE_DIR": "pip", "TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor",
            "HF_HOME": "huggingface", "TORCH_HOME": "torch", "XDG_CACHE_HOME": "xdg", "TEMP": "temp", "TMP": "temp"}
    env = dict(os.environ)
    for var, sub in dirs.items():
        (cache / sub).mkdir(parents=True, exist_ok=True)
        env[var] = str(cache / sub)
    # On Russian Windows (cp1251) ComfyUI's log cannot print emoji (the upscaler node prints a check mark):
    # the error handler itself then crashes, the worker thread dies and Lipsync hangs forever.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def running_core(py, first=CORE_PORTS[0], last=CORE_PORTS[1]):
    """Port of a ComfyUI started by THIS python, how long it has been alive, and whether we looked at all.

    Only processes of this install are considered, so the user's own ComfyUI is never touched.
    Returns (port, age, seen); seen is False when PowerShell did not answer - that is not the same as
    "no core is running", and acting on it the wrong way kills a core another copy is working with.
    """
    exe = str(py).replace("'", "''")
    ps = ("$py = '" + exe + "'; $best = $null; "
          "foreach ($c in Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue) { "
          f"if ($c.LocalPort -lt {first} -or $c.LocalPort -gt {last}) {{ continue }}; "
          "$p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $c.OwningProcess) -ErrorAction SilentlyContinue; "
          "if (-not $p -or $p.ExecutablePath -ne $py) { continue }; "
          "$age = [int]((Get-Date) - $p.CreationDate).TotalSeconds; "
          "if (-not $best -or $c.LocalPort -lt $best[0]) { $best = @($c.LocalPort, $age) } }; "
          "if ($best) { '{0} {1}' -f $best[0], $best[1] } else { 'none' }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=40,
                             capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)
    except Exception:
        return None, None, False          # antivirus, group policy, a slow machine: we do not know
    if out.returncode != 0:
        return None, None, False
    parts = (out.stdout or "").strip().split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1]), True
    return None, None, True               # the look succeeded and found nothing


class ComfyProcess:
    def __init__(self, comfy_root, log_path):
        self.root = Path(comfy_root)
        self.log_path = log_path
        self.proc = None
        self.port = None
        self.attached = False   # the core was started elsewhere (another copy of the program): we may not kill it
        self._log = None        # handle of the ComfyUI log, closed in stop()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        if self.proc and self.proc.poll() is None:
            return
        py = self.root / "python_embeded" / "python.exe"
        port, _age, seen = running_core(py)
        if port:
            # Another copy of the program already runs this ComfyUI or is still waiting for it to answer.
            # Killing it left that copy with "ComfyUI does not see models" (the log stopped mid-start), and
            # starting our own on the next port would put the same 200 GB of weights into video memory twice.
            # How long that core has been up is not our call: it listens, therefore it is alive.
            self.port, self.proc, self.attached = port, None, True
            return
        if seen:
            self.kill_orphans(py)   # we looked and found no live core: leftovers hold the port and comfyui.db
        # When the look itself failed we do not kill anything - a core another copy works with could be there.
        self.port, self.attached = free_port(), False
        args = [str(py), "-s", "ComfyUI\\main.py", "--windows-standalone-build", "--disable-auto-launch",
                "--listen", "127.0.0.1", "--port", str(self.port), "--preview-method", "auto"]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # a append, not "w": the previous run is exactly what log_tail and the error report are asked about,
        # and the handle is kept as an attribute and closed in stop() instead of leaking on every restart
        self._log = open(self.log_path, "a", encoding="utf-8", errors="replace")
        self._log.write("\n=== новый запуск ComfyUI ===\n")
        self._log.flush()
        self.proc = subprocess.Popen(args, cwd=self.root, stdout=self._log, stderr=subprocess.STDOUT,
                                     creationflags=CREATE_NO_WINDOW, env=local_env(self.root))

    @staticmethod
    def kill_orphans(py):
        """After a crash the app's own ComfyUI may stay in the background and hold comfyui.db and the port.
        Kill only python processes started from THIS install's python_embeded (never the user's own ComfyUI)."""
        exe = str(py).replace("'", "''")
        ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
              f"Where-Object {{ $_.ExecutablePath -eq '{exe}' }} | "
              "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }")
        try:
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
            time.sleep(1)
        except Exception:
            pass

    def alive(self):
        if self.attached:
            # not ours: "alive" can only mean the port still answers, otherwise wait_comfy would keep waiting
            # the whole 300 s on a core that died in someone else's window
            if not self.port:
                return False
            with socket.socket() as s:
                s.settimeout(1)
                return s.connect_ex(("127.0.0.1", self.port)) == 0
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self._log:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None
        if self.attached:   # not ours to kill: whoever started that core owns it (another copy, or the user)
            self.attached, self.port = False, None
            return
        if self.alive():
            self.proc.terminate()
            try:
                self.proc.wait(15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            try:  # whole tree: a helper process left behind would keep comfyui.db and the port locked
                subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], timeout=15,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
            except Exception:
                pass
        self.proc = None

    def log_tail(self, n=15):
        try:
            return "\n".join(self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
        except OSError:
            return ""
