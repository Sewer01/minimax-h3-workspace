"""Minimax H3 Workspace local server.

Serves the UI and talks to ComfyUI over its HTTP/WebSocket API: uploads inputs,
queues workflows, relays progress and live previews to the browser, and keeps a
small library of finished videos. On first run it either connects to the user's
own ComfyUI or installs one (plus node packs and models) and manages it.

    python app/server.py [--port 8765] [--open]
"""

import argparse
import asyncio
import json
import os
import random
import struct
import subprocess
import sys
import uuid
import webbrowser
from datetime import datetime
from pathlib import Path

import aiohttp
from aiohttp import web

import agent as ag
import catalog
import cloud
import engine
import workflows as wf

# frozen builds keep code inside the bundle but user data next to the exe
ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
HOME = Path(sys.executable).parent if getattr(sys, "frozen", False) else ROOT
UI_DIR = ROOT / "ui"
DATA_DIR = Path(os.environ.get("H3STUDIO_DATA", HOME / "data"))
LIBRARY_FILE = DATA_DIR / "library.json"

CLIENT_ID = "h3studio-" + uuid.uuid4().hex[:8]

# ComfyUI binary websocket event types
BIN_PREVIEW = 1
BIN_PREVIEW_META = 4


class Library:
    def __init__(self, path):
        self.path = path
        self.items = []
        if path.exists():
            self.items = json.loads(path.read_text(encoding="utf-8"))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, item_id):
        return next((x for x in self.items if x["id"] == item_id), None)

    def add(self, item):
        self.items.insert(0, item)
        self.save()

    def remove(self, item_id):
        self.items = [x for x in self.items if x["id"] != item_id]
        self.save()


class App:
    def __init__(self, comfy_override=None):
        self.config = engine.Config(DATA_DIR / "config.json")
        self.comfy_override = comfy_override.rstrip("/") if comfy_override else None
        self.session = None
        self.browsers = set()
        self.jobs = {}  # prompt_id -> job dict
        self.library = Library(LIBRARY_FILE)
        self.output_dir = None
        self.comfy_online = False
        self.process = None  # engine.ComfyProcess in managed mode
        self.installer = None
        self.setup_log = []  # last setup events, replayed to a reopened window
        self.model_names = {}  # role -> name as ComfyUI lists it
        self.missing = []  # catalog entries ComfyUI does not see
        self.skills = ag.Skills(ROOT / "skills", DATA_DIR / "skills", self.config)
        self.llm = ag.LlamaServer(self.llm_root(), DATA_DIR / "llama.log")
        self.agent_installer = None
        self.hw = None
        self.rh_models = None  # model paths found in the user's RunningHub workflow

    def llm_root(self):
        if self.config.get("llm_dir"):
            return Path(self.config.get("llm_dir"))
        if self.config.get("engine") == "managed" and self.config.get("install_dir"):
            return Path(self.config.get("install_dir")) / "llm"
        return DATA_DIR.parent / "llm"

    async def llm_variant(self):
        if not self.llm.alive():
            self.llm.root = self.llm_root()  # follows the install folder chosen in setup
            self.llm.custom = self.config.get("llm_custom")
        if self.config.get("llm_variant") in ag.LLM_VARIANTS:
            return self.config.get("llm_variant")
        if self.hw is None:
            self.hw = await asyncio.to_thread(engine.detect_hardware)
        return ag.pick_variant(self.hw)

    async def stop_llm(self):
        if self.llm.alive():
            await asyncio.to_thread(self.llm.stop)

    @property
    def comfy(self):
        if self.comfy_override:
            return self.comfy_override
        if self.process:
            return self.process.url
        return self.config.get("comfy_url", "http://127.0.0.1:8188").rstrip("/")

    def comfy_root(self):
        d = self.config.get("install_dir")
        return Path(d) / catalog.COMFY_DIRNAME if d else None

    def choice(self):
        return {**catalog.DEFAULT_CHOICE, **(self.config.get("choice") or {})}

    def modes(self):
        return self.config.get("modes") or list(catalog.MODE_ROLES)

    # ---------- ComfyUI connection ----------

    async def start(self, _app):
        self.session = aiohttp.ClientSession()
        if self.config.get("engine") == "managed" and self.config.get("setup_done") and not self.comfy_override:
            self.start_managed()
        self.ws_task = asyncio.create_task(self.comfy_ws_loop())

    async def stop(self, _app):
        self.ws_task.cancel()
        if self.installer:
            self.installer.cancelled = True
        if self.process:
            await asyncio.to_thread(self.process.stop)
        await asyncio.to_thread(self.llm.stop)
        await self.session.close()

    def start_managed(self):
        root = self.comfy_root()
        self.process = engine.ComfyProcess(root, DATA_DIR / "comfyui.log")
        self.process.start()
        self.output_dir = root / "ComfyUI" / "output"

    async def wait_comfy(self, timeout=300):
        """Wait until ComfyUI answers; raise with its log if the process died."""
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout
        while loop.time() < end:
            if self.process and not self.process.alive():
                raise RuntimeError("ComfyUI не запустился:\n" + self.process.log_tail())
            try:
                await self.system_stats()
                return
            except Exception:
                await asyncio.sleep(2)
        raise RuntimeError("ComfyUI долго не отвечает:\n" + (self.process.log_tail() if self.process else ""))

    async def resolve_models(self):
        """Match the chosen model files against what ComfyUI sees (any subfolder)."""
        visible = {}
        for cat in {v["category"] for variants in catalog.VARIANTS.values() for v in variants.values()}:
            try:
                _, lst = await self.comfy_json("GET", f"/models/{cat}")
            except Exception:
                lst = []
            visible[cat] = lst or []
        names, missing = {}, []
        needed = catalog.roles_for(self.modes())
        for role, variant in self.choice().items():
            entry = catalog.model_entry(role, variant)
            hit = next((n for n in visible[entry["category"]]
                        if n.replace("/", "\\").split("\\")[-1] == entry["file"]), None)
            if hit:
                names[role] = hit
            else:
                names[role] = entry["name"]
                if role in needed:
                    missing.append(entry)
        self.model_names, self.missing = names, missing
        return names, missing

    async def comfy_json(self, method, path, **kw):
        async with self.session.request(method, self.comfy + path, **kw) as r:
            body = await r.text()
            data = json.loads(body) if body else None
            return r.status, data

    async def system_stats(self):
        status, data = await self.comfy_json("GET", "/system_stats", timeout=aiohttp.ClientTimeout(total=4))
        argv = data["system"].get("argv", [])
        if "--output-directory" in argv:
            self.output_dir = Path(argv[argv.index("--output-directory") + 1])
        return data

    async def comfy_ws_loop(self):
        while True:
            if not self.config.get("setup_done") and not self.comfy_override:
                await asyncio.sleep(2)
                continue
            ws_url = self.comfy.replace("http", "ws", 1) + f"/ws?clientId={CLIENT_ID}"
            try:
                async with self.session.ws_connect(ws_url, max_msg_size=0, heartbeat=20) as ws:
                    self.comfy_online = True
                    await self.resolve_models()
                    await self.broadcast({"type": "engine", "online": True})
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self.on_comfy_event(json.loads(msg.data))
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            await self.on_comfy_binary(msg.data)
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            if self.comfy_online:
                self.comfy_online = False
                await self.broadcast({"type": "engine", "online": False})
            await asyncio.sleep(2)

    async def on_comfy_binary(self, data):
        if len(data) < 8:
            return
        kind = struct.unpack(">I", data[:4])[0]
        if kind == BIN_PREVIEW:
            image = data[8:]
        elif kind == BIN_PREVIEW_META:
            meta_len = struct.unpack(">I", data[4:8])[0]
            meta = json.loads(data[8:8 + meta_len])
            if meta.get("prompt_id") not in self.jobs:
                return
            image = data[8 + meta_len:]
        else:
            return
        if not any(j["state"] == "running" for j in self.jobs.values()):
            return
        for b in list(self.browsers):
            try:
                await b.send_bytes(image)
            except Exception:
                self.browsers.discard(b)

    async def on_comfy_event(self, msg):
        t, d = msg.get("type"), msg.get("data") or {}
        job = self.jobs.get(d.get("prompt_id"))
        if not job:
            return
        pid = job["prompt_id"]
        if t == "execution_start":
            job["state"] = "running"
            await self.send_stage(job, "load")
        elif t == "executing" and d.get("node"):
            stage = job["stages"].get(d["node"])
            if stage:
                await self.send_stage(job, stage)
        elif t == "progress":
            stage = job["stages"].get(str(d.get("node")), job.get("stage"))
            if stage in ("sample", "sample2"):
                await self.send_stage(job, stage, d.get("value"), d.get("max"))
        elif t == "execution_success":
            await self.finish_job(job)
        elif t == "execution_error":
            job["state"] = "error"
            await self.broadcast({"type": "error", "prompt_id": pid,
                                  "message": f'{d.get("node_type", "")}: {d.get("exception_message", "").strip()}'})
            self.jobs.pop(pid, None)
        elif t == "execution_interrupted":
            await self.broadcast({"type": "interrupted", "prompt_id": pid})
            self.jobs.pop(pid, None)

    async def send_stage(self, job, stage, step=None, total=None):
        job["stage"] = stage
        await self.broadcast({"type": "stage", "prompt_id": job["prompt_id"],
                              "stage": stage, "step": step, "max": total})

    async def finish_job(self, job):
        pid = job["prompt_id"]
        self.jobs.pop(pid, None)
        _, hist = await self.comfy_json("GET", f"/history/{pid}")
        outputs = (hist or {}).get(pid, {}).get("outputs", {})
        files = [f for out in outputs.values() for lst in out.values() if isinstance(lst, list)
                 for f in lst if isinstance(f, dict) and "filename" in f]
        videos = [f for f in files if f["filename"].lower().endswith((".mp4", ".webm", ".mkv", ".mov"))]
        if not videos:
            await self.broadcast({"type": "error", "prompt_id": pid, "message": "ComfyUI не вернул видеофайл"})
            return
        item = dict(job["meta"], id=uuid.uuid4().hex[:12], prompt_id=pid,
                    date=datetime.now().isoformat(timespec="seconds"), fav=False, file=videos[0])
        meta = videos[0].get("metadata") or {}
        if meta.get("width") and meta.get("height"):
            item["res"] = f'{meta["width"]}×{meta["height"]}'
        self.library.add(item)
        await self.broadcast({"type": "done", "prompt_id": pid, "item": self.public(item)})

    async def broadcast(self, payload):
        text = json.dumps(payload, ensure_ascii=False)
        for b in list(self.browsers):
            try:
                await b.send_str(text)
            except Exception:
                self.browsers.discard(b)

    # ---------- helpers ----------

    def public(self, item):
        return {k: v for k, v in item.items() if k != "file"} | {"url": f"/api/file/{item['id']}"}

    def file_path(self, item):
        if item["file"].get("type") == "local":
            return Path(item["file"]["path"])
        if not self.output_dir or item["file"].get("type") != "output":
            return None
        return self.output_dir / item["file"].get("subfolder", "") / item["file"]["filename"]

    async def upload_input(self, field, rh=None):
        """Put an image/audio/video file into ComfyUI's input/h3studio folder (or upload it to RunningHub)."""
        data = aiohttp.FormData()
        name = f"h3s_{uuid.uuid4().hex[:8]}{Path(field.filename or 'img.png').suffix or '.png'}"
        content_type = field.headers.get("Content-Type", "application/octet-stream")
        if rh:
            try:
                return await rh.upload(name, await field.read(), content_type)
            except cloud.CloudError as e:
                raise web.HTTPBadGateway(text=str(e))
        data.add_field("image", await field.read(), filename=name, content_type=content_type)
        data.add_field("subfolder", "h3studio")
        data.add_field("type", "input")
        status, res = await self.comfy_json("POST", "/upload/image", data=data)
        if status != 200:
            raise web.HTTPBadGateway(text="Не удалось загрузить файл в ComfyUI")
        return f'{res["subfolder"]}/{res["name"]}' if res.get("subfolder") else res["name"]

    # ---------- HTTP handlers ----------

    async def h_status(self, _req):
        base = {"configured": bool(self.config.configured or self.comfy_override),
                "engine": self.config.get("engine"), "installing": bool(self.installer)}
        try:
            stats = await self.system_stats()
        except Exception:
            return web.json_response({**base, "online": False})
        dev = (stats.get("devices") or [{}])[0]
        await self.resolve_models()
        missing = {m["role"] for m in self.missing}
        needed = catalog.roles_for(self.modes())
        models = [{"role": role, "title": catalog.ROLE_TITLES[role], "name": self.model_names[role],
                   "category": catalog.VARIANTS[role][variant]["category"],
                   "size": catalog.VARIANTS[role][variant]["size"], "have": role not in missing,
                   "needed": role in needed,
                   "modes": [m for m, roles in catalog.MODE_ROLES.items() if role in roles]}
                  for role, variant in self.choice().items() if role in needed]
        choice = self.choice()
        add_modes = {m: round(catalog.plan_size(choice, self.modes() + [m]) - catalog.plan_size(choice, self.modes()))
                     for m in catalog.MODE_ROLES if m not in self.modes()}
        return web.json_response({
            **base, "online": True, "modes": self.modes(), "add_modes": add_modes,
            "gpu": gpu_name(dev.get("name", "")),
            "vram_total": dev.get("vram_total"), "vram_free": dev.get("vram_free"),
            "comfy_version": stats["system"].get("comfyui_version"),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "models": models,
        })

    # ---------- first-run setup ----------

    async def setup_emit(self, ev):
        if ev.get("state") != "progress" or ev.get("step") != getattr(self, "_last_step", None):
            self.setup_log = (self.setup_log + [ev])[-40:]
        self._last_step = ev.get("step")
        self.setup_last = ev
        await self.broadcast(ev)

    async def h_setup_info(self, req):
        install_dir = req.query.get("install_dir") or self.config.get("install_dir") or engine.default_install_dir()
        modes = [m for m in (req.query.get("modes") or ",".join(catalog.MODE_ROLES)).split(",") if m in catalog.MODE_ROLES]
        hw = await asyncio.to_thread(engine.detect_hardware, install_dir)
        choice, notes, verdict, profile = catalog.choose_models(hw, req.query.get("profile"))
        plan = [catalog.model_entry(r, choice[r]) for r in catalog.roles_for(modes or ["video"])]
        need_gb = round((catalog.COMFY_SIZE * 2.5 + sum(m["size"] for m in plan)) / 1e9)
        # download size of each profile for each set of modes, for the size picker
        sizes = {}
        for p in catalog.PROFILES:
            c = catalog.choose_models(hw, p)[0]
            sizes[p] = {m: round(catalog.plan_size(c, [m]) / 1e9) for m in catalog.MODE_ROLES}
            sizes[p]["selected"] = round(catalog.plan_size(c, modes or ["video"]) / 1e9)
        return web.json_response({
            "hardware": hw, "choice": choice, "notes": notes, "verdict": verdict,
            "profile": profile, "recommended": catalog.recommended_profile(hw), "modes": modes,
            "profiles": {k: {"title": v["title"], "about": v["about"], "sizes": sizes[k]} for k, v in catalog.PROFILES.items()},
            "install_dir": install_dir, "need_gb": need_gb,
            "plan": [{k: m[k] for k in ("role", "title", "file", "size", "variant")} for m in plan],
            "config": {k: self.config.get(k) for k in ("engine", "install_dir", "models_dir", "comfy_url", "setup_done")},
            "installing": bool(self.installer), "log": self.setup_log,
            "last": getattr(self, "setup_last", None),
        })

    async def h_setup_external(self, req):
        body = await req.json()
        url = (body.get("url") or "http://127.0.0.1:8188").strip().rstrip("/")
        if not url.startswith("http"):
            url = "http://" + url
        old = self.config.get("comfy_url")
        self.config.data["comfy_url"] = url
        self.process = None
        try:
            stats = await self.system_stats()
        except Exception:
            self.config.data["comfy_url"] = old
            return web.json_response({"error": f"ComfyUI не отвечает по адресу {url}. Он запущен?"}, status=400)
        self.config.update(engine="external", comfy_url=url, setup_done=True, choice=catalog.DEFAULT_CHOICE,
                           modes=list(catalog.MODE_ROLES))
        names, missing = await self.resolve_models()
        return web.json_response({"ok": True, "version": stats["system"].get("comfyui_version"),
                                  "missing": [m["file"] for m in missing]})

    async def h_setup_install(self, req):
        if self.installer:
            return web.json_response({"error": "Установка уже идёт"}, status=409)
        body = await req.json()
        install_dir = str(Path(body.get("install_dir") or engine.default_install_dir()))
        models_dir = (body.get("models_dir") or "").strip() or None
        if models_dir and not Path(models_dir).is_dir():
            return web.json_response({"error": f"Папка не найдена: {models_dir}"}, status=400)
        hw = await asyncio.to_thread(engine.detect_hardware, install_dir)
        profile = body.get("profile") or self.config.get("profile")
        choice, _, verdict, profile = catalog.choose_models(hw, profile)
        modes = [m for m in (body.get("modes") or self.modes()) if m in catalog.MODE_ROLES] or ["video"]
        if body.get("add_mode"):
            modes = sorted(set(self.modes()) | {body["add_mode"]}, key=list(catalog.MODE_ROLES).index)
        self.config.update(engine="managed", install_dir=install_dir, models_dir=models_dir,
                           choice=choice, profile=profile, modes=modes, setup_done=False)
        self.setup_log = []
        self.installer = engine.Installer(ROOT, self.session, self.setup_emit)
        self.installer.task = asyncio.create_task(self.run_install(install_dir, models_dir))
        return web.json_response({"ok": True})

    async def run_install(self, install_dir, models_dir):
        inst = self.installer
        try:
            Path(install_dir).mkdir(parents=True, exist_ok=True)
            root = await inst.install_comfy(install_dir)
            await inst.install_nodes(root)
            if models_dir:
                inst.write_extra_paths(root, models_dir)
            await inst.step("start", "progress", label="Запускаем ComfyUI…")
            if self.process:
                await asyncio.to_thread(self.process.stop)
            self.start_managed()
            await self.wait_comfy()
            await inst.step("start", "done", label="ComfyUI запущен")
            _, missing = await self.resolve_models()
            # shared files and the video mode first, so the first mode is usable sooner
            order = ["text_encoder", "video_vae", "audio_vae", "preview", "fast_unet"]
            missing.sort(key=lambda m: order.index(m["role"]) if m["role"] in order else len(order))
            if missing:
                gb = sum(m["size"] for m in missing) / 1e9
                await inst.step("models", "progress", label=f"Скачиваем модели: {len(missing)} файлов, {gb:.0f} ГБ")
                await inst.download_models(root / "ComfyUI" / "models", missing)
                await self.resolve_models()
            else:
                await inst.step("models", "done", label="Все модели уже есть")
            self.config.update(setup_done=True)
            await self.setup_emit({"type": "setup", "step": "finish", "state": "done", "label": "Готово!"})
        except engine.Cancelled:
            await self.setup_emit({"type": "setup", "step": "finish", "state": "cancelled",
                                   "label": "Установка остановлена. Скачанное сохранено — можно продолжить позже."})
        except Exception as e:
            await self.setup_emit({"type": "setup", "step": "finish", "state": "error", "label": str(e)})
        finally:
            self.installer = None

    async def h_setup_cancel(self, _req):
        if self.installer:
            self.installer.cancelled = True
        return web.json_response({"ok": True})

    async def h_setup_reset(self, _req):
        """Back to the first-run wizard; installed files stay where they are."""
        if self.installer:
            return web.json_response({"error": "Сначала остановите установку"}, status=409)
        if self.process:
            await asyncio.to_thread(self.process.stop)
            self.process = None
        self.config.update(setup_done=False)
        return web.json_response({"ok": True})

    # ---------- cloud (RunningHub) ----------

    def rh(self):
        """A RunningHub client when cloud generation is switched on, else None."""
        if self.config.get("rh_enabled") and self.config.get("rh_key"):
            return cloud.RunningHub(self.session, self.config.get("rh_key"), self.config.get("rh_host", "ai"))
        return None

    async def h_cloud_get(self, _req):
        c = self.config
        return web.json_response({"enabled": bool(c.get("rh_enabled")), "has_key": bool(c.get("rh_key")),
                                  "host": c.get("rh_host", "ai"), "workflow_id": c.get("rh_workflow_id", ""),
                                  "instance": c.get("rh_instance", "default")})

    async def h_cloud_set(self, req):
        b = await req.json()
        upd = {}
        if b.get("key"):
            upd["rh_key"] = b["key"].strip()
        if b.get("clear_key"):
            upd["rh_key"] = None
            upd["rh_enabled"] = False
        if b.get("host") in cloud.HOSTS:
            upd["rh_host"] = b["host"]
        if "workflow_id" in b:
            upd["rh_workflow_id"] = str(b["workflow_id"]).strip()
        if b.get("instance") in cloud.INSTANCE_TYPES:
            upd["rh_instance"] = b["instance"]
        if "enabled" in b:
            upd["rh_enabled"] = bool(b["enabled"]) and bool(upd.get("rh_key", self.config.get("rh_key")))
        self.config.update(**upd)
        self.rh_models = None
        return await self.h_cloud_get(req)

    async def h_setup_cloud(self, _req):
        """First-run choice "cloud only": nothing installed locally."""
        if not (self.config.get("rh_key") and self.config.get("rh_enabled")):
            return web.json_response({"error": "Сначала сохраните ключ и включите облако"}, status=400)
        self.config.update(engine="cloud", setup_done=True, modes=list(catalog.MODE_ROLES))
        return web.json_response({"ok": True})

    async def h_cloud_test(self, _req):
        key = self.config.get("rh_key")
        if not key:
            return web.json_response({"error": "Вставьте API-ключ"}, status=400)
        rh = cloud.RunningHub(self.session, key, self.config.get("rh_host", "ai"))
        try:
            acc = await rh.account()
            names = await rh.model_names(self.config.get("rh_workflow_id")) if self.config.get("rh_workflow_id") else set()
        except cloud.CloudError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True, "account": acc, "models": sorted(names)[:40]})

    async def queue_cloud(self, rh, graph, stages, meta):
        if not self.config.get("rh_workflow_id"):
            return web.json_response({"error": "Укажите ID любого вашего workflow на RunningHub (раздел «Модели» → «Облако»)"},
                                     status=400)
        if getattr(self, "rh_models", None) is None:
            try:
                self.rh_models = await rh.model_names(self.config.get("rh_workflow_id"))
            except cloud.CloudError:
                self.rh_models = set()
        cloud.remap_models(graph, self.rh_models)
        try:
            task = await rh.create(graph, self.config.get("rh_workflow_id"), self.config.get("rh_instance", "default"))
        except cloud.CloudError as e:
            return web.json_response({"error": str(e)}, status=400)
        pid = "rh-" + task
        self.jobs[pid] = {"prompt_id": pid, "state": "queued", "stages": {}, "stage": None,
                          "meta": dict(meta, cloud=True), "task": task}
        asyncio.create_task(self.poll_cloud(rh, pid))
        return web.json_response({"prompt_id": pid, "seed": meta["seed"]})

    async def poll_cloud(self, rh, pid):
        job = self.jobs.get(pid)
        last = None
        try:
            while pid in self.jobs:
                status = await rh.status(job["task"])
                if status != last:
                    last = status
                    stage = {"QUEUED": "cloud_queued", "RUNNING": "cloud_run"}.get(status)
                    if stage:
                        await self.send_stage(job, stage)
                if status == "SUCCESS":
                    outs = await rh.outputs(job["task"]) or []
                    video = next((o for o in outs if str(o.get("fileType", "")).lower() in ("mp4", "webm", "mov", "mkv")
                                  or str(o.get("fileUrl", "")).lower().split("?")[0].endswith((".mp4", ".webm", ".mov"))), None)
                    if not video:
                        raise cloud.CloudError("RunningHub не вернул видео")
                    dest = DATA_DIR / "cloud" / f'{pid}.mp4'
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    async with self.session.get(video["fileUrl"]) as r:
                        dest.write_bytes(await r.read())
                    self.jobs.pop(pid, None)
                    item = dict(job["meta"], id=uuid.uuid4().hex[:12], prompt_id=pid,
                                date=datetime.now().isoformat(timespec="seconds"), fav=False,
                                file={"type": "local", "path": str(dest), "filename": dest.name})
                    if video.get("consumeCoins"):
                        item["sub"] = f'{item.get("sub", "")} · ☁ {video["consumeCoins"]} RH'.strip(" ·")
                    self.library.add(item)
                    await self.broadcast({"type": "done", "prompt_id": pid, "item": self.public(item)})
                    return
                if status == "FAILED":
                    await rh.outputs(job["task"])  # raises with the failure reason
                    raise cloud.CloudError("Задание на RunningHub завершилось ошибкой")
                await asyncio.sleep(5)
        except Exception as e:
            if pid in self.jobs:
                self.jobs.pop(pid, None)
                await self.broadcast({"type": "error", "prompt_id": pid, "message": str(e)})

    async def read_form(self, req, file_fields):
        form, files = {}, {}
        rh = self.rh()
        async for field in await req.multipart():
            if field.name in file_fields and field.filename:
                files[field.name] = await self.upload_input(field, rh)
            else:
                form[field.name] = await field.text()
        prompt = form.get("prompt", "").strip()
        aspect, quality = form.get("aspect", "16:9"), form.get("quality", "720p")
        if not prompt:
            raise web.HTTPBadRequest(text="Напишите промпт")
        if aspect not in wf.ASPECTS or quality not in wf.QUALITY:
            raise web.HTTPBadRequest(text="Неизвестный формат или качество")
        seed = int(form["seed"]) if form.get("seed", "").isdigit() else random.randrange(2 ** 50)
        return form, files, prompt, aspect, quality, seed

    async def queue(self, graph, stages, meta):
        rh = self.rh()
        if rh:
            return await self.queue_cloud(rh, graph, stages, meta)
        await self.stop_llm()  # the agent's model and H3 do not fit in VRAM together
        status, res = await self.comfy_json("POST", "/prompt", json={
            "prompt": graph, "client_id": CLIENT_ID,
            "extra_data": {"preview_method": "auto"}})
        if status != 200:
            return web.json_response({"error": comfy_error_text(res)}, status=400)
        pid = res["prompt_id"]
        self.jobs[pid] = {"prompt_id": pid, "state": "queued", "stages": stages, "stage": None, "meta": meta}
        return web.json_response({"prompt_id": pid, "seed": meta["seed"], "queue_number": res.get("number")})

    async def need_models(self, mode):
        if self.rh():  # models live on RunningHub
            if not self.model_names:
                self.model_names = {r: catalog.VARIANTS[r][v]["name"] for r, v in self.choice().items()}
            return
        if mode not in self.modes():
            raise web.HTTPBadRequest(text="Этот режим не установлен. Добавьте его в разделе «Модели».")
        if not self.model_names:
            await self.resolve_models()
        lost = [m["file"] for m in self.missing if m["role"] in catalog.MODE_ROLES[mode]]
        if lost:
            raise web.HTTPBadRequest(text="ComfyUI не видит модели: " + ", ".join(lost)
                                     + ". Откройте раздел «Модели».")

    async def h_generate_video(self, req):
        await self.need_models("video")
        form, images, prompt, aspect, quality, seed = await self.read_form(req, ("first", "last"))
        seconds = min(15.0, max(3.0, float(form.get("duration", 5))))
        width, height = wf.resolution(aspect, quality)
        first, last = images.get("first"), images.get("last") if images.get("first") else None

        graph, stages = wf.video_graph(self.model_names, prompt, width, height, seconds, seed, first, last)
        sub = "Кадр → кадр" if last else "Картинка → видео" if first else "Текст → видео"
        return await self.queue(graph, stages, {
            "mode": "video", "sub": sub, "prompt": prompt, "aspect": aspect, "q": quality,
            "len": seconds, "res": f"{width}×{height}", "seed": seed})

    async def h_generate_lipsync(self, req):
        await self.need_models("lipsync")
        ref_fields = tuple(f"ref{i}" for i in range(9))
        form, files, prompt, aspect, quality, seed = await self.read_form(req, ref_fields + ("audio",))
        if "audio" not in files:
            raise web.HTTPBadRequest(text="Загрузите аудио")
        refs = [files[f] for f in ref_fields if f in files]
        if not refs:
            raise web.HTTPBadRequest(text="Добавьте хотя бы один референс — лицо персонажа")
        seconds = float(form.get("audio_len", 0))
        if not 1 <= seconds <= 15:
            raise web.HTTPBadRequest(text="Фрагмент аудио должен быть от 1 до 15 секунд")

        ref_size = form.get("ref_size") if form.get("ref_size") in ("match", "max") else "max"
        scale = wf.lipsync_scale(quality)
        out_w, out_h = wf.resolution(aspect, quality)
        width, height = wf.r32(out_w / scale), wf.r32(out_h / scale)
        graph, stages = wf.lipsync_graph(self.model_names, wf.lipsync_prompt(prompt, len(refs)), width, height,
                                         seconds, seed, files["audio"], refs, scale, ref_size)
        speed = "максимум" if ref_size == "max" else "быстро"
        return await self.queue(graph, stages, {
            "mode": "lipsync", "sub": f"{len(refs)} реф. ({speed}) · аудио {seconds:.1f} с", "prompt": prompt,
            "aspect": aspect, "q": quality, "len": seconds, "res": f"{out_w}×{out_h}", "seed": seed,
            "ref_size": ref_size})

    async def read_ref_form(self, req, max_images):
        fields = tuple(f"img{i}" for i in range(max_images)) + tuple(f"vid{i}" for i in range(3)) + \
            tuple(f"aud{i}" for i in range(3))
        form, files, prompt, aspect, quality, seed = await self.read_form(req, fields)
        pick = lambda p, n: [files[f"{p}{i}"] for i in range(n) if f"{p}{i}" in files]
        images, videos, audios = pick("img", max_images), pick("vid", 3), pick("aud", 3)
        if not images and not videos:
            raise web.HTTPBadRequest(text="Добавьте хотя бы одну референс-картинку")
        if len(images) + len(videos) + len(audios) > 12:
            raise web.HTTPBadRequest(text="Всего не больше 12 референсов (картинки + видео + аудио)")
        seconds = min(15.0, max(3.0, float(form.get("duration", 6))))
        return form, prompt, aspect, quality, seed, images, videos, audios, seconds

    async def h_generate_refs(self, req):
        await self.need_models("refs")
        form, prompt, aspect, quality, seed, images, videos, audios, seconds = await self.read_ref_form(req, 9)
        steps = int(form.get("steps", 6)) if form.get("steps", "6").isdigit() else 6
        steps = min(8, max(4, steps))
        ref_size = form.get("ref_size") if form.get("ref_size") in ("match", "max") else "max"
        extra = []
        for flag, role, strength in (("realism", "realism_lora", 0.5), ("motion_repair", "motion_repair_lora", 0.5)):
            if form.get(flag) == "1" and role not in {m["role"] for m in self.missing}:
                extra.append((self.model_names[role], strength))
        width, height = wf.resolution(aspect, quality)
        graph, stages = wf.ref_graph(self.model_names, wf.lipsync_prompt(prompt, len(images)), width, height, seconds, seed, images, videos, audios,
                                     steps, ref_size, extra)
        sub = f"{len(images)} карт." + (f", {len(videos)} видео" if videos else "") + (f", {len(audios)} аудио" if audios else "") \
            + f" · {steps} шагов"
        return await self.queue(graph, stages, {
            "mode": "refs", "sub": sub, "prompt": prompt, "aspect": aspect, "q": quality, "len": seconds,
            "res": f"{width}×{height}", "seed": seed, "steps": steps, "ref_size": ref_size})

    async def action_setup(self):
        """What the original action workflow needs and what this ComfyUI has."""
        if not self.model_names:
            await self.resolve_models()
        _, oi = await self.comfy_json("GET", "/object_info/SelfLiftH3Sampler")
        _, oi2 = await self.comfy_json("GET", "/object_info/H3SigmaRefiner")
        _, loras = await self.comfy_json("GET", "/models/loras")
        dmad = next((n for n in loras or [] if catalog.ACTION_DMAD_LORA in n), None)
        missing = {m["role"] for m in self.missing}
        have_up = "action_upscaler" not in missing
        nodes = bool(oi) and bool(oi2)
        return {
            "faithful": nodes and have_up,
            "selflift": bool(oi), "refiner": bool(oi2), "refiner_def": (oi2 or {}).get("H3SigmaRefiner"),
            "upscaler": have_up, "dmad": dmad,
            "speed": None if "speed_slider_lora" in missing else self.model_names.get("speed_slider_lora"),
        }

    async def h_action_info(self, _req):
        s = await self.action_setup()
        s.pop("refiner_def", None)
        return web.json_response(s)

    async def h_generate_action(self, req):
        await self.need_models("action")
        form, prompt, aspect, quality, seed, images, videos, audios, seconds = await self.read_ref_form(req, 6)
        if self.rh():
            # on RunningHub only built-in nodes are assumed; LoRAs are used if the user's RH workflow has them
            rh_files = {n.replace("\\", "/").split("/")[-1].lower(): n for n in (getattr(self, "rh_models", None) or [])}
            speed = rh_files.get(catalog.VARIANTS["speed_slider_lora"]["v1.1"]["file"].lower())
            dmad = next((n for b, n in rh_files.items() if catalog.ACTION_DMAD_LORA in b), None)
            s = {"faithful": False, "dmad": dmad, "speed": speed}
        else:
            s = await self.action_setup()
        loras = ([(s["dmad"], 0.8)] if s["dmad"] else []) + ([(s["speed"], 4.0)] if s["speed"] else [])
        names = dict(self.model_names)
        width, height = wf.resolution(aspect, quality)
        graph, stages = wf.action_graph(names, wf.lipsync_prompt(prompt, len(images)), width, height, seconds, seed, images, audios,
                                        s["faithful"], loras)
        if s["faithful"]:
            # the refiner's input names come from the installed node; values as in the original workflow
            req_inputs = [k for k in (s["refiner_def"]["input"].get("required") or {}) if k != "sigmas"]
            graph["13"]["inputs"] = {"sigmas": ["10", 0], **dict(zip(req_inputs, [1, 0.7, 0.0, "cosine"]))}
        sub = ("точная копия workflow" if s["faithful"] else "упрощённая схема") + f" · {len(images)} карт."
        return await self.queue(graph, stages, {
            "mode": "action", "sub": sub, "prompt": prompt, "aspect": aspect, "q": quality, "len": seconds,
            "res": f"{width}×{height}", "seed": seed, "faithful": s["faithful"]})

    async def h_interrupt(self, req):
        pid = (await req.json()).get("prompt_id")
        job = self.jobs.get(pid)
        if job and job.get("task"):  # cloud task
            self.jobs.pop(pid, None)
            rh = self.rh()
            if rh:
                try:
                    await rh.cancel(job["task"])
                except cloud.CloudError:
                    pass
            await self.broadcast({"type": "interrupted", "prompt_id": pid})
        elif job and job["state"] == "queued":
            await self.comfy_json("POST", "/queue", json={"delete": [pid]})
            self.jobs.pop(pid, None)
            await self.broadcast({"type": "interrupted", "prompt_id": pid})
        else:
            await self.comfy_json("POST", "/interrupt", json={"prompt_id": pid})
        return web.json_response({"ok": True})

    async def h_library(self, _req):
        return web.json_response([self.public(x) for x in self.library.items])

    async def h_library_patch(self, req):
        item = self.library.get(req.match_info["id"])
        if not item:
            raise web.HTTPNotFound()
        body = await req.json()
        if "fav" in body:
            item["fav"] = bool(body["fav"])
            self.library.save()
        return web.json_response(self.public(item))

    async def h_library_delete(self, req):
        item = self.library.get(req.match_info["id"])
        if not item:
            raise web.HTTPNotFound()
        path = self.file_path(item)
        recycled = False
        if path and path.exists():
            recycled = await asyncio.to_thread(send_to_recycle_bin, path)
            if not recycled:
                return web.json_response({"error": "Не удалось переместить файл в корзину"}, status=500)
        self.library.remove(item["id"])
        return web.json_response({"ok": True, "recycled": recycled})

    async def h_file(self, req):
        item = self.library.get(req.match_info["id"])
        if not item:
            raise web.HTTPNotFound()
        f = item["file"]
        headers = {}
        if req.query.get("download"):
            headers["Content-Disposition"] = f'attachment; filename="{f["filename"]}"'
        path = self.file_path(item)
        if path and path.exists():
            return web.FileResponse(path, headers=headers)
        # output folder unknown: stream through ComfyUI
        params = {"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")}
        async with self.session.get(self.comfy + "/view", params=params) as r:
            if r.status != 200:
                raise web.HTTPNotFound(text="Файл не найден")
            resp = web.StreamResponse(headers={"Content-Type": r.headers.get("Content-Type", "video/mp4"), **headers})
            await resp.prepare(req)
            async for chunk in r.content.iter_chunked(1 << 16):
                await resp.write(chunk)
            return resp

    async def h_events(self, req):
        ws = web.WebSocketResponse(heartbeat=20)
        await ws.prepare(req)
        self.browsers.add(ws)
        await ws.send_str(json.dumps({"type": "engine", "online": self.comfy_online}))
        try:
            async for _ in ws:
                pass
        finally:
            self.browsers.discard(ws)
        return ws

    # ---------- agent ----------

    def skill_public(self, s):
        return {k: s[k] for k in ("id", "name", "description", "builtin", "mode", "keywords")} | \
               {"refs": [r.name for r in s["refs"]]}

    async def h_agent_status(self, _req):
        variant = await self.llm_variant()
        v = ag.LLM_VARIANTS[variant]
        custom = self.config.get("llm_custom")
        vram = (self.hw or {}).get("vram_gb") or 0
        candidates = []
        if custom and custom.get("dir") and Path(custom["dir"]).is_dir():
            _, _, found = await asyncio.to_thread(ag.find_gguf, custom["dir"])
            candidates = [{"path": str(p), "name": p.name, "gb": round(p.stat().st_size / 1e9, 1),
                           "fits": ag.fits(p, vram), "current": str(p) == custom["model"]} for p in found]
        return web.json_response({
            "variant": variant, "title": Path(custom["model"]).stem if custom else v["title"],
            "size": 37e6 if custom else v["model_size"] + v["mmproj_size"], "candidates": candidates, "vram": vram,
            "custom": custom, "vision": self.llm.vision(variant), "has_model": self.llm.has_model(variant),
            "installed": self.llm.installed(variant), "running": self.llm.alive(),
            "installing": bool(self.agent_installer), "dir": str(self.llm.root),
            "skills": [self.skill_public(s) for s in self.skills.list()],
        })

    async def h_agent_install(self, _req):
        if self.agent_installer:
            return web.json_response({"error": "Уже скачивается"}, status=409)
        variant = await self.llm_variant()

        async def emit(ev):
            await self.broadcast(dict(ev, type="agent_setup"))

        self.agent_installer = engine.Installer(ROOT, self.session, emit)

        async def run():
            try:
                await ag.install(self.agent_installer, self.llm.root, variant,
                                 runtime_only=bool(self.config.get("llm_custom")))
                await emit({"step": "agent", "state": "done", "label": "Модель агента установлена"})
            except engine.Cancelled:
                await emit({"step": "agent", "state": "cancelled", "label": "Скачивание остановлено"})
            except Exception as e:
                await emit({"step": "agent", "state": "error", "label": str(e)})
            finally:
                self.agent_installer = None

        asyncio.create_task(run())
        return web.json_response({"ok": True})

    async def h_agent_model(self, req):
        """Use a GGUF model the user already has (or go back to the app's own Gemma)."""
        body = await req.json()
        if body.get("reset"):
            self.config.update(llm_custom=None)
        else:
            folder = (body.get("dir") or "").strip().strip('"')
            if not Path(folder).is_dir():
                return web.json_response({"error": f"Папка не найдена: {folder}"}, status=400)
            if self.hw is None:
                self.hw = await asyncio.to_thread(engine.detect_hardware)
            model, proj, models = await asyncio.to_thread(ag.find_gguf, folder, body.get("model"), self.hw.get("vram_gb"))
            if not model:
                return web.json_response({"error": "В папке нет моделей .gguf. Нужна обычная Gemma в формате GGUF "
                                                   "(версия для LTX в text_encoders не подойдёт)."}, status=400)
            self.config.update(llm_custom={"model": str(model), "mmproj": str(proj) if proj else None, "dir": folder})
        await self.stop_llm()
        return await self.h_agent_status(req)

    async def h_agent_install_cancel(self, _req):
        if self.agent_installer:
            self.agent_installer.cancelled = True
        return web.json_response({"ok": True})

    async def h_agent_chat(self, req):
        body = await req.json()
        messages = body.get("messages") or []
        target = body.get("target") if body.get("target") in catalog.MODE_ROLES else "video"
        ctx = body.get("ctx") or {}
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        await resp.prepare(req)

        async def send(obj):
            await resp.write(f"data: {ag.dumps(obj)}\n\n".encode())

        try:
            if self.jobs:
                raise RuntimeError("Сейчас идёт генерация видео — агент запустится, когда она закончится.")
            variant = await self.llm_variant()
            if not self.llm.installed(variant):
                raise RuntimeError("Модель агента ещё не скачана.")
            if not self.llm.alive():
                await send({"type": "status", "text": "Освобождаем видеопамять…"})
                try:
                    await self.comfy_json("POST", "/free", json={"unload_models": True, "free_memory": True})
                except Exception:
                    pass
                await send({"type": "status", "text": f'Загружаем {ag.LLM_VARIANTS[variant]["title"]}… (до минуты)'})
                self.llm.start(variant)
                await ag.wait_ready(self.session, self.llm)

            if not self.llm.vision(variant) and any(m.get("images") for m in messages):
                # without a vision projector the model cannot take pictures
                messages = [dict(m, images=None) for m in messages]
                await send({"type": "status", "text": "У этой модели нет файла «зрения» (mmproj) — картинки она не увидит"})
            ctx["images"] = sum(len(m.get("images") or []) for m in messages if m["role"] == "user")
            active = self.skills.active(ag.chat_text(messages))
            await send({"type": "skills", "names": [s["name"] for s in active]})
            system = ag.system_prompt("\n\n".join(self.skills.text(s) for s in active), target, ctx)
            await send({"type": "status", "text": "Читаю скиллы и думаю…"})

            thinking = False
            payload = {"messages": [{"role": "system", "content": system}] + ag.to_openai(messages),
                       "stream": True, "temperature": 0.6, "cache_prompt": True}
            async with self.session.post(self.llm.url + "/v1/chat/completions", json=payload,
                                         timeout=aiohttp.ClientTimeout(total=None, sock_read=600)) as r:
                if r.status != 200:
                    raise RuntimeError(f"Модель ответила ошибкой {r.status}: {(await r.text())[:300]}")
                async for raw in r.content:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    delta = (json.loads(line[5:]).get("choices") or [{}])[0].get("delta", {})
                    if delta.get("content"):
                        await send({"type": "delta", "text": delta["content"]})
                    elif delta.get("reasoning_content") and not thinking:
                        thinking = True
                        await send({"type": "status", "text": "Думаю над сценой…"})
            await send({"type": "done"})
        except (ConnectionResetError, asyncio.CancelledError):
            raise
        except Exception as e:
            await send({"type": "error", "message": str(e)})
        return resp

    # agent chat history: one JSON file per conversation in data/chats

    def chat_path(self, cid):
        if not re_chat_id(cid):
            raise web.HTTPBadRequest(text="bad chat id")
        return DATA_DIR / "chats" / f"{cid}.json"

    async def h_chats(self, _req):
        out = []
        for p in (DATA_DIR / "chats").glob("*.json") if (DATA_DIR / "chats").exists() else []:
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out.append({"id": p.stem, "title": c.get("title", ""), "updated": c.get("updated", ""),
                        "target": c.get("target", "video"), "count": len(c.get("messages", []))})
        return web.json_response(sorted(out, key=lambda c: c["updated"], reverse=True))

    async def h_chat_get(self, req):
        p = self.chat_path(req.match_info["id"])
        if not p.exists():
            raise web.HTTPNotFound()
        return web.Response(text=p.read_text(encoding="utf-8"), content_type="application/json")

    async def h_chat_put(self, req):
        p = self.chat_path(req.match_info["id"])
        body = await req.json()
        msgs = [{k: m.get(k) for k in ("role", "content", "images", "skills", "error") if m.get(k)}
                for m in body.get("messages") or [] if m.get("role") in ("user", "assistant")]
        first = next((m.get("content", "") for m in msgs if m["role"] == "user"), "") or "Без названия"
        data = {"title": (body.get("title") or first).strip().replace("\n", " ")[:80], "target": body.get("target", "video"),
                "updated": datetime.now().isoformat(timespec="seconds"), "messages": msgs}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return web.json_response({"ok": True, "title": data["title"]})

    async def h_chat_delete(self, req):
        self.chat_path(req.match_info["id"]).unlink(missing_ok=True)
        return web.json_response({"ok": True})

    async def h_chats_clear(self, _req):
        for p in (DATA_DIR / "chats").glob("*.json") if (DATA_DIR / "chats").exists() else []:
            p.unlink(missing_ok=True)
        return web.json_response({"ok": True})

    async def h_skill_upload(self, req):
        files = []
        async for field in await req.multipart():
            if field.filename:
                files.append((field.filename.replace("\\", "/"), await field.read()))
        try:
            sid = await asyncio.to_thread(self.skills.save_upload, files)
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=400)
        return web.json_response({"ok": True, "id": sid})

    async def h_skill_patch(self, req):
        body = await req.json()
        self.skills.save_settings(req.match_info["id"], body.get("mode"), body.get("keywords"))
        return web.json_response({"ok": True})

    async def h_skill_delete(self, req):
        self.skills.delete(req.match_info["id"])
        return web.json_response({"ok": True})

    async def h_index(self, _req):
        return web.FileResponse(UI_DIR / "index.html", headers={"Cache-Control": "no-store"})


def re_chat_id(cid):
    return cid.isalnum() and 4 <= len(cid) <= 40


def gpu_name(raw):
    # "cuda:0 NVIDIA GeForce RTX 4060 Ti : cudaMallocAsync" -> "NVIDIA GeForce RTX 4060 Ti"
    name = raw.split(" : ")[0]
    return name.split(" ", 1)[1] if name.startswith("cuda:") and " " in name else name


def comfy_error_text(res):
    if not isinstance(res, dict):
        return "ComfyUI отклонил задание"
    err = res.get("error") or {}
    parts = [err.get("message", "") + (": " + err.get("details", "") if err.get("details") else "")]
    for node_id, ne in (res.get("node_errors") or {}).items():
        for e in ne.get("errors", []):
            parts.append(f'{ne.get("class_type", node_id)}: {e.get("message")} {e.get("details", "")}'.strip())
    return "\n".join(p for p in parts if p) or "ComfyUI отклонил задание"


def send_to_recycle_bin(path):
    ps = ("Add-Type -AssemblyName Microsoft.VisualBasic; "
          "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($env:H3_DEL, 'OnlyErrorDialogs', 'SendToRecycleBin')")
    r = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                       env={**os.environ, "H3_DEL": str(path)}, capture_output=True)
    return r.returncode == 0 and not path.exists()


def build_app(comfy_override=None):
    state = App(comfy_override)
    app = web.Application(client_max_size=200 * 1024 * 1024)
    app.on_startup.append(state.start)
    app.on_cleanup.append(state.stop)
    app.add_routes([
        web.get("/", state.h_index),
        web.get("/api/status", state.h_status),
        web.get("/api/setup/info", state.h_setup_info),
        web.post("/api/setup/external", state.h_setup_external),
        web.post("/api/setup/install", state.h_setup_install),
        web.post("/api/setup/cancel", state.h_setup_cancel),
        web.post("/api/setup/reset", state.h_setup_reset),
        web.post("/api/generate/video", state.h_generate_video),
        web.post("/api/generate/lipsync", state.h_generate_lipsync),
        web.post("/api/generate/refs", state.h_generate_refs),
        web.post("/api/generate/action", state.h_generate_action),
        web.get("/api/action/info", state.h_action_info),
        web.get("/api/cloud", state.h_cloud_get),
        web.post("/api/cloud", state.h_cloud_set),
        web.post("/api/cloud/test", state.h_cloud_test),
        web.post("/api/setup/cloud", state.h_setup_cloud),
        web.post("/api/interrupt", state.h_interrupt),
        web.get("/api/agent/status", state.h_agent_status),
        web.post("/api/agent/install", state.h_agent_install),
        web.post("/api/agent/install/cancel", state.h_agent_install_cancel),
        web.post("/api/agent/model", state.h_agent_model),
        web.post("/api/agent/chat", state.h_agent_chat),
        web.get("/api/agent/chats", state.h_chats),
        web.delete("/api/agent/chats", state.h_chats_clear),
        web.get("/api/agent/chats/{id}", state.h_chat_get),
        web.put("/api/agent/chats/{id}", state.h_chat_put),
        web.delete("/api/agent/chats/{id}", state.h_chat_delete),
        web.post("/api/agent/skills", state.h_skill_upload),
        web.patch("/api/agent/skills/{id}", state.h_skill_patch),
        web.delete("/api/agent/skills/{id}", state.h_skill_delete),
        web.get("/api/library", state.h_library),
        web.patch("/api/library/{id}", state.h_library_patch),
        web.delete("/api/library/{id}", state.h_library_delete),
        web.get("/api/file/{id}", state.h_file),
        web.get("/api/events", state.h_events),
        web.static("/ui", UI_DIR),
    ])
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", help="always use this ComfyUI URL (skips the setup wizard)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="open the UI in the default browser")
    args = ap.parse_args()

    app = build_app(args.comfy)
    if args.open:
        async def open_browser(_app):
            asyncio.get_running_loop().call_later(1, webbrowser.open, f"http://127.0.0.1:{args.port}")
        app.on_startup.append(open_browser)
    web.run_app(app, host="127.0.0.1", port=args.port, print=lambda *_: print(f"Minimax H3 Workspace: http://127.0.0.1:{args.port}"))


if __name__ == "__main__":
    main()
