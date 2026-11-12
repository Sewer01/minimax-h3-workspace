"""Updates of the engine and the node packs, separate from the program's own update.

The program update replaces the application files. This module answers a different question:
ComfyUI itself is 0.44.0 today, and when 0.44.1 appears the person should not have to wait for
a rebuild of our release. Two rules here:

  - the core is updated from the source archive of the release tag (tens of megabytes) and only
    over the program's own code: models, custom_nodes, input, output and user data are never
    touched, and the files we overwrite are zipped into the data folder first, so a bad update
    can be rolled back;
  - node packs are pinned to a commit on purpose: a newer commit is not "better", it is simply
    newer, so the check only reports it and the person updates a pack by hand, keeping the old
    folder as .old-<time> for the rollback button.
"""

import asyncio
import os
import re
import shutil
import tarfile
import time
import aiohttp
import zipfile
from pathlib import Path

import catalog
import engine

GH = "https://api.github.com"
KEEP_BACKUPS = 2          # how many core backups to keep in the data folder
_EXCLUDE = {"custom_nodes", "models", "input", "output", "user", ".ci", ".github", ".git",
            "__pycache__", "tests", "docs", ".venv"}


def version_of(text):
    """"v0.44.1" or "ComfyUI 0.44" -> (0, 44, 1); anything without a dotted number -> None."""
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", str(text or ""))
    if not m:
        return None
    return tuple(int(x) for x in m.groups() if x is not None)


def installed_version(root):
    """Version of the core that lives in this portable folder, without asking it to run."""
    f = Path(root) / "ComfyUI" / "comfyui_version.py" if root else None
    try:
        return version_of(f.read_text(encoding="utf-8", errors="replace"))
    except (OSError, AttributeError):
        return None


async def gh_json(session, path):
    """A GET to the GitHub API; None when there is no internet or the API says no."""
    try:
        async with session.get(GH + path, timeout=aiohttp.ClientTimeout(total=12),
                               headers={"Accept": "application/vnd.github+json", "User-Agent": "h3"}) as r:
            return await r.json() if r.status == 200 else None
    except Exception:
        return None


async def check(session, comfy_root, installed_running=None, data_dir=None):
    """What is newer upstream: the core and every node pack the catalog knows."""
    out = {"core": {}, "packs": [], "at": time.strftime("%H:%M:%S")}
    mine = installed_running or installed_version(comfy_root) or catalog.COMFY_VERSION
    rel = await gh_json(session, "/repos/Comfy-Org/ComfyUI/releases/latest")
    latest = version_of((rel or {}).get("tag_name"))
    marks = sorted((Path(data_dir) / "backups").glob("comfyui-*.zip"), key=lambda p: p.stat().st_mtime) if data_dir else []
    last = marks[-1].name if marks else None
    prev = None
    if last:
        prev = version_of(last.replace("comfyui-", ""))
    out["core"] = {"installed": ".".join(map(str, mine or ())),
                   "backup": last, "backup_version": ".".join(map(str, prev or ())),
                   "latest": ".".join(map(str, latest or ())),
                   "newer": bool(latest and mine and latest > mine),
                   "url": (rel or {}).get("html_url") or "https://github.com/Comfy-Org/ComfyUI/releases",
                   "asset": portable_asset(rel)}
    for pack in catalog.NODE_PACKS:
        head = await gh_json(session, f"/repos/{pack['repo']}/commits?per_page=1")
        sha = ((head or [{}])[0]).get("sha") or ""
        when = (((head or [{}])[0]).get("commit") or {}).get("author", {}).get("date", "")[:10]
        folder = Path(comfy_root) / "ComfyUI" / "custom_nodes" if comfy_root else None
        olds = sorted(folder.glob(f'{pack["name"]}.old-*')) if folder else []
        applied = ""
        try:
            applied = ((folder / pack["name"] / ".h3-applied").read_text(encoding="utf-8").split() or [""])[0][:10]
        except (OSError, TypeError):
            pass
        out["packs"].append({"name": pack["name"], "repo": pack["repo"], "pinned": pack["commit"][:10], "applied": applied,
                             "latest": sha[:10], "newer": bool(sha) and sha[:10] != (applied or pack["commit"][:10]), "date": when,
                             "installed": bool(folder and (folder / pack["name"]).exists()),
                             "backup": olds[-1].name if olds else None})
    return out


def portable_asset(release):
    """Size of the whole portable package, so the person knows what a full re-install costs."""
    for a in (release or {}).get("assets", []):
        if "windows_portable" in (a.get("name") or ""):
            return int(a.get("size") or 0)
    return catalog.COMFY_SIZE


def source_url(version):
    return f"https://github.com/Comfy-Org/ComfyUI/archive/refs/tags/v{version}.tar.gz"


async def download(session, url, dest, on_progress=None, attempts=3):
    dest.parent.mkdir(parents=True, exist_ok=True)
    # a half-finished download of the same file may still be held open by the previous try, so
    # this one writes to its own name and only then takes the intended one
    tmp = dest.with_suffix(f'{dest.suffix}.{os.getpid()}.{int(time.time() * 1000)}.part')
    last = None
    for _ in range(attempts):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=1800),
                                   headers={"User-Agent": "h3"}) as r:
                if r.status != 200:
                    raise RuntimeError(f"Файл не скачался: сервер ответил {r.status}")
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                with tmp.open("wb") as f:
                    async for chunk in r.content.iter_chunked(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
                        if on_progress and (got >> 22) % 8 == 0:
                            await on_progress(got, total)
                for _ in range(4):
                    try:
                        tmp.replace(dest)
                        return dest
                    except OSError as e:              # Windows: the old archive is still busy
                        last = e
                        await asyncio.sleep(1.5)
                keep = dest.with_name(dest.stem + f'-{int(time.time())}' + dest.suffix)
                tmp.replace(keep)
                return keep
        except Exception as e:
            last = e
            await asyncio.sleep(3)
    raise RuntimeError(f"Не удалось скачать {url.split('/')[-1]}: {last}")


def backup_source(comfy_root, data_dir):
    """Zip the code we are about to overwrite - not models, not nodes, not user data.

    The version goes into the name: a person looking at the folder must see what a copy would
    bring back, and the settings page offers the rollback whenever such a copy exists - not only
    right after the update, when the button would vanish on the next check.
    """
    root = Path(comfy_root) / "ComfyUI"
    was = ".".join(map(str, installed_version(comfy_root) or ())) or "unknown"
    out = Path(data_dir) / "backups" / f"comfyui-{was}-{int(time.time())}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in root.rglob("*"):
            if not f.is_file() or any(part in _EXCLUDE for part in f.parts):
                continue
            z.write(f, f.relative_to(root).as_posix())
    for old in sorted(out.parent.glob("comfyui-*.zip"), key=lambda p: p.stat().st_mtime)[:-KEEP_BACKUPS]:
        old.unlink(missing_ok=True)
    return out


def apply_source(comfy_root, archive):
    """Unpack the release source over the core code, keeping what belongs to the person."""
    root = Path(comfy_root) / "ComfyUI"
    with tarfile.open(archive, "r:gz") as t:
        names = [m for m in t.getmembers() if m.isfile()]
        top = names[0].name.split("/")[0] if names else ""
        done = 0
        for m in names:
            rel = m.name[len(top) + 1:] if m.name.startswith(top + "/") else m.name
            if not rel or any(part in _EXCLUDE for part in rel.split("/")):
                continue
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with t.extractfile(m) as src, dest.open("wb") as out:
                shutil.copyfileobj(src, out)
            done += 1
        return done


def rollback(comfy_root, data_dir):
    """Put back the newest backup of the core code."""
    marks = sorted((Path(data_dir) / "backups").glob("comfyui-*.zip"), key=lambda p: p.stat().st_mtime)
    if not marks:
        raise RuntimeError("Резервных копий ядра нет")
    root = Path(comfy_root) / "ComfyUI"
    with zipfile.ZipFile(marks[-1]) as z:
        for n in z.namelist():
            dest = root / n
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(n))
    return marks[-1].name


def update_pack(comfy_root, pack, commit, archive=None):
    """Replace one node pack with another commit of its repository, keeping the old folder.

    `archive` is the file the download actually landed on: when the usual name is held by another
    program, it gets a name of its own, and the caller knows it, this function does not guess.
    """
    nodes = Path(comfy_root) / "ComfyUI" / "custom_nodes"
    target = nodes / pack["name"]
    archive = Path(archive) if archive else Path(comfy_root) / "downloads" / f'{pack["name"]}-{commit[:8]}.zip'
    z = None
    for _ in range(4):
        try:
            z = zipfile.ZipFile(archive)
            break
        except FileNotFoundError:
            raise RuntimeError("Архив ноды не скачался - попробуйте «Обновить ноду» ещё раз")
        except OSError:
            time.sleep(1.0)
    if z is None:
        raise RuntimeError("Файл архива занят другой программой: закройте ComfyUI и нажмите «Обновить ноду» ещё раз")
    with z:
        names = z.namelist()
        top = names[0].split("/")[0]
        if f"{top}/__init__.py" not in names:
            raise RuntimeError("в архиве нет __init__.py - это не папка нод")
        stage = nodes / f'.new-{pack["name"]}-{commit[:8]}'
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        z.extractall(stage)
        fresh = stage / top
        if not (fresh / "__init__.py").exists():
            shutil.rmtree(stage, ignore_errors=True)
            raise RuntimeError("распакованная папка без __init__.py")
        backup = None
        if target.exists():
            backup = nodes / f'{pack["name"]}.old-{time.strftime("%Y%m%d-%H%M%S")}'
            target.rename(backup)
        fresh.rename(target)
        shutil.rmtree(stage, ignore_errors=True)
        # a note inside the folder: which commit was put here by hand, so the settings can say it
        try:
            (target / ".h3-applied").write_text(commit + " " + time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
        except OSError:
            pass
        result = backup.name if backup else None
    # the downloaded archive and any half-finished file of the same pack are not needed anymore;
    # only after the zip is closed - Windows does not let an open file be deleted
    for left in sorted(archive.parent.glob(f'{pack["name"]}-*.zip*')):
        try:
            left.unlink()
        except OSError:
            pass
    return result


def rollback_pack(comfy_root, name):
    """Put back the newest .old-<time> copy of a node pack."""
    nodes = Path(comfy_root) / "ComfyUI" / "custom_nodes"
    olds = sorted(nodes.glob(f"{name}.old-*"), key=lambda p: p.name)
    if not olds:
        raise RuntimeError("Старой копии этой ноды нет")
    target = nodes / name
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    olds[-1].rename(target)
    (target / ".h3-applied").unlink(missing_ok=True)     # it is the pinned version again
    return olds[-1].name
