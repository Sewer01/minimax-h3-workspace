"""Updates from GitHub Releases: look for a newer release, download its installer, check it, run it.

The release must carry one file named Minimax.H3.Workspace-Setup.exe (GitHub shows its sha256 next to it).
"""

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import aiohttp

from version import APP_VERSION, UPDATE_REPO

ASSET = "Minimax.H3.Workspace-Setup.exe"
HEADERS = {"User-Agent": "Minimax-H3-Workspace", "Accept": "application/vnd.github+json"}


def repo():
    """Whose GitHub releases this program follows.

    This build lives in a fork, so updates come from the fork and not from the original author.
    A file update_repo.txt next to the program (one line, owner/name) or the variable H3_UPDATE_REPO
    change it without a rebuild - handy to look at a release before publishing it.
    """
    env = (os.environ.get("H3_UPDATE_REPO") or "").strip()
    if env:
        return env
    here = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
    for f in (here / "update_repo.txt", here.parent / "update_repo.txt"):
        try:
            line = f.read_text(encoding="utf-8-sig").strip().splitlines()[0].strip()
        except (OSError, IndexError):
            continue
        if re.fullmatch(r"[\w.-]+/[\w.-]+", line):
            return line
    return UPDATE_REPO


def api_url():
    return f"https://api.github.com/repos/{repo()}/releases/latest"


def download_prefix():
    return f"https://github.com/{repo()}/releases/download/"


def parse(v):
    """'v1.2.3' -> (1, 2, 3); anything unreadable -> None."""
    m = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?", str(v or "").strip())
    return tuple(int(x or 0) for x in m.groups()) if m else None


async def check(session):
    """{"available": bool, "current", "version", "url", "asset", "size", "sha256"}; never raises (no internet is normal)."""
    out = {"available": False, "current": APP_VERSION}
    try:
        async with session.get(api_url(), headers=HEADERS, timeout=aiohttp.ClientTimeout(total=8)) as r:
            if r.status != 200:
                return out
            rel = await r.json()
    except Exception:
        return out
    new, cur = parse(rel.get("tag_name")), parse(APP_VERSION)
    asset = next((a for a in rel.get("assets", []) if a.get("name") == ASSET), None)
    if not new or not cur or not asset or rel.get("draft") or rel.get("prerelease"):
        return out
    url = asset.get("browser_download_url", "")
    if new > cur and url.startswith(download_prefix()):
        digest = str(asset.get("digest") or "")
        out.update(available=True, version=".".join(map(str, new)), url=rel.get("html_url", ""), asset=url,
                   size=int(asset.get("size") or 0), sha256=digest.split(":", 1)[1] if digest.startswith("sha256:") else "")
    return out


async def download(session, info):
    """Save the installer to a temp folder and check it against the size and sha256 GitHub published. Returns its path."""
    if not info.get("asset", "").startswith(download_prefix()):
        raise ValueError("Адрес обновления не с GitHub")
    want = str(info.get("sha256") or "")
    if not want:
        # No checksum from GitHub (an API change, a proxy that rewrote the answer). The installer runs with the
        # rights of this program and replaces its files, so an unverified one is not worth the risk.
        raise ValueError("GitHub не прислал контрольную сумму обновления, поэтому программа не ставит его сама. "
                         "Скачайте установщик со страницы релизов и запустите его вручную.")
    folder = Path(tempfile.gettempdir()) / "h3studio_update"
    folder.mkdir(exist_ok=True)
    dest = folder / ASSET
    sha = hashlib.sha256()
    size = 0
    async with session.get(info["asset"], headers=HEADERS, timeout=aiohttp.ClientTimeout(total=900)) as r:
        if r.status != 200:
            raise ValueError(f"GitHub ответил {r.status}")
        with open(dest, "wb") as f:
            async for chunk in r.content.iter_chunked(1 << 20):
                f.write(chunk)
                sha.update(chunk)
                size += len(chunk)
    if size != int(info.get("size") or size) or sha.hexdigest() != want.lower():
        dest.unlink(missing_ok=True)
        raise ValueError("Файл обновления скачался с ошибкой или не совпал с контрольной суммой, попробуйте ещё раз")
    return dest


def run(installer):
    """Start the installer detached and quietly; it closes this app, replaces the files and opens the app again."""
    flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    subprocess.Popen([str(installer), "/SILENT", "/SUPPRESSMSGBOXES", "/CLOSEAPPLICATIONS", "/NORESTART"],
                     creationflags=flags, close_fds=True)


def can_update():
    """Only an installed build updates itself: a run from source or the maker's folder does not."""
    return bool(getattr(sys, "frozen", False))
