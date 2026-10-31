"""Speech to text for the agent: lyrics from a lipsync audio and voice input from the microphone.

Uses whisper.cpp (CPU build, ~10 MB) with the multilingual "small" model (~190 MB). Both files are
downloaded by the app next to the agent's model; nothing is installed system-wide.
"""

import asyncio
import re
import subprocess
import tempfile
from pathlib import Path

import engine

WHISPER_BUILD = "b5454"
WHISPER_URL = f"https://github.com/ggml-org/whisper.cpp/releases/download/{WHISPER_BUILD}/whisper-bin-x64.zip"
WHISPER_ZIP_SIZE = 9e6
MODEL_FILE = "ggml-small-q5_1.bin"
MODEL_URL = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{MODEL_FILE}"
MODEL_SIZE = 190e6

LANG_NAMES = {"ru": "Russian", "en": "English", "uk": "Ukrainian", "de": "German", "fr": "French", "es": "Spanish",
              "it": "Italian", "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
              "tr": "Turkish", "pl": "Polish", "ar": "Arabic", "hi": "Hindi"}
NOISE = re.compile(r"[\[\(][^\]\)]{0,40}[\]\)]")  # [BLANK_AUDIO], (music), [Music] ...


def cli(root):
    base = Path(root) / "whisper" / "bin"
    hits = list(base.rglob("whisper-cli.exe")) if base.exists() else []
    return hits[0] if hits else None


def model_path(root):
    return Path(root) / "whisper" / "models" / MODEL_FILE


def ready(root):
    return bool(cli(root)) and model_path(root).exists()


async def install(installer, root):
    """Download whisper.cpp and the model unless they are already there."""
    import zipfile
    root = Path(root)
    if not cli(root):
        zpath = root / "downloads" / f"whisper-{WHISPER_BUILD}.zip"
        await installer.download(WHISPER_URL, zpath, "agent", "распознавание речи (программа)", WHISPER_ZIP_SIZE)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root / "whisper" / "bin")
        zpath.unlink(missing_ok=True)
    await installer.download(MODEL_URL, model_path(root), "agent", f"распознавание речи: {MODEL_FILE}", MODEL_SIZE)


def _run(root, wav_bytes, lang):
    exe = cli(root)
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "in.wav"
        wav.write_bytes(wav_bytes)
        args = [str(exe), "-m", str(model_path(root)), "-f", str(wav), "-l", lang or "auto", "-nt", "-t", "6"]
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=300, creationflags=engine.CREATE_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError("Не удалось распознать речь: " + (r.stderr.strip().splitlines() or ["ошибка"])[-1])
    m = re.search(r"auto-detected language:\s*([a-z]{2,3})", r.stderr)
    code = m.group(1) if m else (lang if lang and lang != "auto" else "")
    text = " ".join(line.strip() for line in r.stdout.splitlines() if line.strip())
    text = re.sub(r"\s+", " ", NOISE.sub("", text)).strip()
    return text, code


async def transcribe(root, wav_bytes, lang="auto"):
    """Returns (text, language_code, language_name_for_prompt)."""
    text, code = await asyncio.to_thread(_run, root, wav_bytes, lang)
    return text, code, LANG_NAMES.get(code, code.upper() if code else "")
