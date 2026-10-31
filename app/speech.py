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


# multilingual models only (".en" ones cannot do Russian), best first
MODEL_PREFERENCE = ["large-v3-turbo", "large-v3", "large-v2", "large", "medium", "small", "base", "tiny"]


def _find_exe(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return None
    hits = [p for p in folder.rglob("whisper-cli.exe") if len(p.relative_to(folder).parts) <= 5]
    return hits[0] if hits else None


def _find_model(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return None
    files = [p for p in folder.rglob("ggml-*.bin") if len(p.relative_to(folder).parts) <= 4
             and ".en" not in p.name and "silero" not in p.name.lower() and "vad" not in p.name.lower()
             and not p.name.endswith(("-encoder.bin",))]
    if not files:
        return None

    def rank(p):
        n = p.name.lower()
        return next((i for i, k in enumerate(MODEL_PREFERENCE) if k in n), len(MODEL_PREFERENCE)), p.stat().st_size
    return sorted(files, key=rank)[0]


def locate(root, custom_dir=None):
    """Where whisper is: the user's own folder first, then the app's own one. Returns a status dict."""
    own = Path(root) / "whisper"
    exe = (custom_dir and _find_exe(custom_dir)) or _find_exe(own / "bin")
    model = (custom_dir and _find_model(custom_dir)) or _find_model(own / "models")
    missing = ([] if exe else ["whisper-cli.exe"]) + ([] if model else ["модель ggml-*.bin"])
    return {"ready": not missing, "exe": exe, "model": model, "missing": missing, "custom": custom_dir or None}


def ready(root, custom_dir=None):
    return locate(root, custom_dir)["ready"]


async def install(installer, root, custom_dir=None):
    """Download whatever of whisper.cpp / its model is missing (the user's own files are never touched)."""
    import zipfile
    root = Path(root)
    found = locate(root, custom_dir)
    if not found["exe"]:
        zpath = root / "downloads" / f"whisper-{WHISPER_BUILD}.zip"
        await installer.download(WHISPER_URL, zpath, "agent", "распознавание речи (программа)", WHISPER_ZIP_SIZE)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(root / "whisper" / "bin")
        zpath.unlink(missing_ok=True)
    if not found["model"]:
        await installer.download(MODEL_URL, root / "whisper" / "models" / MODEL_FILE, "agent",
                                 f"распознавание речи: {MODEL_FILE}", MODEL_SIZE)


def _run(root, wav_bytes, lang, custom_dir=None):
    found = locate(root, custom_dir)
    exe = found["exe"]
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "in.wav"
        wav.write_bytes(wav_bytes)
        args = [str(exe), "-m", str(found["model"]), "-f", str(wav), "-l", lang or "auto", "-nt", "-t", "6"]
        r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=300, creationflags=engine.CREATE_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError("Не удалось распознать речь: " + (r.stderr.strip().splitlines() or ["ошибка"])[-1])
    m = re.search(r"auto-detected language:\s*([a-z]{2,3})", r.stderr)
    code = m.group(1) if m else (lang if lang and lang != "auto" else "")
    text = " ".join(line.strip() for line in r.stdout.splitlines() if line.strip())
    text = re.sub(r"\s+", " ", NOISE.sub("", text)).strip()
    return text, code


async def transcribe(root, wav_bytes, lang="auto", custom_dir=None):
    """Returns (text, language_code, language_name_for_prompt)."""
    text, code = await asyncio.to_thread(_run, root, wav_bytes, lang, custom_dir)
    return text, code, LANG_NAMES.get(code, code.upper() if code else "")
