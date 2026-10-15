#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""say.py - 配音台. Text in, pick an actor, mp3 out. Standalone.

.\say --list                                  the actors in voices/
    .\say -v narrator_male "崇祯六年秋，苏州城厢，日头偏了。"
    .\say -v 3 -f chapter01.txt -o ch01.mp3       actor by number, text from a file
    .\say "天色渐暗。"                             no -v: pick from a numbered list
    .\say --audition "一句试听的话。"              the same line in EVERY voice
    .\say --web                                   a one-page form on http://127.0.0.1:9110

(`.\say` is say.bat in this folder; on Linux/macOS use `venv/bin/python say.py`.)

An actor is a PAIR of files in voices/:  <name>.wav  (3-10 s of clean speech,
one speaker, no music) and <name>.txt (its exact transcript). CosyVoice is
zero-shot: there is no speaker table, the clip IS the voice. Optional
<name>.rhythm holds the name of another voice whose DELIVERY this one
borrows (spoken in that voice first, then converted to this timbre).

Engine. models/ holds the CosyVoice weights; the first of cosyvoice3-rl,
cosyvoice3, cosyvoice2 that exists is used (or MODEL=<dir name> / --model).
Runs on the GPU when torch sees one, --device cpu otherwise. With
--server http://host:9100 the tool talks to a running tts service instead of
loading anything (the Bookreel tts_server contract).

Text. Anything longer than one breath is cut at sentence ends into pieces of
at most MAX_CHUNK characters, each spoken on its own and joined with a short
pause; a blank line is a longer pause. CosyVoice drifts on very long inputs,
so the cut is not optional. Numbers are its weak spot: write 三钱, not 3钱.

Output. mp3 at 160 kb/s through ffmpeg (PATH, or the imageio-ffmpeg build in
the venv), or --format wav for the raw 24 kHz pcm with no ffmpeg at all.
Default file: output/<voice>-<time>.mp3.
"""

import argparse
import datetime
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave

ROOT = os.path.dirname(os.path.abspath(__file__))
VOICES = os.path.join(ROOT, "voices")
MODELS = os.path.join(ROOT, "models")
OUT_DIR = os.path.join(ROOT, "output")
SAMPLE_TEXT = os.environ.get("SAY_SAMPLE", "崇祯六年秋，苏州城厢，日头偏了。")

MAX_CHUNK = 180          # characters per synthesis call
GAP_SENTENCE = 0.18      # seconds of silence between pieces
GAP_PARAGRAPH = 0.55     # ... and between paragraphs (a blank line)
BITRATE = "160k"
MODEL_ORDER = ("cosyvoice3-rl", "cosyvoice3", "cosyvoice2")


# ----------------------------------------------------------------- voices --

def list_voices():
    """[{name, text, work, rhythm, seconds}] for every .wav + .txt pair in voices/.

    Names starting with "_" are scratch, not actors. `work` is an optional
    tag from <name>.json {"work": "..."}; `rhythm` the voice this one borrows
    its delivery from.
    """
    out = []
    if not os.path.isdir(VOICES):
        return out
    for fn in sorted(os.listdir(VOICES)):
        if not fn.lower().endswith(".wav") or fn.startswith("_"):
            continue
        name = fn[:-4]
        txt = os.path.join(VOICES, name + ".txt")
        if not os.path.isfile(txt):
            continue
        try:
            with open(txt, "r", encoding="utf-8-sig") as fh:
                text = fh.read().strip()
        except OSError:
            text = ""
        work, rhythm = "", ""
        try:
            with open(os.path.join(VOICES, name + ".json"), "r", encoding="utf-8") as fh:
                work = str((json.load(fh) or {}).get("work") or "")
        except (OSError, ValueError):
            pass
        try:
            with open(os.path.join(VOICES, name + ".rhythm"), "r", encoding="utf-8") as fh:
                rhythm = fh.read().strip().splitlines()[0].strip()
        except (OSError, IndexError):
            pass
        out.append({"name": name, "text": text, "work": work, "rhythm": rhythm,
                    "seconds": _wav_seconds(os.path.join(VOICES, fn))})
    return out


def voice_files(name):
    return os.path.join(VOICES, name + ".wav"), os.path.join(VOICES, name + ".txt")


def _wav_seconds(path):
    """Length of a wav off its header. Clips are often 32-bit float (format
    3), which the wave module refuses, so walk the RIFF chunks by hand."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
            if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return 0.0
            bps, pos, size = 0, 12, os.path.getsize(path)
            while pos + 8 <= size:
                fh.seek(pos)
                cid, n = fh.read(4), int.from_bytes(fh.read(4), "little")
                if cid == b"fmt ":
                    f = fh.read(16)
                    bps = int.from_bytes(f[8:12], "little")      # avg bytes/sec
                elif cid == b"data":
                    n = min(n, size - pos - 8)
                    return n / float(bps) if bps else 0.0
                pos += 8 + n + (n & 1)
    except OSError:
        pass
    return 0.0


def print_voices(voices):
    if not voices:
        print("voices/ 里没有可用的声音（需要 name.wav + name.txt 成对）")
        return
    import unicodedata

    def width(s):                       # CJK characters take two columns
        return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    w = max(width(v["name"]) for v in voices)
    for i, v in enumerate(voices, 1):
        tag = []
        if v["work"]:
            tag.append(v["work"])
        if v["rhythm"]:
            tag.append("节奏 " + v["rhythm"])
        line = "%3d  %s%s  %4.1fs  %s" % (i, v["name"], " " * (w - width(v["name"])),
                                          v["seconds"], v["text"][:36])
        if tag:
            line += "  [" + " · ".join(tag) + "]"
        print(line)


def pick_voice(voices, want):
    """A voice by name or 1-based number; None when it does not resolve."""
    if not want:
        return None
    if want.isdigit():
        i = int(want)
        return voices[i - 1] if 1 <= i <= len(voices) else None
    return next((v for v in voices if v["name"] == want), None)


def ask_voice(voices):
    """Interactive pick when -v was not given and there is a terminal."""
    if not sys.stdin.isatty():
        return None
    print_voices(voices)
    while True:
        try:
            s = input("选哪个（编号或名字，回车取消）: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not s:
            return None
        v = pick_voice(voices, s)
        if v:
            return v
        print("没有这个声音")


# ------------------------------------------------------------------- text --

_ENDERS = "。！？!?；;…"


def split_text(text, limit=MAX_CHUNK):
    """[(piece, gap_after)] - sentences packed to `limit`, paragraphs marked.

    A paragraph is text between blank lines (a single newline is just a wrap).
    Within one, cut after sentence enders, then pack neighbours while they fit;
    a single sentence longer than `limit` is cut at commas, and as a last
    resort at `limit` characters flat.
    """
    out = []
    paras = [p for p in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if p.strip()]
    for pi, para in enumerate(paras):
        flat = " ".join(para.split())
        flat = re.sub(r"\s+([，。！？；：、”’）])", r"\1", flat)
        sents = [s for s in re.findall(r"[^%s]+[%s]*[”’）]*" % (_ENDERS, _ENDERS), flat) if s.strip()]
        pieces, cur = [], ""
        for s in sents:
            s = s.strip()
            if len(s) > limit:
                if cur:
                    pieces.append(cur)
                    cur = ""
                pieces.extend(_hard_cut(s, limit))
                continue
            if cur and len(cur) + len(s) > limit:
                pieces.append(cur)
                cur = s
            else:
                cur += s
        if cur:
            pieces.append(cur)
        for i, p in enumerate(pieces):
            last_in_para = i == len(pieces) - 1
            gap = (GAP_PARAGRAPH if pi < len(paras) - 1 else 0.0) if last_in_para else GAP_SENTENCE
            out.append((p, gap))
    return out


def _hard_cut(s, limit):
    parts, cur = [], ""
    for frag in re.findall(r"[^，,、：:]+[，,、：:]?", s):
        if cur and len(cur) + len(frag) > limit:
            parts.append(cur)
            cur = ""
        cur += frag
        while len(cur) > limit:
            parts.append(cur[:limit])
            cur = cur[limit:]
    if cur:
        parts.append(cur)
    return parts


def read_text(args):
    """-f FILE, -f - (stdin), or the words on the command line. stdin is read
    only when asked for: a tool that reads it whenever it is "not a terminal"
    hangs under any launcher that leaves the pipe open."""
    if args.text_file == "-":
        return sys.stdin.read()
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8-sig") as fh:
            return fh.read()
    if args.text:
        return " ".join(args.text) if len(args.text) > 1 else args.text[0]
    return ""


# ------------------------------------------------------------- the engine --

def _to_wav_bytes(tensor, sr):
    """A float tensor in [-1, 1] -> 16-bit mono pcm wav bytes."""
    import numpy as np
    arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    arr = np.clip(arr.reshape(-1), -1.0, 1.0)
    pcm = (arr * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class LocalEngine:
    """CosyVoice loaded in this process. synth(text, voice, speed, instruct) -> wav bytes.

    Generation matters in two places, both learned the hard way on the film:
    CosyVoice3 wants the reference transcript behind a system line
    (`You are a helpful assistant.<|endofprompt|>` + transcript) and asserts
    that marker is present on the instruct path too; CosyVoice2 must NOT be
    given either - it reads the line aloud as words to speak.
    """

    def __init__(self, model_dir, device="auto"):
        if device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.model_dir = model_dir
        self.gen = 3 if os.path.isfile(os.path.join(model_dir, "cosyvoice3.yaml")) else 2
        self.name = "cosyvoice%d" % self.gen
        self.prompt_form = None
        for cand in (os.path.join(ROOT, "third_party", "CosyVoice"),):
            if os.path.isdir(cand):
                sys.path.insert(0, cand)
                matcha = os.path.join(cand, "third_party", "Matcha-TTS")
                if os.path.isdir(matcha):
                    sys.path.insert(0, matcha)
        try:
            from cosyvoice.cli.cosyvoice import AutoModel as loader
        except ImportError:
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2 as loader
            except ImportError as exc:
                raise SystemExit("CosyVoice 导入不了（third_party/CosyVoice 没有，或依赖没装）：%s\n"
                                 "先跑 setup.ps1" % exc)
        # The inference loop wraps its sentence list in tqdm; one progress bar
        # per chunk is noise next to our own per-chunk line.
        import cosyvoice.cli.cosyvoice as _cv
        _cv.tqdm = lambda it, *a, **k: it
        import logging
        import warnings
        os.environ.setdefault("TQDM_DISABLE", "1")           # one bar per chunk is noise
        os.environ.setdefault("MODELSCOPE_LOG_LEVEL", "40")   # wetext's "Downloading 42 files"
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        # Before the cosyvoice import: its file_utils calls basicConfig(DEBUG)
        # at import time, and basicConfig is a no-op once the root logger has
        # a handler - so claim it first at WARNING. torch's flop counter
        # (triton not found) and transformers (sliding window attention) warn
        # about things that are true and harmless here.
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        logging.getLogger("torch").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("modelscope").setLevel(logging.ERROR)
        logging.getLogger("modelscope_hub").setLevel(logging.ERROR)
        # wetext re-checks its 42 fst files against modelscope on every
        # engine start (instant once cached) and announces it with a bar and
        # an INFO line on its own logger. Both go.
        import tqdm as _tqdm
        _tqdm_init = _tqdm.tqdm.__init__

        def _quiet_init(self, *a, **k):
            k["disable"] = True
            _tqdm_init(self, *a, **k)
        _tqdm.tqdm.__init__ = _quiet_init
        import torch
        import torchaudio
        # torchaudio 2.9+ no longer decodes files itself: torchaudio.load
        # wants torchcodec, which has no clean Windows install. CosyVoice
        # reads every reference clip through torchaudio.load(wav,
        # backend='soundfile'), so give it one that is soundfile.
        if not getattr(torchaudio, "_say_patched", False):
            import soundfile

            def _load(path, *a, **k):
                data, sr = soundfile.read(path, dtype="float32", always_2d=True)
                return torch.from_numpy(data.T.copy()), sr
            torchaudio.load = _load
            torchaudio._say_patched = True
        # cosyvoice.utils.file_utils calls logging.basicConfig(DEBUG) at import,
        # which then prints every modelscope lock and HTTP line.
        logging.getLogger().setLevel(logging.WARNING)
        t0 = time.time()
        where = "cuda:" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        print("[say] 加载 %s（%s）…" % (self.name, where), flush=True)
        try:
            self.engine = loader(model_dir=model_dir, fp16=False)
        except TypeError:
            self.engine = loader(model_dir)
        self.sr = int(getattr(self.engine, "sample_rate", 24000))
        print("[say] %s 就绪，%.0fs，%d Hz" % (self.name, time.time() - t0, self.sr), flush=True)

    def _prompt_text(self, text):
        return "You are a helpful assistant.<|endofprompt|>" + text if self.gen == 3 else text

    def _instruct_text(self, text):
        if self.gen == 3 and "<|endofprompt|>" not in text:
            return text + "<|endofprompt|>"
        return text

    def _prompt_forms(self, wav_path):
        """The reference clip as a path (current checkouts) or a 16 k tensor
        (older ones); whichever works first is remembered."""
        def tensor():
            from cosyvoice.utils.file_utils import load_wav
            return load_wav(wav_path, 16000)
        if self.prompt_form == "path":
            return [wav_path]
        if self.prompt_form == "tensor":
            return [tensor()]
        return [wav_path, tensor()]

    def _rhythm_of(self, name):
        p = os.path.join(VOICES, name + ".rhythm")
        try:
            with open(p, "r", encoding="utf-8") as fh:
                src = fh.read().strip().splitlines()[0].strip()
        except (OSError, IndexError):
            return ""
        if not src or src == name:
            return ""
        w, t = voice_files(src)
        return src if (os.path.isfile(w) and os.path.isfile(t)) else ""

    def _run(self, gen):
        import torch
        chunks = [piece["tts_speech"] for piece in gen]
        if not chunks:
            raise RuntimeError("模型没有返回音频")
        return torch.cat(chunks, dim=-1) if len(chunks) > 1 else chunks[0]

    def synth(self, text, voice, speed=1.0, instruct=""):
        wav_path, txt_path = voice_files(voice)
        with open(txt_path, "r", encoding="utf-8-sig") as fh:
            prompt_text = fh.read().strip()
        rhythm = self._rhythm_of(voice)
        if rhythm and not instruct:
            # Spoken in the rhythm voice first (that fixes the cadence), then
            # converted to this voice's timbre. Speed belongs to the first pass.
            rw, rt = voice_files(rhythm)
            with open(rt, "r", encoding="utf-8-sig") as fh:
                rtext = fh.read().strip()
            mid = self._run(self.engine.inference_zero_shot(
                text, self._prompt_text(rtext), rw, stream=False, speed=speed))
            fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="say-vc-")
            os.close(fd)
            try:
                with open(tmp, "wb") as fh:
                    fh.write(_to_wav_bytes(mid, self.sr))
                audio = self._run(self.engine.inference_vc(tmp, wav_path, stream=False, speed=1.0))
            finally:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            return _to_wav_bytes(audio, self.sr)
        last = None
        for prompt_wav in self._prompt_forms(wav_path):
            try:
                if instruct:
                    gen = self.engine.inference_instruct2(
                        text, self._instruct_text(instruct), prompt_wav, stream=False, speed=speed)
                else:
                    gen = self.engine.inference_zero_shot(
                        text, self._prompt_text(prompt_text), prompt_wav, stream=False, speed=speed)
                audio = self._run(gen)
                self.prompt_form = "path" if prompt_wav is wav_path else "tensor"
                return _to_wav_bytes(audio, self.sr)
            except Exception as exc:                     # noqa: BLE001
                last = exc
        raise RuntimeError("合成失败：%s" % last)


class ServerEngine:
    """A running tts service (POST /tts {text, voice, speed, format, instruct})."""

    def __init__(self, server):
        self.server = server.rstrip("/")
        try:
            with urllib.request.urlopen(self.server + "/health", timeout=5) as r:
                h = json.load(r) or {}
        except Exception as exc:                         # noqa: BLE001
            raise SystemExit("语音服务 %s 没有回应：%s" % (self.server, exc))
        if h.get("status") != "ok":
            raise SystemExit("语音服务 %s 不正常：%s" % (self.server, h))
        self.name = h.get("engine", "server")

    def synth(self, text, voice, speed=1.0, instruct=""):
        body = {"text": text, "voice": voice, "speed": speed, "format": "wav"}
        if instruct:
            body["instruct"] = instruct
        req = urllib.request.Request(
            self.server + "/tts", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            raise RuntimeError("HTTP %d: %s" % (exc.code, exc.read().decode("utf-8", "replace")[:300]))


def find_model(want):
    if want:
        p = want if os.path.isdir(want) else os.path.join(MODELS, want)
        if not os.path.isdir(p):
            raise SystemExit("没有这个模型目录：%s" % want)
        return p
    for c in MODEL_ORDER:
        p = os.path.join(MODELS, c)
        if os.path.isdir(p):
            return p
    raise SystemExit("models/ 里没有 CosyVoice 权重（cosyvoice3-rl / cosyvoice3 / cosyvoice2）。先跑 setup.ps1")


def make_engine(args):
    if args.server:
        return ServerEngine(args.server)
    return LocalEngine(find_model(args.model), args.device)


# ------------------------------------------------------------------ audio --

def wav_pcm(data):
    """(pcm16 bytes, sample rate) from wav bytes; mono 16-bit expected."""
    with wave.open(io.BytesIO(data), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise RuntimeError("不是 16 位单声道 wav")
        return w.readframes(w.getnframes()), w.getframerate()


def join_pcm(parts, sr):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for pcm, gap in parts:
            w.writeframes(pcm)
            if gap > 0:
                w.writeframes(b"\x00\x00" * int(sr * gap))
    return buf.getvalue()


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for cand in (os.path.join(ROOT, "bin", "ffmpeg.exe"), os.path.join(ROOT, "bin", "ffmpeg")):
        if os.path.isfile(cand):
            return cand
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                   # noqa: BLE001
        return ""


def write_out(wav_bytes, out_path, fmt):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if fmt == "wav":
        with open(out_path, "wb") as fh:
            fh.write(wav_bytes)
        return
    exe = find_ffmpeg()
    if not exe:
        raise SystemExit("没有 ffmpeg，出不了 mp3：pip install imageio-ffmpeg，或 --format wav")
    # <name>.part.mp3, never <name>.mp3.part: ffmpeg picks the muxer off the extension.
    base, _ = os.path.splitext(out_path)
    part = base + ".part.mp3"
    pr = subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", "-f", "wav", "-i", "pipe:0",
                         "-c:a", "libmp3lame", "-b:a", BITRATE, part],
                        input=wav_bytes, capture_output=True, timeout=600)
    if pr.returncode != 0:
        raise SystemExit("ffmpeg：%s" % pr.stderr.decode("utf-8", "replace").strip()[-300:])
    os.replace(part, out_path)


# ------------------------------------------------------------ add a voice --

def _media_seconds(path):
    exe = find_ffmpeg()
    if not exe:
        return 0.0
    pr = subprocess.run([exe, "-hide_banner", "-i", path], capture_output=True, timeout=120)
    m = re.search(rb"Duration: (\d+):(\d\d):(\d\d(?:\.\d+)?)", pr.stderr)
    return (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) if m else 0.0


def transcribe(wav_path, model_name):
    """The clip's transcript via openai-whisper (installed with CosyVoice).
    Returns (text, language). The person must still read it against the
    clip: a wrong transcript gives a wrong voice."""
    try:
        import whisper
    except ImportError:
        raise SystemExit("没有 whisper，转写不了：给 --text 手写转写，或 pip install openai-whisper")
    import torch
    import soundfile
    from scipy.signal import resample_poly
    print("[say] whisper %s 转写中…（首次会下载模型）" % model_name, flush=True)
    model = whisper.load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    # Hand whisper the samples, not the path: given a path it shells out to
    # an `ffmpeg` on PATH, which this venv does not have (imageio's build
    # has a different file name). Our clips are 24 kHz mono; whisper wants 16 kHz.
    data, sr = soundfile.read(wav_path, dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if sr != 16000:
        from math import gcd
        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype("float32")
    res = model.transcribe(audio, fp16=torch.cuda.is_available())
    text = " ".join(str(res.get("text") or "").split())
    return text, str(res.get("language") or "")


def add_voice(src, name, start, end, text, work, norm, whisper_model):
    """A recording -> voices/<name>.wav (+ .txt, .json). Any format ffmpeg
    reads; -ss/-to cut a range out of a longer file. Mono 24 kHz 16-bit,
    loudness-normalised unless --raw, with a short fade at both ends."""
    if not os.path.isfile(src):
        raise SystemExit("没有这个文件：%s" % src)
    exe = find_ffmpeg()
    if not exe:
        raise SystemExit("没有 ffmpeg：pip install imageio-ffmpeg")
    name = re.sub(r"[^A-Za-z0-9_\-一-鿿]+", "_", name or "").strip("_")[:40]
    if not name:
        raise SystemExit("--name 不能为空（字母、数字、_、- 或汉字）")
    wav, txt = voice_files(name)
    if os.path.isfile(wav):
        raise SystemExit("voices/%s.wav 已经有了，换个名字或先删掉它" % name)
    os.makedirs(VOICES, exist_ok=True)
    args = [exe, "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        args += ["-ss", "%.3f" % start]
    if end is not None:
        args += ["-to", "%.3f" % end]
    af = ("loudnorm=I=-20:TP=-2:LRA=11," if norm else "") + \
         "afade=t=in:d=0.02,areverse,afade=t=in:d=0.05,areverse"
    args += ["-i", src, "-vn", "-af", af, "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", wav]
    pr = subprocess.run(args, capture_output=True, timeout=600)
    if pr.returncode != 0:
        raise SystemExit("ffmpeg 读不了：%s" % pr.stderr.decode("utf-8", "replace").strip()[-300:])
    secs = _wav_seconds(wav)
    if secs < 1.0:
        os.unlink(wav)
        raise SystemExit("切出来不到 1 秒，检查 --start/--end")
    if secs > 30:
        os.unlink(wav)
        raise SystemExit("%.1f 秒太长：参考音要 3-10 秒，用 --start/--end 切一段" % secs)
    lang = ""
    if not text:
        try:
            text, lang = transcribe(wav, whisper_model)
        except BaseException:
            os.unlink(wav)              # no half-made actor (a .wav with no .txt)
            raise
    if not text.strip():
        os.unlink(wav)
        raise SystemExit("转写是空的（没听出话来）：给 --text 手写转写")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(text.strip() + "\n")
    if work:
        with open(os.path.join(VOICES, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"work": work}, fh, ensure_ascii=False)
    return {"name": name, "seconds": secs, "text": text.strip(), "lang": lang, "work": work}


def report_added(r):
    print("[say] 加入演员 %s  %.1fs%s"
          % (r["name"], r["seconds"], "  （长度建议 3-10 秒）" if not 3 <= r["seconds"] <= 10 else ""))
    print("[say]   转写%s：%s" % ((" (whisper, %s)" % r["lang"]) if r["lang"] else "", r["text"]))
    if r["lang"]:
        print("[say]   对着音频核一遍 voices/%s.txt - 转写错一个字，声音就错一点。" % r["name"])


def set_voice_text(name, text):
    """Overwrite an actor's transcript (after the person has checked it)."""
    wav, txt = voice_files(name)
    if not os.path.isfile(wav):
        raise SystemExit("没有这个演员：%s" % name)
    text = " ".join(text.split())
    if not text:
        raise SystemExit("转写不能为空")
    with open(txt, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def trash_voice(name):
    """Deleting is moving: the pair goes to voices/_trash/<name>-<time>.*"""
    wav, txt = voice_files(name)
    if not os.path.isfile(wav):
        raise SystemExit("没有这个演员：%s" % name)
    tdir = os.path.join(VOICES, "_trash")
    os.makedirs(tdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    moved = []
    for ext in (".wav", ".txt", ".json", ".rhythm"):
        p = os.path.join(VOICES, name + ext)
        if os.path.isfile(p):
            os.replace(p, os.path.join(tdir, "%s-%s%s" % (name, stamp, ext)))
            moved.append(ext)
    aud = os.path.join(OUT_DIR, "_audition", name + ".mp3")
    if os.path.isfile(aud):
        os.unlink(aud)
    return moved


# --------------------------------------------------------------- the work --

def speak(engine, text, voice, speed, instruct, out_path, fmt, quiet=False, progress=None):
    """Synthesize `text` in `voice` to out_path. Returns seconds of audio.

    `progress(done, total, seconds_so_far, stage)` is called before the first
    piece, after every piece, and once more while the file is being written;
    the web form draws its bar from it.
    """
    pieces = split_text(text)
    if not pieces:
        raise SystemExit("没有要念的文字")
    n = len(pieces)
    parts, sr, t0, got = [], 24000, time.time(), 0.0
    if progress:
        progress(0, n, 0.0, "合成")
    for i, (piece, gap) in enumerate(pieces, 1):
        t1 = time.time()
        try:
            pcm, sr = wav_pcm(engine.synth(piece, voice, speed, instruct))
        except Exception as exc:                         # noqa: BLE001
            raise SystemExit("第 %d/%d 段出不了声（%s）：%s\n    %s" % (i, n, voice, exc, piece[:60]))
        if not pcm:
            raise SystemExit("第 %d/%d 段返回了空音频（%s）：%s" % (i, n, voice, piece[:60]))
        parts.append((pcm, gap))
        got += len(pcm) / (2.0 * sr) + gap
        if progress:
            progress(i, n, got, "合成")
        if not quiet:
            print("[say] %s %d/%d  %d 字 → %.1fs 音频，用时 %.1fs"
                  % (voice, i, n, len(piece), len(pcm) / (2.0 * sr), time.time() - t1), flush=True)
    if progress:
        progress(n, n, got, "写 " + fmt)
    wav_bytes = join_pcm(parts, sr)
    secs = (len(wav_bytes) - 44) / (2.0 * sr)
    write_out(wav_bytes, out_path, fmt)
    if not quiet:
        print("[say] %s  %.1fs 音频，合成用时 %.0fs" % (out_path, secs, time.time() - t0), flush=True)
    return secs


def default_out(voice, fmt):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(OUT_DIR, "%s-%s.%s" % (voice, stamp, fmt))


# ------------------------------------------------------------------- web --

_PAGE = """<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>配音台</title>
<style>
:root{--bg:#f6f1e7;--fg:#2b2620;--line:#d9cfbd;--acc:#8a5a2b;--card:#fffdf8}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#1d1a16;--fg:#eae2d3;--line:#3b342b;--acc:#d2a06a;--card:#26221c}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 "Noto Serif SC",Georgia,serif}
main{max-width:760px;margin:0 auto;padding:28px 16px 60px}
h1{font-weight:500;font-size:24px;margin:0 0 18px}
label{display:block;font-size:12px;letter-spacing:.1em;opacity:.7;margin:14px 0 4px}
textarea,select,input{width:100%;box-sizing:border-box;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:9px 10px;font:inherit}
textarea{min-height:170px;resize:vertical}
.row{display:flex;gap:14px}.row>div{flex:1}
button{margin-top:16px;background:var(--acc);color:#fff;border:0;border-radius:6px;padding:10px 22px;font:inherit;cursor:pointer}
button[disabled]{opacity:.5;cursor:wait}
.aud{background:none;color:var(--acc);padding:4px 8px;margin:0 0 0 8px;border:1px solid var(--line)}
#status{margin-top:14px;min-height:1.6em;opacity:.8}
audio{width:100%;margin-top:12px}
.small{font-size:12.5px;opacity:.65}
</style></head><body><main>
<h1>配音台</h1>
<label>台词 / 旁白</label>
<textarea id="text" placeholder="长文会按句切开逐段合成，空行是一个较长的停顿。数字请写成汉字。"></textarea>
<div class="row"><div>
<label>演员</label>
<div style="display:flex;align-items:center"><select id="voice"></select><button class="aud" id="aud" type="button" title="听一句">试听</button></div>
<div class="small" id="vinfo"></div>
</div><div style="flex:0 0 120px">
<label>语速</label><input id="speed" type="number" min="0.5" max="2" step="0.05" value="1.0">
</div></div>
<label>念法（可选，如：请用低沉、克制、缓慢的语气讲述）</label>
<input id="instruct" placeholder="留空按参考音的原样念">
<button id="go" type="button">合成 mp3</button>
<div id="prog" style="display:none;margin-top:14px">
<div style="height:10px;border:1px solid var(--line);border-radius:5px;background:var(--card);overflow:hidden"><div id="pbar" style="height:100%;width:0%;background:var(--acc);transition:width .3s"></div></div>
<div id="ptext" class="small" style="margin-top:4px"></div>
</div>
<div id="status"></div>
<audio id="player" controls style="display:none"></audio>
<div id="dl" class="small"></div>

<h2 style="font-weight:500;font-size:19px;margin:44px 0 4px">加演员</h2>
<div class="small">一段 3–10 秒的干净人声（任何格式，长文件给起止秒数切一段）。不填转写就用 whisper 转，转完<b>对着音频核一遍</b>再保存。</div>
<label>录音文件</label><input id="afile" type="file" accept="audio/*,video/*,.mp3,.m4a,.wav,.flac,.ogg,.opus,.mp4,.mkv,.webm">
<div class="row"><div><label>演员名</label><input id="aname" placeholder="如 乔峰"></div>
<div><label>作品（可选）</label><input id="awork" placeholder="如 天龙八部"></div>
<div style="flex:0 0 100px"><label>起（秒）</label><input id="astart" type="number" step="0.1" min="0" placeholder="0"></div>
<div style="flex:0 0 100px"><label>止（秒）</label><input id="aend" type="number" step="0.1" min="0" placeholder="到结尾"></div></div>
<label>转写（可选；留空用 whisper）</label><input id="atext" placeholder="这段话的逐字转写">
<button id="aadd" type="button">上传并加入</button>
<div id="astatus"></div>
<div id="acheck" style="display:none">
<label>核对转写：<span id="acname"></span></label>
<textarea id="actext" style="min-height:70px"></textarea>
<button id="asave" type="button">保存转写</button>
<button id="aplay" type="button" class="aud" style="margin-top:16px">试听这个演员</button>
</div>
<h2 style="font-weight:500;font-size:19px;margin:44px 0 4px">删演员</h2>
<div class="small">移到 voices/_trash/，不真删。</div>
<button id="adel" type="button" style="background:#8a2b2b">删除上面选中的演员</button>
<div id="dstatus"></div>
</main><script>
var V=[];var $=function(id){return document.getElementById(id)};
function setStatus(s){$("status").textContent=s}
function info(){var v=V.find(function(x){return x.name===$("voice").value});$("vinfo").textContent=v?((v.work?v.work+" · ":"")+v.seconds.toFixed(1)+"s · "+v.text):""}
function loadVoices(want){return fetch("/api/voices").then(function(r){return r.json()}).then(function(j){V=j.voices||[];var s=$("voice");s.innerHTML="";V.forEach(function(v){var o=document.createElement("option");o.value=v.name;o.textContent=v.name+(v.work?"  ("+v.work+")":"");s.appendChild(o)});
var last=want;try{last=last||localStorage.getItem("say.voice")}catch(e){}if(last&&V.some(function(v){return v.name===last}))s.value=last;info()})}
loadVoices("");
$("aadd").addEventListener("click",function(){var f=$("afile").files[0];if(!f){$("astatus").textContent="先选一个录音文件";return}
var q="filename="+encodeURIComponent(f.name)+"&name="+encodeURIComponent($("aname").value.trim())+"&work="+encodeURIComponent($("awork").value.trim())+"&start="+encodeURIComponent($("astart").value)+"&end="+encodeURIComponent($("aend").value)+"&text="+encodeURIComponent($("atext").value.trim());
$("aadd").disabled=true;$("acheck").style.display="none";$("astatus").textContent="上传、切音、"+($("atext").value.trim()?"":"whisper 转写")+"中…";
fetch("/api/voice/add?"+q,{method:"POST",body:f}).then(function(r){return r.json()}).then(function(j){$("aadd").disabled=false;if(!j.ok){$("astatus").textContent="失败："+j.error;return}
$("astatus").textContent="已加入 "+j.name+"，"+j.seconds.toFixed(1)+" 秒"+(j.lang?"（whisper 转写，语言 "+j.lang+"）":"")+(j.seconds<3||j.seconds>10?" — 长度建议 3–10 秒":"");
$("acname").textContent=j.name;$("actext").value=j.text;$("acheck").style.display="";loadVoices(j.name)})
.catch(function(e){$("aadd").disabled=false;$("astatus").textContent="失败："+e.message})});
$("asave").addEventListener("click",function(){var n=$("acname").textContent;fetch("/api/voice/text",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:n,text:$("actext").value})}).then(function(r){return r.json()}).then(function(j){$("astatus").textContent=j.ok?"已保存 "+n+" 的转写":"失败："+j.error;if(j.ok)loadVoices(n)})});
$("aplay").addEventListener("click",function(){var n=$("acname").textContent;$("voice").value=n;info();$("aud").click()});
$("adel").addEventListener("click",function(){var n=$("voice").value;if(!n||!confirm("把 "+n+" 移到 voices/_trash/？"))return;fetch("/api/voice/delete",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({name:n})}).then(function(r){return r.json()}).then(function(j){$("dstatus").textContent=j.ok?"已移走 "+n:"失败："+j.error;if(j.ok)loadVoices("")})});
$("voice").addEventListener("change",function(){info();try{localStorage.setItem("say.voice",$("voice").value)}catch(e){}});
$("aud").addEventListener("click",function(){var v=$("voice").value;if(!v)return;setStatus("试听合成中…");var a=$("player");a.style.display="";a.src="/api/audition?voice="+encodeURIComponent(v)+"&t="+Date.now();a.play().then(function(){setStatus("")}).catch(function(){setStatus("试听出不了声")})});
$("go").addEventListener("click",function(){var t=$("text").value.trim();if(!t){setStatus("先写点要念的");return}
$("go").disabled=true;setStatus("");$("dl").innerHTML="";$("prog").style.display="";$("pbar").style.width="0%";$("ptext").textContent="提交中…";
fetch("/api/say",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({text:t,voice:$("voice").value,speed:parseFloat($("speed").value)||1,instruct:$("instruct").value.trim()})})
.then(function(r){return r.json()}).then(function(j){if(!j.ok){$("go").disabled=false;$("prog").style.display="none";setStatus("失败："+j.error);return}poll(j.job)})
.catch(function(e){$("go").disabled=false;$("prog").style.display="none";setStatus("失败："+e.message)})});
function poll(id){fetch("/api/job?id="+id).then(function(r){return r.json()}).then(function(j){
if(!j.ok){$("go").disabled=false;$("prog").style.display="none";setStatus("失败："+j.error);return}
var pct=j.total?Math.round(100*j.done/j.total):0;
if(j.status==="done"){$("pbar").style.width="100%";$("ptext").textContent=j.total+"/"+j.total+" 段 · 完成";$("go").disabled=false;
setStatus(j.seconds.toFixed(1)+" 秒音频，合成用时 "+Math.round(j.took)+" 秒");var a=$("player");a.style.display="";a.src=j.url;a.play().catch(function(){});
$("dl").innerHTML='<a href="'+j.url+'" download="'+j.name+'">下载 '+j.name+'</a> · 已存到 '+j.path;return}
if(j.status==="failed"){$("go").disabled=false;$("ptext").textContent="";setStatus("失败："+j.error);return}
if(j.status==="queued"){$("ptext").textContent="排队中，前面还有 "+j.ahead+" 个任务";}
else{$("pbar").style.width=pct+"%";var eta=j.done>0?Math.round(j.elapsed/j.done*(j.total-j.done)):0;
$("ptext").textContent=j.stage+" · 第 "+j.done+"/"+j.total+" 段 · 已出 "+j.seconds.toFixed(1)+" 秒音频 · 用时 "+Math.round(j.elapsed)+" 秒"+(j.done<j.total&&eta?" · 约剩 "+eta+" 秒":"");}
setTimeout(function(){poll(id)},500)}).catch(function(){setTimeout(function(){poll(id)},1500)})}
</script></body></html>"""


def serve(engine, port):
    """A one-page form on 127.0.0.1:<port>; synthesis is serialized."""
    import threading
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse

    lock = threading.Lock()
    jobs = {}                       # id -> state; the page polls /api/job
    aud_dir = os.path.join(OUT_DIR, "_audition")

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            sys.stderr.write("[say] %s\n" % (fmt % a))

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _file(self, p):
            with open(p, "rb") as fh:
                body = fh.read()
            self._send(200, body, "audio/mpeg" if p.endswith(".mp3") else "audio/wav")

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if u.path == "/api/voices":
                return self._json({"ok": True, "voices": list_voices()})
            if u.path == "/api/job":
                job = jobs.get(parse_qs(u.query).get("id", [""])[0])
                if not job:
                    return self._json({"ok": False, "error": "没有这个任务"}, 404)
                j = dict(job)
                j["ok"] = True
                j["elapsed"] = time.time() - job["t0"]
                # Queue position: jobs ahead of this one still queued or running.
                j["ahead"] = sum(1 for o in jobs.values()
                                 if o["t0"] < job["t0"] and o["status"] in ("queued", "running"))
                return self._json(j)
            if u.path == "/api/audition":
                name = parse_qs(u.query).get("voice", [""])[0]
                if name.isdigit() or not pick_voice(list_voices(), name):
                    return self._json({"ok": False, "error": "没有这个声音"}, 404)
                dst = os.path.join(aud_dir, name + ".mp3")
                stamp = max([os.path.getmtime(os.path.join(VOICES, name + e))
                             for e in (".wav", ".txt", ".rhythm")
                             if os.path.isfile(os.path.join(VOICES, name + e))] or [0])
                if not os.path.isfile(dst) or os.path.getmtime(dst) < stamp:
                    with lock:
                        try:
                            speak(engine, SAMPLE_TEXT, name, 1.0, "", dst, "mp3", quiet=True)
                        except SystemExit as exc:
                            return self._json({"ok": False, "error": str(exc)}, 503)
                return self._file(dst)
            if u.path.startswith("/out/"):
                p = os.path.join(OUT_DIR, os.path.basename(u.path[5:]))
                if not os.path.isfile(p):
                    return self._json({"ok": False, "error": "没有这个文件"}, 404)
                return self._file(p)
            return self._json({"ok": False, "error": "not found"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            if u.path == "/api/voice/add":
                # Raw file body; the fields ride in the query string. The
                # upload lands in output/_uploads and is removed once cut.
                q = parse_qs(u.query)
                fld = lambda k: str(q.get(k, [""])[0]).strip()           # noqa: E731
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    n = 0
                if not 0 < n <= 500 * 1024 * 1024:
                    return self._json({"ok": False, "error": "文件为空或超过 500MB"}, 400)
                ext = os.path.splitext(fld("filename") or "a.wav")[1].lower()[:8] or ".wav"
                updir = os.path.join(OUT_DIR, "_uploads")
                os.makedirs(updir, exist_ok=True)
                tmp = os.path.join(updir, "%d%s" % (int(time.time() * 1000), ext))
                with open(tmp, "wb") as fh:
                    remaining = n
                    while remaining > 0:
                        chunk = self.rfile.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        fh.write(chunk)
                        remaining -= len(chunk)
                name = fld("name") or os.path.splitext(fld("filename"))[0]

                def num(k):
                    try:
                        return float(fld(k)) if fld(k) else None
                    except ValueError:
                        return None
                try:
                    with lock:
                        r = add_voice(tmp, name, num("start"), num("end"), fld("text"),
                                      fld("work"), fld("raw") not in ("1", "true"),
                                      fld("whisper") or os.environ.get("SAY_WHISPER", "small"))
                except SystemExit as exc:
                    return self._json({"ok": False, "error": str(exc)}, 400)
                finally:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
                r["ok"] = True
                return self._json(r)
            if u.path in ("/api/voice/text", "/api/voice/delete"):
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                except ValueError:
                    return self._json({"ok": False, "error": "bad json"}, 400)
                name = str(req.get("name") or "").strip()
                if not pick_voice(list_voices(), name) or name.isdigit():
                    return self._json({"ok": False, "error": "没有这个演员"}, 404)
                try:
                    if u.path.endswith("text"):
                        return self._json({"ok": True, "name": name,
                                           "text": set_voice_text(name, str(req.get("text") or ""))})
                    return self._json({"ok": True, "name": name, "moved": trash_voice(name)})
                except SystemExit as exc:
                    return self._json({"ok": False, "error": str(exc)}, 400)
            if u.path != "/api/say":
                return self._json({"ok": False, "error": "not found"}, 404)
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            except ValueError:
                return self._json({"ok": False, "error": "bad json"}, 400)
            text = str(req.get("text") or "").strip()
            voice = pick_voice(list_voices(), str(req.get("voice") or ""))
            if not text or not voice:
                return self._json({"ok": False, "error": "缺台词或没有这个声音"}, 400)
            try:
                speed = min(2.0, max(0.5, float(req.get("speed") or 1.0)))
            except (TypeError, ValueError):
                speed = 1.0
            # A job, not a blocking request: the page polls /api/job for the
            # bar, and a long chapter no longer hangs one HTTP call for minutes.
            jid = "%x" % int(time.time() * 1000)
            job = {"id": jid, "status": "queued", "done": 0, "total": len(split_text(text)),
                   "seconds": 0.0, "stage": "等待", "t0": time.time()}
            jobs[jid] = job
            out = default_out(voice["name"], "mp3")

            def run():
                def prog(done, total, secs, stage):
                    job.update(done=done, total=total, seconds=secs, stage=stage, status="running")
                with lock:
                    try:
                        secs = speak(engine, text, voice["name"], speed,
                                     str(req.get("instruct") or "").strip(), out, "mp3",
                                     progress=prog)
                    except SystemExit as exc:
                        job.update(status="failed", error=str(exc))
                        return
                    except Exception as exc:                 # noqa: BLE001
                        job.update(status="failed", error="合成出错：%s" % exc)
                        return
                name = os.path.basename(out)
                job.update(status="done", url="/out/" + name, name=name, path=out,
                           seconds=secs, took=time.time() - job["t0"])
            threading.Thread(target=run, name="say-" + jid, daemon=True).start()
            return self._json({"ok": True, "job": jid, "total": job["total"]})

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    url = "http://127.0.0.1:%d" % port
    print("[say] 配音台 %s  （引擎 %s）  Ctrl+C 退出" % (url, engine.name), flush=True)
    try:
        webbrowser.open(url)
    except Exception:                                   # noqa: BLE001
        pass
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()


# ------------------------------------------------------------------- main --

def main():
    # Windows consoles default to a legacy code page and every Chinese line
    # then dies in print(); the file names and text are utf-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="配音台：文字 + 演员 → mp3",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("text", nargs="*", help="要念的文字（也可以 -f 文件）")
    ap.add_argument("-v", "--voice", help="演员：voices/ 里的名字，或 --list 里的编号")
    ap.add_argument("-f", "--text-file", help="从文件读要念的文字（utf-8）；「-」表示标准输入")
    ap.add_argument("-o", "--out", help="输出文件（默认 output/<演员>-<时间>.mp3）")
    ap.add_argument("--format", choices=["mp3", "wav"], default="mp3")
    ap.add_argument("--speed", type=float, default=1.0, help="0.5-2.0，默认 1.0")
    ap.add_argument("--instruct", default="", help="念法，如「请用低沉、克制、缓慢的语气讲述」（CosyVoice3 听得懂，v2 会把它念出来）")
    ap.add_argument("--list", action="store_true", help="列出演员")
    ap.add_argument("--audition", action="store_true",
                    help="同一句话用每个演员各出一个文件到 output/audition-<时间>/，用来挑人")
    ap.add_argument("--model", default=os.environ.get("MODEL", ""),
                    help="models/ 里的权重目录名（默认按 cosyvoice3-rl, cosyvoice3, cosyvoice2 找）")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto", help="auto：有 GPU 就用")
    ap.add_argument("--server", default=os.environ.get("SAY_SERVER", ""),
                    help="改用一个已经在跑的 tts 服务，如 http://127.0.0.1:9100")
    ap.add_argument("--web", action="store_true", help="开一个网页表单")
    ap.add_argument("--port", type=int, default=9110, help="--web 用的端口")
    g = ap.add_argument_group("加演员", "把一段录音变成 voices/ 里的一对文件")
    g.add_argument("--add", metavar="录音", help="任何 ffmpeg 读得了的音频/视频文件")
    g.add_argument("--name", help="演员名（默认取文件名）")
    g.add_argument("--start", type=float, help="从第几秒开始截（可选）")
    g.add_argument("--end", type=float, help="截到第几秒（可选）")
    g.add_argument("--text", help="这段话的逐字转写；不给就用 whisper 转，之后务必核对")
    g.add_argument("--work", help="来源作品，只是列表里的一个标签")
    g.add_argument("--raw", action="store_true", help="不做响度归一")
    g.add_argument("--whisper", default=os.environ.get("SAY_WHISPER", "small"),
                   help="转写用的 whisper 模型：tiny/base/small/medium/large（默认 small）")
    g.add_argument("--set-text", metavar="演员", help="改某个演员的转写（配 --text）")
    g.add_argument("--remove", metavar="演员", help="把某个演员移到 voices/_trash/（不真删）")
    args = ap.parse_args()

    if args.add:
        stem = os.path.splitext(os.path.basename(args.add))[0]
        report_added(add_voice(args.add, args.name or stem, args.start, args.end,
                               (args.text or "").strip(), (args.work or "").strip(),
                               not args.raw, args.whisper))
        return
    if args.set_text:
        print("[say] %s 的转写改为：%s" % (args.set_text, set_voice_text(args.set_text, args.text or "")))
        return
    if args.remove:
        print("[say] %s 移到 voices/_trash/（%s）" % (args.remove, " ".join(trash_voice(args.remove))))
        return

    voices = list_voices()
    if args.list:
        print_voices(voices)
        return
    if not voices:
        raise SystemExit("voices/ 里没有可用的声音：需要 name.wav（3-10 秒干净人声）+ name.txt（逐字转写）成对")

    if args.web:
        return serve(make_engine(args), args.port)

    text = read_text(args).strip()
    if args.audition:
        text = text or SAMPLE_TEXT
        chosen = [pick_voice(voices, args.voice)] if args.voice else voices
        if chosen == [None]:
            raise SystemExit("没有这个声音：%s（--list 看有哪些）" % args.voice)
        engine = make_engine(args)
        d = os.path.join(OUT_DIR, "audition-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        for v in chosen:
            try:
                speak(engine, text, v["name"], args.speed, args.instruct,
                      os.path.join(d, v["name"] + "." + args.format), args.format)
            except SystemExit as exc:
                print("[say] × %s：%s" % (v["name"], exc), flush=True)
        print("[say] 试听都在 %s" % d)
        return

    if not text:
        ap.print_usage()
        raise SystemExit("没有要念的文字：直接写在命令后面，或 -f 文件，或 -f - 从管道进来")
    voice = pick_voice(voices, args.voice) if args.voice else ask_voice(voices)
    if not voice:
        raise SystemExit(("没有这个声音：%s（--list 看有哪些）" % args.voice) if args.voice else "没选演员")

    engine = make_engine(args)
    out = args.out or default_out(voice["name"], args.format)
    if args.out and not args.out.lower().endswith("." + args.format):
        out = args.out + "." + args.format
    speak(engine, text, voice["name"], args.speed, args.instruct, out, args.format)


if __name__ == "__main__":
    main()
