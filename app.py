import base64
import gc
import html as html_lib
import importlib.util
import json
import math
import os
import re
import shutil
import struct
import subprocess
import time
import unicodedata
import uuid
import wave
from datetime import datetime
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECTS_DIR = APP_DIR / "projects"
PROJECTS_DIR.mkdir(exist_ok=True)


def _register_cuda_dlls():
    """Make pip-installed cuBLAS / cuDNN DLLs visible to faster-whisper on Windows."""
    if os.name != "nt":
        return []
    found = []
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(pkg)
        except Exception:
            continue
        if not spec or not spec.submodule_search_locations:
            continue
        for loc in spec.submodule_search_locations:
            bin_dir = Path(loc) / "bin"
            if bin_dir.exists():
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                found.append(pkg)
    return found


CUDA_DLL_PACKAGES = _register_cuda_dlls()

import imageio_ffmpeg  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

st.set_page_config(page_title="Reel Studio AI", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{max-width:1200px;padding:1rem .9rem 4rem}
div[data-testid="stColumn"]:has(.sticky-preview){position:sticky;top:3.5rem;align-self:flex-start}
h1{font-size:1.5rem!important} h2{font-size:1.25rem!important} h3{font-size:1.1rem!important}
@media (max-width:640px){.stButton>button,.stDownloadButton>button{width:100%}}
</style>""", unsafe_allow_html=True)

STEPS = ["1 · Remove pauses", "2 · Remove repeats", "3 · Subtitles & title"]
VIDEO_TYPES = ["mp4", "mov", "m4v", "mkv"]

PROFILES = {
    "Kokni": {
        "description": "Conversational, warm, expressive. Mixed-language transcription is expected; review dialect spellings carefully.",
        "language": "mr",
        "prompt": ("Kokni (Konkan, Mumbai) boli. Sab Roman script mein, Devanagari nahi. Jaise: Shadichya kamaat ek "
                   "kaam aste saglyaat bhaari, daawti chilis banauna ani daoti karna puri. Paile te hotan meeting "
                   "yo ke kunna varecha, kaisa varecha, kava varecha. Mang decided hota loka."),
    },
    "Hindi": {
        "description": "Mumbai Hindi / Hinglish. Preserve colloquial phrasing, English words and natural sentence structure.",
        "language": "hi",
        "prompt": "Yeh ek Hinglish video hai. Hindi aur English words mix hote hain, sab Roman script mein likhna hai. Jaise: Aaj kal log kaam mein busy rehte hain, but deadline pe sab complete kar dete hain.",
    },
}
LANGUAGE_CHOICES = {"Profile default": None, "Marathi (mr)": "mr", "Hindi (hi)": "hi", "English (en)": "en",
                    "Urdu (ur)": "ur", "Auto-detect (not recommended)": "auto"}
WHISPER_MODELS = ["small", "medium", "large-v3-turbo", "large-v3"]

CUT_PRESETS = {
    "Tight (best for reels)": dict(min_silence=0.30, pad=0.08),
    "Normal": dict(min_silence=0.45, pad=0.12),
    "Gentle (keeps natural pauses)": dict(min_silence=0.80, pad=0.18),
    "Aggressive": dict(min_silence=0.18, pad=0.05),
    "Custom": None,
}
MY_STYLE_FILE = APP_DIR / "my_style.json"
MY_STYLE_NAME = "My style (learned from my reels)"
REF_DIR = APP_DIR / "reference_videos"
REF_DIR.mkdir(exist_ok=True)


def cut_presets():
    presets = {k: v for k, v in CUT_PRESETS.items() if k != "Custom"}
    mine = load_json(MY_STYLE_FILE, None)
    if isinstance(mine, dict) and "min_silence" in mine and "pad" in mine:
        presets[MY_STYLE_NAME] = dict(min_silence=float(mine["min_silence"]), pad=float(mine["pad"]))
    presets["Custom"] = None
    return presets

ZOOM_STYLES = [
    "Slow push-in on every sentence",
    "Alternate push-in / pull-out",
    "Alternate wide / zoomed (jump-cut style)",
    "Slow pull-out on every sentence",
    "No zoom",
]

RATIOS = {
    "9:16 · Reels / Shorts / TikTok": (1080, 1920),
    "4:5 · Instagram feed": (1080, 1350),
    "1:1 · Square": (1080, 1080),
    "16:9 · YouTube / landscape": (1920, 1080),
}
FITS = {"Crop to fill the frame": "crop", "Fit with black bars": "pad", "Fit with blurred background": "blur"}
FONT_CHOICES = ["Arial", "Arial Black", "Impact", "Segoe UI", "Verdana", "Tahoma", "Georgia", "Nirmala UI"]
DEFAULT_STYLE = dict(show=True, font="Arial", size=64, bold=True, upper=False, color="#FFFFFF",
                     outline_color="#000000", outline=4, box=False, box_color="#000000", box_opacity=60,
                     x_pct=50, y_pct=75, max_words=4)
FRAME = 0.02  # silence analysis frame (seconds)


# =============================================================== basic helpers
def safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name[:100] or "video"


def load_json(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_project(project_dir: Path, data: dict):
    (project_dir / "project.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fmt_time(seconds):
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def ass_time(t):
    t = max(0.0, float(t))
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def cuda_status():
    try:
        import ctranslate2
        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:
        return False, f"Could not query CUDA: {exc}"
    if count <= 0:
        return False, "No CUDA device visible. Update the NVIDIA driver."
    if os.name == "nt" and len(CUDA_DLL_PACKAGES) < 2:
        return False, ("GPU found, but cuBLAS/cuDNN libraries are missing. In the app's .venv run: "
                       "pip install nvidia-cublas-cu12 \"nvidia-cudnn-cu12>=9,<10\"")
    return True, f"NVIDIA GPU ready ({count} device)."


def list_projects():
    found = []
    for folder in sorted(PROJECTS_DIR.iterdir(), reverse=True):
        p = folder / "project.json"
        if p.exists():
            found.append((folder, load_json(p, {})))
    return found


# ================================================================ ffmpeg layer
def run_ffmpeg(args, cwd=None):
    r = subprocess.run([FFMPEG, "-y", *args], capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError("FFmpeg failed: " + r.stderr[-1800:])
    return r.stderr


def run_graph(pre_args, graph, post_args, workdir, graph_file):
    """Run a filter graph from a file (no command-line length limit). Tries the option spelling that
    the installed FFmpeg understands: -/filter_complex (7+), -filter_complex_script (older), inline."""
    graph_file = Path(graph_file)
    graph_file.write_text(graph, encoding="utf-8")
    last = ""
    for flag in (["-/filter_complex", str(graph_file)], ["-filter_complex_script", str(graph_file)],
                 ["-filter_complex", graph]):
        started = time.time()
        r = subprocess.run([FFMPEG, "-y", *pre_args, *flag, *post_args], capture_output=True, text=True,
                           cwd=workdir)
        if r.returncode == 0:
            return r.stderr
        last = r.stderr
        option_problem = re.search(r"Unrecognized option|Option not found|Error splitting the argument list",
                                   r.stderr)
        if not option_problem and time.time() - started > 8:
            break  # a real failure after real work: do not repeat a long render
    raise RuntimeError("FFmpeg failed: " + last[-1800:])


def norm_fps(f):
    if not f:
        return 30.0
    for k, v in ((23.976, 24000 / 1001), (29.97, 30000 / 1001), (59.94, 60000 / 1001), (119.88, 120000 / 1001)):
        if abs(f - k) < 0.02:
            return v
    return round(float(f), 3)


def probe_video(path: Path):
    r = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True)
    err = r.stderr
    info = {"duration_seconds": None, "size_mb": round(path.stat().st_size / (1024 * 1024), 1),
            "width": None, "height": None, "fps": None, "has_audio": "Audio:" in err}
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if m:
        info["duration_seconds"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    v = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", err)
    if v:
        w, h = int(v.group(1)), int(v.group(2))
        rot = re.search(r"rotation of (-?\d+(?:\.\d+)?) degrees", err)
        if rot and abs(float(rot.group(1))) in (90.0, 270.0):
            w, h = h, w
        info["width"], info["height"] = w, h
    f = re.search(r"(\d+(?:\.\d+)?) fps", err)
    if f:
        info["fps"] = norm_fps(float(f.group(1)))
    return info


def extract_audio(video: Path, wav: Path):
    run_ffmpeg(["-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-af", "highpass=f=80",
                "-c:a", "pcm_s16le", str(wav)])


def make_proxy(video: Path, out: Path, fps):
    """Small, fast-to-seek copy used for every preview. Same timeline and frame rate as the original."""
    vf = (f"scale='if(gt(iw,ih),-2,540)':'if(gt(iw,ih),540,-2)',fps={fps},format=yuv420p")
    run_ffmpeg(["-i", str(video), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out)])


# ============================================================ audio / cut logic
@st.cache_data(show_spinner=False)
def audio_levels(wav_path, stamp):
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    frame = max(1, int(rate * FRAME))
    n = len(audio) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32), 0.0
    frames = audio[: n * frame].reshape(n, frame)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    return db.astype(np.float32), n * frame / rate


def compute_keep(db, duration, offset_db, min_silence, pad, min_speech=0.10):
    """Where the person is talking. Threshold is relative to this clip's own speech level."""
    if len(db) == 0:
        return [], None
    ref = float(np.percentile(db, 95))
    thr = float(np.clip(ref + offset_db, -60.0, -25.0))
    speech = (db >= thr).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], speech, [0]))))
    runs = [(a * FRAME, b * FRAME) for a, b in zip(edges[0::2], edges[1::2]) if (b - a) * FRAME >= min_speech]
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < min_silence:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    keep = []
    for a, b in merged:
        a, b = max(0.0, a - pad), min(duration, b + pad)
        if b <= a:
            continue
        if keep and a <= keep[-1][1]:
            keep[-1][1] = max(keep[-1][1], b)
        else:
            keep.append([a, b])
    return [(a, b) for a, b in keep], thr


def snap_keep(keep, fps, duration=None, min_len=0.1):
    """Put every cut exactly on a video frame, so picture and sound stay in sync across hundreds of cuts."""
    out = []
    last = math.floor(duration * fps) / fps if duration else None
    for a, b in keep:
        a, b = round(a * fps) / fps, round(b * fps) / fps
        if last is not None:
            b = min(b, last)
        if b - a < min_len:
            continue
        if out and a <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return [(round(a, 5), round(b, 5)) for a, b in out]


def _offsets(keep):
    offs, c = [], 0.0
    for a, b in keep:
        offs.append(c)
        c += b - a
    return offs, c


def out_to_src(t, keep):
    offs, _ = _offsets(keep)
    for (a, b), o in zip(keep, offs):
        if o <= t <= o + (b - a):
            return a + (t - o)
    return max(0.0, keep[-1][1] - 0.05)


def remove_output_ranges(keep, ranges):
    """Delete time ranges expressed in the review video's own timeline (what you see in the player)."""
    offs, _ = _offsets(keep)
    ranges = sorted((max(0.0, a), b) for a, b in ranges if b > a)
    new = []
    for (a, b), o in zip(keep, offs):
        pieces = [(a, b)]
        for r0, r1 in ranges:
            lo, hi = a + (r0 - o), a + (r1 - o)
            if hi <= a or lo >= b:
                continue
            lo, hi = max(lo, a), min(hi, b)
            nxt = []
            for p0, p1 in pieces:
                if hi <= p0 or lo >= p1:
                    nxt.append((p0, p1))
                    continue
                if lo > p0:
                    nxt.append((p0, lo))
                if hi < p1:
                    nxt.append((hi, p1))
            pieces = nxt
        new += pieces
    return new


def levels_image(db, thr, keep, duration, width=1100, height=150):
    n = len(db)
    img = Image.new("RGB", (width, height), (246, 246, 246))
    if n == 0 or duration <= 0:
        return img
    draw = ImageDraw.Draw(img)
    for a, b in keep:
        x0 = int(a / duration * width)
        x1 = max(x0 + 1, int(b / duration * width))
        draw.rectangle([x0, 0, x1, height], fill=(214, 240, 222))
    cols = min(width, n)
    k = int(math.ceil(n / cols))
    padded = np.concatenate([db, np.full(k * cols - n, float(db.min()))]).reshape(cols, k).max(axis=1)
    lo, hi = -80.0, 0.0
    for i, v in enumerate(padded):
        x = int(i * width / cols)
        y = int((1 - (float(np.clip(v, lo, hi)) - lo) / (hi - lo)) * (height - 1))
        draw.line([x, height - 1, x, y], fill=(70, 110, 170))
    if thr is not None:
        ty = int((1 - (float(np.clip(thr, lo, hi)) - lo) / (hi - lo)) * (height - 1))
        draw.line([0, ty, width, ty], fill=(220, 50, 50), width=2)
    return img


def analyse_reference(video: Path):
    """Measure how long the pauses are in one of the user's finished reels."""
    tmp = REF_DIR / "_analysis"
    tmp.mkdir(exist_ok=True)
    wav = tmp / (safe_name(video.stem) + ".wav")
    extract_audio(video, wav)
    db, dur = audio_levels(str(wav), wav.stat().st_mtime)
    if len(db) == 0:
        return None
    thr = float(np.clip(np.percentile(db, 95) - 30, -60.0, -25.0))
    speech = (db >= thr).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], speech, [0]))))
    runs = [(a * FRAME, b * FRAME) for a, b in zip(edges[0::2], edges[1::2]) if (b - a) * FRAME >= 0.10]
    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    talk = sum(b - a for a, b in runs)
    return dict(name=video.name, duration=round(dur, 1), speech_pct=round(100 * talk / max(dur, 1e-6)),
                runs=len(runs), gaps=[round(g, 3) for g in gaps])


def learn_style(results):
    gaps = [g for r in results for g in r["gaps"] if g >= 0.06]
    if len(gaps) < 5:
        return None
    med, p90 = float(np.median(gaps)), float(np.percentile(gaps, 90))
    return dict(min_silence=round(float(np.clip(p90 * 1.15 + 0.05, 0.15, 1.0)), 2),
                pad=round(float(np.clip(med / 2, 0.03, 0.25)), 2),
                median_pause=round(med, 2), p90_pause=round(p90, 2), based_on=[r["name"] for r in results])


# ===================================================== filter graph statements
def denoise_filter(project):
    dn = (project or {}).get("denoise") or {}
    if not dn.get("on", False):
        return ""
    return f"highpass=f=80,afftdn=nr={int(dn.get('level', 12))}:nf=-30:tn=1"


def cut_statements(keep, fps, afilter=""):
    """Frame-exact video selection + sample-exact audio, so A/V never drift. Ends with labels [vc] and [ac]."""
    e = 0.5 / fps
    expr = "+".join(f"between(t,{a - e:.5f},{b - e:.5f})" for a, b in keep)
    stm = [f"[0:v]select='{expr}',setpts=N/({fps:.6f}*TB)[vc]"]
    for i, (a, b) in enumerate(keep):
        stm.append(f"[0:a]atrim=start={a:.5f}:end={b:.5f},asetpts=PTS-STARTPTS[a{i}]")
    cat = "".join(f"[a{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=0:a=1"
    stm.append(cat + (f"[acr];[acr]{afilter}[ac]" if afilter else "[ac]"))
    return stm


def fit_statements(src, fit, w, h, fps, out="vb"):
    if fit == "crop":
        return [f"{src}scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},fps={fps},"
                f"format=yuv420p[{out}]"]
    if fit == "blur":
        return [f"{src}split[bg][fg]",
                f"[bg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=25:5[bgb]",
                f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fgs]",
                f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,fps={fps},format=yuv420p[{out}]"]
    return [f"{src}scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black,fps={fps},format=yuv420p[{out}]"]


# ======================================================================= zoom
def zoom_plan(keep, zcfg):
    """[(start, end, mode, amount)] on the output timeline, from the user's From / To / % table."""
    ranges = (zcfg or {}).get("ranges") or []
    if not keep or not ranges:
        return []
    _, total = _offsets(keep)
    plan = []
    for r in sorted(ranges, key=lambda r: float(r["from"])):
        a, b = max(0.0, float(r["from"])), min(float(r["to"]), total)
        if b - a >= 0.2 and float(r["pct"]) > 0:
            plan.append((a, b, f"pulse:{float((zcfg or {}).get('ease', 0.4)):.2f}", float(r["pct"]) / 100.0))
    return plan


def sentence_spans(project, keep, pause=0.5):
    """Sentence start/end times on the output timeline. Uses the transcript when it matches the current cut
    (sentence-ending punctuation or a gap), otherwise the pauses that silence removal left between clips."""
    _, total = _offsets(keep)
    segs = project.get("transcript") or []
    spans = []
    if segs and project.get("transcript_keep_version") == project.get("keep_version"):
        cur = None
        for sg in segs:
            if cur and sg["start"] - cur[1] > max(pause, 0.6):
                spans.append(cur)
                cur = None
            cur = [cur[0] if cur else sg["start"], sg["end"]]
            if re.search(r"[.?!।]\s*$", sg["text"].strip()):
                spans.append(cur)
                cur = None
        if cur:
            spans.append(cur)
    else:
        offs, _t = _offsets(keep)
        cur = None
        for i, (a, b) in enumerate(keep):
            s0, e0 = offs[i], offs[i] + (b - a)
            if cur and keep[i][0] - keep[i - 1][1] >= pause + 0.25:
                spans.append(cur)
                cur = None
            cur = [cur[0] if cur else s0, e0]
        if cur:
            spans.append(cur)
    merged = []                       # join very short sentences with the next one
    for a, b in spans:
        if merged and merged[-1][1] - merged[-1][0] < 1.6:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if len(merged) > 1 and merged[-1][1] - merged[-1][0] < 1.6:
        merged[-2][1] = merged[-1][1]
        merged.pop()
    return [(round(max(0.0, a), 2), round(min(b, total), 2)) for a, b in merged if b - a >= 0.8]


def suggest_zoom(project, keep, pct, every_other, pause):
    spans = sentence_spans(project, keep, pause)
    out = []
    for i, (a, b) in enumerate(spans):
        if every_other and i % 2 == 0:
            continue
        out.append({"from": a, "to": round(min(b, a + 9.0), 2), "pct": float(pct)})
    return out


def zoom_active(plan):
    return any(m != "none" for _, _, m, _ in plan)


def zoom_expr(plan, var="t"):
    expr = "1"
    for s, e, mode, amt in reversed(plan):
        d = max(e - s, 0.05)
        u = f"(({var}-{s:.3f})/{d:.3f})"
        smooth = f"({u}*{u}*(3-2*{u}))"
        mode, _, ez = mode.partition(":")
        r = min(float(ez or 0.4), d / 2)
        ua = f"clip(({var}-{s:.3f})/{r:.3f},0,1)"
        ub = f"clip(({var}-{e - r:.3f})/{r:.3f},0,1)"
        pulse = f"(({ua}*{ua}*(3-2*{ua}))*(1-{ub}*{ub}*(3-2*{ub})))"
        f = {"in": f"(1+{amt:.4f}*{smooth})", "out": f"(1+{amt:.4f}*(1-{smooth}))",
             "hold": f"{1 + amt:.4f}", "none": "1", "pulse": f"(1+{amt:.4f}*{pulse})"}[mode]
        expr = f"if(lt({var},{e:.3f}),{f},{expr})"
    return expr


def zoom_at(plan, t):
    """Python twin of zoom_expr, used by the live previews."""
    for s, e, mode, amt in plan:
        if t < e:
            d = max(e - s, 0.05)
            u = min(max((t - s) / d, 0.0), 1.0)
            sm = u * u * (3 - 2 * u)
            mode, _, ez = mode.partition(":")
            r = min(float(ez or 0.4), d / 2)

            def ss(x):
                x = min(max(x, 0.0), 1.0)
                return x * x * (3 - 2 * x)
            pulse = ss((t - s) / r) * (1 - ss((t - (e - r)) / r))
            return {"in": 1 + amt * sm, "out": 1 + amt * (1 - sm), "hold": 1 + amt, "none": 1.0,
                    "pulse": 1 + amt * pulse}[mode]
    return 1.0


def zoom_chain(src, w, h, z, fx, fy, out):
    """Smooth zoom: render at 2x, scale per frame by the zoom curve z (an FFmpeg expression in t), crop on the
    focus point, scale back."""
    return [f"{src}scale={2 * w}:{2 * h}:flags=bilinear[zb_{out}]",
            f"[zb_{out}]scale=w='2*trunc({w}*({z}))':h='2*trunc({h}*({z}))':eval=frame:flags=bicubic,"
            f"crop={2 * w}:{2 * h}:x='(iw-ow)*{fx:.3f}':y='(ih-oh)*{fy:.3f}',"
            f"scale={w}:{h}:flags=lanczos,format=yuv420p[{out}]"]


def zoom_statements(src, w, h, plan, fx, fy, out="vz"):
    return zoom_chain(src, w, h, zoom_expr(plan), fx, fy, out)


VIDEO_OUT = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
PROXY_OUT = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p"]


def render_review(project_dir: Path, project: dict, plan=None, fx=0.5, fy=0.4, name="review_cut.mp4"):
    """Preview of the current cut (optionally with zoom), built from the small proxy copy."""
    keep, fps = project["keep"], project["fps"]
    stm = cut_statements(keep, fps, denoise_filter(project))
    last = "vc"
    if plan and zoom_active(plan):
        pw, ph = project["proxy_size"]
        stm += zoom_statements("[vc]", pw, ph, plan, fx, fy, "vz")
        last = "vz"
    stm.append(f"[{last}]format=yuv420p[vout]")
    out = project_dir / name
    run_graph(["-i", str(project_dir / project["proxy_file"])], ";".join(stm),
              ["-map", "[vout]", "-map", "[ac]", *PROXY_OUT, "-c:a", "aac", "-b:a", "128k",
               "-movflags", "+faststart", str(out)], project_dir, project_dir / "graph_review.txt")
    return out


# ========================================================= fonts and subtitles
def _font_dirs():
    if os.name == "nt":
        dirs = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"]
        local = os.environ.get("LOCALAPPDATA")
        if local:
            dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")
        return [d for d in dirs if d.exists()]
    cands = [Path("/Library/Fonts"), Path.home() / "Library" / "Fonts", Path("/usr/share/fonts"),
             Path.home() / ".fonts", Path.home() / ".local" / "share" / "fonts"]
    return [d for d in cands if d.exists()]


def _read_family_names(path):
    found = set()
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            offsets = [0]
            if head[:4] == b"ttcf":
                n = min(struct.unpack(">I", head[8:12])[0], 64)
                offsets = list(struct.unpack(">" + "I" * n, f.read(4 * n)))
            for off in offsets:
                f.seek(off)
                hdr = f.read(12)
                if len(hdr) < 12:
                    continue
                num = struct.unpack(">H", hdr[4:6])[0]
                tbl = f.read(16 * num)
                for i in range(num):
                    rec = tbl[i * 16:(i + 1) * 16]
                    if rec[:4] != b"name":
                        continue
                    o, ln = struct.unpack(">II", rec[8:16])
                    f.seek(o)
                    data = f.read(min(ln, 1 << 18))
                    count, str_off = struct.unpack(">HH", data[2:6])
                    english, other = set(), set()
                    for j in range(count):
                        pid, _eid, lid, nid, length, so = struct.unpack(">HHHHHH", data[6 + 12 * j: 18 + 12 * j])
                        if nid not in (1, 16):
                            continue
                        raw = data[str_off + so: str_off + so + length]
                        if pid in (0, 3):
                            txt = raw.decode("utf-16-be", "ignore")
                        elif pid == 1:
                            txt = raw.decode("mac_roman", "ignore")
                        else:
                            continue
                        txt = txt.strip()
                        if txt:
                            (english if (pid == 1 or lid == 0x409) else other).add(txt)
                    found |= (english or other)
    except Exception:
        pass
    return found


@st.cache_data(show_spinner="Scanning installed fonts...")
def installed_fonts(refresh_token=0):
    index = {}
    for d in _font_dirs():
        for p in d.rglob("*"):
            if p.suffix.lower() in (".ttf", ".otf", ".ttc", ".otc"):
                for name in _read_family_names(p):
                    index.setdefault(name, []).append(str(p))
    return index


def _font_em_ratio(path):
    """CSS em size of one ASS font-size unit for this font file: libass sizes text by the OS/2 win metrics."""
    with open(path, "rb") as f:
        head = f.read(12)
        off = 0
        if head[:4] == b"ttcf":
            off = struct.unpack(">I", f.read(4))[0]
        f.seek(off)
        hdr = f.read(12)
        num = struct.unpack(">H", hdr[4:6])[0]
        tbl = f.read(16 * num)
        where = {}
        for i in range(num):
            rec = tbl[i * 16:(i + 1) * 16]
            where[rec[:4]] = struct.unpack(">I", rec[8:12])[0]
        f.seek(where[b"head"] + 18)
        upm = struct.unpack(">H", f.read(2))[0]
        f.seek(where[b"OS/2"] + 74)
        asc, desc = struct.unpack(">HH", f.read(4))
    return upm / (asc + desc) if asc + desc else 0.89


@st.cache_data(show_spinner=False)
def font_em_ratio(family, refresh_token=0):
    """Multiply an ASS font size by this to get the same text size in CSS pixels (Arial: about 0.895)."""
    index = installed_fonts(refresh_token)
    lower = {k.lower(): v for k, v in index.items()}
    for fp in lower.get(str(family).lower(), []):
        try:
            r = _font_em_ratio(fp)
            if 0.5 < r < 1.3:
                return r
        except Exception:
            continue
    return 0.895


def ass_font_size(family, size, scale):
    """ASS font size for a size slider value. The slider means what the preview always showed:
    size * 0.71 CSS pixels at 1080p, so the export is sized to match the preview, whatever the font."""
    ratio = font_em_ratio(family, st.session_state.get("font_refresh", 0))
    return max(10, int(round(size * scale * 0.71 / ratio)))


def css_text_size(family, ass_size, k):
    """(font-size px, line-height px) in a preview scaled by k that matches libass drawing at ass_size."""
    return ass_size * k * font_em_ratio(family, st.session_state.get("font_refresh", 0)), ass_size * k


def prepare_export_fonts(families):
    if isinstance(families, str):
        families = [families]
    index = installed_fonts(st.session_state.get("font_refresh", 0))
    lower = {k.lower(): v for k, v in index.items()}
    dest = APP_DIR / "fonts_export"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(exist_ok=True)
    seen = set()
    for fam in list(families) + ["Arial", "Nirmala UI", "Segoe UI"]:
        for fp in lower.get(fam.lower(), []):
            if fp not in seen:
                seen.add(fp)
                try:
                    shutil.copy2(fp, dest / Path(fp).name)
                except Exception:
                    pass
    return dest


def _fontsdir_variants():
    d = APP_DIR / "fonts_export"
    if d.exists() and any(d.iterdir()):
        return [":fontsdir=fonts_export", ""]
    return [""]


def hex_to_ass(hex_color, alpha=0):
    h = hex_color.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper().replace("&H", "&H", 1)


def make_cues(segments, max_words=6):
    cues = []
    for s in segments:
        words = s.get("words") or []
        if words:
            for i in range(0, len(words), max_words):
                chunk = words[i:i + max_words]
                text = " ".join(w["word"].strip() for w in chunk).strip()
                if text:
                    cues.append({"start": chunk[0]["start"], "end": chunk[-1]["end"], "text": text})
        elif s.get("text", "").strip():
            cues.append({"start": s["start"], "end": s["end"], "text": s["text"].strip()})
    cues.sort(key=lambda c: c["start"])
    for i, c in enumerate(cues):
        c["end"] = max(c["end"], c["start"] + 0.3)
        if i + 1 < len(cues) and c["end"] > cues[i + 1]["start"]:
            c["end"] = max(cues[i + 1]["start"], c["start"] + 0.05)
    return cues


def caption_segments(project):
    if project.get("caption_source") == "english" and project.get("english_segments"):
        return project["english_segments"]
    return project.get("transcript", [])


def write_vtt(cues, path: Path):
    lines = ["WEBVTT", ""]
    for c in cues:
        lines += [f"{fmt_time(c['start'])} --> {fmt_time(c['end'])}", c["text"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_srt(rows, path: Path):
    lines, idx = [], 0
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        idx += 1
        lines.extend([str(idx), f"{fmt_time(row['start']).replace('.', ',')} --> "
                                f"{fmt_time(row['end']).replace('.', ',')}", text, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_text_exports(project_dir: Path, segments):
    write_srt(segments, project_dir / "transcript.srt")
    (project_dir / "transcript.txt").write_text("\n".join(s["text"] for s in segments), encoding="utf-8")


MOTION_ANIMS = ["None", "Pop (bounce)", "Slam", "Zoom-in", "Slide up", "Fade"]
DEFAULT_MOTION_STYLE = dict(font="Arial", bold=True, upper=True, color="#FFD400", outline_color="#000000",
                            outline=6, box=False, box_color="#000000")


def motion_tags(anim, x, y):
    """ASS override tags for one animated text pop (times in milliseconds)."""
    pos = "\\an5\\pos(%d,%d)" % (x, y)
    if anim == "None":
        return pos
    if anim.startswith("Pop"):
        return pos + "\\fscx1\\fscy1\\t(0,140,\\fscx125\\fscy125)\\t(140,260,\\fscx100\\fscy100)\\fad(0,150)"
    if anim.startswith("Slam"):
        return pos + "\\fscx260\\fscy260\\t(0,130,\\fscx100\\fscy100)\\fad(0,120)"
    if anim.startswith("Zoom"):
        return pos + "\\fscx55\\fscy55\\t(0,380,\\fscx100\\fscy100)\\fad(140,140)"
    if anim.startswith("Slide"):
        return "\\an5\\move(%d,%d,%d,%d,0,260)\\fad(160,140)" % (x, y + 90, x, y)
    return pos + "\\fad(260,260)"


def _ass_text(text, upper):
    text = text.replace("{", "(").replace("}", ")").replace("\n", "\\N")
    return text.upper() if upper else text


def build_ass(cues, width, height, style, offset=0.0, motion=None):
    scale = min(width, height) / 1080
    fs = ass_font_size(style["font"], style["size"], scale)
    margin = int(width * 0.06)
    primary = hex_to_ass(style["color"])
    if style["box"]:
        border_style = 3
        outline_col = hex_to_ass(style["box_color"], int((100 - style["box_opacity"]) * 255 / 100))
        outline_w = max(6, int(12 * scale))
    else:
        border_style = 1
        outline_col = hex_to_ass(style["outline_color"])
        outline_w = style["outline"] * scale
    bold = -1 if style["bold"] else 0
    x = int(width * style["x_pct"] / 100)
    y = int(height * style["y_pct"] / 100)
    ms = {**DEFAULT_MOTION_STYLE, **((motion or {}).get("style") or {})}
    m_border = 3 if ms.get("box") else 1
    m_col = hex_to_ass(ms["box_color"]) if ms.get("box") else hex_to_ass(ms["outline_color"])
    m_w = (max(8, int(ms["outline"] * 2.5 * scale)) if ms.get("box") else ms["outline"] * scale)
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}", "WrapStyle: 0", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{style['font']},{fs},{primary},&H000000FF,{outline_col},&H00000000,{bold},0,0,0,100,100,0,0,"
        f"{border_style},{outline_w:.1f},0,5,{margin},{margin},0,1",
        f"Style: Motion,{ms['font']},{int(110 * scale)},{hex_to_ass(ms['color'])},&H000000FF,"
        f"{m_col},&H80000000,{-1 if ms['bold'] else 0},0,0,0,100,100,0,0,{m_border},"
        f"{m_w:.1f},3,5,{margin},{margin},0,1",
        "", "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for c in cues:
        s0, e0 = c["start"] - offset, c["end"] - offset
        if e0 <= 0:
            continue
        lines.append(f"Dialogue: 0,{ass_time(max(0.0, s0))},{ass_time(e0)},Default,,0,0,0,,"
                     "{\\an5\\pos(%d,%d)}%s" % (x, y, _ass_text(c["text"], style["upper"])))
    for it in (motion or {}).get("items") or []:
        s0 = float(it["start"]) - offset
        e0 = s0 + float(it["dur"])
        if e0 <= 0 or not str(it.get("text", "")).strip():
            continue
        mx, my = int(width * float(it.get("x", 50)) / 100), int(height * float(it.get("y", 35)) / 100)
        fsz = ass_font_size(ms["font"], float(it.get("size", 110)), scale)
        tags = motion_tags(it.get("anim", MOTION_ANIMS[0]), mx, my) + "\\fs%d" % fsz
        lines.append(f"Dialogue: 1,{ass_time(max(0.0, s0))},{ass_time(e0)},Motion,,0,0,0,,"
                     "{%s}%s" % (tags, _ass_text(str(it["text"]), ms["upper"])))
    return "\n".join(lines) + "\n"


def grab_frame(video_path: Path, t, out_png: Path):
    subprocess.run([FFMPEG, "-y", "-ss", f"{max(0.0, t):.2f}", "-i", str(video_path), "-frames:v", "1",
                    str(out_png)], capture_output=True, text=True)
    return out_png if out_png.exists() else None


# --------------------------------------------------------- live style preview
@st.cache_data(show_spinner=False)
def live_base_frame(video_path, keep_key, size, fit, t_out, z, fx, fy, stamp):
    """JPEG of the preview copy at output time t_out, in the chosen framing and zoom (no subtitles)."""
    keep = [tuple(k) for k in keep_key]
    src_t = out_to_src(t_out, keep)
    w, h = size
    stm = fit_statements("[0:v]", fit, w, h, 30, "vb")
    if z > 1.001:
        stm.append(f"[vb]scale=iw*{z:.4f}:ih*{z:.4f}:flags=bicubic,"
                   f"crop={w}:{h}:x='(iw-ow)*{fx:.3f}':y='(ih-oh)*{fy:.3f}'[vz]")
        last = "vz"
    else:
        last = "vb"
    stm.append(f"[{last}]format=yuvj420p[vj]")
    out = Path(video_path).parent / "live_base.jpg"
    r = subprocess.run([FFMPEG, "-y", "-ss", f"{src_t:.3f}", "-i", str(video_path), "-filter_complex",
                        ";".join(stm), "-map", "[vj]", "-frames:v", "1", "-q:v", "4", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError("Could not make the live preview frame: " + r.stderr[-400:])
    return out.read_bytes()


def pick_cue_text(cues, t):
    if not cues:
        return "Your subtitles appear here"
    for c in cues:
        if c["start"] <= t <= c["end"]:
            return c["text"]
    later = [c for c in cues if c["start"] > t]
    return (later[0] if later else cues[-1])["text"]


def live_preview_html(img_bytes, size, style, text):
    w, h = size
    k = 1 / 3
    pw, ph = int(w * k), int(h * k)
    scale = min(w, h) / 1080
    fs, lh = css_text_size(style["font"], ass_font_size(style["font"], style["size"], scale), k)
    b64 = base64.b64encode(img_bytes).decode()
    fam = style["font"].replace('"', "").replace("'", "")
    shown = html_lib.escape(text.upper() if style["upper"] else text).replace("\n", "<br>")
    stroke = ""
    if not style["box"] and style["outline"] > 0:
        stroke = (f"-webkit-text-stroke:{style['outline'] * scale * k * 2:.2f}px {style['outline_color']};"
                  "paint-order:stroke fill;")
    span_bg = ""
    if style["box"]:
        c = style["box_color"].lstrip("#")
        rr, gg, bb = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        span_bg = (f"background:rgba({rr},{gg},{bb},{style['box_opacity'] / 100:.2f});"
                   f"padding:{12 * scale * k:.1f}px {16 * scale * k:.1f}px;"
                   "box-decoration-break:clone;-webkit-box-decoration-break:clone;")
    return (
        f"<div style=\"position:relative;width:{pw}px;height:{ph}px;overflow:hidden;background:#000 "
        f"url(data:image/jpeg;base64,{b64}) center/cover no-repeat;\">"
        f"<div style=\"position:absolute;left:{style['x_pct']}%;top:{style['y_pct']}%;"
        f"transform:translate(-50%,-50%);width:max-content;max-width:88%;text-align:center;"
        f"font-family:'{fam}',Arial,sans-serif;font-size:{fs:.1f}px;"
        f"font-weight:{700 if style['bold'] else 400};color:{style['color']};line-height:{lh:.2f}px;"
        f"text-wrap:balance;{stroke}\">"
        f"<span style=\"{span_bg}\">{shown}</span></div></div>"
    )


def render_title_base(project_dir: Path, project: dict, keep, upto):
    """Small, fast clip of the opening of the cut video for the real-time title player (any source size)."""
    _, total = _offsets(keep)
    length = round(min(total, max(upto, 4.0)), 1) + 0.1
    out = project_dir / "title_base.mp4"
    tag = (project.get("keep_version"), length, (project.get("denoise") or {}).get("on"))
    if out.exists() and project.get("title_base") == list(tag):
        return out
    stm = cut_statements(keep, project["fps"], denoise_filter(project))
    stm.append("[vc]scale=288:-2,format=yuv420p[vs]")
    run_graph(["-i", str(project_dir / project["proxy_file"])], ";".join(stm),
              ["-map", "[vs]", "-map", "[ac]", "-t", f"{length:.1f}", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "30", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", str(out)],
              str(project_dir), project_dir / "graph_titlebase.txt")
    project["title_base"] = list(tag)
    save_project(project_dir, project)
    return out


def _hex_rgba(hex_color, opacity_pct):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{opacity_pct / 100:.2f})"


def title_player_html(video_path: Path, size, fit, item, ms, style_title, width=340, caps=None):
    """Real-time preview: the cut video plays with the captions and the title drawn over it."""
    w, h = size
    pw = width
    ph = int(width * h / w)
    scale = min(w, h) / 1080
    k = pw / w
    b64 = base64.b64encode(video_path.read_bytes()).decode()
    fit_css = "cover" if fit == "crop" else "contain"

    title_div, cfg = "", "null"
    if item:
        fs, lh = css_text_size(ms["font"], ass_font_size(ms["font"], item["size"], scale), k)
        text = html_lib.escape(item["text"].upper() if ms["upper"] else item["text"]).replace("\n", "<br>")
        fam = ms["font"].replace('"', "").replace("'", "")
        stroke, bg, shadow = "", "", ""
        if not ms.get("box"):     # the export's Motion style has a 3 px half-transparent shadow
            shadow = f"text-shadow:{3 * scale * k:.2f}px {3 * scale * k:.2f}px 0 rgba(0,0,0,.5);"
        if ms.get("box"):
            bg = (f"background:{ms['box_color']};padding:{max(6, ms['outline'] * 2.5) * scale * k:.1f}px "
                  f"{max(8, ms['outline'] * 3) * scale * k:.1f}px;")
        elif ms["outline"] > 0:
            stroke = (f"-webkit-text-stroke:{ms['outline'] * scale * k * 2:.2f}px {ms['outline_color']};"
                      "paint-order:stroke fill;")
        title_div = (f'<div id="ti" style="position:absolute;left:{item["x"]}%;top:{item["y"]}%;'
                     f'transform:translate(-50%,-50%);width:max-content;max-width:88%;text-align:center;'
                     f'pointer-events:none;opacity:0;font-family:\'{fam}\',Arial,sans-serif;font-size:{fs:.1f}px;'
                     f'font-weight:{700 if ms["bold"] else 400};color:{ms["color"]};line-height:{lh:.2f}px;'
                     f'text-wrap:balance;{stroke}{shadow}">'
                     f'<span style="{bg}">{text}</span></div>')
        cfg = json.dumps(dict(start=item["start"], dur=item["dur"], anim=item["anim"], slide=90 * scale * k))

    cap_div, cues = "", "[]"
    if caps and caps.get("cues"):
        cs = caps["style"]
        fam = cs["font"].replace('"', "").replace("'", "")
        cstroke, cbg = "", ""
        if cs["box"]:
            cbg = (f"background:{_hex_rgba(cs['box_color'], cs['box_opacity'])};"
                   f"padding:{6 * scale * k:.1f}px {10 * scale * k:.1f}px;")
        elif cs["outline"] > 0:
            cstroke = (f"-webkit-text-stroke:{cs['outline'] * scale * k * 2:.2f}px {cs['outline_color']};"
                       "paint-order:stroke fill;")
        cfs, clh = css_text_size(cs["font"], ass_font_size(cs["font"], cs["size"], scale), k)
        cap_div = (f'<div id="cap" style="position:absolute;left:{cs["x_pct"]}%;top:{cs["y_pct"]}%;'
                   f'transform:translate(-50%,-50%);width:max-content;max-width:88%;text-align:center;'
                   f'pointer-events:none;font-family:\'{fam}\',Arial,sans-serif;'
                   f'font-size:{cfs:.2f}px;font-weight:{700 if cs["bold"] else 400};'
                   f'color:{cs["color"]};line-height:{clh:.2f}px;text-wrap:balance;{cstroke}"><span id="capt" style="{cbg}"></span></div>')
        cues = json.dumps([{"s": c["start"], "e": c["end"],
                            "t": (c["text"].upper() if cs["upper"] else c["text"])} for c in caps["cues"]])

    return """
<div style="font-family:Arial,sans-serif;color:#ddd;width:%(pw)dpx">
 <div style="position:relative;width:%(pw)dpx;height:%(ph)dpx;overflow:hidden;background:#000;border-radius:6px">
  <video id="v" src="data:video/mp4;base64,%(b64)s" controls playsinline style="width:100%%;height:100%%;object-fit:%(fit)s;display:block"></video>
  %(title)s%(cap)s
 </div>
</div>
<script>
const C=%(cfg)s, CUES=%(cues)s, v=document.getElementById('v'), ti=document.getElementById('ti'),
      capt=document.getElementById('capt');
const clamp=(x,a,b)=>Math.min(Math.max(x,a),b), ease=x=>1-Math.pow(1-x,3);
function frame(){
 const now=v.currentTime;
 if(capt){let txt='';for(const c of CUES){if(now>=c.s&&now<c.e){txt=c.t;break;}} 
  capt.textContent=txt;capt.parentNode.style.display=txt?'block':'none';}
 if(C&&ti){
  const t=now-C.start, ms=t*1000, D=C.dur*1000;
  if(t<0||t>C.dur){ti.style.opacity=0;}
  else{
   let sc=1,op=1,dy=0;const a=C.anim;
   if(a.startsWith('Pop')){sc=ms<140?1+.25*ease(ms/140):(ms<260?1.25-.25*ease((ms-140)/120):1);op=ms>D-150?(D-ms)/150:1;}
   else if(a.startsWith('Slam')){sc=ms<130?2.6-1.6*ease(ms/130):1;op=ms>D-120?(D-ms)/120:1;}
   else if(a.startsWith('Zoom')){sc=.55+.45*ease(clamp(ms/380,0,1));op=Math.min(clamp(ms/140,0,1),clamp((D-ms)/140,0,1));}
   else if(a.startsWith('Slide')){dy=C.slide*(1-ease(clamp(ms/260,0,1)));op=Math.min(clamp(ms/160,0,1),clamp((D-ms)/140,0,1));}
   else if(a==='None'){op=1;}
   else{op=Math.min(clamp(ms/260,0,1),clamp((D-ms)/260,0,1));}
   ti.style.opacity=clamp(op,0,1);
   ti.style.transform='translate(-50%%,calc(-50%% + '+dy+'px)) scale('+sc+')';
  }
 }
 requestAnimationFrame(frame);}
frame();
</script>""" % dict(pw=pw, ph=ph, b64=b64, fit=fit_css, title=title_div, cap=cap_div, cfg=cfg, cues=cues)


# ------------------------------------------------------------------- exports
TITLE_KEYS = ("font", "color", "outline_color", "outline", "upper", "bold", "box", "box_color")


def motion_style(project):
    t = project.get("title") or {}
    return {**DEFAULT_MOTION_STYLE, **{k: t[k] for k in TITLE_KEYS if k in t}}


def motion_items(project):
    t = project.get("title") or {}
    if t.get("off_by_user") or not str(t.get("text", "")).strip():
        return []
    return [dict(text=t["text"], start=0.0, dur=36000.0,
                 anim=(t.get("anim", "None") if t.get("anim_chosen") else "None"), x=float(t.get("x", 50)), y=float(t.get("y", 15)),
                 size=float(t.get("size", 100)))]


def render_audio_preview(project_dir: Path, project: dict, keep, processed: bool, name: str):
    """Listenable audio: the original, or the audio after silence removal and noise reduction."""
    src = project_dir / project["proxy_file"]
    out = project_dir / name
    if processed:
        stm = [f"[0:a]atrim=start={a:.5f}:end={b:.5f},asetpts=PTS-STARTPTS[a{i}]" for i, (a, b) in enumerate(keep)]
        dn = denoise_filter(project)
        stm.append("".join(f"[a{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=0:a=1"
                   + (f"[acr];[acr]{dn}[aout]" if dn else "[aout]"))
    else:
        stm = ["[0:a]anull[aout]"]
    run_graph(["-i", str(src)], ";".join(stm), ["-map", "[aout]", "-ac", "1", "-ar", "32000", "-c:a", "pcm_s16le",
                                               str(out)], str(project_dir), project_dir / "graph_aud.txt")
    return out


def render_title_clip(project_dir: Path, project: dict, size, fit, style):
    """Short exact preview (with sound) of the opening of the video: title animation plus captions."""
    keep, fps = project["keep"], project["fps"]
    w, h = (size[0] // 3) // 2 * 2, (size[1] // 3) // 2 * 2
    cues = make_cues(caption_segments(project), style["max_words"]) if (style["show"] and caption_segments(project)) else []
    motion = {"style": motion_style(project), "items": motion_items(project)}
    if not motion["items"]:
        raise RuntimeError("Turn on the title and type some text first.")
    ass_name = (project_dir / "title_prev.ass").relative_to(APP_DIR).as_posix()
    (APP_DIR / ass_name).write_text(build_ass(cues, w, h, style, motion=motion), encoding="utf-8")
    prepare_export_fonts([style["font"], motion["style"]["font"]])
    _, total = _offsets(keep)
    it = motion["items"][0]
    length = min(total, 12.0)
    out = project_dir / "title_preview.mp4"
    err = ""
    for fd in _fontsdir_variants():
        stm = cut_statements(keep, fps, denoise_filter(project)) + fit_statements("[vc]", fit, w, h, 30, "vb")
        stm.append(f"[vb]ass=filename={ass_name}{fd}[vout]")
        try:
            run_graph(["-i", str(project_dir / project["proxy_file"])], ";".join(stm),
                      ["-map", "[vout]", "-map", "[ac]", "-t", f"{length:.2f}", *PROXY_OUT, "-c:a", "aac",
                       "-movflags", "+faststart", str(out)], str(APP_DIR), project_dir / "graph_title.txt")
            return out
        except RuntimeError as exc:
            err = str(exc)
    raise RuntimeError("Title preview failed. " + err[-1200:])


def export_still(project_dir: Path, project: dict, size, fit, style, t_out, plan, fx, fy):
    """One exact frame (zoom, captions and motion text; b-roll is not drawn here)."""
    keep = project["keep"]
    w, h = size
    proxy = project_dir / project["proxy_file"]
    src_t = out_to_src(t_out, keep)
    cues = make_cues(caption_segments(project), style["max_words"]) if (style["show"] and caption_segments(project)) else []
    motion = {"style": motion_style(project), "items": motion_items(project)}
    ass_name = None
    if cues or motion["items"]:
        ass_name = (project_dir / "still_subs.ass").relative_to(APP_DIR).as_posix()
        (APP_DIR / ass_name).write_text(build_ass(cues, w, h, style, motion=motion), encoding="utf-8")
        prepare_export_fonts([style["font"], motion["style"]["font"]])
    z = zoom_at(plan, t_out) if plan else 1.0
    stm = fit_statements("[0:v]", fit, w, h, 30, "vb")
    last = "vb"
    if z > 1.001:
        stm.append(f"[vb]scale=iw*{z:.4f}:ih*{z:.4f}:flags=bicubic,"
                   f"crop={w}:{h}:x='(iw-ow)*{fx:.3f}':y='(ih-oh)*{fy:.3f}'[vz]")
        last = "vz"
    stm.append(f"[{last}]setpts=PTS+{t_out:.3f}/TB[vs]")   # the frame carries its real time, so animations are right
    out = project_dir / "style_preview.png"
    err = ""
    for fd in (_fontsdir_variants() if ass_name else [""]):
        tail = f"[vs]ass=filename={ass_name}{fd}[vout]" if ass_name else "[vs]null[vout]"
        r = subprocess.run([FFMPEG, "-y", "-ss", f"{src_t:.3f}", "-i", str(proxy), "-filter_complex",
                            ";".join(stm + [tail]), "-map", "[vout]", "-frames:v", "1", str(out)],
                           capture_output=True, text=True, cwd=str(APP_DIR))
        if r.returncode == 0:
            return out
        err = r.stderr
    raise RuntimeError("Preview failed. FFmpeg said: " + err[-1800:])


def export_final(project_dir: Path, project: dict, size, fit, style, plan, fx, fy):
    keep, fps = project["keep"], project["fps"]
    w, h = size
    video = project_dir / project["video_file"]
    cues = make_cues(caption_segments(project), style["max_words"]) if (style["show"] and caption_segments(project)) else []
    motion = {"style": motion_style(project), "items": motion_items(project)}
    ass_name = None
    if cues or motion["items"]:
        ass_name = (project_dir / "export_subs.ass").relative_to(APP_DIR).as_posix()
        (APP_DIR / ass_name).write_text(build_ass(cues, w, h, style, motion=motion), encoding="utf-8")
        prepare_export_fonts([style["font"], motion["style"]["font"]])
    out = project_dir / f"export_{w}x{h}_{fit}.mp4"
    err = ""
    for fd in (_fontsdir_variants() if ass_name else [""]):
        stm = cut_statements(keep, fps, denoise_filter(project)) + fit_statements("[vc]", fit, w, h, 30, "vb")
        last = "[vb]"
        if plan and zoom_active(plan):
            stm += zoom_statements("[vb]", w, h, plan, fx, fy, "vz")
            last = "[vz]"
        stm.append(f"{last}ass=filename={ass_name}{fd}[vout]" if ass_name else f"{last}null[vout]")
        try:
            log = run_graph(["-i", str(video)], ";".join(stm),
                            ["-map", "[vout]", "-map", "[ac]", *VIDEO_OUT, "-c:a", "aac", "-b:a", "192k",
                             "-movflags", "+faststart", str(out)], str(APP_DIR), project_dir / "graph_export.txt")
        except RuntimeError as exc:
            err = str(exc)
            continue
        if motion["items"]:
            t_chk = float(motion["items"][0]["start"]) + 0.45
        else:
            t_chk = cues[0]["start"] + 0.3 if cues else 0.5
        check = grab_frame(out, t_chk, project_dir / "export_check.png")
        return out, f"fontsdir option used: {fd or 'none'}\n" + log[-3000:], check
    raise RuntimeError("Export failed. " + err[-1800:])


# ============================================================= transcription
def spread_words(text, start, end):
    tokens = text.split()
    if not tokens:
        return []
    weights = [max(1, len(t)) for t in tokens]
    total = sum(weights)
    dur = max(0.0, end - start)
    words, cur = [], start
    for tok, w in zip(tokens, weights):
        nxt = cur + dur * w / total
        words.append({"start": round(cur, 3), "end": round(nxt, 3), "word": " " + tok, "estimated": True})
        cur = nxt
    return words


def rebuild_segments(edited_df, original_segments):
    lookup = {(round(s["start"], 2), round(s["end"], 2)): s for s in original_segments}
    cleaned = []
    for _, row in edited_df.iterrows():
        if pd.isna(row["Start (sec)"]) or pd.isna(row["End (sec)"]):
            continue
        start = max(0.0, float(row["Start (sec)"]))
        end = max(start, float(row["End (sec)"]))
        text = "" if pd.isna(row["Text"]) else str(row["Text"]).strip()
        if not text:
            continue
        orig = lookup.get((round(start, 2), round(end, 2)))
        tokens = text.split()
        if orig and orig.get("words") and orig["text"].strip() == text:
            words = orig["words"]
        elif orig and orig.get("words") and len(orig["words"]) == len(tokens):
            words = [dict(w, word=" " + t) for w, t in zip(orig["words"], tokens)]
        else:
            words = spread_words(text, start, end)
        low = bool(orig and orig.get("low") and orig["text"].strip() == text)
        cleaned.append({"start": round(start, 3), "end": round(end, 3), "text": text, "words": words, "low": low})
    return cleaned


def read_wav_f32(path):
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def split_chunks(audio, sr, lo=14.0, hi=28.0):
    """Cut the audio into ~15-28 s pieces, always at the quietest moment, so every piece is transcribed on its own."""
    n, frame = len(audio), int(sr * 0.02)
    chunks, start = [], 0
    while start < n:
        if (n - start) / sr <= hi:
            chunks.append((start, n))
            break
        a0, b0 = start + int(lo * sr), start + int(hi * sr)
        seg = audio[a0:b0]
        m = len(seg) // frame
        rms = np.sqrt((seg[: m * frame].reshape(m, frame) ** 2).mean(axis=1))
        smooth = np.convolve(rms, np.ones(3) / 3, mode="same")
        cut = a0 + int(np.argmin(smooth)) * frame + frame // 2
        chunks.append((start, cut))
        start = cut
    return chunks


def script_ok(text, language):
    """False when a line contains a script that cannot be right (Bengali, Hangul, Cyrillic...) or broken characters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return True
    allowed = ("LATIN", "DEVANAGARI") + (("ARABIC",) if language == "ur" else ())
    bad = sum(1 for c in letters if not unicodedata.name(c, "").startswith(allowed))
    return bad == 0 and "\ufffd" not in text


def transcribe(wav_path, profile, model_size, device_choice, custom_prompt, language_choice, use_vad,
               max_chunks=None, task="transcribe"):
    cfg = PROFILES[profile]
    lang_setting = LANGUAGE_CHOICES.get(language_choice) or cfg["language"]
    language = None if lang_setting == "auto" else lang_setting
    base_prompt = (cfg["prompt"] + " " + custom_prompt).strip()
    audio, sr = read_wav_f32(wav_path)
    chunks = split_chunks(audio, sr)
    if max_chunks:
        chunks = chunks[:max_chunks]
    attempts = []
    if device_choice in ("Auto (GPU if available)", "NVIDIA GPU"):
        attempts.append(("cuda", "int8_float16"))
    if device_choice in ("Auto (GPU if available)", "CPU"):
        attempts.append(("cpu", "int8"))
    errors = []
    for device, compute_type in attempts:
        model = None
        try:
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
            segments, tail, detected, prob = [], "", None, None
            for a0, b0 in chunks:
                prompt = (base_prompt + " " + tail).strip()[-700:]
                kwargs = dict(language=language, task=task, beam_size=5, initial_prompt=prompt, word_timestamps=True,
                              condition_on_previous_text=False, temperature=[0.0, 0.2, 0.4],
                              compression_ratio_threshold=2.4, log_prob_threshold=-1.0, no_speech_threshold=0.6,
                              hallucination_silence_threshold=2.0, repetition_penalty=1.05, vad_filter=use_vad)
                if use_vad:
                    kwargs["vad_parameters"] = dict(threshold=0.3, min_silence_duration_ms=700, speech_pad_ms=300)
                try:
                    it, info = model.transcribe(audio[a0:b0], **kwargs)
                except TypeError:  # older faster-whisper without the newer options
                    kwargs.pop("hallucination_silence_threshold", None)
                    kwargs.pop("repetition_penalty", None)
                    it, info = model.transcribe(audio[a0:b0], **kwargs)
                t0 = a0 / sr
                added = []
                for seg in it:
                    text = seg.text.strip()
                    if not text:
                        continue
                    words = [{"start": round(float(w.start) + t0, 3), "end": round(float(w.end) + t0, 3),
                              "word": w.word} for w in (seg.words or [])]
                    low = (getattr(seg, "avg_logprob", 0.0) < -1.0 or getattr(seg, "no_speech_prob", 0.0) > 0.6
                           or getattr(seg, "compression_ratio", 1.0) > 2.4 or (task == "transcribe" and not script_ok(text, language)))
                    added.append({"start": round(float(seg.start) + t0, 3), "end": round(float(seg.end) + t0, 3),
                                  "text": text, "words": words, "low": bool(low)})
                segments += added
                good = [x["text"] for x in added if not x["low"]]
                tail = " ".join(good[-2:])[-250:] if good else ""
                if detected is None:
                    detected, prob = info.language, float(info.language_probability)
            return segments, {
                "detected_language": detected, "language_probability": round(prob or 0.0, 3),
                "engine": "faster-whisper", "model": model_size, "device_used": device,
                "compute_type": compute_type, "vad_filter": use_vad, "chunks": len(chunks),
                "fallback_reason": "; ".join(errors) or None,
            }
        except Exception as exc:
            errors.append(f"{device}/{compute_type}: {str(exc)[:300]}")
        finally:
            del model
            gc.collect()
    raise RuntimeError("Transcription failed. " + " | ".join(errors))


def render_cut_audio(project_dir: Path, project: dict):
    """Speech-only audio of the reviewed cut: this is what gets transcribed, so timings match the cut."""
    keep = project["keep"]
    stm = [f"[0:a]atrim=start={a:.5f}:end={b:.5f},asetpts=PTS-STARTPTS[a{i}]" for i, (a, b) in enumerate(keep)]
    stm.append("".join(f"[a{i}]" for i in range(len(keep))) + f"concat=n={len(keep)}:v=0:a=1[ac]")
    dn = denoise_filter(project)
    stm.append("[ac]" + (dn + "," if dn else "") + "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11[aout]")
    out = project_dir / "cut_audio_16khz.wav"
    run_graph(["-i", str(project_dir / project["audio_file"])], ";".join(stm),
              ["-map", "[aout]", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out)],
              str(project_dir), project_dir / "graph_audio.txt")
    return out


def translate_to_english(texts, api_key, model):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        "Translate each line of this spoken Hindi / Hinglish / Kokni video transcript into natural, short, "
        "punchy English suitable for on-screen captions. Keep names and English words as they are. "
        "Return ONLY a JSON array of strings with exactly " + str(len(texts)) + " items, in the same order.\n\n"
        + json.dumps(texts, ensure_ascii=False))
    msg = client.messages.create(model=model, max_tokens=4000, messages=[{"role": "user", "content": prompt}])
    match = re.search(r"\[.*\]", msg.content[0].text, re.DOTALL)
    if not match:
        raise RuntimeError("Translation reply was not a JSON list.")
    result = json.loads(match.group(0))
    if len(result) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} lines back, got {len(result)}. Try again.")
    return [str(x).strip() for x in result]


# ============================================================ project actions
def clean_path(text):
    """Accepts a pasted path in any common form: with quotes, ~, or a file:/// link."""
    t = text.strip().strip('"').strip("'").strip()
    if t.lower().startswith("file:"):
        from urllib.parse import unquote, urlparse
        u = urlparse(t)
        t = unquote(u.path)
        if re.match(r"^/[A-Za-z]:", t):
            t = t[1:]
        if u.netloc and u.netloc != "localhost":
            t = "//" + u.netloc + t
    return os.path.expanduser(t)


def new_project(uploaded, path_text):
    if uploaded is None:
        raw = path_text.strip().strip('"').strip("'").strip()
        if raw.lower().startswith(("http://", "https://")):
            raise RuntimeError("That is a web link. Download the video to this computer first, then paste its "
                               "file path (right-click the file, Copy as path).")
        src = Path(clean_path(path_text))
        if src.is_dir():
            raise RuntimeError(f"That is a folder, not a video file: {src}")
        if not src.is_file():
            raise RuntimeError(f"File not found: {src}\nCheck the path. In File Explorer, right-click the video "
                               "and choose Copy as path, then paste it here.")
        if src.suffix.lower().lstrip(".") not in VIDEO_TYPES + ["mkv", "webm", "avi", "mts", "m2ts", "wmv"]:
            raise RuntimeError(f"Unsupported file type: {src.suffix}")
        try:
            with open(src, "rb") as fh:
                fh.read(1024)
        except OSError as exc:
            raise RuntimeError(f"The file cannot be read ({exc}). Close any app using it, or copy it to a local "
                               "folder first.")
        name = src.name
    else:
        name = uploaded.name
    project_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    project_dir = PROJECTS_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    video_name = safe_name(name)
    dest = project_dir / video_name
    if uploaded is not None:
        uploaded.seek(0)
        with open(dest, "wb") as f:
            shutil.copyfileobj(uploaded, f, 16 * 1024 * 1024)
    else:
        try:
            os.link(src, dest)          # instant on the same drive, no second copy of a large file
        except Exception:
            shutil.copyfile(src, dest)
    project = {"project_id": project_id, "created_at": datetime.now().isoformat(timespec="seconds"),
               "original_filename": name, "video_file": video_name, "status": "Uploaded", "keep_version": 0}
    save_project(project_dir, project)
    return project_dir, project


def prepare_project(project_dir: Path, project: dict):
    video = project_dir / project["video_file"]
    info = probe_video(video)
    if not info["has_audio"]:
        raise RuntimeError("This video has no audio track, so silence cannot be detected.")
    project["video_info"] = info
    project["fps"] = info["fps"] or 30.0
    project["audio_file"] = "audio_16khz.wav"
    project["proxy_file"] = "proxy.mp4"
    extract_audio(video, project_dir / project["audio_file"])
    make_proxy(video, project_dir / project["proxy_file"], project["fps"])
    pinfo = probe_video(project_dir / project["proxy_file"])
    project["proxy_size"] = [pinfo["width"], pinfo["height"]]
    project["status"] = "Ready for draft cut"
    save_project(project_dir, project)


def apply_keep(project_dir: Path, project: dict, new_keep, as_draft=False):
    new_keep = snap_keep(new_keep, project["fps"], project["video_info"].get("duration_seconds"))
    if not new_keep:
        raise RuntimeError("Nothing would be left after this change.")
    if as_draft:
        project["draft_keep"] = new_keep
        project["keep_history"] = []
    else:
        hist = project.setdefault("keep_history", [])
        hist.append(project["keep"])
        del hist[:-30]
    project["keep"] = new_keep
    project["keep_is_full"] = False
    project["keep_version"] = project.get("keep_version", 0) + 1
    if isinstance(project.get("zoom"), dict):
        project["zoom"].pop("overrides", None)
    render_review(project_dir, project)
    project["status"] = "Cut ready – review it"
    save_project(project_dir, project)


def ensure_keep(project_dir: Path, project: dict):
    """Every step can run on its own: with no cut made, the cut is simply the whole video."""
    if project.get("keep"):
        return project["keep"]
    if "proxy_file" not in project:
        return None
    dur = project["video_info"].get("duration_seconds") or 0
    full = snap_keep([(0.0, dur)], project["fps"], dur)
    if not full:
        return None
    project["keep"] = full
    project["keep_is_full"] = True
    project["keep_version"] = project.get("keep_version", 0) + 1
    shutil.copyfile(project_dir / project["proxy_file"], project_dir / "review_cut.mp4")
    save_project(project_dir, project)
    return full


def reset_to_full(project_dir: Path, project: dict):
    project.pop("keep", None)
    project.pop("draft_keep", None)
    project["keep_history"] = []
    ensure_keep(project_dir, project)


NO_PROJECT_MSG = "This project has no preview copy (made with an older version). Import the video again in this section."
FULL_NOTE = "No cut applied: this uses the whole video. You can still remove pauses in step 2."


# ===================================================================== steps
def goto(step_label):
    st.session_state["_goto"] = step_label
    st.rerun()


def step_draft(project_dir, project):
    pid = project["project_id"]
    st.subheader("Remove silent parts and background noise")
    if "audio_file" not in project:
        st.warning("This project was made with an older version and has no preview copy. "
                   "Create a new project from the Upload step.")
        return
    wav = project_dir / project["audio_file"]
    fps = project["fps"]
    db, dur_a = audio_levels(str(wav), wav.stat().st_mtime)
    duration = project["video_info"].get("duration_seconds") or dur_a
    cs = project.get("cut_settings") or {}
    presets_all = cut_presets()
    presets = list(presets_all)
    preset = st.selectbox("Cut style", presets,
                          index=presets.index(cs["preset"]) if cs.get("preset") in presets else 0,
                          key=f"preset_{pid}")
    with st.expander("Fine-tune", expanded=preset == "Custom"):
        offset_db = st.slider("Sensitivity (dB below your normal speech level)", -45, -15,
                              int(cs.get("offset_db", -30)), 1, key=f"offdb_{pid}",
                              help="Closer to -15 treats more as silence. If words are being cut, move it left.")
        if preset == "Custom":
            min_silence = st.slider("Remove pauses longer than (seconds)", 0.10, 1.5,
                                    float(cs.get("min_silence", 0.30)), 0.05, key=f"minsil_{pid}")
            pad = st.slider("Breathing room kept around speech (seconds)", 0.0, 0.4,
                            float(cs.get("pad", 0.08)), 0.01, key=f"pad_{pid}")
        else:
            min_silence, pad = presets_all[preset]["min_silence"], presets_all[preset]["pad"]
            st.caption(f"This style removes pauses longer than {min_silence}s and keeps {pad}s around speech. "
                       "Choose Custom to change these.")
    with st.expander("Optional: learn my editing style from my own finished reels"):
        st.caption("Put 2 to 6 of your finished talking-head reels in this folder, then click Analyse.")
        st.code(str(REF_DIR), language="text")
        st.caption("It measures how long the pauses are in your reels and saves a cut style that matches. "
                   "Works best on reels without loud background music.")
        vids = sorted(v for v in REF_DIR.iterdir() if v.is_file() and v.suffix.lower().lstrip(".") in VIDEO_TYPES)
        st.write(f"{len(vids)} reference video(s) found.")
        if st.button("Analyse my reference reels", disabled=not vids, key=f"ref_{pid}"):
            try:
                with st.spinner("Analysing your reels..."):
                    results = [r for r in (analyse_reference(v) for v in vids) if r]
                    learned = learn_style(results)
                st.session_state["ref_results"] = (results, learned)
            except Exception as exc:
                st.error(str(exc))
        ref_res = st.session_state.get("ref_results")
        if ref_res:
            results, learned = ref_res
            st.dataframe(pd.DataFrame([{
                "Video": r["name"], "Length (s)": r["duration"], "Speech %": r["speech_pct"], "Phrases": r["runs"],
                "Median pause (s)": round(float(np.median(r["gaps"])), 2) if r["gaps"] else None}
                for r in results]), hide_index=True, use_container_width=True)
            if learned:
                st.success(f"Learned: remove pauses longer than {learned['min_silence']}s and keep {learned['pad']}s "
                           f"around speech (your typical pause is {learned['median_pause']}s).")
                if st.button("Save as my cut style", key=f"refsave_{pid}"):
                    MY_STYLE_FILE.write_text(json.dumps(learned), encoding="utf-8")
                    st.rerun()
            else:
                st.warning("Not enough phrases found to learn from. Try longer talking-head reels.")
    st.markdown("**Background noise reduction (optional)**")
    dn_old = project.get("denoise") or {"on": False, "level": 12}
    d1, d2 = st.columns([1, 2])
    dn_on = d1.checkbox("Reduce background noise", value=bool(dn_old.get("on", False)), key=f"dn_on_{pid}")
    dn_level = d2.slider("Strength (dB of noise removed)", 6, 30, int(dn_old.get("level", 12)), key=f"dn_lv_{pid}",
                         help="12 is gentle. Raise it for fan or street noise; too high makes the voice sound watery.")
    dn_new = dict(on=dn_on, level=dn_level)
    if dn_new != project.get("denoise"):
        project["denoise"] = dn_new
        save_project(project_dir, project)
    if project.get("keep") and st.button("Apply noise setting to the preview", key=f"dn_apply_{pid}"):
        with st.spinner("Rendering..."):
            render_review(project_dir, project)
        st.rerun()
    raw, thr = compute_keep(db, duration, offset_db, min_silence, pad)
    keep = snap_keep(raw, fps, duration)
    if not keep:
        st.warning("No speech detected with these settings. Move the sensitivity slider towards -45.")
        return
    _, kept = _offsets(keep)
    m1, m2 = st.columns(2)
    m1.metric("Original", f"{duration:.1f}s")
    m2.metric("After cut", f"{kept:.1f}s")
    m3, m4 = st.columns(2)
    m3.metric("Removed", f"{duration - kept:.1f}s")
    m4.metric("Clips", len(keep))
    st.image(levels_image(db, thr, keep, duration), use_container_width=True,
             caption="Loudness over time. Green = kept, red line = silence threshold.")
    if project.get("keep") and not project.get("keep_is_full"):
        st.info("A cut already exists for this project. Creating the draft again replaces it and discards "
                "your earlier cut.")
    if st.button("Create draft cut and preview", type="primary"):
        try:
            with st.spinner("Rendering the draft preview..."):
                project["cut_settings"] = dict(preset=preset, offset_db=offset_db, min_silence=min_silence,
                                               pad=pad, threshold_db=thr)
                apply_keep(project_dir, project, keep, as_draft=True)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if st.button("Skip: use the whole video without cutting", key=f"skipcut_{pid}"):
        reset_to_full(project_dir, project)
        st.rerun()
    review = project_dir / "review_cut.mp4"
    st.markdown("**Listen**")
    la, lb = st.columns(2)
    if la.button("Hear original audio (first 60 s)", key=f"aud_o_{pid}"):
        try:
            with st.spinner("Preparing..."):
                st.session_state[f"aud_o_path_{pid}"] = str(render_audio_preview(
                    project_dir, project, [(0.0, min(duration, 60.0))], False, "audio_original.wav"))
        except Exception as exc:
            st.error(str(exc))
    if lb.button("Hear audio after cut + noise reduction (first 60 s)", key=f"aud_p_{pid}"):
        try:
            with st.spinner("Preparing..."):
                clipped, acc = [], 0.0
                for a, b in keep:
                    if acc >= 60.0:
                        break
                    b2 = min(b, a + (60.0 - acc))
                    clipped.append((a, b2))
                    acc += b2 - a
                st.session_state[f"aud_p_path_{pid}"] = str(render_audio_preview(
                    project_dir, project, clipped, True, "audio_processed.wav"))
        except Exception as exc:
            st.error(str(exc))
    for lab, key, col in (("Original", f"aud_o_path_{pid}", la), ("After cut + noise reduction", f"aud_p_path_{pid}", lb)):
        pth = st.session_state.get(key)
        if pth and Path(pth).exists():
            col.caption(lab)
            col.audio(Path(pth).read_bytes(), format="audio/wav")
    if project.get("keep") and not project.get("keep_is_full") and review.exists():
        st.markdown("**Current draft**")
        vc, _ = st.columns([1, 2])
        vc.video(str(review))


def step_review(project_dir, project):
    pid = project["project_id"]
    st.subheader("Review the cut and remove retakes")
    keep = ensure_keep(project_dir, project)
    if not keep:
        st.info(NO_PROJECT_MSG)
        return
    if project.get("keep_is_full"):
        st.info(FULL_NOTE + " You can also trim the whole video here.")
    offs, total = _offsets(keep)
    dur = project["video_info"].get("duration_seconds") or total
    review = project_dir / "review_cut.mp4"
    if not review.exists():
        with st.spinner("Rendering the preview..."):
            render_review(project_dir, project)
    kv = project.get("keep_version", 0)
    st.caption(f"{len(keep)} clips · {total:.1f}s (original {dur:.1f}s). Watch the preview, note the times "
               "where you repeated yourself, then remove them on the right.")
    left, right = st.columns([1, 2])
    with left:
        jump = st.number_input("Play from (seconds)", 0.0, float(max(total - 0.1, 0.1)), 0.0, 1.0,
                               key=f"jump_{pid}_{kv}")
        st.video(str(review), start_time=int(jump))
        b1, b2 = st.columns(2)
        if b1.button("Undo last change", disabled=not project.get("keep_history")):
            project["keep"] = project["keep_history"].pop()
            project["keep_version"] = kv + 1
            with st.spinner("Rendering..."):
                render_review(project_dir, project)
            save_project(project_dir, project)
            st.rerun()
        if b2.button("Reset to draft", disabled=not project.get("draft_keep")):
            project["keep"] = project["draft_keep"]
            project["keep_history"] = []
            project["keep_version"] = kv + 1
            with st.spinner("Rendering..."):
                render_review(project_dir, project)
            save_project(project_dir, project)
            st.rerun()
    with right:
        st.markdown("**A. Remove a part you want gone (time range in the preview)**")
        st.caption("Example: you repeated a line between 12.4 s and 15.9 s in the player, so add From 12.4, To 15.9.")
        empty = pd.DataFrame({"From (s)": pd.Series(dtype=float), "To (s)": pd.Series(dtype=float)})
        ranges_df = st.data_editor(empty, num_rows="dynamic", hide_index=True, use_container_width=True,
                                   key=f"ranges_{pid}_{kv}")
        if st.button("Remove these ranges"):
            rng = []
            for _, r in ranges_df.iterrows():
                if pd.notna(r["From (s)"]) and pd.notna(r["To (s)"]) and r["To (s)"] > r["From (s)"]:
                    rng.append((float(r["From (s)"]), float(r["To (s)"])))
            if not rng:
                st.warning("Add at least one range with To greater than From.")
            else:
                try:
                    with st.spinner("Applying and rendering..."):
                        apply_keep(project_dir, project, remove_output_ranges(keep, rng))
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        st.markdown("**B. Remove whole clips**")
        rows = [{"Clip": i + 1, "Starts at (s)": round(offs[i], 2), "Length (s)": round(b - a, 2), "Remove": False}
                for i, (a, b) in enumerate(keep)]
        clips_df = st.data_editor(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=260,
                                  disabled=["Clip", "Starts at (s)", "Length (s)"], key=f"clips_{pid}_{kv}")
        if st.button("Remove ticked clips"):
            drop = {int(r["Clip"]) - 1 for _, r in clips_df.iterrows() if r["Remove"]}
            if not drop:
                st.warning("Tick at least one clip.")
            else:
                try:
                    with st.spinner("Applying and rendering..."):
                        apply_keep(project_dir, project, [k for i, k in enumerate(keep) if i not in drop])
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


def step_transcript(project_dir, project):
    pid = project["project_id"]
    st.subheader("Transcribe")
    keep = ensure_keep(project_dir, project)
    if not keep:
        st.info(NO_PROJECT_MSG)
        return
    if project.get("keep_is_full"):
        st.caption(FULL_NOTE)
    stale = project.get("transcript") and project.get("transcript_keep_version") != project.get("keep_version")
    if stale:
        st.warning("You changed the cut after transcribing. The transcript below no longer matches the video. "
                   "Run the transcription again.")
    with st.expander("Transcription settings", expanded=not project.get("transcript") or bool(stale)):
        a, b, c = st.columns(3)
        profile = a.radio("Profile", list(PROFILES), horizontal=True,
                          index=list(PROFILES).index(project.get("profile", "Hindi")), key=f"tprof_{pid}")
        a.caption(PROFILES[profile]["description"])
        model_size = b.selectbox("Whisper model", WHISPER_MODELS, index=1, key=f"tmodel_{pid}",
                                 help="medium suits a 4 GB GPU. large-v3-turbo is a good step up and still fits. "
                                      "large-v3 is the most accurate but may run out of memory.")
        device_choice = b.selectbox("Processing device", ["Auto (GPU if available)", "CPU", "NVIDIA GPU"],
                                    key=f"tdev_{pid}")
        language_choice = c.selectbox("Language", list(LANGUAGE_CHOICES), key=f"tlang_{pid}",
                                      help="Kokni defaults to Marathi. Use the quick test to compare Marathi and "
                                           "Hindi on the first 25 seconds. Avoid Auto-detect: it jumps between languages.")
        use_vad = c.checkbox("Skip non-speech (VAD)", value=False, key=f"tvad_{pid}",
                             help="Not needed: silence is already removed.")
        custom_prompt = st.text_area("Custom vocabulary / names (optional)", value=project.get("custom_prompt", ""),
                                     height=80, key=f"tprompt_{pid}")
    bt1, bt2 = st.columns(2)
    if bt1.button("Transcribe the video", type="primary"):
        try:
            with st.spinner("Preparing audio and transcribing..."):
                wav = render_cut_audio(project_dir, project)
                segments, info = transcribe(wav, profile, model_size, device_choice, custom_prompt,
                                            language_choice, use_vad)
            project.update(profile=profile, custom_prompt=custom_prompt, transcript=segments,
                           transcript_info=info, transcript_keep_version=project.get("keep_version", 0),
                           status="Transcribed – review required")
            project.pop("english_segments", None)
            save_project(project_dir, project)
            write_text_exports(project_dir, segments)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if bt2.button("Quick test: first 25 seconds only"):
        try:
            with st.spinner("Testing the first 25 seconds..."):
                wav = render_cut_audio(project_dir, project)
                t_segs, t_info = transcribe(wav, profile, model_size, device_choice, custom_prompt,
                                            language_choice, use_vad, max_chunks=1)
            st.session_state[f"ttest_{pid}"] = {"lang": t_info.get("detected_language"),
                                                "device": t_info.get("device_used"),
                                                "lines": [s["text"] for s in t_segs]}
        except Exception as exc:
            st.error(str(exc))
    tt = st.session_state.get(f"ttest_{pid}")
    if tt:
        st.caption(f"Quick test result (language {tt['lang']}, {tt['device']}). Change the language or model above "
                   "and test again until this looks right, then transcribe the whole video.")
        st.code("\n".join(tt["lines"]) or "(nothing recognised)", language="text")
    segs = project.get("transcript")
    if not segs:
        return
    info = project.get("transcript_info") or {}
    st.caption(f"Model: {info.get('model')} · Device: {info.get('device_used')} ({info.get('compute_type')}) · "
               f"Language: {info.get('detected_language')} ({info.get('language_probability')})")
    if info.get("fallback_reason"):
        st.warning("GPU was skipped: " + info["fallback_reason"])
    low_n = sum(1 for x in segs if x.get("low"))
    if low_n:
        st.warning(f"{low_n} line(s) look unreliable (marked in the Check column). Listen to them and correct the "
                   "text, or run the quick test again with another language or model.")
    _, total = _offsets(keep)
    if segs and total - segs[-1]["end"] > 3:
        st.warning(f"The transcript ends {total - segs[-1]['end']:.1f}s before the video does.")
    review = project_dir / "review_cut.mp4"
    vcol, ecol = st.columns([1, 2])
    with vcol:
        if review.exists():
            vtt = project_dir / "preview_subtitles.vtt"
            write_vtt(make_cues(caption_segments(project)), vtt)
            st.video(str(review), subtitles={"Subtitles": str(vtt)})
            st.caption("Press CC in the player if subtitles do not show.")
    with ecol:
        st.markdown("**Transcript editor** (times are in the cut video)")
        rows = [{"Start (sec)": s["start"], "End (sec)": s["end"], "Text": s["text"],
                 "Check": "⚠ check" if s.get("low") else ""} for s in segs]
        edited = st.data_editor(
            pd.DataFrame(rows), use_container_width=True, num_rows="dynamic", height=300, disabled=["Check"],
            column_config={"Start (sec)": st.column_config.NumberColumn(format="%.3f", min_value=0.0),
                           "End (sec)": st.column_config.NumberColumn(format="%.3f", min_value=0.0),
                           "Text": st.column_config.TextColumn(width="large")},
            key=f"editor_{pid}_{project.get('transcript_keep_version')}")
        if st.button("Save transcript corrections", type="primary"):
            cleaned = rebuild_segments(edited, segs)
            project["transcript"] = cleaned
            project.pop("english_segments", None)
            save_project(project_dir, project)
            write_text_exports(project_dir, cleaned)
            st.success("Saved. Word timings were preserved.")
            st.rerun()
    d1, d2 = st.columns(2)
    for col, fname in ((d1, "transcript.srt"), (d2, "transcript.txt")):
        p = project_dir / fname
        if p.exists():
            col.download_button(f"Download {fname.split('.')[-1].upper()}", p.read_bytes(), file_name=fname)



def go_next(label, idx):
    st.button(label, type="primary", key=f"next_{idx}", on_click=lambda: st.session_state.update(_goto=STEPS[idx]))


def render_setup(project_dir, project):
    keep = ensure_keep(project_dir, project)
    if not keep:
        st.info(NO_PROJECT_MSG)
        return None
    zoom_cfg = project.get("zoom") or {}
    plan = zoom_plan(keep, zoom_cfg)
    return keep, [], 0.5, 0.5


def ratios_for(project):
    info = project.get("video_info") or {}
    w, h = info.get("width"), info.get("height")
    out = {}
    if w and h:
        out[f"Same as original ({w // 2 * 2}x{h // 2 * 2})"] = (w // 2 * 2, h // 2 * 2)
    out.update(RATIOS)
    return out


def shared_format(project):
    """The aspect ratio and framing chosen once; used by the captions preview, the title preview and every export."""
    R = ratios_for(project)
    saved = project.get("format") or {}
    ratio_label = saved.get("ratio") if saved.get("ratio") in R else list(R)[0]
    fit_label = saved.get("fit") if saved.get("fit") in FITS else list(FITS)[0]
    return R[ratio_label], FITS[fit_label]


def format_choice(project_dir, project, key, show=True):
    """Aspect ratio / framing selectors that remember the choice for the whole project."""
    pid = project["project_id"]
    R = ratios_for(project)
    saved = project.get("format") or {}
    ratio_label = saved.get("ratio") if saved.get("ratio") in R else list(R)[0]
    fit_label = saved.get("fit") if saved.get("fit") in FITS else list(FITS)[0]
    if show:
        a, b = st.columns(2)
        ratio_label = a.selectbox("Aspect ratio", list(R), index=list(R).index(ratio_label), key=f"{key}_ratio_{pid}")
        fit_label = b.selectbox("Framing", list(FITS), index=list(FITS).index(fit_label), key=f"{key}_fit_{pid}")
        new = {"ratio": ratio_label, "fit": fit_label}
        if new != project.get("format"):
            project["format"] = new
            save_project(project_dir, project)
    return R[ratio_label], FITS[fit_label]


def font_picker(label, current, key):
    font_index = installed_fonts(st.session_state.get("font_refresh", 0))
    font_list = sorted(font_index.keys(), key=str.lower) or FONT_CHOICES
    default_font = current if current in font_list else ("Arial" if "Arial" in font_list else font_list[0])
    return st.selectbox(label, font_list, index=font_list.index(default_font), key=key)


def translation_panel(project_dir, project):
    pid = project["project_id"]
    st.subheader("English subtitles (Kokni and Hindi)")
    if not project.get("transcript"):
        return
    use_en = st.checkbox("Use English subtitles instead of the original", value=project.get("caption_source") == "english",
                         key=f"use_en_{pid}")
    new_src = "english" if use_en else "original"
    if project.get("caption_source") != new_src:
        project["caption_source"] = new_src
        save_project(project_dir, project)
        st.rerun()
    if not use_en:
        return
    method = st.radio("Translate with", ["Whisper on this laptop (free, no key)", "Claude (needs an API key)"],
                      key=f"tr_method_{pid}", horizontal=True)
    if method.startswith("Whisper"):
        model_size = st.selectbox("Model", WHISPER_MODELS, index=WHISPER_MODELS.index("medium"), key=f"tr_model_{pid}")
        if st.button("Create English subtitles", type="primary", key=f"tr_go_{pid}"):
            try:
                with st.spinner("Translating from the audio. This takes a few minutes..."):
                    wav = render_cut_audio(project_dir, project)
                    segs, _info = transcribe(wav, project.get("profile") or list(PROFILES)[0], model_size,
                                             "Auto (GPU if available)", project.get("custom_prompt", ""),
                                             "Profile default", True, task="translate")
                    project["english_segments"] = segs
                    save_project(project_dir, project)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    else:
        api_key = st.text_input("Anthropic API key", type="password", value=os.environ.get("ANTHROPIC_API_KEY", ""),
                                help="Used only for this call. Never saved to the project.", key=f"tr_key_{pid}")
        tmodel = st.text_input("Translation model", value="claude-sonnet-5-5", key=f"tr_cm_{pid}")
        if st.button("Translate with Claude", disabled=not api_key, key=f"tr_cgo_{pid}"):
            try:
                with st.spinner("Translating..."):
                    english = translate_to_english([x["text"] for x in project["transcript"]], api_key, tmodel)
                    project["english_segments"] = [
                        {"start": x["start"], "end": x["end"], "text": en, "words": spread_words(en, x["start"], x["end"])}
                        for x, en in zip(project["transcript"], english)]
                    save_project(project_dir, project)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    eng = project.get("english_segments")
    if not eng:
        st.warning("Create the English subtitles first. Until then, captions use the original transcript.")
        return
    rows = [{"Start": e["start"], "End": e["end"], "English": e["text"]} for e in eng]
    ed = st.data_editor(pd.DataFrame(rows), use_container_width=True, hide_index=True, disabled=["Start", "End"],
                        key=f"en_{pid}_{len(eng)}")
    if st.button("Save English edits", key=f"en_save_{pid}"):
        for e, txt in zip(eng, ed["English"]):
            txt = str(txt).strip()
            if txt != e["text"]:
                e["text"], e["words"] = txt, spread_words(txt, e["start"], e["end"])
        save_project(project_dir, project)
        st.success("English subtitles saved.")
        st.rerun()


def step_captions(project_dir, project):
    pid = project["project_id"]
    st.subheader("Captions")
    setup = render_setup(project_dir, project)
    if not setup:
        return None
    keep, plan, zfx, zfy = setup
    if project.get("transcript") and project.get("transcript_keep_version") != project.get("keep_version"):
        st.warning("The cut changed after the transcript was made. Transcribe again so captions line up.")
    if not caption_segments(project):
        st.info("No transcript yet, so there are no captions. Transcribe first.")
    sv = {**DEFAULT_STYLE, **(project.get("style") or {})}
    with st.expander("Text", expanded=True):
        show_subs = st.checkbox("Burn subtitles into the export", value=sv["show"], key=f"m3_show_{pid}")
        font_pick = font_picker("Font (type to search installed fonts)", sv["font"], f"m3_font_{pid}")
        if st.button("Refresh fonts", key=f"m3_fontrefresh_{pid}",
                     help="Click after installing a new font, then pick it from the list."):
            st.session_state["font_refresh"] = st.session_state.get("font_refresh", 0) + 1
            st.rerun()
        size = st.slider("Size", 20, 140, int(sv["size"]), key=f"m3_size_{pid}")
        ga, gb = st.columns(2)
        bold = ga.checkbox("Bold", value=sv["bold"], key=f"m3_bold_{pid}")
        upper = gb.checkbox("UPPERCASE", value=sv["upper"], key=f"m3_upper_{pid}")
        mw = st.slider("Max words per subtitle", 2, 8, int(sv["max_words"]), key=f"m3_mw_{pid}")
    with st.expander("Colours and outline"):
        ca, cb = st.columns(2)
        color = ca.color_picker("Text colour", sv["color"], key=f"m3_color_{pid}")
        ocolor = cb.color_picker("Outline colour", sv["outline_color"], key=f"m3_ocolor_{pid}")
        outline = st.slider("Outline thickness", 0, 10, int(sv["outline"]), key=f"m3_outline_{pid}")
        box = st.checkbox("Background box", value=sv["box"], key=f"m3_box_{pid}")
        if box:
            ba, bb = st.columns(2)
            bcolor = ba.color_picker("Box colour", sv["box_color"], key=f"m3_bcolor_{pid}")
            bop = bb.slider("Box opacity %", 0, 100, int(sv["box_opacity"]), key=f"m3_bop_{pid}")
        else:
            bcolor, bop = sv["box_color"], sv["box_opacity"]
    with st.expander("Position"):
        xp = st.slider("Horizontal position (%)", 0, 100, int(sv["x_pct"]), key=f"m3_x_{pid}")
        yp = st.slider("Vertical position (%)", 0, 100, int(sv["y_pct"]), key=f"m3_y_{pid}",
                       help="0 = top, 100 = bottom. Around 70-78 keeps text clear of Reels buttons.")
    style = dict(show=show_subs, font=font_pick, size=size, bold=bold, upper=upper, color=color,
                 outline_color=ocolor, outline=outline, box=box, box_color=bcolor, box_opacity=bop,
                 x_pct=xp, y_pct=yp, max_words=mw)
    if style != project.get("style"):
        project["style"] = style
        save_project(project_dir, project)
    st.divider()
    translation_panel(project_dir, project)
    return style


TITLE_PRESETS = {
    "Bold hook at top": dict(y=14.0, size=100.0, color="#FFD400", outline_color="#000000", outline=6,
                             upper=True, bold=True, box=False),
    "Centre hook": dict(y=40.0, size=120.0, color="#FFFFFF", outline_color="#000000", outline=7,
                        upper=True, bold=True, box=False),
    "Boxed title": dict(y=18.0, size=76.0, color="#FFFFFF", outline_color="#000000", outline=5,
                        upper=True, bold=True, box=True, box_color="#E11D48"),
    "Clean": dict(y=16.0, size=84.0, color="#FFFFFF", outline_color="#000000", outline=4,
                       upper=False, bold=True, box=False),
}
FULL_TITLE = 36000.0
TITLE_DEFAULT = dict(on=True, text="", start=0.0, dur=3.0, font="Arial", ver=0, x=50.0, anim="None", **TITLE_PRESETS["Bold hook at top"])


def step_title(project_dir, project):
    pid = project["project_id"]
    st.caption("Optional. The title appears at the start and stays for the whole video. "
               "It uses the aspect ratio and framing chosen in the captions preview above.")
    setup = render_setup(project_dir, project)
    if not setup:
        return None
    keep, plan, zfx, zfy = setup
    t = {**TITLE_DEFAULT, **(project.get("title") or {})}
    t["on"] = not t.get("off_by_user", False)      # on by default; off only if you switch it off yourself
    if not t.get("anim_chosen"):
        t["anim"] = "None"
    ver = int(t.get("ver", 0))

    def apply_preset(name):
        cur = {**TITLE_DEFAULT, **(project.get("title") or {})}
        cur.update(TITLE_PRESETS[name])
        cur["ver"] = int(cur.get("ver", 0)) + 1
        project["title"] = cur
        save_project(project_dir, project)

    with st.expander("Text and effect", expanded=True):
        on = st.checkbox("Add a title", value=bool(t["on"]), key=f"ti_on_{pid}_{ver}")
        text = st.text_area("Title text (Enter makes a new line)", value=t["text"], height=80, key=f"ti_text_{pid}_{ver}")
        anim = st.selectbox("Effect", MOTION_ANIMS, index=MOTION_ANIMS.index(t["anim"]) if t["anim"] in MOTION_ANIMS else 0,
                            key=f"ti_anim_{pid}_{ver}", help="None = the title simply stays on screen.")
    start, dur = 0.0, FULL_TITLE      # the title stays on screen for the whole video
    with st.expander("Font and colours"):
        pa = st.selectbox("Start from a style", list(TITLE_PRESETS), key=f"ti_preset_{pid}")
        st.button("Apply style", key=f"ti_apply_{pid}", on_click=apply_preset, args=(pa,))
        font = font_picker("Font", t["font"], f"ti_font_{pid}_{ver}")
        size = st.slider("Size", 30, 220, int(t["size"]), key=f"ti_size_{pid}_{ver}")
        g1, g2 = st.columns(2)
        color = g1.color_picker("Text colour", t["color"], key=f"ti_color_{pid}_{ver}")
        ocolor = g2.color_picker("Outline colour", t["outline_color"], key=f"ti_ocolor_{pid}_{ver}")
        g3, g4 = st.columns(2)
        bold = g3.checkbox("Bold", value=bool(t["bold"]), key=f"ti_bold_{pid}_{ver}")
        upper = g4.checkbox("UPPERCASE", value=bool(t["upper"]), key=f"ti_upper_{pid}_{ver}")
        outline = st.slider("Outline / box padding", 0, 12, int(t["outline"]), key=f"ti_out_{pid}_{ver}")
        box = st.checkbox("Coloured box behind text", value=bool(t["box"]), key=f"ti_box_{pid}_{ver}")
        bcolor = (st.color_picker("Box colour", t.get("box_color", "#E11D48"), key=f"ti_bcolor_{pid}_{ver}")
                  if box else t.get("box_color", "#000000"))
    with st.expander("Position"):
        x = st.slider("Horizontal position (%)", 0, 100, int(t["x"]), key=f"ti_x_{pid}_{ver}")
        y = st.slider("Vertical position (%)", 0, 100, int(t["y"]), key=f"ti_y_{pid}_{ver}",
                      help="0 = top, 100 = bottom. 12-20 sits under the top of a Reel.")
    new_t = dict(on=on, text=text, start=float(start), dur=float(dur), anim=anim, font=font, size=float(size),
                 color=color, outline_color=ocolor, bold=bold, upper=upper, outline=outline, box=box,
                 box_color=bcolor, x=float(x), y=float(y), ver=ver, off_by_user=not on,
                 anim_chosen=bool(t.get("anim_chosen")) or anim != "None")
    if new_t != project.get("title"):
        project["title"] = new_t
        save_project(project_dir, project)
    ms_now = dict(font=font, color=color, outline_color=ocolor, outline=outline, upper=upper, bold=bold, box=box,
                  box_color=bcolor)
    item_now = dict(text=text.strip() or "Your title here", start=float(start), dur=float(dur), anim=anim, x=float(x),
                    y=float(y), size=float(size))
    return (item_now if (on and text.strip()) else None), ms_now


def import_panel(idx, expanded):
    with st.expander("Import a new video for this section", expanded=expanded):
        st.caption("Videos up to 1 GB. Nothing leaves this computer. This creates a new project; "
                   "you can switch projects in the sidebar.")
        uploaded = st.file_uploader("Choose a video file", type=VIDEO_TYPES, key=f"imp_file_{idx}")
        path_text = st.text_input("Or paste the full path of a video on this computer (faster for very large files)",
                                  placeholder=r"C:\Users\you\Videos\raw.mov", key=f"imp_path_{idx}")
        have = uploaded is not None or bool(path_text.strip())
        if st.button("Import video", type="primary", disabled=not have, key=f"imp_go_{idx}"):
            ok = False
            try:
                with st.spinner("Saving the video..."):
                    project_dir, project = new_project(uploaded, path_text)
                with st.spinner("Reading audio and building the preview copy. Large files take a few minutes..."):
                    prepare_project(project_dir, project)
                ok = True
            except Exception as exc:
                st.error(str(exc))
            if ok:
                st.session_state["_select_pid"] = project["project_id"]
                goto(STEPS[idx])


def section_footer(project_dir, project, idx, next_label):
    pid = project["project_id"]
    st.divider()
    st.subheader("Finish this section")
    keep = ensure_keep(project_dir, project)
    if not keep:
        st.info(NO_PROJECT_MSG)
        return
    style = {**DEFAULT_STYLE, **(project.get("style") or {})}
    parts = [f"pauses/repeats removed: {'yes' if not project.get('keep_is_full', True) else 'no (full video)'}",
             f"subtitles: {'on' if style['show'] and caption_segments(project) else 'none'}",
             f"title: {'yes' if motion_items(project) else 'none'}"]
    if (project.get("denoise") or {}).get("on"):
        parts.append("noise reduction: on")
    st.caption("Export includes: " + " · ".join(parts))
    size_px, fit_val = format_choice(project_dir, project, f"sf{idx}", show=idx != len(STEPS) - 1)
    if idx == len(STEPS) - 1:
        st.caption(f"Export size and framing: as chosen in the captions preview above ({size_px[0]}x{size_px[1]}).")
    c1, c2 = st.columns(2)
    if c1.button("Export video now", type="primary", key=f"sf_exp_{pid}_{idx}"):
        try:
            with st.spinner("Exporting from the original video. Large files take a few minutes..."):
                out, elog, check = export_final(project_dir, project, size_px, fit_val, style, [], 0.5, 0.5)
            st.session_state[f"sf_out_{pid}"] = (str(out), elog)
        except Exception as exc:
            st.error(str(exc))
    if idx + 1 < len(STEPS):
        c2.button(next_label, key=f"sf_next_{pid}_{idx}",
                  on_click=lambda: st.session_state.update(_goto=STEPS[idx + 1]))
    res = st.session_state.get(f"sf_out_{pid}")
    if res and Path(res[0]).exists():
        ev, ed_ = st.columns([1, 1])
        ev.video(res[0])
        ed_.success(f"Exported: {Path(res[0]).name}")
        ed_.caption(f"Saved in: {Path(res[0]).parent}")
        ed_.download_button("Download export", Path(res[0]).read_bytes(), file_name=Path(res[0]).name,
                            key=f"sf_dl_{pid}_{idx}")
        with st.expander("Export log (for troubleshooting)"):
            st.code(res[1], language="text")


def step_subtitles(project_dir, project):
    st.header("A. Transcribe")
    step_transcript(project_dir, project)
    st.divider()
    keep = ensure_keep(project_dir, project)
    if not keep:
        return
    left, right = st.columns([3, 2])
    with left:
        st.header("B. Captions style and English")
        cap_style = step_captions(project_dir, project)
        st.divider()
        st.header("C. Title")
        title_res = step_title(project_dir, project)
    with right:
        st.markdown('<div class="sticky-preview"></div>', unsafe_allow_html=True)
        st.markdown("**Preview** - watch the video while you change captions and title.")
        size_px, fit = format_choice(project_dir, project, "m3")
        st.caption("This aspect ratio is used for the preview and every export.")
        try:
            base_clip = render_title_base(project_dir, project, keep, 1e9)
            item, ms = title_res if title_res else (None, None)
            style = cap_style or {**DEFAULT_STYLE, **(project.get("style") or {})}
            caps = None
            if style.get("show") and caption_segments(project):
                caps = {"cues": make_cues(caption_segments(project), style["max_words"]), "style": style}
            components.html(title_player_html(base_clip, size_px, fit, item, ms, None, caps=caps),
                            height=int(340 * size_px[1] / size_px[0]) + 20)
            if not item:
                st.caption("No title yet. Type the title text to see it here.")
        except Exception as exc:
            st.error(f"Preview could not be prepared: {exc}")


SECTIONS = [(step_draft, "Continue: remove repeats →"), (step_review, "Continue: subtitles and title →"),
            (step_subtitles, "")]


# ================================================================ main script
projects = list_projects()
label_to_dir = {}
for folder, d in projects:
    label = (f"{d.get('created_at', '')[:16].replace('T', ' ')} · {d.get('original_filename', '')[:34]} · "
             f"{folder.name[-6:]}")
    label_to_dir[label] = (folder, d)

if "_select_pid" in st.session_state:
    wanted = st.session_state.pop("_select_pid")
    for lbl, (folder, _d) in label_to_dir.items():
        if folder.name == wanted:
            st.session_state["project_select"] = lbl
if "_goto" in st.session_state:
    st.session_state["step"] = st.session_state.pop("_goto")

gpu_ok, gpu_msg = cuda_status()
current = None
with st.expander("📁 Project and settings", expanded=False):
    (st.success if gpu_ok else st.warning)(("🟢 " if gpu_ok else "🟠 ") + gpu_msg)
    st.caption("RTX 2050 (4 GB): use the medium model. large-v3 may run out of memory.")
    if label_to_dir:
        chosen = st.selectbox("Project", list(label_to_dir), key="project_select")
        current = label_to_dir[chosen]
        with st.expander("Delete this project"):
            confirm = st.checkbox("Yes, permanently delete this project folder", key="del_confirm")
            if st.button("Delete project", disabled=not confirm):
                shutil.rmtree(current[0], ignore_errors=True)
                st.session_state.pop("project_select", None)
                st.session_state.pop("del_confirm", None)
                st.rerun()
    else:
        st.caption("No projects yet.")
    st.caption("Your videos stay on this computer.")

st.title("🎬 Reel Studio AI")
step = st.radio("Workflow", STEPS, horizontal=True, key="step", label_visibility="collapsed")

idx = STEPS.index(step)
import_panel(idx, expanded=current is None)
if current is None:
    st.info("Import a video above to start this section.")
else:
    pdir, pdata = current
    fn, nxt = SECTIONS[idx]
    fn(pdir, pdata)
    section_footer(pdir, pdata, idx, nxt)
