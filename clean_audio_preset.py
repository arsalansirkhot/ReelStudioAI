"""Clean audio step for Reel Studio AI: background noise removal with DeepFilterNet3.

This step comes first, right after importing. It cleans the audio once and makes the cleaned video
the project's main video, so every later step (remove pauses, repeats, subtitles, Talking1,
Template 2 and all exports) uses the clean audio automatically. The picture is not re-encoded.

Chain (tested on the user's phone a-rolls, about 11 dB cleaner than DaVinci Resolve's result):
  highpass 65 Hz x2 -> DeepFilterNet3 (deep-filter.exe -D -a 100) -> gentle 2:1 compressor
  -> soft expander for the pauses -> linear gain to -14 LUFS -> limiter at -2 dBFS.
The compressor sits after the denoiser on purpose: before it, the denoiser removes about 9 dB less.

Kept in its own file so app updates keep it (update.ps1 installs it). The denoiser program lives in
tools/deep-filter.exe, which updates never touch. If it is missing, this step offers to download it.
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
ENGINE = APP_DIR / "tools" / "deep-filter.exe"
ENGINE_URL = ("https://github.com/Rikorose/DeepFilterNet/releases/download/v0.5.6/"
              "deep-filter-0.5.6-x86_64-pc-windows-msvc.exe")
ENGINE_SIZE = 26912256
CLEAN_VIDEO = "video_clean.mp4"
CLEAN_WAV = "audio_clean_48k.wav"
PREVIEW_LEN = 20.0
TARGET_LUFS = -14.0
STRENGTHS = {
    "Full (recommended)": 100,
    "Natural (keeps a little room sound)": 24,
}
POST = ("acompressor=threshold=0.125:ratio=2:attack=15:release=150:knee=6,"
        "agate=threshold=0.003:ratio=2:range=0.25:attack=10:release=250:knee=3")
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ------------------------------------------------------------------ engine
def engine_ok():
    return ENGINE.is_file() and ENGINE.stat().st_size > 1_000_000


def download_engine():
    ENGINE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ENGINE.with_suffix(".part")
    urllib.request.urlretrieve(ENGINE_URL, tmp)
    if tmp.stat().st_size != ENGINE_SIZE:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("The download was incomplete. Check the internet connection and try again.")
    tmp.replace(ENGINE)


# ------------------------------------------------------------------ audio processing
def integrated_lufs(A, wav):
    log = A["run_ffmpeg"](["-hide_banner", "-nostats", "-i", str(wav), "-af", "ebur128", "-f", "null", "-"])
    found = re.findall(r"I:\s+(-?[\d.]+) LUFS", log)
    if not found:
        raise RuntimeError("Could not measure the loudness.")
    return float(found[-1])


def to_target(A, src, out):
    """Linear gain to -14 LUFS plus a limiter, so pauses are not lifted the way dynamic loudnorm does."""
    lufs = integrated_lufs(A, src)
    gain = max(-40.0, min(40.0, TARGET_LUFS - lufs)) if lufs > -70 else 0.0
    A["run_ffmpeg"](["-i", str(src), "-af", f"volume={gain:.2f}dB,alimiter=limit=0.79:attack=2:release=50:level=0",
                     "-ar", "48000", "-c:a", "pcm_s16le", str(out)])


def clean_wav(A, media, out, atten, start=None, length=None):
    """Clean the audio of `media` (optionally only a piece of it) into a 48 kHz mono WAV at -14 LUFS."""
    work = Path(tempfile.mkdtemp(prefix="rs_clean_"))
    try:
        cut = (["-ss", f"{start:.3f}", "-t", f"{length:.3f}"] if start is not None else [])
        A["run_ffmpeg"]([*cut, "-i", str(media), "-vn", "-ac", "1", "-ar", "48000",
                         "-af", "highpass=f=65:p=2,highpass=f=65:p=2", "-c:a", "pcm_s16le", str(work / "in.wav")])
        r = subprocess.run([str(ENGINE), "-D", "-a", str(int(atten)), "-o", str(work / "dfn"), str(work / "in.wav")],
                           capture_output=True, text=True, creationflags=NO_WINDOW)
        done = work / "dfn" / "in.wav"
        if r.returncode != 0 or not done.exists():
            raise RuntimeError("Noise removal failed: " + (r.stderr or r.stdout or "no output")[-1500:])
        A["run_ffmpeg"](["-i", str(done), "-af", POST, "-c:a", "pcm_s16le", str(work / "post.wav")])
        to_target(A, work / "post.wav", out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


def original_video(project):
    return project.get("video_file_original") or project["video_file"]


# ------------------------------------------------------------------ preview (a short piece, before vs after)
def make_preview(project_dir, project, A, start, atten):
    orig = project_dir / original_video(project)
    proxy = project_dir / project["proxy_file"]
    clean = project_dir / "preview_clean.wav"
    raw = project_dir / "preview_raw_tmp.wav"
    raw_n = project_dir / "preview_raw.wav"
    clean_wav(A, orig, clean, atten, start, PREVIEW_LEN)
    # the "before" audio gets the same loudness, so the comparison is fair
    A["run_ffmpeg"](["-ss", f"{start:.3f}", "-t", f"{PREVIEW_LEN:.3f}", "-i", str(orig), "-vn", "-ac", "1",
                     "-ar", "48000", "-c:a", "pcm_s16le", str(raw)])
    to_target(A, raw, raw_n)
    raw.unlink(missing_ok=True)
    outs = []
    for name, wav in (("preview_before.mp4", raw_n), ("preview_after.mp4", clean)):
        out = project_dir / name
        A["run_ffmpeg"](["-ss", f"{start:.3f}", "-t", f"{PREVIEW_LEN:.3f}", "-i", str(proxy), "-i", str(wav),
                         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ac", "2", "-shortest",
                         "-movflags", "+faststart", str(out)])
        outs.append(out)
    return outs


# ------------------------------------------------------------------ apply / undo on the whole project
def _swap_proxy_audio(project_dir, project, A, audio_src):
    """Give the preview copy new audio without re-encoding its picture (fast)."""
    proxy = project_dir / project["proxy_file"]
    tmp = project_dir / "proxy_swap_tmp.mp4"
    A["run_ffmpeg"](["-i", str(proxy), "-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                     "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-movflags", "+faststart", str(tmp)])
    tmp.replace(proxy)


def _refresh_derived(project_dir, project, A):
    """Rebuild what the later steps read, and drop cached renders made with the old audio."""
    video = project_dir / project["video_file"]
    A["extract_audio"](video, project_dir / project["audio_file"])
    for name in ("talking1_source.mp4", "title_base.mp4", "audio_original.wav", "audio_processed.wav"):
        (project_dir / name).unlink(missing_ok=True)
    project.pop("title_base", None)
    if project.get("keep") and not project.get("keep_is_full"):
        A["render_review"](project_dir, project)
    elif project.get("keep"):
        shutil.copyfile(project_dir / project["proxy_file"], project_dir / "review_cut.mp4")


def apply_clean(project_dir, project, A, strength):
    atten = STRENGTHS[strength]
    orig = original_video(project)
    wav = project_dir / CLEAN_WAV
    clean_wav(A, project_dir / orig, wav, atten)
    tmp = project_dir / "video_clean_tmp.mp4"
    # picture copied untouched; cleaned audio as high-quality AAC so later re-encodes lose nothing audible
    A["run_ffmpeg"](["-i", str(project_dir / orig), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "320k", "-ac", "2", "-movflags", "+faststart", str(tmp)])
    tmp.replace(project_dir / CLEAN_VIDEO)
    project["video_file_original"] = orig
    project["video_file"] = CLEAN_VIDEO
    _swap_proxy_audio(project_dir, project, A, wav)
    dn = project.get("denoise") or {}
    project["denoise"] = {**dn, "on": False}   # never denoise twice: the old afftdn option stays off
    project["audio_clean"] = {"strength": strength, "atten": atten, "lufs": TARGET_LUFS,
                              "at": time.strftime("%Y-%m-%d %H:%M")}
    _refresh_derived(project_dir, project, A)
    A["save_project"](project_dir, project)


def undo_clean(project_dir, project, A):
    orig = original_video(project)
    project["video_file"] = orig
    _swap_proxy_audio(project_dir, project, A, project_dir / orig)
    project.pop("audio_clean", None)
    _refresh_derived(project_dir, project, A)
    A["save_project"](project_dir, project)


# ------------------------------------------------------------------ the step UI
def step(project_dir, project, A):
    pid = project["project_id"]
    st.subheader("Clean the audio (remove background noise)")
    st.caption("Removes fan, room and street noise from your voice. Clean the audio once here; every next step "
               "and every export then uses the cleaned audio. The picture is not changed.")
    if "proxy_file" not in project:
        st.info(A["NO_PROJECT_MSG"])
        return
    if not engine_ok():
        st.warning("The noise removal program (DeepFilterNet, 27 MB, free and open source) is not installed yet. "
                   "It is downloaded once from its official GitHub page and saved in the tools folder.")
        if st.button("Download noise removal program (27 MB)", type="primary", key=f"ca_dl_{pid}"):
            try:
                with st.spinner("Downloading..."):
                    download_engine()
                st.rerun()
            except Exception as exc:
                st.error(f"Download failed: {exc}")
        return

    cfg = project.get("audio_clean")
    if cfg:
        st.success(f"Audio cleaned ({cfg['strength']}, {cfg['at']}). Every next step uses the cleaned audio.")

    names = list(STRENGTHS)
    cur = cfg["strength"] if cfg and cfg.get("strength") in names else names[0]
    strength = st.radio("Strength", names, index=names.index(cur), horizontal=True, key=f"ca_str_{pid}",
                        help="Full removes the noise completely. Natural leaves a little room sound in the pauses.")
    atten = STRENGTHS[strength]

    # ---- 1. preview
    st.markdown("#### 1 · Preview 20 seconds (before vs after)")
    dur = float(project["video_info"].get("duration_seconds") or 0)
    max_start = max(0, int(dur - PREVIEW_LEN))
    start = st.slider("Preview starts at (seconds)", 0, max_start, min(max_start, 0), 1, key=f"ca_st_{pid}",
                      disabled=max_start == 0) if max_start > 0 else 0
    if st.button("Make preview", key=f"ca_prev_{pid}"):
        try:
            with st.spinner("Cleaning 20 seconds..."):
                outs = make_preview(project_dir, project, A, float(start), atten)
            st.session_state[f"ca_prev_{pid}_out"] = ([str(o) for o in outs], strength, start)
        except Exception as exc:
            st.error(str(exc))
    prev = st.session_state.get(f"ca_prev_{pid}_out")
    if prev and all(Path(p).exists() for p in prev[0]):
        b, a, _sp = st.columns([1, 1, 1])
        b.caption("Before (original audio)")
        b.video(Path(prev[0][0]).read_bytes())
        a.caption(f"After ({prev[1]})")
        a.video(Path(prev[0][1]).read_bytes())
        st.caption("Both are set to the same loudness so you can compare fairly. Headphones help.")

    # ---- 2. clean the whole video
    st.markdown("#### 2 · Clean the whole video")
    if cfg and cfg.get("strength") == strength:
        label = "Clean the whole video again"
    elif cfg:
        label = f"Clean again with {strength.split(' (')[0]} strength"
    else:
        label = "Clean the whole video"
    c1, c2 = st.columns(2)
    if c1.button(label, type="primary" if not cfg else "secondary", key=f"ca_go_{pid}"):
        try:
            with st.spinner("Removing background noise from the whole video..."):
                apply_clean(project_dir, project, A, strength)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if cfg and c2.button("Go back to the original audio", key=f"ca_undo_{pid}"):
        try:
            with st.spinner("Restoring the original audio..."):
                undo_clean(project_dir, project, A)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    steps = A["STEPS"]
    nxt = steps[1] if len(steps) > 1 else None
    if not cfg:
        if nxt:
            st.button("Skip: keep the original audio and continue →", key=f"ca_skip_{pid}",
                      on_click=lambda: st.session_state.update(_goto=nxt))
        return

    # ---- 3. watch, then export or continue
    st.markdown("#### 3 · Watch the cleaned video")
    vc, _ = st.columns([1, 2])
    vc.video(str(project_dir / project["proxy_file"]))
    st.caption("This is a small preview copy. The export below is full quality.")

    st.markdown("#### 4 · Export now, or continue editing")
    clean = project_dir / CLEAN_VIDEO
    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Export the cleaned video now**")
        st.caption("Full length, original picture quality, only the audio cleaned. Use this if you don't need "
                   "the other steps.")
        stem = Path(project.get("original_filename") or "video").stem
        if clean.exists():
            st.download_button("Download cleaned video", clean.read_bytes(), file_name=f"{stem}_clean.mp4",
                               mime="video/mp4", type="primary", key=f"ca_save_{pid}")
            st.caption(f"Also saved in: {clean}")
    with e2:
        st.markdown("**Continue editing**")
        st.caption("Remove pauses, repeats, add subtitles and templates. All of them use the cleaned audio.")
        if nxt:
            st.button(f"Continue: {nxt.split(' · ', 1)[-1].lower()} →", key=f"ca_next_{pid}",
                      on_click=lambda: st.session_state.update(_goto=nxt))
