"""Talking1 style preset for Reel Studio AI.

Big condensed punch words at eye level, tiny lowercase subtitles, warm grade with black & white
emphasis moments, the user's linear push-in zoom (resets at every cut) and a "follow karo" end card.

The look itself is the Remotion template made with Claude in D:\\ClaudeAIVideoEdit\\remotion
(composition "Talking1"). This file only prepares the data and calls Remotion, so the preset looks
exactly like the edits made there.

Kept in its own file on purpose: update.ps1 replaces app.py on every update but never touches this
file. app.py only needs a few lines to show it as step 4 (see the "Talking1 add-on" comments there).
Set the environment variable REELSTUDIO_REMOTION_DIR if the Remotion project moves.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REMOTION_DIR = Path(os.environ.get("REELSTUDIO_REMOTION_DIR", r"D:\ClaudeAIVideoEdit\remotion"))
COMPOSITION = "Talking1"
SOURCE_NAME = "talking1_source.mp4"
COLOURS = ["white", "red", "cream"]
FORMATS = {"1:1 · Square": "1:1", "9:16 · Reels": "9:16", "Same as the video": "source"}
DEFAULTS = dict(format="1:1", push_to=1.55, eye_line=0.33, sub_line=0.86, end_cta="follow karo",
                romanize=True, min_shot=1.0)

# Words that never become punch words (Kokni / Hindi / Marathi / English fillers)
STOP = set("""a an the and or but so to of in on at for is are was were be been it this that these those i you he
she we they me my your our us them his her its am do does did have has had yes ok okay hai hain ho hota hoti hote
tha thi the ki ka ke ko se mein me main mai par pe bhi toh ye yeh wo woh aur ya kya kyu kyun kaise jo jab tab ab
hum tum aap apan apun mana tula mala tyane tyala tine ani pan tar te ti to ha he hi ahe aahe asa ase aise aisa
aisach ek don kahi kai karun kela keli kelan zala jhala mhanje matlab like just really very actually basically
yaar bhai re la ga paije pahije hota thoda thode jar kar karu karna""".split())
# Words that get the red colour (negatives / problems)
NEG = {"nahi", "nahin", "nahe", "nahi", "nai", "nay", "naay", "nako", "no", "not", "never", "problem", "problems", "galat", "mat",
       "wrong", "fail", "mistake", "bad", "stop", "maybe"}
# Words worth a punch even if short
EMPHASIS = {"problem", "confidence", "camera", "video", "videos", "topic", "comment", "content", "creator", "try",
            "roj", "daily", "job", "time", "money", "paisa", "life", "success", "secret", "tomorrow", "today",
            "follow", "subscribe", "business", "corporate", "vlog", "idea", "dream", "goal", "important", "best",
            "favourite", "favorite", "first", "always", "never", "baat", "charcha"}


# ------------------------------------------------------------------ helpers
def npx_path():
    return shutil.which("npx.cmd") or shutil.which("npx") or r"C:\Program Files\nodejs\npx.cmd"


def remotion_problem():
    """None when Remotion can be used, otherwise a message for the user."""
    if not (REMOTION_DIR / "src" / "index.ts").exists():
        return (f"The Remotion project was not found at {REMOTION_DIR}. Set the environment variable "
                "REELSTUDIO_REMOTION_DIR to its folder and restart the app.")
    if not (REMOTION_DIR / "node_modules").exists():
        return f"Remotion is not installed yet. Run 'npm install' in {REMOTION_DIR}."
    if not Path(npx_path()).exists() and not shutil.which("npx"):
        return "Node.js (npx) was not found. Install Node.js and restart the app."
    return None


def settings(project):
    return {**DEFAULTS, **(project.get("talking1") or {})}


def save_settings(project_dir, project, A, **changes):
    project["talking1"] = {**(project.get("talking1") or {}), **changes}
    A["save_project"](project_dir, project)


def has_devanagari(text):
    return any("\u0900" <= c <= "\u097f" for c in text)


def roman(text, A):
    """Devanagari words -> rough Roman spelling (the user corrects the rest in the transcript)."""
    return " ".join(A["romanize"](w) if has_devanagari(w) else w for w in text.split())


def clean_word(w):
    return re.sub(r"[^\w']", "", w, flags=re.UNICODE)


# ------------------------------------------------------------------ source video (the cut, no captions)
def source_video(project_dir, project, A):
    """The reviewed cut as one plain video (no captions). Remotion reads it from the project folder."""
    if project.get("keep_is_full") and not (project.get("denoise") or {}).get("on"):
        return project["video_file"]
    key = [project.get("keep_version", 0), A["denoise_filter"](project)]
    out = project_dir / SOURCE_NAME
    if out.exists() and settings(project).get("source_key") == key:
        return SOURCE_NAME
    stm = A["cut_statements"](project["keep"], project["fps"], A["denoise_filter"](project))
    stm.append("[vc]format=yuv420p[vout]")
    A["run_graph"](["-i", str(project_dir / project["video_file"])], ";".join(stm),
                   ["-map", "[vout]", "-map", "[ac]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)],
                   str(project_dir), project_dir / "graph_talking1.txt")
    save_settings(project_dir, project, A, source_key=key)
    return SOURCE_NAME


def video_fps(path, A):
    err = subprocess.run([A["FFMPEG"], "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"(\d+(?:\.\d+)?) fps", err)
    return float(m.group(1)) if m else 30.0


# ------------------------------------------------------------------ cuts (where the zoom resets)
def detect_cuts(path, A, size=64):
    """Jump-cut candidates: a one-frame spike in picture change. Real cuts and fast hand moves both
    show up, so the user confirms them with the before/after thumbnails."""
    r = subprocess.run([A["FFMPEG"], "-v", "error", "-i", str(path), "-vf", f"scale={size}:{size},format=gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    frames = np.frombuffer(r.stdout, np.uint8)
    if frames.size < size * size * 5:
        return []
    frames = frames[: frames.size // (size * size) * size * size].reshape(-1, size, size).astype(np.float32)
    fps = video_fps(path, A)
    d = np.concatenate([[0.0], np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))])
    med = float(np.median(d))
    out = []
    for i in range(2, len(d) - 2):
        nb = max(d[i - 1], d[i + 1], 0.5)
        if d[i] / nb > 1.8 and d[i] > med * 2:
            out.append({"t": round(i / fps, 3), "strength": round(float(d[i]), 1),
                        "likely": bool(d[i] >= max(7.0, med * 2.8))})
    return out


def keep_boundaries(project, A):
    """Cuts made by pause/repeat removal, in the cut video's timeline."""
    keep = project.get("keep") or []
    if project.get("keep_is_full") or len(keep) < 2:
        return []
    offs, _ = A["_offsets"](keep)
    return [round(o, 3) for o in offs[1:]]


def thumb_pair(project_dir, src, t, fps, A):
    """Small 'just before | just after' image for one cut candidate."""
    folder = project_dir / "talking1_thumbs"
    folder.mkdir(exist_ok=True)
    png = folder / f"cut_{t:08.3f}.png"
    if not png.exists():
        before = max(0.0, t - 1.0 / fps)
        subprocess.run([A["FFMPEG"], "-y", "-v", "error", "-ss", f"{before:.3f}", "-i", str(project_dir / src),
                        "-ss", f"{t + 0.5 / fps:.3f}", "-i", str(project_dir / src), "-filter_complex",
                        "[0:v]scale=150:-2,trim=end_frame=1[a];[1:v]scale=150:-2,trim=end_frame=1[b];"
                        "[a][b]hstack=inputs=2[v]", "-map", "[v]", "-frames:v", "1", str(png)],
                       capture_output=True)
    return png if png.exists() else None


def merge_short(cuts, min_shot):
    out = []
    for c in sorted(set(round(float(x), 3) for x in cuts)):
        if c <= 0.05:
            continue
        if out and c - out[-1] < min_shot:
            continue
        out.append(c)
    return out


# ------------------------------------------------------------------ captions and punch words
def subtitles(project, A, use_roman):
    cues = A["make_cues"](A["caption_segments"](project), 4)
    return [{"start": round(c["start"], 3), "end": round(c["end"], 3),
             "text": roman(c["text"], A) if use_roman else c["text"]} for c in cues]


def suggest_punch(project, A, use_roman):
    """A first proposal the user edits: about one strong word per short phrase, at least 1.2 s apart."""
    words = []
    for seg in A["caption_segments"](project):
        for k, w in enumerate(seg.get("words") or []):
            raw = w["word"].strip()
            text = roman(raw, A) if use_roman else raw
            words.append({"start": float(w["start"]), "end": float(w["end"]), "raw": text,
                          "first": k == 0})
    picks, last = [], -9.0
    i = 0
    while i < len(words):
        w = words[i]
        base = clean_word(w["raw"])
        low = base.lower()
        score = 0
        if low in NEG:
            score = 9
        elif len(low) >= 3 and low not in STOP:
            score = min(len(low), 9) + (4 if low in EMPHASIS else 0)
            if base[:1].isupper() and not w["first"]:
                score += 3  # names, as Whisper capitalises them
        text, end, j = base, w["end"], i
        # join a following capitalised word (e.g. "Irfan Junejo")
        if base[:1].isupper() and i + 1 < len(words) and clean_word(words[i + 1]["raw"])[:1].isupper():
            text = base + " " + clean_word(words[i + 1]["raw"])
            end, j = words[i + 1]["end"], i + 1
            score += 3
        if score >= 8 and w["start"] - last >= 1.2:
            colour = "red" if low in NEG else ("cream" if len(picks) % 3 == 2 else "white")
            shown = text.upper() if len(text) <= 9 and " " not in text else text
            cream = text.split()[-1] if " " in text and colour == "white" else ""
            picks.append({"Start": round(w["start"], 2), "End": round(end, 2), "Text": shown,
                          "Colour": colour, "Cream word": cream})
            last = w["start"]
            i = j
        i += 1
    return picks


def suggest_bw(project, A, punch):
    """Black & white on the transcript lines that hold the first two red words (after the opening)."""
    segs = A["caption_segments"](project)
    out = []
    for p in punch:
        if p["Colour"] != "red" or p["Start"] < 3.0:
            continue
        for s in segs:
            if s["start"] <= p["Start"] < s["end"] and not any(abs(b["Start"] - s["start"]) < 0.01 for b in out):
                out.append({"Start": round(s["start"], 2), "End": round(s["end"], 2)})
        if len(out) >= 2:
            break
    return out


# ------------------------------------------------------------------ Remotion
def build_props(project_dir, project, A, src):
    s = settings(project)
    punch = []
    for p in s.get("punch") or []:
        if not str(p.get("Text", "")).strip():
            continue
        item = {"start": float(p["Start"]), "end": float(p["End"]), "text": str(p["Text"]).strip()}
        if p.get("Colour") in ("red", "cream"):
            item["color"] = p["Colour"]
        if str(p.get("Cream word") or "").strip():
            item["cream"] = str(p["Cream word"]).split()
        punch.append(item)
    punch.sort(key=lambda x: x["start"])
    bw = [[float(b["Start"]), float(b["End"])] for b in s.get("bw") or [] if float(b["End"]) > float(b["Start"])]
    props = {"src": src, "subs": subtitles(project, A, s["romanize"]), "punch": punch,
             "cuts": merge_short(s.get("cuts") or [], s["min_shot"]), "bw": bw, "format": s["format"],
             "zoomMode": "push", "pushTo": float(s["push_to"]), "pushHolds": [], "eyeLine": float(s["eye_line"]),
             "subLine": float(s["sub_line"]), "endCta": s["end_cta"]}
    path = project_dir / "talking1_props.json"
    path.write_text(json.dumps(props, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def run_remotion(args, project_dir):
    cmd = [npx_path(), "remotion", *args, "--public-dir", str(project_dir), "--log", "error"]
    r = subprocess.run(cmd, cwd=str(REMOTION_DIR), capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        raise RuntimeError("Remotion failed:\n" + (r.stderr or r.stdout)[-2500:])


def render_still(project_dir, project, A, t):
    src = source_video(project_dir, project, A)
    props = build_props(project_dir, project, A, src)
    out = project_dir / "talking1_still.png"
    run_remotion(["still", "src/index.ts", COMPOSITION, str(out), f"--frame={int(t * 30)}", "--scale=0.4",
                  f"--props={props}"], project_dir)
    return out


def render_video(project_dir, project, A):
    src = source_video(project_dir, project, A)
    props = build_props(project_dir, project, A, src)
    s = settings(project)
    out = project_dir / f"talking1_{s['format'].replace(':', 'x')}.mp4"
    run_remotion(["render", "src/index.ts", COMPOSITION, str(out), f"--props={props}", "--crf=18"], project_dir)
    return out


# ------------------------------------------------------------------ the step (UI)
def step(project_dir, project, A):
    pid = project["project_id"]
    st.subheader("Talking1 style")
    st.caption("Big punch words at eye level, small lowercase subtitles, warm grade with black & white moments, "
               "your push-in zoom on every shot and a 'follow karo' end card. Rendered with the same Remotion "
               "template Claude uses, so it matches those edits.")
    problem = remotion_problem()
    if problem:
        st.error(problem)
        return
    keep = A["ensure_keep"](project_dir, project)
    if not keep:
        st.info(A["NO_PROJECT_MSG"])
        return
    if not project.get("transcript"):
        st.info("First transcribe the video in step 3 (A. Transcribe) and correct the words there. "
                "The subtitles here come from that transcript.")
        return
    if project.get("transcript_keep_version") != project.get("keep_version"):
        st.warning("You changed the cut after transcribing. Transcribe again in step 3 before using this style.")
        return
    s = settings(project)
    _, total = A["_offsets"](keep)

    # ---- 1. subtitles + punch words (review before rendering)
    st.markdown("#### 1 · Review subtitles and punch words")
    use_roman = st.checkbox("Write Devanagari words in Roman letters", value=s["romanize"], key=f"t1_rom_{pid}",
                            help="Rough spelling. Correct the words in the step 3 transcript editor.")
    if use_roman != s["romanize"]:
        save_settings(project_dir, project, A, romanize=use_roman)
    subs = subtitles(project, A, use_roman)
    with st.expander(f"Subtitles ({len(subs)} lines, from the step 3 transcript)", expanded=False):
        st.code("\n".join(f"{c['start']:6.2f}  {c['text']}" for c in subs) or "(none)", language="text")
        st.caption("To fix a word, correct it in step 3 and save. It updates here.")
    if "punch" not in s or st.button("Suggest punch words again", key=f"t1_sugg_{pid}"):
        punch = suggest_punch(project, A, use_roman)
        save_settings(project_dir, project, A, punch=punch, bw=suggest_bw(project, A, punch))
        s = settings(project)
        st.rerun()
    st.caption("Punch words replace the subtitle while they are on screen. Change the text (e.g. 'IRFAN JUNEJO'), "
               "times, colour, or add / delete rows. 'Cream word' colours one word of a white phrase cream.")
    pdf = pd.DataFrame(s.get("punch") or [], columns=["Start", "End", "Text", "Colour", "Cream word"])
    edited = st.data_editor(
        pdf, num_rows="dynamic", use_container_width=True, height=320, key=f"t1_punch_{pid}",
        column_config={"Start": st.column_config.NumberColumn(format="%.2f", min_value=0.0, step=0.05),
                       "End": st.column_config.NumberColumn(format="%.2f", min_value=0.0, step=0.05),
                       "Text": st.column_config.TextColumn(width="medium"),
                       "Colour": st.column_config.SelectboxColumn(options=COLOURS, required=True),
                       "Cream word": st.column_config.TextColumn()})
    st.markdown("**Black & white moments** (seconds)")
    bdf = pd.DataFrame(s.get("bw") or [], columns=["Start", "End"])
    bw_edit = st.data_editor(bdf, num_rows="dynamic", use_container_width=True, key=f"t1_bw_{pid}",
                             column_config={"Start": st.column_config.NumberColumn(format="%.2f", min_value=0.0),
                                            "End": st.column_config.NumberColumn(format="%.2f", min_value=0.0)})
    if st.button("Save punch words and B&W", type="primary", key=f"t1_psave_{pid}"):
        rows = [r for r in edited.to_dict("records") if pd.notna(r.get("Start")) and pd.notna(r.get("End"))
                and str(r.get("Text") or "").strip()]
        for r in rows:
            r["Colour"] = r.get("Colour") if r.get("Colour") in COLOURS else "white"
            r["Cream word"] = "" if pd.isna(r.get("Cream word")) else str(r.get("Cream word") or "")
        bws = [r for r in bw_edit.to_dict("records") if pd.notna(r.get("Start")) and pd.notna(r.get("End"))]
        save_settings(project_dir, project, A, punch=sorted(rows, key=lambda r: r["Start"]), bw=bws)
        st.success(f"Saved {len(rows)} punch words and {len(bws)} black & white moment(s).")

    # ---- 2. cuts
    st.markdown("#### 2 · Confirm the cuts (the zoom restarts at each one)")
    src = None
    try:
        with st.spinner("Preparing the cut video..."):
            src = source_video(project_dir, project, A)
    except Exception as exc:
        st.error(str(exc))
        return
    silence_cuts = keep_boundaries(project, A)
    if "candidates" not in s or s.get("candidates_key") != project.get("keep_version"):
        with st.spinner("Looking for jump cuts..."):
            cands = detect_cuts(project_dir / src, A)
        save_settings(project_dir, project, A, candidates=cands, candidates_key=project.get("keep_version"),
                      cuts=merge_short(silence_cuts + [c["t"] for c in cands if c["likely"]], 0.3))
        s = settings(project)
    chosen = set(round(c, 3) for c in s.get("cuts") or [])
    if silence_cuts:
        st.caption(f"{len(silence_cuts)} cuts come from removing pauses / repeats. They are always used "
                   f"(shots shorter than {s['min_shot']} s are joined so the zoom has time to move).")
    cands = s.get("candidates") or []
    if cands:
        st.caption("Jump cuts found in the video. Tick the real cuts (the person jumps between the two pictures). "
                   "Untick hand movements, or the zoom will jump in the middle of a shot.")
        fps = video_fps(project_dir / src, A)
        new_chosen = set(c for c in chosen if c in silence_cuts)
        cols = st.columns(4)
        for k, c in enumerate(cands):
            with cols[k % 4]:
                png = thumb_pair(project_dir, src, c["t"], fps, A)
                if png:
                    st.image(str(png), use_container_width=True)
                if st.checkbox(f"{c['t']:.2f}s cut", value=round(c["t"], 3) in chosen, key=f"t1_cut_{pid}_{c['t']}"):
                    new_chosen.add(round(c["t"], 3))
        if new_chosen != chosen:
            save_settings(project_dir, project, A, cuts=sorted(new_chosen))
            s = settings(project)
    elif not silence_cuts:
        st.caption("No cuts found: the zoom pushes in slowly across the whole video.")

    # ---- 3. look
    st.markdown("#### 3 · Look")
    a, b = st.columns(2)
    fmt_labels = list(FORMATS)
    cur_fmt = next((k for k, v in FORMATS.items() if v == s["format"]), fmt_labels[0])
    fmt = FORMATS[a.radio("Output", fmt_labels, index=fmt_labels.index(cur_fmt), horizontal=True,
                          key=f"t1_fmt_{pid}")]
    push = a.slider("Zoom at the end of each shot", 1.0, 2.0, float(s["push_to"]), 0.05, key=f"t1_push_{pid}",
                    help="1.55 matches your Resolve zooms. 1.0 = no zoom.")
    eye = b.slider("Punch word height (put it on the eyes)", 0.15, 0.75, float(s["eye_line"]), 0.01,
                   key=f"t1_eye_{pid}", help="0.33 suits a square video, about 0.5 a 9:16 video.")
    sub = b.slider("Subtitle height", 0.6, 0.95, float(s["sub_line"]), 0.01, key=f"t1_sub_{pid}")
    cta = a.text_input("End card text", s["end_cta"], key=f"t1_cta_{pid}")
    new = dict(format=fmt, push_to=push, eye_line=eye, sub_line=sub, end_cta=cta)
    if any(s[k] != v for k, v in new.items()):
        save_settings(project_dir, project, A, **new)
        s = settings(project)
    p1, p2 = st.columns([2, 1])
    first_punch = (s.get("punch") or [{"Start": 1.0}])[0]["Start"]
    t_prev = p1.slider("Preview frame at (seconds)", 0.0, max(0.1, round(total - 0.1, 1)),
                       float(min(round(first_punch + 0.1, 1), max(0.0, total - 0.2))), 0.1, key=f"t1_tprev_{pid}")
    if p2.button("Show this frame", key=f"t1_still_{pid}"):
        try:
            with st.spinner("Rendering one frame (about 15 seconds)..."):
                st.session_state[f"t1_stillimg_{pid}"] = str(render_still(project_dir, project, A, t_prev))
        except Exception as exc:
            st.error(str(exc))
    still = st.session_state.get(f"t1_stillimg_{pid}")
    if still and Path(still).exists():
        st.image(still, width=360, caption="Exact frame from the final render")

    # ---- 4. render
    st.markdown("#### 4 · Render")
    st.caption("Uses the saved punch words. Save them above first. A 35 s video takes about 2-4 minutes.")
    if st.button("Render Talking1 video", type="primary", key=f"t1_render_{pid}"):
        try:
            with st.spinner("Rendering with Remotion..."):
                out = render_video(project_dir, project, A)
            st.session_state[f"t1_out_{pid}"] = str(out)
        except Exception as exc:
            st.error(str(exc))
    res = st.session_state.get(f"t1_out_{pid}")
    if res and Path(res).exists():
        v, d = st.columns([1, 1])
        v.video(res)
        d.success(f"Rendered: {Path(res).name}")
        d.caption(f"Saved in: {Path(res).parent}")
        d.download_button("Download video", Path(res).read_bytes(), file_name=Path(res).name, key=f"t1_dl_{pid}")
