"""Talking1 style preset for Reel Studio AI.

Big condensed punch words at eye level, tiny lowercase subtitles, warm grade with black & white
emphasis moments, the user's linear push-in zoom (resets at every cut) and a "follow karo" end card.

The look itself is the Remotion template made with Claude in D:\\ClaudeAIVideoEdit\\remotion
(composition "Talking1"). This file only prepares the data and calls Remotion, so the preset looks
exactly like the edits made there.

Kept in its own file on purpose: update.ps1 replaces app.py on every update but never touches this
file. app.py only needs a few lines to show it as step 5 (see the "Talking1 add-on" comments there).
Set the environment variable REELSTUDIO_REMOTION_DIR if the Remotion project moves.
"""
import hashlib
import http.server
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
                romanize=True, min_shot=1.0, push_rate=0.12, pop_on=True,
                pop_volume=0.5)
# Zoom speed limit (zoom added per second): short shots zoom in less instead of rushing to the full zoom.
ZOOM_SPEEDS = {"Slow": 0.06, "Gentle": 0.12, "Medium": 0.2, "Fast (full zoom in every shot)": 0.0}

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
    if not Path(node_path()).exists():
        return "Node.js was not found. Install Node.js and restart the app."
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


# ------------------------------------------------------------------ colours & frames (shared with Template 2)
# Ready-made colour styles for Talking1: punch words, the "red" and "cream" punch colours, subtitles, end card.
T1_FIELDS = [("punch", "Main (punch words)"), ("red", "Colour 2"), ("cream", "Colour 3"),
             ("subs", "Subtitles"), ("end", "End card text"), ("endBg", "End card background")]
T1_PALETTES = {
    "Classic (red & cream)": dict(punch="#FFFFFF", red="#D21F26", cream="#F1DDBF", subs="#FFFFFF", end="#F1DDBF", endBg="#000000"),
    "Bold yellow": dict(punch="#FFFFFF", red="#FF3B30", cream="#FFD60A", subs="#FFFFFF", end="#FFD60A", endBg="#000000"),
    "Ocean": dict(punch="#FFFFFF", red="#FF5A5F", cream="#7FDBFF", subs="#FFFFFF", end="#7FDBFF", endBg="#001F3F"),
    "Candy": dict(punch="#FFFFFF", red="#FF2E88", cream="#FFC2E2", subs="#FFFFFF", end="#FFC2E2", endBg="#2A0A33"),
    "Lime": dict(punch="#FFFFFF", red="#FF4D00", cream="#C6FF00", subs="#FFFFFF", end="#C6FF00", endBg="#0B0B0B"),
    "Mono": dict(punch="#FFFFFF", red="#A8A8A8", cream="#E6E6E6", subs="#FFFFFF", end="#FFFFFF", endBg="#000000"),
}


def swatch_html(colours, labels):
    cells = "".join(
        f'<div style="text-align:center;font-size:11px;opacity:.85"><div style="width:44px;height:28px;border-radius:6px;'
        f'background:{c};border:1px solid rgba(128,128,128,.45);margin:0 auto 3px"></div>{l}</div>'
        for c, l in zip(colours, labels))
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:2px 0 8px">{cells}</div>'


def look_picker(pid, prefix, looks, s, on_save, help_text=""):
    """One 'Look' dropdown (a ready-made set of colours and effects) + '🎲 Surprise me'.
    looks: {name: {settings to store}}. The chosen name is stored as 'look'; colours changed by hand
    show as 'Custom'. Widget keys carry a revision so a new look starts fresh widgets."""
    import random
    rev_key = f"{prefix}_rev_{pid}"
    rev = st.session_state.get(rev_key, 0)
    names = list(looks)
    cur = s.get("look") or names[0]
    options = names + (["Custom"] if cur not in names else [])
    a, b = st.columns([3, 1])
    choice = a.selectbox("Look", options, index=options.index(cur if cur in options else names[0]),
                         key=f"{prefix}_look_{pid}_{rev}", help=help_text or None)
    b.write("")
    surprise = b.button("🎲 Surprise me", key=f"{prefix}_surprise_{pid}", use_container_width=True,
                        help="A random look, so this video looks different from the last one.")
    pick = None
    if surprise:
        pick = random.choice([n for n in names if n != cur] or names)
    elif choice != cur and choice in looks:
        pick = choice
    if pick:
        st.session_state[rev_key] = rev + 1
        on_save(look=pick, **looks[pick])
        st.rerun()


def custom_colours(pid, prefix, fields, pal, on_save):
    """One colour picker per role (for 'More settings'). Any change stores the look as 'Custom'."""
    rev_key = f"{prefix}_rev_{pid}"
    rev = st.session_state.get(rev_key, 0)
    cols = st.columns(4)
    new = {k: cols[i % 4].color_picker(label, pal.get(k, "#FFFFFF"), key=f"{prefix}_col_{k}_{pid}_{rev}")
           for i, (k, label) in enumerate(fields)}
    if any(new[k].lower() != str(pal.get(k, "")).lower() for k in new):
        st.session_state[rev_key] = rev + 1
        on_save(palette=new, look="Custom")
        st.rerun()


# ------------------------------------------------------------------ word plan (shared with Template 2)
# Both templates show one row per spoken word: what it shows as, its colour and effect. The saved data
# stays as before (punch words / text events with Start-End times); the grid is made from it and turned
# back into it, so older projects keep working.
def spoken_words(project, A, use_roman):
    """Every word of the (corrected) step 4 transcript with its own time. When the corrected line has
    the same number of words as Whisper heard, Whisper's word times are used; otherwise the line's
    time is shared between its words by their length."""
    out = []
    line, count = -1, 0
    for seg in A["caption_segments"](project):
        line, count = line + 1, 0
        text = roman(seg["text"], A) if use_roman else seg["text"]
        parts = text.split()
        if not parts:
            continue
        heard = [w for w in seg.get("words") or [] if str(w.get("word", "")).strip()]
        s0, s1 = float(seg["start"]), float(seg["end"])
        if len(heard) == len(parts):
            times = [(max(float(w["start"]), s0), min(float(w["end"]), s1)) for w in heard]
        else:
            a = max(s0, float(heard[0]["start"])) if heard else s0
            b = min(s1, float(heard[-1]["end"])) if heard else s1
            if b <= a:
                a, b = s0, s1
            weights = [max(2, len(p)) for p in parts]
            total, cur, times = sum(weights), a, []
            for wt in weights:
                nxt = cur + (b - a) * wt / total
                times.append((cur, nxt))
                cur = nxt
        for p, (s, e) in zip(parts, times):
            # caption-sized lines (as in the grid): a new line after , . ! ? or every 7 words
            if count >= 7:
                line, count = line + 1, 0
            out.append({"start": round(s, 3), "end": round(max(e, s + 0.08), 3), "text": p, "line": line})
            count += 1
            if re.search(r"[,.!?;:।]$", p):
                line, count = line + 1, 0
        line += 1
    return out


def norm_word(w):
    return clean_word(str(w)).lower()


def assign_words(words, spans):
    """Which span (index) owns each word, or None. A span owns the words whose middle is inside it;
    with 'tokens' only the matching words (and the words between them), so an accent 'pressure.'
    that stays on screen a bit longer doesn't take the next word too."""
    owner = [None] * len(words)
    mids = [(w["start"] + w["end"]) / 2 for w in words]
    for k, sp in enumerate(spans):
        a, b = float(sp["start"]), float(sp["end"])
        cand = [i for i in range(len(words)) if owner[i] is None and a - 0.05 <= mids[i] < b + 0.02]
        toks = sp.get("tokens")
        if toks and cand:
            hit = [i for i in cand if norm_word(words[i]["text"]) in toks]
            cand = [i for i in range(hit[0], hit[-1] + 1) if owner[i] is None] if hit else cand[:1]
        if not cand:  # nothing inside (e.g. a gap between words): the nearest free word
            free = [i for i in range(len(words)) if owner[i] is None]
            if free:
                i = min(free, key=lambda i: abs(words[i]["start"] - a))
                if abs(words[i]["start"] - a) < 0.6:
                    cand = [i]
        for i in cand:
            owner[i] = k
    return owner


def group_runs(n, key_of, owner, text_of):
    """Consecutive rows with the same (non-None) key form one item. A new item also starts where two
    different old items meet, or where a new word (not part of an old item) has its own on-screen text."""
    runs, cur, cur_key = [], [], None
    for i in range(n):
        k = key_of(i)
        brk = (k is None or k != cur_key or not cur
               or (owner[i] is not None and owner[cur[-1]] is not None and owner[i] != owner[cur[-1]])
               or (owner[i] is None and bool(text_of(i))))
        if brk and cur:
            runs.append((cur_key, cur))
            cur = []
        if k is not None:
            cur.append(i)
        cur_key = k
    if cur:
        runs.append((cur_key, cur))
    return runs


def owned_by(owner, k):
    return [i for i, o in enumerate(owner) if o == k]


def cell(v, default=""):
    """A data editor value without NaN / None."""
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    return v


def plan_timeline(words, kinds, bars, colour_domain, colour_range, rows, view, key, height=150):
    """Words coloured by what they show as, plus labelled bars (one row each in 'rows').
    Drag across it to select words. Returns the selected (from, to) seconds or None."""
    import altair as alt
    v0, v1 = view
    px_per_s = 1000 / max(0.5, v1 - v0)

    def fit(text, dur):
        n = int(dur * px_per_s / 7)
        return text if len(text) <= n else (text[:max(0, n - 1)] + "…" if n > 3 else "")

    wrows = [{"start": w["start"], "end": max(w["end"], w["start"] + 0.12), "row": "Words", "kind": kd,
              "label": fit(w["text"], w["end"] - w["start"]), "word": w["text"]}
             for w, kd in zip(words, kinds) if w["end"] > v0 and w["start"] < v1]
    brows = [{**b, "label": fit(b["label"], b["end"] - b["start"])} for b in bars if b["end"] > v0 and b["start"] < v1]
    order = ["Words"] + rows
    x = alt.X("start:Q", title="seconds", scale=alt.Scale(domain=[v0, v1], nice=False))
    y = alt.Y("row:N", title=None, sort=order, scale=alt.Scale(domain=order))
    colours = alt.Scale(domain=colour_domain, range=colour_range)
    brush = alt.selection_interval(encodings=["x"], name="brush")
    base = (alt.Chart(alt.Data(values=wrows)).mark_rect(cornerRadius=2, stroke="white", strokeWidth=1)
            .encode(x=x, x2="end:Q", y=y,
                    color=alt.Color("kind:N", scale=colours, legend=alt.Legend(orient="top", title=None)),
                    tooltip=[alt.Tooltip("word:N", title="Word"), alt.Tooltip("start:Q", format=".2f", title="At (s)"),
                             alt.Tooltip("kind:N", title="Shows as")])
            .add_params(brush))
    layers = [base, alt.Chart(alt.Data(values=wrows)).mark_text(align="left", dx=2, color="white", fontSize=10)
              .encode(x=x, y=y, text="label:N")]
    if brows:
        layers.append(alt.Chart(alt.Data(values=brows)).mark_rect(cornerRadius=3, stroke="white", strokeWidth=1)
                      .encode(x=x, x2="end:Q", y=y, color=alt.Color("kind:N", scale=colours)))
        layers.append(alt.Chart(alt.Data(values=brows)).mark_text(align="left", dx=3, color="white", fontSize=11)
                      .encode(x=x, y=y, text="label:N"))
    chart = alt.layer(*layers).properties(height=height, width="container").configure_view(strokeWidth=0)
    event = st.altair_chart(chart, width="stretch", height=height + 60, on_select="rerun",
                            selection_mode=["brush"], key=key)
    try:
        sel = event.selection if hasattr(event, "selection") else event["selection"]
        for v in (sel.get("brush") or {}).values():
            if isinstance(v, (list, tuple)) and len(v) == 2 and abs(v[1] - v[0]) >= 0.05:
                return float(min(v)), float(max(v))
    except Exception:
        pass
    return None


def timeline_view(pid, prefix, total):
    """Long videos: show a window of the timeline so the words stay readable."""
    if total <= 30:
        return 0.0, total
    w = st.slider("Part of the video shown on the timeline (seconds)", 0.0, float(round(total, 1)),
                  (0.0, float(min(25.0, round(total, 1)))), 0.5, key=f"{prefix}_view_{pid}")
    return (w[0], w[1]) if w[1] - w[0] >= 2 else (w[0], min(total, w[0] + 2))


def grid_editor(df, key, column_config, disabled):
    """The word grid at full height (no scrolling inside it)."""
    return st.data_editor(df, key=key, hide_index=True, use_container_width=True, num_rows="fixed",
                          height=35 * (len(df) + 1) + 3, column_config=column_config, disabled=disabled)


def grid_keep(state_key, edited, changed):
    """After a grid edit: keep the same grid (True), so edits made while the page reloads aren't lost, or
    rebuild it (False) when the edit changed nothing (e.g. a refused tick) or the same grid content was
    already applied once (stops a reload loop)."""
    sig = edited.to_json()
    if not changed or st.session_state.get(state_key) == sig:
        st.session_state.pop(state_key, None)
        return False
    st.session_state[state_key] = sig
    return True


def grid_changed(edited, base, cols):
    a = [[cell(edited.iloc[i][c]) for c in cols] for i in range(len(edited))]
    b = [[cell(base.iloc[i][c]) for c in cols] for i in range(len(base))]
    return a != b


def selected_words(words, sel):
    return [i for i, w in enumerate(words) if sel[0] <= (w["start"] + w["end"]) / 2 <= sel[1]]


def line_groups(words):
    """Word indexes per caption line (the step 4 transcript lines)."""
    groups = {}
    for i, w in enumerate(words):
        groups.setdefault(w.get("line", 0), []).append(i)
    return [groups[k] for k in sorted(groups)]


def split_phrases(text):
    return [p.strip() for p in str(text or "").split("/") if p.strip()]


def phrase_words(words, idx, phrase):
    """The words of a line (idx) that a typed phrase points at: an exact run first, else first..last match."""
    toks = [norm_word(t) for t in str(phrase).split() if norm_word(t)]
    if not toks:
        return []
    lw = [norm_word(words[i]["text"]) for i in idx]
    n = len(toks)
    for s in range(len(idx) - n + 1):
        if lw[s:s + n] == toks:
            return idx[s:s + n]
    hits = [idx[j] for j, t in enumerate(lw) if t in toks]
    return [i for i in idx if hits[0] <= i <= hits[-1]] if hits else []


def best_word(words, idx):
    """A sensible word to pop when none is typed: the longest word that isn't a filler."""
    cands = [i for i in idx if norm_word(words[i]["text"]) not in STOP] or idx
    return clean_word(words[max(cands, key=lambda i: len(clean_word(words[i]["text"])))]["text"])


def line_changed(o, n, cols):
    return any(cell(o[c]) != cell(n[c]) for c in cols)


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
        if p.get("Pop") is False:
            item["pop"] = False
        punch.append(item)
    punch.sort(key=lambda x: x["start"])
    bw = [[float(b["Start"]), float(b["End"])] for b in s.get("bw") or [] if float(b["End"]) > float(b["Start"])]
    props = {"src": src, "subs": subtitles(project, A, s["romanize"]), "punch": punch,
             "cuts": merge_short(s.get("cuts") or [], s["min_shot"]), "bw": bw, "format": s["format"],
             "zoomMode": "push", "pushTo": float(s["push_to"]), "pushRate": float(s["push_rate"]), "pushHolds": [], "eyeLine": float(s["eye_line"]),
             "subLine": float(s["sub_line"]), "endCta": s["end_cta"],
             "palette": s.get("palette") or T1_PALETTES["Classic (red & cream)"]}
    pop = pop_props(project_dir, s)
    if pop:
        props["pop"] = pop
    path = project_dir / "talking1_props.json"
    path.write_text(json.dumps(props, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


# ------------------------------------------------------------------ pop sound on pop-out words (all templates)
POP_FILE = "pop_sfx.wav"  # made in the project folder (Remotion's public dir), so no sound file is needed


POP_SOURCE = Path(__file__).resolve().parent / "sfx" / "pop.wav"  # the user's pop (pop-4, lead-in trimmed)


def ensure_pop(project_dir):
    """The pop sound in the project folder: the app's sfx/pop.wav (kept in step with it), or, if that file
    is missing, a short synthesised 'bubble pop' made once."""
    path = project_dir / POP_FILE
    if POP_SOURCE.exists():
        data = POP_SOURCE.read_bytes()
        if not path.exists() or path.read_bytes() != data:
            path.write_bytes(data)
        return path
    if path.exists():
        return path
    import wave
    sr = 48000
    t = np.arange(int(0.12 * sr)) / sr
    freq = 300 + 1100 * np.exp(-t / 0.012)
    tone = np.sin(2 * np.pi * np.cumsum(freq) / sr) * (1 - np.exp(-t / 0.0006)) * np.exp(-t / 0.024)
    rng = np.random.default_rng(7)
    click = np.diff(rng.standard_normal(len(t) + 1)) * np.exp(-t / 0.0015) * 0.25
    y = tone + click
    y = y / np.max(np.abs(y)) * 0.9
    pcm = (np.repeat(y[:, None], 2, axis=1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


CLICK_FILE = "click_sfx.wav"
CLICK_SOURCE = Path(__file__).resolve().parent / "sfx" / "click.wav"  # the user's mouse click (lead-in trimmed)


def click_props(project_dir, s):
    """The `click` prop (Template 3: a click when each B-roll clip appears), or None when off / missing."""
    if not s.get("click_on", True) or float(s.get("click_volume", 0.5)) <= 0 or not CLICK_SOURCE.exists():
        return None
    path = project_dir / CLICK_FILE
    data = CLICK_SOURCE.read_bytes()
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)
    return {"src": CLICK_FILE, "volume": round(float(s.get("click_volume", 0.5)), 2)}


def pop_props(project_dir, s):
    """The `pop` prop for the templates, or None when the sound is off."""
    if not s.get("pop_on", True) or float(s.get("pop_volume", 0.5)) <= 0:
        return None
    ensure_pop(project_dir)
    return {"src": POP_FILE, "volume": round(float(s.get("pop_volume", 0.5)), 2)}


def pop_controls(pid, prefix, s, save, what="pop words"):
    """One row: pop sound on/off + volume. Changes save straight away."""
    a, b = st.columns([1, 2])
    on = a.checkbox(f"🔊 Pop sound on {what}", value=bool(s.get("pop_on", True)), key=f"{prefix}_popon_{pid}",
                    help=f"A short pop each time one of the {what} appears.")
    vol = b.slider("Pop volume", 0.1, 1.0, float(s.get("pop_volume", 0.5)), 0.05, key=f"{prefix}_popvol_{pid}",
                   disabled=not on)
    if on != bool(s.get("pop_on", True)) or abs(vol - float(s.get("pop_volume", 0.5))) > 1e-6:
        save(pop_on=on, pop_volume=vol)


# ------------------------------------------------------------------ per-line pop sound (grid column)
POP_COL = "Pop SFX"
POP_COL_HELP = "Play the pop sound on this line's pop words. Untick to keep this line silent."


def _pop_items(words, groups, g, items, is_pop):
    a = float(words[groups[g][0]]["start"]) - 0.01
    b = float(words[groups[g][-1]]["end"])
    return [k for k, it in enumerate(items) if is_pop(it) and a <= float(it["Start"]) <= b]


def pop_values(words, groups, items, is_pop):
    """Per line: ticked when it has pop words and none of them is silenced."""
    out = []
    for g in range(len(groups)):
        ks = _pop_items(words, groups, g, items, is_pop)
        out.append(bool(ks) and all(items[k].get("Pop", True) is not False for k in ks))
    return out


def apply_pop_col(words, groups, items, old_lines, new_lines, is_pop):
    """Ticks changed in the Pop SFX column -> 'Pop' on the pop items of those lines."""
    items, notes = [dict(x) for x in items], []
    for g, (o, n) in enumerate(zip(old_lines, new_lines)):
        want = bool(cell(n.get(POP_COL), False))
        if want == bool(cell(o.get(POP_COL), False)):
            continue
        ks = _pop_items(words, groups, g, items, is_pop)
        if not ks:
            if want:
                notes.append(f"The line at {float(words[groups[g][0]]['start']):.2f} s has no pop words, so there "
                             "is no pop to play.")
            continue
        for k in ks:
            items[k]["Pop"] = want
    return items, notes


# ------------------------------------------------------------------ carry forward (render once, at the end)
# A template page can carry its settings forward instead of rendering. The finishing pages (Visual hook, and
# later ones) then render the Remotion "Finish" composition: the template drawn live + their touches, in ONE
# render. Saved as project["carry"] = {"template": "talking1" | "template2" | "template3", "at": time}.
CARRY_TEMPLATES = {"talking1": ("Talking1", "Talking1"), "template2": ("Template2", "Template 2"),
                   "template3": ("Template3", "Template 3")}


def carry_button(project_dir, project, A, key, col):
    """'Carry forward' next to a template's Render button: no render now, the Visual hook page renders it."""
    pid = project["project_id"]
    name = CARRY_TEMPLATES[key][1]
    target = next((s for s in A.get("STEPS", []) if s.endswith("Visual hook")), None)
    if not target:
        return
    if col.button("➡ Carry forward to Visual hook (no render)", key=f"carry_{key}_{pid}", use_container_width=True,
                  help=f"Don't render now. The Visual hook page renders {name} with the hook in one go, always "
                       "with this page's latest settings."):
        project["carry"] = {"template": key, "at": __import__("datetime").datetime.now().isoformat(timespec="seconds")}
        A["save_project"](project_dir, project)
        st.session_state["_goto"] = target
        st.rerun()
    if (project.get("carry") or {}).get("template") == key:
        col.caption(f"✓ {name} is carried forward to Visual hook.")


def live_props(project_dir, pid, port, props, video):
    """Props for the browser preview: the cut video, B-roll and sounds as URLs from the local preview server."""
    vname = f"v/{pid}_{int(video.stat().st_mtime)}.mp4"
    _LiveHandler.files[vname] = str(video)
    props = json.loads(json.dumps(props))
    props["src"] = f"http://127.0.0.1:{port}/{vname}"
    for sk in ("pop", "click"):  # sound effects in the project folder
        if props.get(sk) and not str(props[sk]["src"]).startswith("http"):
            f = project_dir / props[sk]["src"]
            if f.exists():
                pname = f"p/{pid}_{sk}_{int(f.stat().st_mtime)}.wav"
                _LiveHandler.files[pname] = str(f)
                props[sk]["src"] = f"http://127.0.0.1:{port}/{pname}"
    for b in props.get("brolls") or []:  # Template 3: B-roll clips in the project folder
        f = project_dir / b["src"]
        if f.exists() and not str(b["src"]).startswith("http"):
            bname = f"b/{pid}_{f.stem}_{int(f.stat().st_mtime)}.mp4"
            _LiveHandler.files[bname] = str(f)
            b["src"] = f"http://127.0.0.1:{port}/{bname}"
    return props


def node_path():
    return shutil.which("node") or r"C:\Program Files\nodejs\node.exe"


REMOTION_CLI = REMOTION_DIR / "node_modules" / "@remotion" / "cli" / "remotion-cli.js"


def run_remotion(args, project_dir):
    # Run Remotion's CLI file with node directly. Going through npx made npm hand the command to
    # cmd.exe, which in some app sessions failed with "'remotion' is not recognized".
    if REMOTION_CLI.exists():
        cmd = [node_path(), str(REMOTION_CLI), *args]
    else:
        cmd = [npx_path(), "remotion", *args]
    cmd += ["--public-dir", str(project_dir), "--log", "error"]
    r = subprocess.run(cmd, cwd=str(REMOTION_DIR), capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        detail = "\n".join(x for x in (r.stderr.strip(), r.stdout.strip()) if x)
        raise RuntimeError("Remotion failed:\n" + detail[-2500:])


def render_still(project_dir, project, A, t):
    src = source_video(project_dir, project, A)
    props = build_props(project_dir, project, A, src)
    out = project_dir / "talking1_still.png"
    run_remotion(["still", "src/index.ts", COMPOSITION, str(out), f"--frame={int(t * 30)}", "--scale=0.4",
                  f"--props={props}"], project_dir)
    return out


def props_key(props_path):
    """Changes whenever anything that affects the picture changes (punch words, cuts, look, subtitles)."""
    import hashlib
    return hashlib.md5(Path(props_path).read_bytes()).hexdigest()


def render_preview(project_dir, project, A, start, length, total, fps=30):
    """Low-resolution preview video from the same Remotion composition, so it looks exactly like the
    final render, only smaller. length=None renders the whole video (with the end card)."""
    src = source_video(project_dir, project, A)
    props = build_props(project_dir, project, A, src)
    out = project_dir / "talking1_preview.mp4"
    args = ["render", "src/index.ts", COMPOSITION, str(out), f"--props={props}", "--scale=0.5", "--crf=26"]
    if length is not None:
        last = max(1, int(total * fps) - 1)
        a = min(int(start * fps), last - 1)
        b = min(int((start + length) * fps), last)
        args.append(f"--frames={a}-{b}")
    if out.exists():
        out.unlink()
    run_remotion(args, project_dir)
    return out, props_key(props)


def render_video(project_dir, project, A):
    src = source_video(project_dir, project, A)
    props = build_props(project_dir, project, A, src)
    s = settings(project)
    out = project_dir / f"talking1_{s['format'].replace(':', 'x')}.mp4"
    run_remotion(["render", "src/index.ts", COMPOSITION, str(out), f"--props={props}", "--crf=18"], project_dir)
    return out


# ------------------------------------------------------------------ live preview (shared with Template 2)
# The same Remotion template plays in the browser with Remotion's Player, so every change in the grid shows
# at once, without rendering. The Player code is bundled once from the Remotion folder with esbuild (rebuilt
# when the templates change) and, with a small low-res copy of the cut, served by a tiny local web server.
LIVE_DIR = Path(__file__).resolve().parent / "live_preview_cache"
LIVE_SOURCE = "live_source.mp4"
LIVE_ENTRY = r'''
import React, { useEffect, useRef } from "react";
import { createRoot } from "react-dom/client";
import { Player, PlayerRef } from "@remotion/player";
import { Poorvay } from "__SRC__/Poorvay";
import { PunchWordEdit } from "__SRC__/PunchWordEdit";
import { VisualHook } from "__SRC__/VisualHook";
import { Finish } from "__SRC__/Finish";

const COMPS: Record<string, React.FC<any>> = { Template2: Poorvay, Template3: Poorvay, Talking1: PunchWordEdit, VisualHook, Finish };
type Opts = { comp: string; props: any; frames: number; w: number; h: number; seek?: number; seekId?: number };

const Live: React.FC<{ o: Opts }> = ({ o }) => {
  const ref = useRef<PlayerRef>(null);
  useEffect(() => {
    if (o.seekId && ref.current && typeof o.seek === "number") {
      ref.current.seekTo(Math.max(0, Math.round(o.seek * 30)));
      ref.current.play();
    }
  }, [o.seekId]);
  return (
    <Player ref={ref} component={COMPS[o.comp]} inputProps={o.props} durationInFrames={Math.max(1, o.frames)}
      fps={30} compositionWidth={o.w} compositionHeight={o.h} controls clickToPlay spaceKeyToPlayOrPause
      style={{ width: "100%", borderRadius: 8, overflow: "hidden", background: "#000" }} acknowledgeRemotionLicense />
  );
};

export function mount(el: HTMLElement) {
  const root = createRoot(el);
  return { update: (o: Opts) => root.render(<Live o={o} />) };
}
'''

LIVE_JS = """
export default function(component) {
  const { data, parentElement } = component;
  let host = parentElement.querySelector('.rs-live');
  if (!host) {
    host = document.createElement('div');
    host.className = 'rs-live';
    host.textContent = 'Loading the live preview...';
    parentElement.appendChild(host);
  }
  const go = (mod) => {
    if (!host.ctl) { host.textContent = ''; host.ctl = mod.mount(host); }
    host.ctl.update(data.opts);
  };
  if (window.__rsLive && window.__rsLiveUrl === data.bundle) { go(window.__rsLive); return; }
  import(data.bundle).then((m) => { window.__rsLive = m; window.__rsLiveUrl = data.bundle; go(m); })
    .catch((e) => { host.textContent = 'The live preview could not load (' + e + '). Use Preview below instead.'; });
}
"""


def live_bundle():
    """Build (or reuse) the browser bundle of the templates. Returns its path."""
    src = REMOTION_DIR / "src"
    newest = max((p.stat().st_mtime for p in src.rglob("*") if p.is_file()), default=0)
    out, stamp = LIVE_DIR / "live.js", LIVE_DIR / "stamp.txt"
    key = f"{newest}|{REMOTION_DIR}|{hashlib.md5(LIVE_ENTRY.encode()).hexdigest()}"
    if out.exists() and stamp.exists() and stamp.read_text(encoding="utf-8") == key:
        return out
    LIVE_DIR.mkdir(exist_ok=True)
    entry = LIVE_DIR / "entry.tsx"
    entry.write_text(LIVE_ENTRY.replace("__SRC__", src.as_posix()), encoding="utf-8")
    esbuild = REMOTION_DIR / "node_modules" / "esbuild" / "bin" / "esbuild"
    env = {**os.environ, "NODE_PATH": str(REMOTION_DIR / "node_modules")}
    res = subprocess.run([node_path(), str(esbuild), str(entry), "--bundle", "--format=esm", "--minify",
                          "--jsx=automatic", "--loader:.woff2=dataurl", '--define:process.env.NODE_ENV="production"',
                          f"--outfile={out}", "--log-level=error"],
                         capture_output=True, text=True, env=env, cwd=str(LIVE_DIR),
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if res.returncode != 0 or not out.exists():
        raise RuntimeError("Could not build the live preview: " + (res.stderr or res.stdout or "")[-600:])
    stamp.write_text(key, encoding="utf-8")
    return out


class _LiveHandler(http.server.BaseHTTPRequestHandler):
    files = {}  # served name -> file path

    def log_message(self, *args):
        pass

    def do_GET(self):
        p = self.files.get(self.path.split("?")[0].lstrip("/"))
        if not p or not Path(p).exists():
            self.send_error(404)
            return
        size = Path(p).stat().st_size
        a, b, status = 0, size - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                a = int(m.group(1))
                b = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                a = max(0, size - int(m.group(2)))
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", "text/javascript" if str(p).endswith(".js")
                         else "audio/wav" if str(p).endswith(".wav") else "video/mp4")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(b - a + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
        self.end_headers()
        try:
            with open(p, "rb") as f:
                f.seek(a)
                left = b - a + 1
                while left > 0:
                    chunk = f.read(min(256 * 1024, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass


@st.cache_resource
def _live_server():
    import threading
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LiveHandler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]


@st.cache_resource
def _live_component():
    return st.components.v2.component("reelstudio_live_preview", html="", js=LIVE_JS, isolate_styles=False)


def live_source(project_dir, project, A, prefix):
    """A small low-res copy of the cut (from the proxy) for the live preview."""
    key = f"{project.get('keep_version', 0)}|{A['denoise_filter'](project)}"
    out = project_dir / LIVE_SOURCE
    stamp = project_dir / "live_source_key.txt"
    if not (out.exists() and stamp.exists() and stamp.read_text(encoding="utf-8") == key):
        A["render_review"](project_dir, project, name=LIVE_SOURCE)
        stamp.write_text(key, encoding="utf-8")
    return out


def frame_size(fmt, project):
    if fmt == "1:1":
        return 1080, 1080
    if fmt == "9:16":
        return 1080, 1920
    if fmt == "16:9":
        return 1920, 1080
    vi = project.get("video_info") or {}
    w, h = vi.get("width") or 1080, vi.get("height") or 1080
    return 1080, int(round(1080 * h / w / 2) * 2)


def live_preview(project_dir, project, A, prefix, comp, props_path, fmt, total, sel):
    """The template playing live in the browser, updated on every change. sel: seek here when it changes."""
    pid = project["project_id"]
    try:
        with st.spinner("Preparing the live preview..."):
            bundle = live_bundle()
            video = live_source(project_dir, project, A, prefix)
            port = _live_server()
    except Exception as exc:
        st.warning(f"Live preview is not available: {exc}")
        return
    _LiveHandler.files["live.js"] = str(bundle)
    props = live_props(project_dir, pid, port, json.loads(Path(props_path).read_text(encoding="utf-8")), video)
    w, h = frame_size(fmt, project)
    seek_key, last_key = f"{prefix}_live_seek_{pid}", f"{prefix}_live_last_{pid}"
    if sel and st.session_state.get(last_key) != sel:
        st.session_state[seek_key] = st.session_state.get(seek_key, 0) + 1
    st.session_state[last_key] = sel
    opts = {"comp": comp, "props": props, "frames": max(1, int(total * 30)), "w": w, "h": h,
            "seek": float(sel[0]) if sel else 0.0, "seekId": st.session_state.get(seek_key, 0)}
    _live_component()(data={"bundle": f"http://127.0.0.1:{port}/live.js?v={int(bundle.stat().st_mtime)}", "opts": opts},
                      key=f"{prefix}_live_{pid}")
    st.caption("Live preview: the same template, small and without rendering. It updates as you edit. "
               "Make a full preview below to check the final quality.")


def sticky_column(key):
    """Keep the column holding st.container(key=key) in view while the page scrolls."""
    st.markdown(f"<style>div[data-testid='stColumn']:has(.st-key-{key}){{position:sticky;top:3.75rem;"
                f"align-self:flex-start;}}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ Talking1 word grid <-> punch words / B&W
T1_SHOW = ["Subtitle", "Punch word"]
T1_KIND_COLOURS = ["#5f6670", "#c99a06", "#8a8a8a"]  # Subtitle, Punch word, B&W
T1_COLOUR_LABELS = {"white": "Main", "red": "Colour 2", "cream": "Colour 3"}
T1_LABEL_TO_COLOUR = {v: k for k, v in T1_COLOUR_LABELS.items()}
T1_GRID_COLS = ["Time", "Word", "Show as", "Colour", "B&W", "Text on screen"]
T1_EDIT_COLS = ["Show as", "Colour", "B&W", "Text on screen"]
DEFAULT_T1_PALETTE = "Classic (red & cream)"


def _t1look(palette, push):
    return {"palette": dict(T1_PALETTES[palette]), "push_to": push}


# A look = colours + how strong the push-in zoom is. Picked from one dropdown.
T1_LOOKS = {
    "Classic": _t1look("Classic (red & cream)", 1.55),
    "Bold yellow": _t1look("Bold yellow", 1.55),
    "Ocean calm": _t1look("Ocean", 1.3),
    "Candy": _t1look("Candy", 1.55),
    "Lime punchy": _t1look("Lime", 1.7),
    "Mono": _t1look("Mono", 1.4),
}


def t1_plan_rows(words, punch, bw):
    owner = assign_words(words, [{"start": float(p["Start"]), "end": float(p["End"]),
                                  "tokens": {norm_word(w) for w in str(p.get("Text") or "").split()} - {""} or None}
                                 for p in punch])
    bw_owner = assign_words(words, [{"start": float(b["Start"]), "end": float(b["End"])} for b in bw])
    rows, seen = [], set()
    for i, w in enumerate(words):
        r = {"Time": round(w["start"], 2), "Word": w["text"], "Show as": "Subtitle", "Colour": None,
             "B&W": bw_owner[i] is not None, "Text on screen": ""}
        k = owner[i]
        if k is not None:
            p = punch[k]
            creams = {norm_word(c) for c in str(p.get("Cream word") or "").split()}
            r["Show as"] = "Punch word"
            r["Colour"] = "Colour 3" if norm_word(w["text"]) in creams else T1_COLOUR_LABELS.get(p.get("Colour"), "Main")
            if k not in seen:
                r["Text on screen"] = str(p.get("Text") or "")
                seen.add(k)
        rows.append(r)
    return rows, owner, bw_owner


def t1_from_rows(words, rows, owner, bw_owner, punch, bw):
    def text_of(i):
        return str(cell(rows[i].get("Text on screen"))).strip()

    new_punch = []
    for _, idx in group_runs(len(rows), lambda i: "p" if cell(rows[i].get("Show as")) == "Punch word" else None,
                             owner, text_of):
        olds = {owner[i] for i in idx}
        k = next(iter(olds)) if len(olds) == 1 else None
        same = k is not None and 0 <= k < len(punch) and owned_by(owner, k) == idx
        k0 = owner[idx[0]]
        old_text = str(punch[k0].get("Text") or "").strip() if k0 is not None and 0 <= k0 < len(punch) else ""
        first_text = text_of(idx[0])
        joined = " ".join(clean_word(words[i]["text"]) or words[i]["text"] for i in idx)
        if first_text and (same or first_text != old_text):
            text = first_text
        else:
            text = joined.upper() if len(joined) <= 9 and " " not in joined else joined
        cols = [T1_LABEL_TO_COLOUR.get(cell(rows[i].get("Colour")), "white") for i in idx]
        if len(set(cols)) == 1:
            colour, cream = cols[0], ""
        else:
            colour = "red" if "red" in cols else "white"
            cream = " ".join(clean_word(words[i]["text"]) for i, c in zip(idx, cols) if c == "cream")
        if same:
            a, b = float(punch[k]["Start"]), float(punch[k]["End"])
        else:
            a, b = float(words[idx[0]]["start"]), float(words[idx[-1]]["end"])
        pop = punch[k0].get("Pop", True) is not False if k0 is not None and 0 <= k0 < len(punch) else True
        new_punch.append({"Start": round(a, 2), "End": round(b, 2), "Text": text, "Colour": colour, "Cream word": cream,
                          "Pop": pop})
    new_bw = []
    for _, idx in group_runs(len(rows), lambda i: "b" if bool(cell(rows[i].get("B&W"), False)) else None,
                             bw_owner, lambda i: ""):
        olds = {bw_owner[i] for i in idx}
        k = next(iter(olds)) if len(olds) == 1 else None
        if k is not None and 0 <= k < len(bw) and owned_by(bw_owner, k) == idx:
            new_bw.append({"Start": float(bw[k]["Start"]), "End": float(bw[k]["End"])})
        else:
            new_bw.append({"Start": round(float(words[idx[0]]["start"]), 2), "End": round(float(words[idx[-1]]["end"]), 2)})
    return sorted(new_punch, key=lambda r: r["Start"]), sorted(new_bw, key=lambda r: r["Start"])


T1_LINE_COLS = ["Time", "Line", "Punch words", "Colour", "Pop SFX", "B&W"]
T1_LINE_EDIT = ["Punch words", "Colour", "B&W"]


def t1_line_rows(words, groups, rows, owner, punch):
    out = []
    for idx in groups:
        evs = []
        for i in idx:
            if owner[i] is not None and owner[i] not in evs:
                evs.append(owner[i])
        first = punch[evs[0]] if evs else None
        out.append({"Time": round(words[idx[0]]["start"], 2), "Line": " ".join(words[i]["text"] for i in idx),
                    "Punch words": " / ".join(str(punch[k].get("Text") or "") for k in evs),
                    "Colour": T1_COLOUR_LABELS.get(first.get("Colour"), "Main") if first else None,
                    "B&W": any(rows[i]["B&W"] for i in idx)})
    return out


def t1_apply_line_rows(words, groups, rows, owner, bw_owner, punch, bw, old_lines, new_lines):
    new_rows, new_owner, new_bw_owner = [dict(r) for r in rows], list(owner), list(bw_owner)
    marker, notes = -1, []
    for g, (o, n) in enumerate(zip(old_lines, new_lines)):
        idx = groups[g]
        if bool(cell(o["B&W"], False)) != bool(cell(n["B&W"], False)):
            for i in idx:
                new_rows[i]["B&W"] = bool(cell(n["B&W"], False))
                new_bw_owner[i] = marker
            marker -= 1
        if not line_changed(o, n, ["Punch words", "Colour"]):
            continue
        for i in idx:
            new_rows[i].update({"Show as": "Subtitle", "Colour": None, "Text on screen": ""})
            new_owner[i] = None
        text = str(cell(n["Punch words"])).strip()
        if not text and not cell(n["Colour"]):
            continue  # punch words cleared: the line is plain subtitles
        colour = cell(n["Colour"]) or "Main"
        if not text:  # a colour picked on a line without punch words: punch its strongest word
            w = best_word(words, idx)
            text = w.upper() if len(w) <= 9 else w
        for ph in split_phrases(text):
            sel = phrase_words(words, idx, ph)
            if not sel:
                sel = idx
                notes.append(f"“{ph}” isn't in the line, so it shows over the whole line.")
            for j, i in enumerate(sel):
                new_rows[i].update({"Show as": "Punch word", "Colour": colour, "Text on screen": ph if j == 0 else ""})
                new_owner[i] = marker
            marker -= 1
    p2, b2 = t1_from_rows(words, new_rows, new_owner, new_bw_owner, punch, bw)
    return p2, b2, notes


def _t1_grid(pid, rev, words, rows, owner, bw_owner, punch, bw, pal, store):
    st.markdown(swatch_html([pal["punch"], pal["red"], pal["cream"]], ["Main", "Colour 2", "Colour 3"]),
                unsafe_allow_html=True)
    groups = line_groups(words)
    lines = t1_line_rows(words, groups, rows, owner, punch)
    for ln, v in zip(lines, pop_values(words, groups, punch, lambda it: True)):
        ln[POP_COL] = v
    df = pd.DataFrame(lines, columns=T1_LINE_COLS)
    edited = grid_editor(
        df, f"t1_grid_{pid}_{rev}",
        {"Time": st.column_config.NumberColumn(format="%.2f", width="small"),
         "Line": st.column_config.TextColumn(width="medium"),
         "Punch words": st.column_config.TextColumn(
             width="medium", help="The word(s) of this line shown big at eye level, as they should appear "
                                  "(e.g. 'IRFAN JUNEJO'). Several: separate with /. Empty = only subtitles."),
         "Colour": st.column_config.SelectboxColumn(options=list(T1_LABEL_TO_COLOUR),
                                                    help="Punch word colour (see the swatches)."),
         POP_COL: st.column_config.CheckboxColumn(help=POP_COL_HELP),
         "B&W": st.column_config.CheckboxColumn(help="Black & white for this line.")},
        disabled=["Time", "Line"])
    if grid_changed(edited, df, T1_LINE_EDIT + [POP_COL]):
        recs = edited.to_dict("records")
        p2, b2, notes = punch, bw, []
        if grid_changed(edited, df, T1_LINE_EDIT):
            p2, b2, notes = t1_apply_line_rows(words, groups, rows, owner, bw_owner, punch, bw, lines, recs)
        if grid_changed(edited, df, [POP_COL]):
            p2, n2 = apply_pop_col(words, groups, p2, lines, recs, lambda it: True)
            notes += n2
        store(p2, b2, ("⚠ " + " ".join(notes)) if notes else None, grid=edited)


# ------------------------------------------------------------------ the step (UI)
def step(project_dir, project, A):
    pid = project["project_id"]
    st.subheader("Talking1 style")
    st.caption("Big punch words at eye level, small lowercase subtitles, warm grade with black & white moments, "
               "a push-in zoom on every shot and an end card.")

    def save(**ch):
        save_settings(project_dir, project, A, **ch)

    s = settings(project)
    if "look" not in s and s.get("palette"):
        save(look=next((n for n, lk in T1_LOOKS.items() if lk["palette"] == s["palette"]
                        and abs(lk["push_to"] - float(s["push_to"])) < 1e-6), "Custom"))
        s = settings(project)
    # ---- 1. look
    st.markdown("#### 1 · Look")
    look_picker(pid, "t1", T1_LOOKS, s, save,
                help_text="Colours for punch words, subtitles and the end card, plus how strong the zoom is. "
                          "Fine-tune them in More settings.")
    s = settings(project)
    pal = s.get("palette") or T1_PALETTES[DEFAULT_T1_PALETTE]
    st.markdown(swatch_html([pal["punch"], pal["red"], pal["cream"], pal["subs"], pal["endBg"]],
                            ["Main", "Colour 2", "Colour 3", "Subtitles", "End card"])
                + f"<div style='font-size:13px;opacity:.8;margin:-4px 0 6px'>Zoom: up to {float(s['push_to']):.2f}×, "
                  f"{next((k for k, v in ZOOM_SPEEDS.items() if abs(v - float(s['push_rate'])) < 1e-6), 'custom').split(' (')[0].lower()} "
                  f"speed (change in More settings)</div>",
                unsafe_allow_html=True)
    fmt_labels = list(FORMATS)
    cur_fmt = next((k for k, v in FORMATS.items() if v == s["format"]), fmt_labels[0])
    fmt = FORMATS[st.radio("Output", fmt_labels, index=fmt_labels.index(cur_fmt), horizontal=True, key=f"t1_fmt_{pid}")]
    if fmt != s["format"]:
        save(format=fmt)
        s = settings(project)

    problem = remotion_problem()
    if problem:
        st.error(problem)
        return
    keep = A["ensure_keep"](project_dir, project)
    if not keep:
        st.info(A["NO_PROJECT_MSG"])
        return
    if not project.get("transcript"):
        st.info("First transcribe the video in step 4 (A. Transcribe) and correct the words there. "
                "The words here come from that transcript.")
        return
    if project.get("transcript_keep_version") != project.get("keep_version"):
        st.warning("You changed the cut after transcribing. Transcribe again in step 4 before using this style.")
        return
    _, total = A["_offsets"](keep)
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
        save(candidates=cands, candidates_key=project.get("keep_version"),
             cuts=merge_short(silence_cuts + [c["t"] for c in cands if c["likely"]], 0.3))
        s = settings(project)

    # ---- 2. words
    st.markdown("#### 2 · Words")
    use_roman = s["romanize"]
    words = spoken_words(project, A, use_roman)
    if "punch" not in s:  # starts as plain subtitles: you pick the punch words (or ask for a suggestion)
        save(punch=[], bw=[])
        s = settings(project)
    punch, bw = s.get("punch") or [], s.get("bw") or []
    rows, owner, bw_owner = t1_plan_rows(words, punch, bw)
    rev_key = f"t1_grid_rev_{pid}"
    rev = st.session_state.get(rev_key, 0)

    def store(new_punch, new_bw, note=None, grid=None):
        changed = (new_punch, new_bw) != (punch, bw)
        if changed:
            save(punch=new_punch, bw=new_bw)
        if grid is None or not grid_keep(f"t1_applied_{pid}", grid, changed):
            st.session_state[rev_key] = rev + 1
        if note:
            st.session_state[f"t1_note_{pid}"] = note
        st.rerun()

    c1, c2, c3 = st.columns([4, 1, 1])
    c1.caption("Every spoken word shows in the small subtitles unless it's a punch word (big, at eye level). "
               "In the grid, type a line's **Punch words**, or **drag across the timeline** to select words. "
               "Changes are saved straight away and show in the live preview.")
    if c2.button("✨ Suggest for me", key=f"t1_sugg_{pid}", use_container_width=True,
                 help="Replace the punch words and B&W moments with a suggestion."):
        p0 = suggest_punch(project, A, use_roman)
        store(p0, suggest_bw(project, A, p0), "New suggestion made.")
    if c3.button("⌫ All subtitles", key=f"t1_clear_{pid}", use_container_width=True,
                 help="Remove every punch word and B&W moment: all lines back to plain subtitles."):
        store([], [], "All lines are plain subtitles now.")
    note = st.session_state.pop(f"t1_note_{pid}", None)
    if note:
        st.toast(note.lstrip("⚠ "), icon="⚠️" if note.startswith("⚠") else "✅")
    view = timeline_view(pid, "t1", total)
    bars = ([{"start": float(p["Start"]), "end": float(p["End"]), "row": "Punch", "kind": "Punch word",
              "label": str(p.get("Text") or "")} for p in punch]
            + [{"start": float(b["Start"]), "end": float(b["End"]), "row": "B&W", "kind": "B&W", "label": "B&W"}
               for b in bw])
    sel = plan_timeline(words, [r["Show as"] for r in rows], bars, T1_SHOW + ["B&W"], T1_KIND_COLOURS,
                        ["Punch", "B&W"], view, key=f"t1_tl_{pid}_{rev}", height=160)
    idx = selected_words(words, sel) if sel else []
    if idx:
        act = st.container()
        said = " ".join(words[i]["text"] for i in idx)
        act.info(f"Selected **{words[idx[0]]['start']:.2f} – {words[idx[-1]]['end']:.2f} s**: “{said}”")
        act.caption("Make these words:")
        bc = act.columns(4)
        for n, (label, change) in enumerate((("Punch word", {"Show as": "Punch word"}),
                                             ("Subtitle", {"Show as": "Subtitle"}),
                                             ("B&W on", {"B&W": True}), ("B&W off", {"B&W": False}))):
            if bc[n].button(label, key=f"t1_make_{n}_{pid}_{rev}", use_container_width=True):
                new_rows = [dict(r) for r in rows]
                new_owner, new_bw_owner = list(owner), list(bw_owner)
                for i in idx:
                    new_rows[i].update(change)
                    if "Show as" in change:
                        new_rows[i]["Text on screen"] = ""
                        new_rows[i]["Colour"] = new_rows[i]["Colour"] or "Main"
                        new_owner[i] = -1
                    else:
                        new_bw_owner[i] = -1
                p2, b2 = t1_from_rows(words, new_rows, new_owner, new_bw_owner, punch, bw)
                store(p2, b2, f"“{said}”: {label}.")
        act.caption("Double-click the timeline to clear the selection.")

    grid_col, live_col = st.columns([5, 3])
    with live_col:
        sticky_column(f"t1_livebox_{pid}")
        live_slot = st.container(key=f"t1_livebox_{pid}")
    with grid_col:
        _t1_grid(pid, rev, words, rows, owner, bw_owner, punch, bw, pal, store)

    # ---- more settings
    pop_controls(pid, "t1", s, save, "punch words")
    s = settings(project)
    with st.expander("⚙ More settings: exact colours, zoom, text heights, end card, jump cuts"):
        st.markdown("**Exact colours**")
        custom_colours(pid, "t1", T1_FIELDS, pal, save)
        a, b = st.columns(2)
        push = a.slider("Zoom at the end of each shot", 1.0, 2.0, float(s["push_to"]), 0.05,
                        key=f"t1_push_{pid}_{st.session_state.get(f't1_rev_{pid}', 0)}",
                        help="1.55 matches your Resolve zooms. 1.0 = no zoom.")
        speeds = list(ZOOM_SPEEDS)
        cur_speed = next((k for k, v in ZOOM_SPEEDS.items() if abs(v - float(s["push_rate"])) < 1e-6), "Gentle")
        speed = a.radio("Zoom speed", speeds, index=speeds.index(cur_speed), horizontal=True, key=f"t1_speed_{pid}",
                        help="How fast the zoom moves. Slower = short shots only zoom in a little; long shots "
                             "still reach the full zoom. 'Fast' zooms all the way in every shot, however short.")
        if ZOOM_SPEEDS[speed] != float(s["push_rate"]):
            save(push_rate=ZOOM_SPEEDS[speed])
            s = settings(project)
        cta = a.text_input("End card text", s["end_cta"], key=f"t1_cta_{pid}")
        eye = b.slider("Punch word height (put it on the eyes)", 0.15, 0.75, float(s["eye_line"]), 0.01,
                       key=f"t1_eye_{pid}", help="0.33 suits a square video, about 0.5 a 9:16 video.")
        sub = b.slider("Subtitle height", 0.6, 0.95, float(s["sub_line"]), 0.01, key=f"t1_sub_{pid}")
        new = dict(push_to=push, end_cta=cta, eye_line=eye, sub_line=sub)
        if any(s[k] != v for k, v in new.items()):
            if push != s["push_to"]:
                new["look"] = "Custom"
            save(**new)
            s = settings(project)
        rom = st.checkbox("Write Devanagari words in Roman letters", value=s["romanize"], key=f"t1_rom_{pid}",
                          help="Rough spelling. Correct the words in the step 4 transcript editor.")
        if rom != s["romanize"]:
            save(romanize=rom)
            st.rerun()
        st.markdown("**Jump cuts** (the zoom restarts at each one)")
        cands = s.get("candidates") or []
        chosen = set(round(c, 3) for c in s.get("cuts") or [])
        st.caption((f"{len(silence_cuts)} cuts come from removing pauses / repeats and are always used "
                    f"(shots shorter than {s['min_shot']} s are joined). " if silence_cuts else "")
                   + ("Tick the real cuts found in the video, untick hand movements." if cands
                      else ("" if silence_cuts else "No cuts found: the zoom pushes in slowly across the whole video.")))
        if cands:
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
                save(cuts=sorted(new_chosen))
                s = settings(project)

    with live_slot:
        try:
            live_preview(project_dir, project, A, "t1", COMPOSITION, build_props(project_dir, project, A, src),
                         s["format"], total, (round(sel[0], 2), round(sel[1], 2)) if idx else None)
        except Exception as exc:
            st.warning(f"Live preview is not available: {exc}")

    # ---- 3. full preview
    st.markdown("#### 3 · Full preview")
    try:
        cur_key = props_key(build_props(project_dir, project, A, src))
    except Exception:
        cur_key = None
    max_t = max(0.1, round(total - 0.1, 1))
    first = float(words[idx[0]]["start"]) if idx else float((punch or [{"Start": 1.0}])[0]["Start"])
    mode = st.radio("Preview", ["A few seconds", "Whole video (takes longer)", "One frame"], horizontal=True,
                    key=f"t1_pmode_{pid}", label_visibility="collapsed")
    p1, p2 = st.columns([2, 1])
    if mode == "One frame":
        t_prev = p1.slider("Frame at (seconds)", 0.0, max_t, float(min(round(first + 0.1, 1), max_t)), 0.1,
                           key=f"t1_tprev_{pid}")
        if p2.button("Show this frame", key=f"t1_still_{pid}", use_container_width=True):
            try:
                with st.spinner("Rendering one frame (about 15 seconds)..."):
                    st.session_state[f"t1_stillimg_{pid}"] = str(render_still(project_dir, project, A, t_prev))
            except Exception as exc:
                st.error(str(exc))
        still = st.session_state.get(f"t1_stillimg_{pid}")
        if still and Path(still).exists():
            st.image(still, width=360, caption="Exact frame from the final render")
    else:
        whole = mode.startswith("Whole")
        if whole:
            start, length = 0.0, None
            p1.caption(f"The whole {total:.0f} s video plus the end card, at half resolution.")
        else:
            start = p1.slider("Start at (seconds)", 0.0, max_t, float(max(0.0, min(first - 0.5, max_t))), 0.5,
                              key=f"t1_pstart_{pid}")
            length = p1.slider("Length (seconds)", 2, 15, 6, 1, key=f"t1_plen_{pid}")
            p1.caption("About 20 to 40 seconds to make.")
        if p2.button("Make preview", type="primary", key=f"t1_pbtn_{pid}", use_container_width=True):
            try:
                with st.spinner("Making the preview with Remotion..."):
                    out, key = render_preview(project_dir, project, A, start, length, total)
                st.session_state[f"t1_prev_{pid}"] = (str(out), key, out.stat().st_mtime)
            except Exception as exc:
                st.error(str(exc))
        prev = st.session_state.get(f"t1_prev_{pid}")
        if prev and Path(prev[0]).exists():
            v, info = st.columns([1, 1])
            v.video(Path(prev[0]).read_bytes())
            if cur_key and prev[1] != cur_key:
                info.warning("You changed something after this preview. Click Make preview again to see it.")
            else:
                info.success("Preview is up to date with your current settings.")

    # ---- 4. render
    st.markdown("#### 4 · Render or carry forward")
    st.caption("Render the video now (about 2-4 minutes for 35 s), or carry it forward to the Visual hook page "
               "and render it there once, with the hook.")
    rc, cc = st.columns(2)
    carry_button(project_dir, project, A, "talking1", cc)
    if rc.button("Render Talking1 video", type="primary", key=f"t1_render_{pid}", use_container_width=True):
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
