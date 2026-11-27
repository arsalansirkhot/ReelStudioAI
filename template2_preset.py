"""Template 2 preset for Reel Studio AI (the "Poorvay" style).

Clean editorial look: small white word-by-word captions beside the head, bigger accent words in
white / dark green / blue, huge top-left quotes revealed word by word, quick jump-cuts with
instant punch-ins (harder ones on emphasis words), a royal-blue text screen, a cream card with the
video shrunk in a frame, and a "video card" (shrunk video with a title on cream).

The look itself is the Remotion template made with Claude in D:\\ClaudeAIVideoEdit\\remotion
(composition "Template2"). This file only prepares the data and calls Remotion. It reuses the
helpers of talking1_preset.py (cut video, cut detection, running Remotion).

Kept in its own file: update.ps1 replaces app.py on updates, app.py only has a few lines that show
this as step 6 (see the "Template 2 add-on" comments there).
"""
import json
import re
from pathlib import Path

import pandas as pd
import streamlit as st

import talking1_preset as t1

COMPOSITION = "Template2"
TYPES = ["accent", "quote", "blue screen", "cream card", "video card"]
TYPE_TO_EVENT = {"accent": "accent", "quote": "quote", "blue screen": "blue", "cream card": "cream",
                 "video card": "player"}
COLOURS = ["white", "green", "blue"]
SIDES = ["right", "left"]
FORMATS = {"1:1 · Square": "1:1", "9:16 · Reels": "9:16", "16:9 · Wide": "16:9", "Same as the video": "source"}
DEFAULTS = dict(format="1:1", punch=1.18, beat=1.6, hard_zoom=1.45, face_x=0.47, face_y=0.33, face_w=0.13,
                romanize=True, min_shot=0.5)
EVENT_COLS = ["Start", "End", "Type", "Text", "Colour", "Side", "Hard zoom"]


def settings(project):
    return {**DEFAULTS, **(project.get("template2") or {})}


def save_settings(project_dir, project, A, **changes):
    project["template2"] = {**(project.get("template2") or {}), **changes}
    A["save_project"](project_dir, project)


# ------------------------------------------------------------------ words (one entry per spoken word)
def spoken_words(project, A, use_roman):
    """Every word of the (corrected) step 4 transcript with its own time. When the corrected line has
    the same number of words as Whisper heard, Whisper's word times are used; otherwise the line's
    time is shared between its words by their length."""
    out = []
    for seg in A["caption_segments"](project):
        text = t1.roman(seg["text"], A) if use_roman else seg["text"]
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
            out.append({"start": round(s, 3), "end": round(max(e, s + 0.08), 3), "text": p})
    return out


def lines(project, A, use_roman):
    return [{"start": float(s["start"]), "end": float(s["end"]),
             "text": (t1.roman(s["text"], A) if use_roman else s["text"]).strip()}
            for s in A["caption_segments"](project) if str(s.get("text", "")).strip()]


def short_title(text, most=5):
    """A title for the video card: the first clause, at most a few words."""
    first = re.split(r"[,.!?;:]", text.strip())[0].split()
    return " ".join(first[:most]) + ("..." if len(first) > most else "")


# ------------------------------------------------------------------ first proposal of text events
def suggest_events(project, A, use_roman, total):
    """A starting plan the user edits: one blue screen, one cream card, one video card, two quotes
    and accent words spread through the video, never overlapping."""
    segs = lines(project, A, use_roman)
    used = []

    def free(a, b):
        return all(b <= u0 + 0.05 or a >= u1 - 0.05 for u0, u1 in used)

    events = []

    def add(row):
        events.append(row)
        used.append((row["Start"], row["End"]))

    def pick_line(target, min_len=1.2, max_len=4.0):
        best = None
        for s in segs:
            ln = s["end"] - s["start"]
            if not (min_len <= ln <= max_len) or s["start"] < 2.0 or s["end"] > total - 1.0:
                continue
            if not free(s["start"], s["end"]):
                continue
            d = abs((s["start"] + s["end"]) / 2 - target)
            if best is None or d < best[0]:
                best = (d, s)
        return best[1] if best else None

    if total >= 12:
        for frac, kind in ((0.30, "blue screen"), (0.22, "cream card"), (0.72, "video card")):
            s = pick_line(total * frac)
            if s:
                text = short_title(s["text"]) if kind == "video card" else ""
                add({"Start": round(s["start"], 2), "End": round(s["end"], 2), "Type": kind, "Text": text,
                     "Colour": "white", "Side": "right", "Hard zoom": False})
    # quotes: short lines (2-4 words), questions first, at least 8 s apart
    quotes = sorted((s for s in segs if 2 <= len(s["text"].split()) <= 5 and s["end"] - s["start"] <= 2.6),
                    key=lambda s: (not s["text"].rstrip().endswith("?"), s["start"]))
    q_times = []
    for s in quotes:
        if len(q_times) >= 2 or s["start"] < 1.5:
            continue
        if all(abs(s["start"] - q) >= 8 for q in q_times) and free(s["start"], s["end"]):
            add({"Start": round(s["start"], 2), "End": round(s["end"], 2), "Type": "quote",
                 "Text": s["text"].strip(" ,"), "Colour": "white", "Side": "right", "Hard zoom": False})
            q_times.append(s["start"])
    # accents: Talking1's punch-word picker (names, problem words, emphasis words)
    n, last = 0, -9.0
    for p in t1.suggest_punch(project, A, use_roman):
        a, b = float(p["Start"]), max(float(p["End"]), float(p["Start"]) + 0.5)
        word = str(p["Text"])
        parts = word.split()
        # two capitalised words are joined as a name ("Eldar Shafir"), but title-case phrases like
        # "Scarcity Why" are not names: keep only the first word then
        if len(parts) > 1 and any(len(w) <= 3 or w.lower() in t1.STOP for w in parts):
            word = parts[0]
            b = min(b, a + 0.6)
        if not free(a, b) or a - last < 2.0:
            continue
        is_name = " " in word and all(w[:1].isupper() for w in word.split())
        text = word if is_name else word.lower()
        if not is_name and len(text.split()) == 1 and n % 2 == 0:
            text += "."  # the style ends key words with a full stop ("problem.", "focus.")
        colour = "green" if is_name else ("white", "white", "blue", "green")[n % 4]
        add({"Start": round(a, 2), "End": round(b, 2), "Type": "accent", "Text": text, "Colour": colour,
             "Side": SIDES[n % 2] if colour == "white" else "left", "Hard zoom": n % 2 == 0})
        n += 1
        last = a
    return sorted(events, key=lambda r: r["Start"])


# ------------------------------------------------------------------ Remotion
def event_rows_to_props(rows, hard_zoom):
    out = []
    for r in rows:
        kind = TYPE_TO_EVENT.get(str(r.get("Type")))
        try:
            a, b = float(r["Start"]), float(r["End"])
        except (TypeError, ValueError):
            continue
        if not kind or b <= a:
            continue
        text = str(r.get("Text") or "").strip()
        if kind in ("accent", "quote") and not text:
            continue
        ev = {"type": kind, "start": a, "end": b}
        if text and kind in ("accent", "quote", "player"):
            ev["text"] = text
        if kind == "accent":
            ev["color"] = r.get("Colour") if r.get("Colour") in COLOURS else "white"
            ev["side"] = r.get("Side") if r.get("Side") in SIDES else "right"
        if bool(r.get("Hard zoom")) and kind in ("accent", "quote"):
            ev["zoom"] = float(hard_zoom)
        out.append(ev)
    return sorted(out, key=lambda e: e["start"])


def build_props(project_dir, project, A, src):
    s = settings(project)
    props = {"src": src, "words": spoken_words(project, A, s["romanize"]),
             "cuts": t1.merge_short(s.get("cuts") or [], s["min_shot"]),
             "events": event_rows_to_props(s.get("events") or [], s["hard_zoom"]),
             "format": s["format"], "face": {"x": float(s["face_x"]), "y": float(s["face_y"]), "w": float(s["face_w"])},
             "punch": float(s["punch"]), "beat": float(s["beat"])}
    path = project_dir / "template2_props.json"
    path.write_text(json.dumps(props, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def render_still(project_dir, project, A, t):
    src = t1.source_video(project_dir, project, A)
    props = build_props(project_dir, project, A, src)
    out = project_dir / "template2_still.png"
    t1.run_remotion(["still", "src/index.ts", COMPOSITION, str(out), f"--frame={int(t * 30)}", "--scale=0.4",
                     f"--props={props}"], project_dir)
    return out


def render_preview(project_dir, project, A, start, length, total, fps=30):
    src = t1.source_video(project_dir, project, A)
    props = build_props(project_dir, project, A, src)
    out = project_dir / "template2_preview.mp4"
    args = ["render", "src/index.ts", COMPOSITION, str(out), f"--props={props}", "--scale=0.5", "--crf=26"]
    if length is not None:
        last = max(1, int(total * fps) - 1)
        a = min(int(start * fps), last - 1)
        b = min(int((start + length) * fps), last)
        args.append(f"--frames={a}-{b}")
    if out.exists():
        out.unlink()
    t1.run_remotion(args, project_dir)
    return out, t1.props_key(props)


def render_video(project_dir, project, A):
    src = t1.source_video(project_dir, project, A)
    props = build_props(project_dir, project, A, src)
    s = settings(project)
    out = project_dir / f"template2_{s['format'].replace(':', 'x')}.mp4"
    t1.run_remotion(["render", "src/index.ts", COMPOSITION, str(out), f"--props={props}", "--crf=18"], project_dir)
    return out


# ------------------------------------------------------------------ the step (UI)
def step(project_dir, project, A):
    pid = project["project_id"]
    st.subheader("Template 2")
    st.caption("Small white captions beside the head, accent words in white / green / blue, big top-left quotes, "
               "quick jump-cuts with punch-ins, a blue text screen, a cream card and a video card. Rendered with "
               "the same Remotion template Claude uses.")
    problem = t1.remotion_problem()
    if problem:
        st.error(problem)
        return
    keep = A["ensure_keep"](project_dir, project)
    if not keep:
        st.info(A["NO_PROJECT_MSG"])
        return
    if not project.get("transcript"):
        st.info("First transcribe the video in step 4 (A. Transcribe) and correct the words there. "
                "The captions here come from that transcript.")
        return
    if project.get("transcript_keep_version") != project.get("keep_version"):
        st.warning("You changed the cut after transcribing. Transcribe again in step 4 before using this template.")
        return
    s = settings(project)
    _, total = A["_offsets"](keep)

    # ---- 1. captions + text plan
    st.markdown("#### 1 · Review captions and the text plan")
    use_roman = st.checkbox("Write Devanagari words in Roman letters", value=s["romanize"], key=f"t2_rom_{pid}",
                            help="Rough spelling. Correct the words in the step 4 transcript editor.")
    if use_roman != s["romanize"]:
        save_settings(project_dir, project, A, romanize=use_roman)
    segs = lines(project, A, use_roman)
    with st.expander(f"Transcript ({len(segs)} lines, from step 4)", expanded=False):
        st.code("\n".join(f"{c['start']:6.2f}  {c['text']}" for c in segs) or "(none)", language="text")
        st.caption("Every word not covered by a row below shows as a small caption beside the head. "
                   "To fix a word, correct it in step 4 and save.")
    if "events" not in s or st.button("Suggest the text plan again", key=f"t2_sugg_{pid}"):
        save_settings(project_dir, project, A, events=suggest_events(project, A, use_roman, total))
        s = settings(project)
        st.rerun()
    st.caption("accent = bigger word beside the head (Colour, Side) · quote = huge top-left text, words appear "
               "as spoken · blue screen = the spoken words on royal blue · cream card = video shrunk in a frame "
               "with a small green caption · video card = shrunk video with the Text as a title. "
               "'Hard zoom' gives an accent or quote a strong punch-in.")
    edf = pd.DataFrame(s.get("events") or [], columns=EVENT_COLS)
    edited = st.data_editor(
        edf, num_rows="dynamic", use_container_width=True, height=360, key=f"t2_ev_{pid}",
        column_config={"Start": st.column_config.NumberColumn(format="%.2f", min_value=0.0, step=0.05),
                       "End": st.column_config.NumberColumn(format="%.2f", min_value=0.0, step=0.05),
                       "Type": st.column_config.SelectboxColumn(options=TYPES, required=True),
                       "Text": st.column_config.TextColumn(width="medium"),
                       "Colour": st.column_config.SelectboxColumn(options=COLOURS),
                       "Side": st.column_config.SelectboxColumn(options=SIDES),
                       "Hard zoom": st.column_config.CheckboxColumn()})
    if st.button("Save text plan", type="primary", key=f"t2_esave_{pid}"):
        rows = []
        for r in edited.to_dict("records"):
            if pd.isna(r.get("Start")) or pd.isna(r.get("End")) or r.get("Type") not in TYPES:
                continue
            rows.append({"Start": float(r["Start"]), "End": float(r["End"]), "Type": r["Type"],
                         "Text": "" if pd.isna(r.get("Text")) else str(r.get("Text") or ""),
                         "Colour": r.get("Colour") if r.get("Colour") in COLOURS else "white",
                         "Side": r.get("Side") if r.get("Side") in SIDES else "right",
                         "Hard zoom": bool(r.get("Hard zoom")) if not pd.isna(r.get("Hard zoom")) else False})
        save_settings(project_dir, project, A, events=sorted(rows, key=lambda r: r["Start"]))
        st.success(f"Saved {len(rows)} text items.")

    # ---- 2. cuts
    st.markdown("#### 2 · Confirm the cuts (the framing changes at each one)")
    try:
        with st.spinner("Preparing the cut video..."):
            src = t1.source_video(project_dir, project, A)
    except Exception as exc:
        st.error(str(exc))
        return
    silence_cuts = t1.keep_boundaries(project, A)
    if "candidates" not in s or s.get("candidates_key") != project.get("keep_version"):
        with st.spinner("Looking for jump cuts..."):
            cands = t1.detect_cuts(project_dir / src, A)
        save_settings(project_dir, project, A, candidates=cands, candidates_key=project.get("keep_version"),
                      cuts=t1.merge_short(silence_cuts + [c["t"] for c in cands if c["likely"]], 0.3))
        s = settings(project)
    chosen = set(round(c, 3) for c in s.get("cuts") or [])
    st.caption("Extra jump-cuts with punch-ins are added automatically inside long shots (see 'Cut pace' below). "
               + (f"{len(silence_cuts)} cuts come from removing pauses / repeats and are always used." if silence_cuts else ""))
    cands = s.get("candidates") or []
    if cands:
        st.caption("Cuts found in the video. Tick the real cuts, untick hand movements.")
        fps = t1.video_fps(project_dir / src, A)
        new_chosen = set(c for c in chosen if c in silence_cuts)
        cols = st.columns(4)
        for k, c in enumerate(cands):
            with cols[k % 4]:
                png = t1.thumb_pair(project_dir, src, c["t"], fps, A)
                if png:
                    st.image(str(png), use_container_width=True)
                if st.checkbox(f"{c['t']:.2f}s cut", value=round(c["t"], 3) in chosen, key=f"t2_cut_{pid}_{c['t']}"):
                    new_chosen.add(round(c["t"], 3))
        if new_chosen != chosen:
            save_settings(project_dir, project, A, cuts=sorted(new_chosen))
            s = settings(project)

    # ---- 3. look
    st.markdown("#### 3 · Look")
    a, b = st.columns(2)
    fmt_labels = list(FORMATS)
    cur_fmt = next((k for k, v in FORMATS.items() if v == s["format"]), fmt_labels[0])
    fmt = FORMATS[a.radio("Output", fmt_labels, index=fmt_labels.index(cur_fmt), horizontal=True,
                          key=f"t2_fmt_{pid}")]
    punch = a.slider("Punch-in at jump-cuts", 1.0, 1.5, float(s["punch"]), 0.02, key=f"t2_punch_{pid}",
                     help="Shots switch between normal and this zoom. 1.18 matches the reference.")
    hard = a.slider("Hard zoom (rows ticked 'Hard zoom')", 1.1, 2.0, float(s["hard_zoom"]), 0.05, key=f"t2_hard_{pid}")
    beat = a.slider("Cut pace (seconds between jump-cuts)", 0.8, 4.0, float(s["beat"]), 0.1, key=f"t2_beat_{pid}",
                    help="Smaller = more cuts. 1.6 is close to the reference.")
    b.caption("Where is the head? Captions sit beside it and zooms centre on it.")
    fx = b.slider("Head centre, left ↔ right", 0.1, 0.9, float(s["face_x"]), 0.01, key=f"t2_fx_{pid}")
    fy = b.slider("Eye height, top ↔ bottom", 0.1, 0.8, float(s["face_y"]), 0.01, key=f"t2_fy_{pid}")
    fw = b.slider("Head half-width", 0.05, 0.3, float(s["face_w"]), 0.01, key=f"t2_fw_{pid}",
                  help="Bigger = captions sit further from the head.")
    new = dict(format=fmt, punch=punch, hard_zoom=hard, beat=beat, face_x=fx, face_y=fy, face_w=fw)
    if any(s[k] != v for k, v in new.items()):
        save_settings(project_dir, project, A, **new)
        s = settings(project)

    # ---- 4. preview
    st.markdown("#### 4 · Preview before rendering")
    st.caption("Same Remotion template as the final video, at lower resolution. Uses the saved text plan, "
               "so click Save above first. Tip: check the head position with 'One frame' first.")
    try:
        cur_key = t1.props_key(build_props(project_dir, project, A, src))
    except Exception:
        cur_key = None
    max_t = max(0.1, round(total - 0.1, 1))
    mode = st.radio("Preview", ["One frame", "A few seconds", "Whole video (takes longer)"], horizontal=True,
                    key=f"t2_pmode_{pid}")
    p1, p2 = st.columns([2, 1])
    if mode == "One frame":
        t_prev = p1.slider("Frame at (seconds)", 0.0, max_t, float(min(1.0, max_t)), 0.1, key=f"t2_tprev_{pid}")
        if p2.button("Show this frame", key=f"t2_still_{pid}"):
            try:
                with st.spinner("Rendering one frame (about 15 seconds)..."):
                    st.session_state[f"t2_stillimg_{pid}"] = str(render_still(project_dir, project, A, t_prev))
            except Exception as exc:
                st.error(str(exc))
        still = st.session_state.get(f"t2_stillimg_{pid}")
        if still and Path(still).exists():
            st.image(still, width=360, caption="Exact frame from the final render")
    else:
        whole = mode.startswith("Whole")
        if whole:
            start, length = 0.0, None
            p1.caption(f"The whole {total:.0f} s video at half resolution.")
        else:
            start = p1.slider("Start at (seconds)", 0.0, max_t, 0.0, 0.5, key=f"t2_pstart_{pid}")
            length = p1.slider("Length (seconds)", 2, 15, 6, 1, key=f"t2_plen_{pid}")
            p1.caption("About 20 to 40 seconds to make.")
        if p2.button("Make preview", type="primary", key=f"t2_pbtn_{pid}"):
            try:
                with st.spinner("Making the preview with Remotion..."):
                    out, key = render_preview(project_dir, project, A, start, length, total)
                st.session_state[f"t2_prev_{pid}"] = (str(out), key, out.stat().st_mtime)
            except Exception as exc:
                st.error(str(exc))
        prev = st.session_state.get(f"t2_prev_{pid}")
        if prev and Path(prev[0]).exists():
            v, info = st.columns([1, 1])
            v.video(Path(prev[0]).read_bytes())
            if cur_key and prev[1] != cur_key:
                info.warning("You changed something after this preview. Click Make preview again to see it.")
            else:
                info.success("Preview is up to date with your current settings.")

    # ---- 5. render
    st.markdown("#### 5 · Render")
    st.caption("Uses the saved text plan. A 35 s video takes about 2-4 minutes.")
    if st.button("Render Template 2 video", type="primary", key=f"t2_render_{pid}"):
        try:
            with st.spinner("Rendering with Remotion..."):
                out = render_video(project_dir, project, A)
            st.session_state[f"t2_out_{pid}"] = str(out)
        except Exception as exc:
            st.error(str(exc))
    res = st.session_state.get(f"t2_out_{pid}")
    if res and Path(res).exists():
        v, d = st.columns([1, 1])
        v.video(res)
        d.success(f"Rendered: {Path(res).name}")
        d.caption(f"Saved in: {Path(res).parent}")
        d.download_button("Download video", Path(res).read_bytes(), file_name=Path(res).name, key=f"t2_dl_{pid}")
