"""Template 3 preset for Reel Studio AI: a vlog made from a voice-over video plus B-roll clips.

The voice-over video goes through the usual steps (clean audio, remove pauses, remove repeats, transcribe
in step 4). Here it gets Template 2's text animations, and muted B-roll clips can be placed over any
caption line: on a coloured card with the spoken line under it, or full screen with the line at the
bottom. A clip shorter than its lines freezes on its last frame.

Rendered with the Remotion composition "Template3" (the Template 2 / Poorvay component with `brolls` and
`lines`), in D:\\ClaudeAIVideoEdit\\remotion. Reuses template2_preset (text plan) and talking1_preset
(cut video, cut detection, live preview, Remotion runner).
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import talking1_preset as t1
import template2_preset as t2

COMPOSITION = "Template3"
KEY = "template3"
DEFAULTS = dict(t2.DEFAULTS, card_colour="Cream", click_on=True, click_volume=0.5)
BROLL_DIR = "broll"
CLIP_TYPES = ["mp4", "mov", "m4v", "webm", "mkv"]
# card colour: background + the colour of the spoken line under the clip
CARD_COLOURS = {
    "From the look": None,
    "Cream": ("#FDFDE4", "#1F5A2B"),
    "Sage": ("#DDE8D5", "#2F4A2B"),
    "Sky": ("#DCEBF7", "#0C447C"),
    "Blush": ("#F7E1E4", "#72243E"),
    "Sand": ("#EFE3CF", "#5A3E1B"),
    "Charcoal": ("#2B2B2B", "#F4F1E8"),
    "White": ("#FFFFFF", "#151515"),
}
MODES = ["Card", "Full"]
BROLL_KIND = "B-roll"
T3_COLS = ["Time", "Line", "Show as", "Pop words", "B-roll", "As", "Start at", "Colour", "Side", "Hard zoom",
           "Pop SFX"]
BROLL_EDIT = ["B-roll", "As", "Start at"]


def settings(project):
    return {**DEFAULTS, **(project.get(KEY) or {})}


def save_settings(project_dir, project, A, **changes):
    project[KEY] = {**(project.get(KEY) or {}), **changes}
    A["save_project"](project_dir, project)


def palette(s):
    pal = dict(s.get("palette") or t2.T2_PALETTES[t2.DEFAULT_PALETTE])
    cc = CARD_COLOURS.get(s.get("card_colour") or "Cream")
    if cc:
        pal["cardBg"], pal["cardText"] = cc
    return pal


# ------------------------------------------------------------------ B-roll clips
def add_clip(project_dir, project, A, upload):
    """Save an uploaded clip as a muted, browser-friendly MP4 (H.264, max 1080 wide) with a thumbnail."""
    s = settings(project)
    clips = list(s.get("clips") or [])
    n = max([int(c["id"][1:]) for c in clips if c["id"][1:].isdigit()], default=0) + 1
    cid = f"B{n}"
    folder = project_dir / BROLL_DIR
    folder.mkdir(exist_ok=True)
    raw = folder / f"{cid}_upload{Path(upload.name).suffix.lower() or '.mp4'}"
    raw.write_bytes(upload.getbuffer())
    out, thumb = folder / f"{cid}.mp4", folder / f"{cid}.jpg"
    try:
        A["run_ffmpeg"](["-i", str(raw), "-an", "-vf", "scale='min(1080,iw)':-2,fps=30,format=yuv420p",
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-movflags", "+faststart", str(out)])
    finally:
        raw.unlink(missing_ok=True)
    info = A["probe_video"](out)
    dur = float(info.get("duration_seconds") or 0.0)
    try:
        A["run_ffmpeg"](["-ss", f"{min(1.0, dur / 2):.2f}", "-i", str(out), "-frames:v", "1", "-vf", "scale=320:-2",
                         str(thumb)])
    except Exception:
        pass
    w, h = info.get("width") or 16, info.get("height") or 9
    clips.append({"id": cid, "name": upload.name, "file": f"{BROLL_DIR}/{cid}.mp4", "thumb": f"{BROLL_DIR}/{cid}.jpg",
                  "dur": round(dur, 3), "aspect": round(w / h, 4)})
    save_settings(project_dir, project, A, clips=clips)
    return cid


def remove_clip(project_dir, project, A, cid):
    s = settings(project)
    clips = [c for c in s.get("clips") or [] if c["id"] != cid]
    brolls = [b for b in s.get("brolls") or [] if b["clip"] != cid]
    for ext in ("mp4", "jpg"):
        (project_dir / BROLL_DIR / f"{cid}.{ext}").unlink(missing_ok=True)
    save_settings(project_dir, project, A, clips=clips, brolls=brolls)


def clip_library(project_dir, project, A, pid):
    s = settings(project)
    clips = s.get("clips") or []
    rev_key = f"t3_up_rev_{pid}"
    rev = st.session_state.get(rev_key, 0)
    ups = st.file_uploader("Add B-roll clips (their sound is removed)", type=CLIP_TYPES, accept_multiple_files=True,
                           key=f"t3_up_{pid}_{rev}")
    if ups:
        done = []
        for i, u in enumerate(ups):
            with st.spinner(f"Preparing {u.name} ({i + 1} of {len(ups)})..."):
                try:
                    done.append(add_clip(project_dir, project, A, u))
                except Exception as exc:
                    st.error(f"{u.name}: {exc}")
        st.session_state[rev_key] = rev + 1
        if done:
            st.session_state[f"t3_note_{pid}"] = f"Added {', '.join(done)}. Pick them in the B-roll column below."
        st.rerun()
    if not clips:
        st.caption("No clips yet. Add the clips you want to show over the voice-over, then pick them per line below.")
        return
    cols = st.columns(6)
    for k, c in enumerate(clips):
        with cols[k % 6]:
            th = project_dir / c["thumb"]
            if th.exists():
                st.image(str(th), use_container_width=True)
            st.caption(f"**{c['id']}** · {c['dur']:.1f} s · {c['name'][:18]}")
            if st.button("Remove", key=f"t3_rm_{c['id']}_{pid}", use_container_width=True):
                remove_clip(project_dir, project, A, c["id"])
                st.rerun()


# ------------------------------------------------------------------ lines and B-roll
def line_info(words, groups):
    return [{"start": float(words[g[0]]["start"]), "end": float(words[g[-1]]["end"]),
             "text": " ".join(words[i]["text"] for i in g)} for g in groups]


def broll_rows(lines, brolls):
    """B-roll / As / Start at for every line, from the saved B-roll items."""
    out = []
    for ln in lines:
        mid = (ln["start"] + ln["end"]) / 2
        b = next((b for b in brolls if b["start"] - 0.01 <= mid < b["end"]), None)
        if not b:
            out.append({"B-roll": None, "As": None, "Start at": None})
            continue
        first = next(x for x in lines if b["start"] - 0.01 <= (x["start"] + x["end"]) / 2 < b["end"])
        out.append({"B-roll": b["clip"], "As": "Full" if b["mode"] == "full" else "Card",
                    "Start at": float(b.get("from") or 0.0) if first is ln and b.get("from") else None})
    return out


def brolls_from_rows(lines, recs, clip_ids, total):
    """Neighbouring lines with the same clip and look become one item, so the clip keeps playing.
    A 'Start at' on a later line starts a new item there."""
    cell = t1.cell
    items = []
    for g, r in enumerate(recs):
        clip = cell(r.get("B-roll"))
        if not clip or clip not in clip_ids:
            continue
        mode = "full" if cell(r.get("As")) == "Full" else "card"
        start_at = cell(r.get("Start at"), None)
        start_at = None if start_at in ("", None) else max(0.0, float(start_at))
        if items and items[-1]["clip"] == clip and items[-1]["mode"] == mode and items[-1]["_g"] == g - 1 \
                and start_at is None:
            items[-1]["_g"] = g
            continue
        items.append({"clip": clip, "mode": mode, "from": start_at or 0.0,
                      "start": 0.0 if g == 0 else round(lines[g]["start"], 3), "_g": g})
    for it in items:
        nxt = it.pop("_g") + 1
        it["end"] = round(lines[nxt]["start"], 3) if nxt < len(lines) else round(total, 3)
    return items


# ------------------------------------------------------------------ Remotion
def build_props(project_dir, project, A, src):
    s = settings(project)
    words = t2.spoken_words(project, A, s["romanize"])
    groups = t1.line_groups(words)
    clips = {c["id"]: c for c in s.get("clips") or []}
    brolls = []
    for b in s.get("brolls") or []:
        c = clips.get(b["clip"])
        if c and (project_dir / c["file"]).exists() and b["end"] > b["start"]:
            brolls.append({"start": float(b["start"]), "end": float(b["end"]), "src": c["file"],
                           "from": float(b.get("from") or 0.0), "dur": float(c["dur"]), "aspect": float(c["aspect"]),
                           "mode": b["mode"]})
    props = {"src": src, "words": [{k: w[k] for k in ("start", "end", "text")} for w in words],
             "cuts": t1.merge_short(s.get("cuts") or [], s["min_shot"]),
             "events": t2.event_rows_to_props(s.get("events") or [], s["hard_zoom"]),
             "format": s["format"], "face": {"x": float(s["face_x"]), "y": float(s["face_y"]), "w": float(s["face_w"])},
             "punch": float(s["punch"]), "beat": float(s["beat"]), "palette": palette(s),
             "lines": line_info(words, groups), "brolls": sorted(brolls, key=lambda b: b["start"])}
    if t2.FRAMES.get(s.get("frame_name") or "As designed"):
        props["frame"] = t2.FRAMES[s["frame_name"]]
    pop = t1.pop_props(project_dir, s)
    if pop:
        props["pop"] = pop
    click = t1.click_props(project_dir, s) if props["brolls"] else None
    if click:
        props["click"] = click
    path = project_dir / "template3_props.json"
    path.write_text(json.dumps(props, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def render_still(project_dir, project, A, t):
    src = t1.source_video(project_dir, project, A)
    props = build_props(project_dir, project, A, src)
    out = project_dir / "template3_still.png"
    t1.run_remotion(["still", "src/index.ts", COMPOSITION, str(out), f"--frame={int(t * 30)}", "--scale=0.4",
                     f"--props={props}"], project_dir)
    return out


def render_preview(project_dir, project, A, start, length, total, fps=30):
    src = t1.source_video(project_dir, project, A)
    props = build_props(project_dir, project, A, src)
    out = project_dir / "template3_preview.mp4"
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
    out = project_dir / f"template3_{s['format'].replace(':', 'x')}.mp4"
    t1.run_remotion(["render", "src/index.ts", COMPOSITION, str(out), f"--props={props}", "--crf=18"], project_dir)
    return out


# ------------------------------------------------------------------ the grid
def _grid(pid, rev, words, groups, lines, rows, owner, events, brolls, clip_ids, total, pal, store):
    st.markdown(t1.swatch_html([pal["text"], pal["accent2"], pal["accent3"], pal["cardBg"]],
                               ["Main", "Colour 2", "Colour 3", "Card"]), unsafe_allow_html=True)
    text_rows = t2.with_pop_col(words, groups, t2.line_rows(words, groups, rows, owner, events), events)
    b_rows = broll_rows(lines, brolls)
    table = [{**a, **b} for a, b in zip(text_rows, b_rows)]
    df = pd.DataFrame(table, columns=T3_COLS)
    edited = t1.grid_editor(
        df, f"t3_grid_{pid}_{rev}",
        {"Time": st.column_config.NumberColumn(format="%.2f", width="small"),
         "Line": st.column_config.TextColumn(width="medium"),
         "Show as": st.column_config.SelectboxColumn(options=t2.SHOW, required=True),
         "Pop words": st.column_config.TextColumn(
             help="Accent: the word(s) of this line that pop out (several: separate with /). Quote / Video card: "
                  "the text shown. Typing here on a caption line makes it an accent."),
         "B-roll": st.column_config.SelectboxColumn(
             options=clip_ids, help="The clip shown over this line. The same clip on the next lines keeps playing."),
         "As": st.column_config.SelectboxColumn(
             options=MODES, help="Card: the clip on the card colour with the spoken line under it. "
                                 "Full: the clip fills the screen, the line at the bottom."),
         "Start at": st.column_config.NumberColumn(
             format="%.1f", min_value=0.0, step=0.5,
             help="Seconds into the clip to start from (empty = its start). On a later line it restarts the clip there."),
         "Colour": st.column_config.SelectboxColumn(options=list(t2.LABEL_TO_COLOUR), help="Accent colour."),
         "Side": st.column_config.SelectboxColumn(options=t2.SIDES, help="Accent: which side of the head."),
         "Hard zoom": st.column_config.CheckboxColumn(help="Accent or quote: a strong punch-in."),
         t1.POP_COL: st.column_config.CheckboxColumn(help=t1.POP_COL_HELP)},
        disabled=["Time", "Line"])
    if not t1.grid_changed(edited, df, t2.LINE_EDIT + BROLL_EDIT + [t1.POP_COL]):
        return
    recs = edited.to_dict("records")
    new_events, notes = events, []
    if t1.grid_changed(edited, df, t2.LINE_EDIT):
        new_events, notes = t2.apply_line_rows(words, groups, rows, owner, events, text_rows, recs)
    if t1.grid_changed(edited, df, [t1.POP_COL]):
        new_events, n2 = t1.apply_pop_col(words, groups, new_events, text_rows, recs, t2.is_accent)
        notes += n2
    new_brolls = brolls
    if t1.grid_changed(edited, df, BROLL_EDIT):
        for r, old in zip(recs, table):  # a clip picked without 'As': a card
            if t1.cell(r.get("B-roll")) and not t1.cell(r.get("As")):
                r["As"] = "Card"
        new_brolls = brolls_from_rows(lines, recs, clip_ids, total)
    store(new_events, new_brolls, ("⚠ " + " ".join(notes)) if notes else None, grid=edited)


# ------------------------------------------------------------------ the step (UI)
def step(project_dir, project, A):
    pid = project["project_id"]
    st.subheader("Template 3 · Vlog")
    st.caption("Your voice-over video with Template 2's text animations, plus B-roll clips over any line: on a "
               "coloured card with the spoken line under it, or full screen. Only the voice-over's sound plays.")

    def save(**ch):
        save_settings(project_dir, project, A, **ch)

    s = settings(project)
    # ---- 1. look
    st.markdown("#### 1 · Look")
    t1.look_picker(pid, "t3", t2.T2_LOOKS, s, save,
                   help_text="Colours for text and accents plus the frame around the clips. Fine-tune in More settings.")
    s = settings(project)
    names = list(CARD_COLOURS)
    cur_cc = s.get("card_colour") if s.get("card_colour") in names else "Cream"
    a, b = st.columns([3, 2])
    cc = a.radio("Card colour (behind the B-roll)", names, index=names.index(cur_cc), horizontal=True,
                 key=f"t3_cc_{pid}")
    if cc != cur_cc:
        save(card_colour=cc)
        s = settings(project)
    frames = list(t2.FRAMES)
    cur_fr = s.get("frame_name") or "As designed"
    fr = b.selectbox("Frame around the clip", frames, index=frames.index(cur_fr) if cur_fr in frames else 0,
                     key=f"t3_frame_{pid}_{st.session_state.get(f't3_rev_{pid}', 0)}")
    if fr != cur_fr:
        save(frame_name=fr, look="Custom")
        st.session_state[f"t3_rev_{pid}"] = st.session_state.get(f"t3_rev_{pid}", 0) + 1
        st.rerun()
    pal = palette(s)
    st.markdown(t1.swatch_html([pal["text"], pal["accent2"], pal["accent3"], pal["cardBg"], pal["cardText"]],
                               ["Main", "Colour 2", "Colour 3", "Card", "Card text"]), unsafe_allow_html=True)
    fmt_labels = list(t2.FORMATS)
    cur_fmt = next((k for k, v in t2.FORMATS.items() if v == s["format"]), fmt_labels[0])
    fmt = t2.FORMATS[st.radio("Output", fmt_labels, index=fmt_labels.index(cur_fmt), horizontal=True, key=f"t3_fmt_{pid}")]
    if fmt != s["format"]:
        save(format=fmt)
        s = settings(project)

    problem = t1.remotion_problem()
    if problem:
        st.error(problem)
        return
    keep = A["ensure_keep"](project_dir, project)
    if not keep:
        st.info(A["NO_PROJECT_MSG"])
        return

    # ---- 2. clips
    st.markdown("#### 2 · B-roll clips")
    clip_library(project_dir, project, A, pid)
    s = settings(project)
    note = st.session_state.pop(f"t3_note_{pid}", None)
    if note:
        st.toast(note.lstrip("⚠ "), icon="⚠️" if note.startswith("⚠") else "✅")

    if not project.get("transcript"):
        st.info("Next, transcribe the voice-over in step 4 (A. Transcribe) and correct the words there. "
                "The lines and text here come from that transcript.")
        return
    if project.get("transcript_keep_version") != project.get("keep_version"):
        st.warning("You changed the cut after transcribing. Transcribe again in step 4 before using this template.")
        return
    _, total = A["_offsets"](keep)
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
        save(candidates=cands, candidates_key=project.get("keep_version"),
             cuts=t1.merge_short(silence_cuts + [c["t"] for c in cands if c["likely"]], 0.3))
        s = settings(project)

    # ---- 3. words and B-roll
    st.markdown("#### 3 · Words and B-roll")
    use_roman = s["romanize"]
    words = t2.spoken_words(project, A, use_roman)
    groups = t1.line_groups(words)
    lines = line_info(words, groups)
    if "events" not in s:  # starts as plain captions: you pick what pops out (or ask for a suggestion)
        save(events=[])
        s = settings(project)
    events, brolls = s.get("events") or [], s.get("brolls") or []
    clip_ids = [c["id"] for c in s.get("clips") or []]
    rows, owner = t2.plan_rows(words, events)
    rev_key = f"t3_grid_rev_{pid}"
    rev = st.session_state.get(rev_key, 0)

    def store(new_events, new_brolls, note=None, grid=None):
        changed = (new_events, new_brolls) != (events, brolls)
        if changed:
            save(events=new_events, brolls=new_brolls)
        if grid is None or not t1.grid_keep(f"t3_applied_{pid}", grid, changed):
            st.session_state[rev_key] = rev + 1
        if note:
            st.session_state[f"t3_note_{pid}"] = note
        st.rerun()

    c1, c2, c3 = st.columns([4, 1, 1])
    c1.caption("For each line choose a **B-roll** clip (and **As**: card or full screen), and how its text shows. "
               "Or **drag across the timeline** to select words. Changes save straight away and show in the live "
               "preview.")
    if c2.button("✨ Suggest text", key=f"t3_sugg_{pid}", use_container_width=True,
                 help="Replace the text plan with a suggestion. B-roll stays."):
        store(t2.suggest_events(project, A, use_roman, total), brolls, "New text suggestion made.")
    if c3.button("⌫ All captions", key=f"t3_clear_{pid}", use_container_width=True,
                 help="Every line's text back to a plain caption. B-roll stays."):
        store([], brolls, "All lines are plain captions now.")
    view = t1.timeline_view(pid, "t3", total)
    bars = ([{"start": float(e["Start"]), "end": float(e["End"]), "row": "On screen",
              "kind": t2.TYPE_TO_SHOW.get(e["Type"], "Caption"),
              "label": str(e.get("Text") or "") or t2.TYPE_TO_SHOW.get(e["Type"], "")} for e in events]
            + [{"start": float(b_["start"]), "end": float(b_["end"]), "row": "B-roll", "kind": BROLL_KIND,
                "label": f"{b_['clip']} · {b_['mode']}"} for b_ in brolls])
    sel = t1.plan_timeline(words, [r["Show as"] for r in rows], bars, t2.SHOW + [BROLL_KIND],
                           t2.SHOW_COLOURS + ["#d4537e"], ["On screen", "B-roll"], view,
                           key=f"t3_tl_{pid}_{rev}", height=170)
    idx = t1.selected_words(words, sel) if sel else []
    if idx:
        said = " ".join(words[i]["text"] for i in idx)
        st.info(f"Selected **{words[idx[0]]['start']:.2f} – {words[idx[-1]]['end']:.2f} s**: “{said}”")
        st.caption("Make these words:")
        bc = st.columns(6)
        for n, show in enumerate(t2.SHOW):
            if bc[n].button(show, key=f"t3_make_{show}_{pid}_{rev}", use_container_width=True):
                new_rows = [dict(r) for r in rows]
                new_owner = list(owner)
                for i in idx:
                    new_rows[i].update({"Show as": show, "Text on screen": ""})
                    if show == "Accent":
                        new_rows[i]["Colour"] = new_rows[i]["Colour"] or "Main"
                        new_rows[i]["Side"] = new_rows[i]["Side"] or "right"
                    new_owner[i] = -1
                store(t2.events_from_rows(words, new_rows, new_owner, events), brolls,
                      f"“{said}” now shows as {show.lower()}.")
        st.caption("Double-click the timeline to clear the selection.")

    grid_col, live_col = st.columns([2, 1])
    with live_col:
        t1.sticky_column(f"t3_livebox_{pid}")
        live_slot = st.container(key=f"t3_livebox_{pid}")
    with grid_col:
        if not clip_ids:
            st.caption("Add B-roll clips above to pick them here.")
        _grid(pid, rev, words, groups, lines, rows, owner, events, brolls, clip_ids, total, pal, store)

    # ---- more settings
    t1.pop_controls(pid, "t3", s, save, "accent words")
    s = settings(project)
    a, b = st.columns([1, 2])
    con = a.checkbox("🖱 Click sound when a B-roll clip appears", value=bool(s.get("click_on", True)),
                     key=f"t3_clickon_{pid}", help="A mouse click each time a B-roll clip comes on screen.")
    cvol = b.slider("Click volume", 0.1, 1.0, float(s.get("click_volume", 0.5)), 0.05, key=f"t3_clickvol_{pid}",
                    disabled=not con)
    if con != bool(s.get("click_on", True)) or abs(cvol - float(s.get("click_volume", 0.5))) > 1e-6:
        save(click_on=con, click_volume=cvol)
        s = settings(project)
    with st.expander("⚙ More settings: exact colours, zoom, head position, jump cuts"):
        st.markdown("**Exact colours** (the card colour above overrides the card colours here; pick 'From the look' to use these)")
        t1.custom_colours(pid, "t3", t2.T2_FIELDS, s.get("palette") or t2.T2_PALETTES[t2.DEFAULT_PALETTE], save)
        a, b = st.columns(2)
        punch = a.slider("Punch-in at jump-cuts", 1.0, 1.5, float(s["punch"]), 0.02, key=f"t3_punch_{pid}")
        hard = a.slider("Hard zoom strength", 1.1, 2.0, float(s["hard_zoom"]), 0.05, key=f"t3_hard_{pid}")
        beat = a.slider("Cut pace (seconds between jump-cuts)", 0.8, 4.0, float(s["beat"]), 0.1, key=f"t3_beat_{pid}")
        b.caption("Where is the head in the voice-over video? Captions sit beside it and zooms centre on it.")
        fx = b.slider("Head centre, left ↔ right", 0.1, 0.9, float(s["face_x"]), 0.01, key=f"t3_fx_{pid}")
        fy = b.slider("Eye height, top ↔ bottom", 0.1, 0.8, float(s["face_y"]), 0.01, key=f"t3_fy_{pid}")
        fw = b.slider("Head half-width", 0.05, 0.3, float(s["face_w"]), 0.01, key=f"t3_fw_{pid}")
        new = dict(punch=punch, hard_zoom=hard, beat=beat, face_x=fx, face_y=fy, face_w=fw)
        if any(s[k] != v for k, v in new.items()):
            save(**new)
            s = settings(project)
        rom = st.checkbox("Write Devanagari words in Roman letters", value=s["romanize"], key=f"t3_rom_{pid}")
        if rom != s["romanize"]:
            save(romanize=rom)
            st.rerun()
        st.markdown("**Jump cuts**")
        cands = s.get("candidates") or []
        chosen = set(round(c, 3) for c in s.get("cuts") or [])
        st.caption((f"{len(silence_cuts)} cuts come from removing pauses / repeats and are always used. " if silence_cuts else "")
                   + ("Tick the real cuts found in the video, untick hand movements." if cands else ""))
        if cands:
            fps = t1.video_fps(project_dir / src, A)
            new_chosen = set(c for c in chosen if c in silence_cuts)
            cols = st.columns(4)
            for k, c in enumerate(cands):
                with cols[k % 4]:
                    png = t1.thumb_pair(project_dir, src, c["t"], fps, A)
                    if png:
                        st.image(str(png), use_container_width=True)
                    if st.checkbox(f"{c['t']:.2f}s cut", value=round(c["t"], 3) in chosen, key=f"t3_cut_{pid}_{c['t']}"):
                        new_chosen.add(round(c["t"], 3))
            if new_chosen != chosen:
                save(cuts=sorted(new_chosen))
                s = settings(project)

    with live_slot:
        try:
            t1.live_preview(project_dir, project, A, "t3", COMPOSITION, build_props(project_dir, project, A, src),
                            s["format"], total, (round(sel[0], 2), round(sel[1], 2)) if idx else None)
        except Exception as exc:
            st.warning(f"Live preview is not available: {exc}")

    # ---- 4. full preview
    st.markdown("#### 4 · Full preview")
    try:
        cur_key = t1.props_key(build_props(project_dir, project, A, src))
    except Exception:
        cur_key = None
    max_t = max(0.1, round(total - 0.1, 1))
    mode = st.radio("Preview", ["A few seconds", "One frame", "Whole video (takes longer)"], horizontal=True,
                    key=f"t3_pmode_{pid}", label_visibility="collapsed")
    p1, p2 = st.columns([2, 1])
    first = float(words[idx[0]]["start"]) if idx else float((brolls or [{"start": 0.0}])[0]["start"])
    if mode == "One frame":
        t_prev = p1.slider("Frame at (seconds)", 0.0, max_t, float(min(round(first + 0.5, 1), max_t)), 0.1,
                           key=f"t3_tprev_{pid}")
        if p2.button("Show this frame", key=f"t3_still_{pid}", use_container_width=True):
            try:
                with st.spinner("Rendering one frame (about 15 seconds)..."):
                    st.session_state[f"t3_stillimg_{pid}"] = str(render_still(project_dir, project, A, t_prev))
            except Exception as exc:
                st.error(str(exc))
        still = st.session_state.get(f"t3_stillimg_{pid}")
        if still and Path(still).exists():
            st.image(still, width=360, caption="Exact frame from the final render")
    else:
        whole = mode.startswith("Whole")
        if whole:
            start, length = 0.0, None
            p1.caption(f"The whole {total:.0f} s video at half resolution.")
        else:
            start = p1.slider("Start at (seconds)", 0.0, max_t, float(min(max(0.0, round(first - 0.5)), max_t)), 0.5,
                              key=f"t3_pstart_{pid}")
            length = p1.slider("Length (seconds)", 2, 15, 6, 1, key=f"t3_plen_{pid}")
            p1.caption("About 20 to 40 seconds to make.")
        if p2.button("Make preview", type="primary", key=f"t3_pbtn_{pid}", use_container_width=True):
            try:
                with st.spinner("Making the preview with Remotion..."):
                    out, key = render_preview(project_dir, project, A, start, length, total)
                st.session_state[f"t3_prev_{pid}"] = (str(out), key, out.stat().st_mtime)
            except Exception as exc:
                st.error(str(exc))
        prev = st.session_state.get(f"t3_prev_{pid}")
        if prev and Path(prev[0]).exists():
            v, info = st.columns([1, 1])
            v.video(Path(prev[0]).read_bytes())
            if cur_key and prev[1] != cur_key:
                info.warning("You changed something after this preview. Click Make preview again to see it.")
            else:
                info.success("Preview is up to date with your current settings.")

    # ---- 5. render
    st.markdown("#### 5 · Render or carry forward")
    st.caption("Render the video now (about 2-4 minutes for 35 s), or carry it forward to the Visual hook page "
               "and render it there once, with the hook.")
    rc, cc = st.columns(2)
    t1.carry_button(project_dir, project, A, "template3", cc)
    if rc.button("Render Template 3 video", type="primary", key=f"t3_render_{pid}", use_container_width=True):
        try:
            with st.spinner("Rendering with Remotion..."):
                out = render_video(project_dir, project, A)
            st.session_state[f"t3_out_{pid}"] = str(out)
        except Exception as exc:
            st.error(str(exc))
    res = st.session_state.get(f"t3_out_{pid}")
    if res and Path(res).exists():
        v, d = st.columns([1, 1])
        v.video(res)
        d.success(f"Rendered: {Path(res).name}")
        d.caption(f"Saved in: {Path(res).parent}")
        d.download_button("Download video", Path(res).read_bytes(), file_name=Path(res).name, key=f"t3_dl_{pid}")
