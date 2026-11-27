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

# ---- colours & frames
T2_FIELDS = [("text", "Main (captions, quotes)"), ("accent2", "Colour 2"), ("accent3", "Colour 3"),
             ("screenBg", "Text screen background"), ("screenText", "Text screen words"),
             ("cardBg", "Card background"), ("cardText", "Cream card caption"), ("titleText", "Video card title")]
T2_PALETTES = {
    "Classic (blue, green & cream)": dict(text="#FFFFFF", accent2="#1F5A2B", accent3="#0715A6", screenBg="#0715A6",
                                          screenText="#FFFFFF", cardBg="#FDFDE4", cardText="#1F5A2B", titleText="#151515"),
    "Sunset": dict(text="#FFFFFF", accent2="#E4572E", accent3="#F3A712", screenBg="#E4572E", screenText="#FFF6E5",
                   cardBg="#FFF1E0", cardText="#B23A1E", titleText="#2B1B12"),
    "Ocean": dict(text="#FFFFFF", accent2="#0E7C86", accent3="#1B4DE0", screenBg="#0B3D91", screenText="#E6F4FF",
                  cardBg="#EAF6F6", cardText="#0E5E66", titleText="#0B1F33"),
    "Candy": dict(text="#FFFFFF", accent2="#D6246E", accent3="#7B2FF7", screenBg="#7B2FF7", screenText="#FFFFFF",
                  cardBg="#FFF0F6", cardText="#A0154F", titleText="#2A0A33"),
    "Forest & gold": dict(text="#FFFFFF", accent2="#2F6B3A", accent3="#C9A227", screenBg="#1E3D2A", screenText="#F4EBD0",
                          cardBg="#F4EFE1", cardText="#2F6B3A", titleText="#1E2A1E"),
    "Night neon": dict(text="#FFFFFF", accent2="#00C2A8", accent3="#FF3D7F", screenBg="#101820", screenText="#00E5C4",
                       cardBg="#101820", cardText="#00E5C4", titleText="#FFFFFF"),
    "Mono": dict(text="#FFFFFF", accent2="#111111", accent3="#6B6B6B", screenBg="#111111", screenText="#FFFFFF",
                 cardBg="#F2F2F2", cardText="#111111", titleText="#111111"),
}
FRAMES = {"As designed": None, "Soft shadow": "shadow", "Rounded": "rounded", "Polaroid": "polaroid",
          "Tilted polaroid": "tilted", "Outline": "outline", "Circle": "circle", "No frame": "none"}
CARD_TYPES = ["blue screen", "cream card", "video card"]
DEFAULT_PALETTE = "Classic (blue, green & cream)"


def _look(palette, frame):
    return {"palette": dict(T2_PALETTES[palette]), "frame_name": frame}


# A look = colours + the frame around the shrunk video. Picked from one dropdown.
T2_LOOKS = {
    "Classic": _look(DEFAULT_PALETTE, "As designed"),
    "Classic polaroid": _look(DEFAULT_PALETTE, "Polaroid"),
    "Sunset rounded": _look("Sunset", "Rounded"),
    "Ocean outline": _look("Ocean", "Outline"),
    "Candy tilted": _look("Candy", "Tilted polaroid"),
    "Forest & gold polaroid": _look("Forest & gold", "Polaroid"),
    "Night neon": _look("Night neon", "Outline"),
    "Mono shadow": _look("Mono", "Soft shadow"),
    "Mono circle": _look("Mono", "Circle"),
}


def settings(project):
    return {**DEFAULTS, **(project.get("template2") or {})}


def save_settings(project_dir, project, A, **changes):
    project["template2"] = {**(project.get("template2") or {}), **changes}
    A["save_project"](project_dir, project)


# ------------------------------------------------------------------ words (one entry per spoken word)
spoken_words = t1.spoken_words  # shared with Talking1


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


# ------------------------------------------------------------------ word grid <-> text plan
SHOW = ["Caption", "Accent", "Quote", "Blue screen", "Cream card", "Video card"]
SHOW_TO_TYPE = {"Accent": "accent", "Quote": "quote", "Blue screen": "blue screen", "Cream card": "cream card",
                "Video card": "video card"}
TYPE_TO_SHOW = {v: k for k, v in SHOW_TO_TYPE.items()}
SHOW_COLOURS = ["#5f6670", "#c99a06", "#8e44ad", "#1f4fd6", "#a8844f", "#1f8a70"]
COLOUR_LABELS = {"white": "Main", "green": "Colour 2", "blue": "Colour 3"}
LABEL_TO_COLOUR = {v: k for k, v in COLOUR_LABELS.items()}
GRID_COLS = ["Time", "Word", "Show as", "Colour", "Side", "Hard zoom", "Text on screen"]
EDIT_COLS = ["Show as", "Colour", "Side", "Hard zoom", "Text on screen"]


def event_spans(events):
    spans = []
    for e in events:
        toks = None
        if e.get("Type") in ("accent", "quote"):
            toks = {t1.norm_word(w) for w in str(e.get("Text") or "").split()} - {""}
        spans.append({"start": float(e["Start"]), "end": float(e["End"]), "tokens": toks or None})
    return spans


def plan_rows(words, events):
    """One grid row per spoken word, made from the saved text plan."""
    owner = t1.assign_words(words, event_spans(events))
    rows, seen = [], set()
    for i, w in enumerate(words):
        r = {"Time": round(w["start"], 2), "Word": w["text"], "Show as": "Caption", "Colour": None, "Side": None,
             "Hard zoom": False, "Text on screen": ""}
        k = owner[i]
        if k is not None:
            e = events[k]
            r["Show as"] = TYPE_TO_SHOW.get(e.get("Type"), "Caption")
            if e.get("Type") == "accent":
                r["Colour"] = COLOUR_LABELS.get(e.get("Colour"), "Main")
                r["Side"] = e.get("Side") if e.get("Side") in SIDES else "right"
            if e.get("Type") in ("accent", "quote"):
                r["Hard zoom"] = bool(e.get("Hard zoom"))
            if k not in seen:
                r["Text on screen"] = str(e.get("Text") or "")
                seen.add(k)
        rows.append(r)
    return rows, owner


def events_from_rows(words, rows, owner, events):
    """The grid back to the text plan: neighbouring words with the same settings become one item.
    Items that didn't change keep their exact times and text."""
    cell = t1.cell

    def key_of(i):
        r = rows[i]
        t = SHOW_TO_TYPE.get(cell(r.get("Show as")))
        if not t:
            return None
        zoom = bool(cell(r.get("Hard zoom"), False))
        if t == "accent":
            return (t, LABEL_TO_COLOUR.get(cell(r.get("Colour")), "white"),
                    cell(r.get("Side")) if cell(r.get("Side")) in SIDES else "right", zoom)
        if t == "quote":
            return (t, zoom)
        return (t,)

    def text_of(i):
        return str(cell(rows[i].get("Text on screen"))).strip()

    out = []
    for key, idx in t1.group_runs(len(rows), key_of, owner, text_of):
        t = key[0]
        olds = {owner[i] for i in idx}
        k = next(iter(olds)) if len(olds) == 1 else None
        same = (k is not None and 0 <= k < len(events) and t1.owned_by(owner, k) == idx
                and events[k].get("Type") == t)
        k0 = owner[idx[0]]
        old_text = str(events[k0].get("Text") or "").strip() if k0 is not None and 0 <= k0 < len(events) else ""
        first_text = text_of(idx[0])
        joined = " ".join(words[i]["text"] for i in idx).strip(" ,")
        if first_text and (same or first_text != old_text):
            text = first_text
        elif t == "accent":
            parts = joined.split()
            text = joined if len(parts) > 1 and all(w[:1].isupper() for w in parts) else joined.lower()
        elif t == "quote":
            text = joined
        elif t == "video card":
            text = short_title(joined)
        else:
            text = ""
        if same:
            a, b = float(events[k]["Start"]), float(events[k]["End"])
        else:
            a, b = float(words[idx[0]]["start"]), float(words[idx[-1]]["end"])
            if t == "accent":
                b = max(b, a + 0.5)
        out.append({"Start": round(a, 2), "End": round(b, 2), "Type": t, "Text": text,
                    "Colour": key[1] if t == "accent" else "white",
                    "Side": key[2] if t == "accent" else "right",
                    "Hard zoom": key[-1] if t in ("accent", "quote") else False})
    return sorted(out, key=lambda r: r["Start"])


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
             "punch": float(s["punch"]), "beat": float(s["beat"]),
             "palette": s.get("palette") or T2_PALETTES[DEFAULT_PALETTE]}
    if FRAMES.get(s.get("frame_name") or "As designed"):
        props["frame"] = FRAMES[s["frame_name"]]
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


LINE_COLS = ["Time", "Line", "Show as", "Pop words", "Colour", "Side", "Hard zoom"]
LINE_EDIT = ["Show as", "Pop words", "Colour", "Side", "Hard zoom"]


def line_rows(words, groups, rows, owner, events):
    """One grid row per caption line, made from the per-word plan."""
    out = []
    for idx in groups:
        kinds = [rows[i]["Show as"] for i in idx]
        big = [k for k in kinds if k not in ("Caption", "Accent")]
        show = big[0] if big else ("Accent" if "Accent" in kinds else "Caption")
        evs = []
        for i in idx:
            if owner[i] is not None and owner[i] not in evs:
                evs.append(owner[i])
        r = {"Time": round(words[idx[0]]["start"], 2), "Line": " ".join(words[i]["text"] for i in idx),
             "Show as": show, "Pop words": "", "Colour": None, "Side": None, "Hard zoom": False}
        if show == "Accent":
            acc = [events[k] for k in evs if events[k].get("Type") == "accent"]
            r["Pop words"] = " / ".join(str(e.get("Text") or "") for e in acc)
            r["Colour"] = COLOUR_LABELS.get(acc[0].get("Colour"), "Main")
            r["Side"] = acc[0].get("Side") if acc[0].get("Side") in SIDES else "right"
            r["Hard zoom"] = bool(acc[0].get("Hard zoom"))
        elif show in ("Quote", "Video card"):
            e = next((events[k] for k in evs if events[k].get("Type") == SHOW_TO_TYPE[show]), {})
            r["Pop words"] = str(e.get("Text") or "")
            r["Hard zoom"] = bool(e.get("Hard zoom")) if show == "Quote" else False
        out.append(r)
    return out


def apply_line_rows(words, groups, rows, owner, events, old_lines, new_lines):
    """Turn edited line rows back into the text plan. Lines that didn't change keep their items exactly."""
    cell = t1.cell
    new_rows, new_owner = [dict(r) for r in rows], list(owner)
    marker, notes = -1, []
    for g, (o, n) in enumerate(zip(old_lines, new_lines)):
        if not t1.line_changed(o, n, LINE_EDIT):
            continue
        idx = groups[g]
        for i in idx:
            new_rows[i].update({"Show as": "Caption", "Colour": None, "Side": None, "Hard zoom": False,
                                "Text on screen": ""})
            new_owner[i] = None
        show = cell(n["Show as"]) or "Caption"
        pop = str(cell(n["Pop words"])).strip()
        colour = cell(n["Colour"]) or "Main"
        side = cell(n["Side"]) or "right"
        zoom = bool(cell(n["Hard zoom"], False))
        # pop words, a colour, side or zoom chosen on a plain caption line: make it an accent
        if show == "Caption" == cell(o["Show as"]) and t1.line_changed(o, n, ["Pop words", "Colour", "Side", "Hard zoom"]):
            show = "Accent"
        if show == "Accent":
            phrases = t1.split_phrases(pop) or [t1.best_word(words, idx).lower()]
            for ph in phrases:
                sel = t1.phrase_words(words, idx, ph)
                if not sel:
                    sel = idx
                    notes.append(f"“{ph}” isn't in the line, so it shows over the whole line.")
                for j, i in enumerate(sel):
                    new_rows[i].update({"Show as": "Accent", "Colour": colour, "Side": side, "Hard zoom": zoom,
                                        "Text on screen": ph if j == 0 else ""})
                    new_owner[i] = marker
                marker -= 1
        elif show in SHOW_TO_TYPE:
            for j, i in enumerate(idx):
                new_rows[i].update({"Show as": show, "Hard zoom": zoom if show == "Quote" else False,
                                    "Text on screen": pop if j == 0 and show in ("Quote", "Video card") else ""})
                new_owner[i] = marker
            marker -= 1
    return events_from_rows(words, new_rows, new_owner, events), notes


def _t2_grid(pid, rev, words, rows, owner, events, pal, store):
    st.markdown(t1.swatch_html([pal["text"], pal["accent2"], pal["accent3"]], ["Main", "Colour 2", "Colour 3"]),
                unsafe_allow_html=True)
    groups = t1.line_groups(words)
    lines = line_rows(words, groups, rows, owner, events)
    df = pd.DataFrame(lines, columns=LINE_COLS)
    edited = t1.grid_editor(
        df, f"t2_grid_{pid}_{rev}",
        {"Time": st.column_config.NumberColumn(format="%.2f", width="small"),
         "Line": st.column_config.TextColumn(width="medium"),
         "Show as": st.column_config.SelectboxColumn(options=SHOW, required=True),
         "Pop words": st.column_config.TextColumn(
             width="medium", help="Accent: the word(s) of this line that pop out, as they should appear (e.g. "
                                  "'problem.'). Several: separate with /. Quote / Video card: the text shown "
                                  "(empty = the spoken line). Typing here on a caption line makes it an accent."),
         "Colour": st.column_config.SelectboxColumn(options=list(LABEL_TO_COLOUR), help="Accent colour (see the swatches)."),
         "Side": st.column_config.SelectboxColumn(options=SIDES, help="Accent: which side of the head."),
         "Hard zoom": st.column_config.CheckboxColumn(help="Accent or quote: a strong punch-in.")},
        disabled=["Time", "Line"])
    if t1.grid_changed(edited, df, LINE_EDIT):
        new_events, notes = apply_line_rows(words, groups, rows, owner, events, lines, edited.to_dict("records"))
        store(new_events, ("⚠ " + " ".join(notes)) if notes else None)


# ------------------------------------------------------------------ the step (UI)
def step(project_dir, project, A):
    pid = project["project_id"]
    st.subheader("Template 2")
    st.caption("Small captions beside the head, bigger accent words, big quotes, quick jump-cuts with punch-ins, "
               "a text screen, a card with the video shrunk in a frame and a video card.")

    def save(**ch):
        save_settings(project_dir, project, A, **ch)

    s = settings(project)
    if "look" not in s and s.get("palette"):
        match = next((n for n, lk in T2_LOOKS.items() if lk["palette"] == s["palette"]
                      and lk["frame_name"] == (s.get("frame_name") or "As designed")), "Custom")
        save(look=match)
        s = settings(project)
    # ---- 1. look
    st.markdown("#### 1 · Look")
    t1.look_picker(pid, "t2", T2_LOOKS, s, save,
                   help_text="Colours for text, accents, the text screen and cards, plus the frame around the "
                             "shrunk video. Fine-tune them in More settings.")
    s = settings(project)
    pal = s.get("palette") or T2_PALETTES[DEFAULT_PALETTE]
    st.markdown(t1.swatch_html([pal["text"], pal["accent2"], pal["accent3"], pal["screenBg"], pal["cardBg"]],
                               ["Main", "Colour 2", "Colour 3", "Text screen", "Card"])
                + f"<div style='font-size:13px;opacity:.8;margin:-4px 0 6px'>Frame: {s.get('frame_name') or 'As designed'}</div>",
                unsafe_allow_html=True)
    fmt_labels = list(FORMATS)
    cur_fmt = next((k for k, v in FORMATS.items() if v == s["format"]), fmt_labels[0])
    fmt = FORMATS[st.radio("Output", fmt_labels, index=fmt_labels.index(cur_fmt), horizontal=True, key=f"t2_fmt_{pid}")]
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
    if not project.get("transcript"):
        st.info("First transcribe the video in step 4 (A. Transcribe) and correct the words there. "
                "The words here come from that transcript.")
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

    # ---- 2. words
    st.markdown("#### 2 · Words")
    use_roman = s["romanize"]
    words = spoken_words(project, A, use_roman)
    if "events" not in s:
        save(events=suggest_events(project, A, use_roman, total))
        s = settings(project)
    events = s.get("events") or []
    rows, owner = plan_rows(words, events)
    rev_key = f"t2_grid_rev_{pid}"
    rev = st.session_state.get(rev_key, 0)

    def store(new_events, note=None):
        save(events=new_events)
        st.session_state[rev_key] = rev + 1
        if note:
            st.session_state[f"t2_note_{pid}"] = note
        st.rerun()

    c1, c2 = st.columns([3, 1])
    c1.caption("Every spoken word shows as a small caption unless you make it something else. In the grid, choose "
               "what each line shows as and type its **Pop words**, or **drag across the timeline** to select "
               "words. Changes are saved straight away and show in the live preview.")
    if c2.button("↺ Suggest again", key=f"t2_sugg_{pid}", use_container_width=True,
                 help="Replace the plan with a fresh suggestion."):
        store(suggest_events(project, A, use_roman, total), "New suggestion made.")
    note = st.session_state.pop(f"t2_note_{pid}", None)
    if note:
        st.toast(note.lstrip("⚠ "), icon="⚠️" if note.startswith("⚠") else "✅")
    view = t1.timeline_view(pid, "t2", total)
    bars = [{"start": float(e["Start"]), "end": float(e["End"]), "row": "On screen",
             "kind": TYPE_TO_SHOW.get(e["Type"], "Caption"),
             "label": str(e.get("Text") or "") or TYPE_TO_SHOW.get(e["Type"], "")} for e in events]
    sel = t1.plan_timeline(words, [r["Show as"] for r in rows], bars, SHOW, SHOW_COLOURS, ["On screen"], view,
                           key=f"t2_tl_{pid}_{rev}")
    idx = t1.selected_words(words, sel) if sel else []
    if idx:
        act = st.container()
        said = " ".join(words[i]["text"] for i in idx)
        act.info(f"Selected **{words[idx[0]]['start']:.2f} – {words[idx[-1]]['end']:.2f} s**: “{said}”")
        act.caption("Make these words:")
        bc = act.columns(6)
        for n, show in enumerate(SHOW):
            if bc[n % 3].button(show, key=f"t2_make_{show}_{pid}_{rev}", use_container_width=True):
                new_rows = [dict(r) for r in rows]
                new_owner = list(owner)
                for i in idx:
                    new_rows[i].update({"Show as": show, "Text on screen": ""})
                    if show == "Accent":
                        new_rows[i]["Colour"] = new_rows[i]["Colour"] or "Main"
                        new_rows[i]["Side"] = new_rows[i]["Side"] or "right"
                    new_owner[i] = -1  # one new item
                store(events_from_rows(words, new_rows, new_owner, events),
                      f"“{said}” now shows as {show.lower()}.")
        act.caption("Double-click the timeline to clear the selection.")

    grid_col, live_col = st.columns([2, 1])
    with live_col:
        t1.sticky_column(f"t2_livebox_{pid}")
        live_slot = st.container(key=f"t2_livebox_{pid}")
    with grid_col:
        _t2_grid(pid, rev, words, rows, owner, events, pal, store)

    # ---- more settings
    with st.expander("⚙ More settings: exact colours, frame, zoom, head position, jump cuts"):
        st.markdown("**Exact colours**")
        t1.custom_colours(pid, "t2", T2_FIELDS, pal, save)
        st.markdown("**Frame around the shrunk video** (cream card and video card)")
        names = list(FRAMES)
        cur = s.get("frame_name") or "As designed"
        frame_name = st.radio("Frame", names, index=names.index(cur) if cur in names else 0, horizontal=True,
                              key=f"t2_frame_{pid}_{st.session_state.get(f't2_rev_{pid}', 0)}",
                              label_visibility="collapsed")
        if frame_name != cur:
            save(frame_name=frame_name, look="Custom")
            st.session_state[f"t2_rev_{pid}"] = st.session_state.get(f"t2_rev_{pid}", 0) + 1
            st.rerun()
        a, b = st.columns(2)
        punch = a.slider("Punch-in at jump-cuts", 1.0, 1.5, float(s["punch"]), 0.02, key=f"t2_punch_{pid}",
                         help="Shots switch between normal and this zoom. 1.18 matches the reference.")
        hard = a.slider("Hard zoom strength", 1.1, 2.0, float(s["hard_zoom"]), 0.05, key=f"t2_hard_{pid}")
        beat = a.slider("Cut pace (seconds between jump-cuts)", 0.8, 4.0, float(s["beat"]), 0.1, key=f"t2_beat_{pid}",
                        help="Smaller = more cuts. 1.6 is close to the reference.")
        b.caption("Where is the head? Captions sit beside it and zooms centre on it. Check with 'One frame'.")
        fx = b.slider("Head centre, left ↔ right", 0.1, 0.9, float(s["face_x"]), 0.01, key=f"t2_fx_{pid}")
        fy = b.slider("Eye height, top ↔ bottom", 0.1, 0.8, float(s["face_y"]), 0.01, key=f"t2_fy_{pid}")
        fw = b.slider("Head half-width", 0.05, 0.3, float(s["face_w"]), 0.01, key=f"t2_fw_{pid}",
                      help="Bigger = captions sit further from the head.")
        new = dict(punch=punch, hard_zoom=hard, beat=beat, face_x=fx, face_y=fy, face_w=fw)
        if any(s[k] != v for k, v in new.items()):
            save(**new)
            s = settings(project)
        rom = st.checkbox("Write Devanagari words in Roman letters", value=s["romanize"], key=f"t2_rom_{pid}",
                          help="Rough spelling. Correct the words in the step 4 transcript editor.")
        if rom != s["romanize"]:
            save(romanize=rom)
            st.rerun()
        st.markdown("**Jump cuts** (the framing changes at each one)")
        cands = s.get("candidates") or []
        chosen = set(round(c, 3) for c in s.get("cuts") or [])
        st.caption("Extra jump-cuts are added automatically inside long shots (Cut pace). "
                   + (f"{len(silence_cuts)} cuts come from removing pauses / repeats and are always used. " if silence_cuts else "")
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
                    if st.checkbox(f"{c['t']:.2f}s cut", value=round(c["t"], 3) in chosen, key=f"t2_cut_{pid}_{c['t']}"):
                        new_chosen.add(round(c["t"], 3))
            if new_chosen != chosen:
                save(cuts=sorted(new_chosen))
                s = settings(project)

    with live_slot:
        try:
            t1.live_preview(project_dir, project, A, "t2", COMPOSITION, build_props(project_dir, project, A, src),
                            s["format"], total, (round(sel[0], 2), round(sel[1], 2)) if idx else None)
        except Exception as exc:
            st.warning(f"Live preview is not available: {exc}")

    # ---- 3. full preview
    st.markdown("#### 3 · Full preview")
    try:
        cur_key = t1.props_key(build_props(project_dir, project, A, src))
    except Exception:
        cur_key = None
    max_t = max(0.1, round(total - 0.1, 1))
    mode = st.radio("Preview", ["One frame", "A few seconds", "Whole video (takes longer)"], horizontal=True,
                    key=f"t2_pmode_{pid}", label_visibility="collapsed")
    p1, p2 = st.columns([2, 1])
    first = float(idx and words[idx[0]]["start"] or min(1.0, max_t))
    if mode == "One frame":
        t_prev = p1.slider("Frame at (seconds)", 0.0, max_t, float(min(round(first + 0.2, 1), max_t)), 0.1,
                           key=f"t2_tprev_{pid}")
        if p2.button("Show this frame", key=f"t2_still_{pid}", use_container_width=True):
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
            start = p1.slider("Start at (seconds)", 0.0, max_t, float(min(max(0.0, round(first - 0.5)), max_t)), 0.5,
                              key=f"t2_pstart_{pid}")
            length = p1.slider("Length (seconds)", 2, 15, 6, 1, key=f"t2_plen_{pid}")
            p1.caption("About 20 to 40 seconds to make.")
        if p2.button("Make preview", type="primary", key=f"t2_pbtn_{pid}", use_container_width=True):
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

    # ---- 4. render
    st.markdown("#### 4 · Render")
    st.caption("A 35 s video takes about 2-4 minutes.")
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
