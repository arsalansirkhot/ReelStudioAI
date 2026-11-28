"""Outro tab for Reel Studio AI: a short end card added after the video.

Reference: outro.mp4 — a white card with a small bold word ("arsalan") and a second word ("sirkhot") that flips
through handwritten fonts every 2 frames. Here the two words are Text 1 and Text 2: "mostly kokni",
"randomly arsalan", or your own.

Two ways in, like the Visual hook tab:
  * Carried forward: the template carried forward from its page (and, if ticked, the visual hook) plus the
    outro, rendered once (Remotion composition "Finish").
  * A finished render: the outro added after a video you already rendered (also "Finish", base "Video").
"""
import datetime
import json
from pathlib import Path

import streamlit as st

import talking1_preset as t1
import visualhook_preset as vh

KEY = "outro"
PRESETS = {"mostly kokni": ("mostly", "kokni"), "randomly arsalan": ("randomly", "arsalan"),
           "My own text": None}
SIZES = {"Like the reference": 1.0, "Bigger": 1.4, "Biggest": 1.9}
COLOURS = {"White card, black text": ("#FDFDFD", "#111111"), "Black card, white text": ("#111111", "#F7F7F7")}
SOURCES = ("talking1_*.mp4", "template2_*.mp4", "template3_*.mp4")
SKIP = ("talking1_source", "_preview", "_still", "_outro")


def settings(project):
    return {"preset": next(iter(PRESETS)), "text1": "mostly", "text2": "kokni", "seconds": 1.5,
            "size": "Like the reference", "colours": next(iter(COLOURS)), "with_hook": True,
            **(project.get(KEY) or {})}


def save_settings(project_dir, project, A, **changes):
    project[KEY] = {**(project.get(KEY) or {}), **changes}
    A["save_project"](project_dir, project)


def finished_videos(project_dir):
    vids = [p for pat in SOURCES for p in project_dir.glob(pat) if not any(s in p.stem for s in SKIP)]
    return sorted(vids, key=lambda p: p.stat().st_mtime, reverse=True)


def label(p):
    name = vh.PRETTY.get(p.stem.split("_")[0], p.stem)
    hook = " + hook" if p.stem.endswith("_hook") else ""
    when = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%d %b %H:%M")
    return f"{name}{hook} · {p.name} · rendered {when}"


def outro_props(s):
    t1_, t2_ = PRESETS.get(s["preset"]) or (s["text1"], s["text2"])
    bg, ink = COLOURS.get(s["colours"]) or next(iter(COLOURS.values()))
    return {"text1": str(t1_).strip(), "text2": str(t2_).strip(), "seconds": round(float(s["seconds"]), 2),
            "size": SIZES.get(s["size"], 1.0), "bg": bg, "ink": ink}


def build(project_dir, project, A, s, pick, key):
    """finish_props.json for the chosen video + outro. Returns (path, props, out_name, fmt or None, total)."""
    if pick == "carry":
        mod = vh._template_module(key)
        src = t1.source_video(project_dir, project, A)
        base_props = json.loads(Path(mod.build_props(project_dir, project, A, src)).read_text(encoding="utf-8"))
        props = {"base": t1.CARRY_TEMPLATES[key][0], "baseProps": base_props}
        if s["with_hook"]:
            props["hook"] = vh.hook_props(project_dir, vh.settings(project))
        fmt = mod.settings(project)["format"]
        out = f"{key}_{fmt.replace(':', 'x')}{'_hook' if s['with_hook'] else ''}_outro.mp4"
        total = float(A["probe_video"](project_dir / src).get("duration_seconds") or 1)
    else:
        props = {"base": "Video", "baseProps": {"src": pick}}
        fmt, out = None, f"{Path(pick).stem}_outro.mp4"
        total = float(A["probe_video"](project_dir / pick).get("duration_seconds") or 1)
    props["outro"] = outro_props(s)
    path = project_dir / "finish_props.json"
    path.write_text(json.dumps(props, ensure_ascii=False, indent=1), encoding="utf-8")
    return path, props, out, fmt, total


def render(project_dir, props_path, out_name):
    out = project_dir / out_name
    if out.exists():
        out.unlink()
    t1.run_remotion(["render", "src/index.ts", "Finish", str(out), f"--props={props_path}", "--crf=18"], project_dir)
    return out


def live(project_dir, project, A, pid, props, fmt, total):
    """The video with its outro playing live (Remotion Player); the button jumps to just before the outro."""
    try:
        with st.spinner("Preparing the live preview..."):
            bundle = t1.live_bundle()
            port = t1._live_server()
            video = t1.live_source(project_dir, project, A, "outro") if props["base"] != "Video" else None
    except Exception as exc:
        st.warning(f"Live preview is not available: {exc}")
        return
    t1._LiveHandler.files["live.js"] = str(bundle)
    props = json.loads(json.dumps(props))
    if props["base"] == "Video":
        f = project_dir / props["baseProps"]["src"]
        name = f"o/{pid}_{f.stem}_{int(f.stat().st_mtime)}.mp4"
        t1._LiveHandler.files[name] = str(f)
        props["baseProps"]["src"] = f"http://127.0.0.1:{port}/{name}"
        info = A["probe_video"](f)
        w, h = int(info.get("width") or 1080), int(info.get("height") or 1080)
    else:
        props["baseProps"] = t1.live_props(project_dir, pid, port, props["baseProps"], video)
        w, h = t1.frame_size(fmt, project)
    if props.get("hook"):
        props["hook"] = vh._sound_url(project_dir, pid, port, props["hook"])
    seek_key = f"outro_seek_{pid}"
    frames = max(1, int(total * 30)) + max(1, round(props["outro"]["seconds"] * 30))
    opts = {"comp": "Finish", "props": props, "frames": frames, "w": w // 2 * 2, "h": h // 2 * 2,
            "seek": max(0.0, total - 1.0), "seekId": st.session_state.get(seek_key, 0)}
    t1._live_component()(data={"bundle": f"http://127.0.0.1:{port}/live.js?v={int(bundle.stat().st_mtime)}",
                               "opts": opts}, key=f"outro_live_{pid}")
    if st.button("▶ Play the ending", key=f"outro_replay_{pid}", use_container_width=True,
                 help="Jump to 1 second before the outro."):
        st.session_state[seek_key] = st.session_state.get(seek_key, 0) + 1
        st.rerun()
    st.caption("Live preview: your video with the outro at the end, without rendering. Render below for the file.")


# ------------------------------------------------------------------ the step (UI)
def step(project_dir, project, A):
    pid = project["project_id"]
    st.subheader("Outro")
    st.caption("A short end card after your video: a steady word and a word that flips through handwritten "
               "fonts, like the outro reference.")

    def save(**ch):
        save_settings(project_dir, project, A, **ch)

    problem = t1.remotion_problem()
    if problem:
        st.error(problem)
        return
    key = vh.carried(project)
    vids = finished_videos(project_dir)
    if not key and not vids:
        st.info("Nothing to add the outro to yet. Carry a template forward (its last step) or render one first.")
        return
    s = settings(project)

    # ---- 1. video
    st.markdown("#### 1 · Video")
    options = (["carry"] if key else []) + [p.name for p in vids]
    names = {p.name: label(p) for p in vids}
    if key:
        names["carry"] = (f"{t1.CARRY_TEMPLATES[key][1]} · carried forward (not rendered yet; renders once here, "
                          "with the outro)")
    cur = s.get("video") if s.get("video") in options else options[0]
    pick = st.selectbox("Add the outro to", options, index=options.index(cur), format_func=names.get,
                        key=f"outro_vid_{pid}", help="The carried-forward template (with its page's latest "
                                                      "settings), then your finished videos, newest first.")
    if pick != s.get("video"):
        save(video=pick)
    if pick == "carry":
        wh = st.checkbox("Include the visual hook at the start (as set on the Visual hook page)",
                         value=bool(s["with_hook"]), key=f"outro_hook_{pid}")
        if wh != bool(s["with_hook"]):
            save(with_hook=wh)
            s = settings(project)

    # ---- 2. outro
    st.markdown("#### 2 · Outro")
    a, b = st.columns([1, 1])
    presets = list(PRESETS)
    preset = a.radio("Text", presets, index=presets.index(s["preset"]) if s["preset"] in presets else 0,
                     key=f"outro_preset_{pid}", horizontal=True,
                     help="The first word stays; the second word flips through handwritten fonts.")
    text1, text2 = s["text1"], s["text2"]
    if PRESETS.get(preset) is None:
        c, d = a.columns(2)
        text1 = c.text_input("Text 1 (steady)", value=s["text1"], key=f"outro_t1_{pid}")
        text2 = d.text_input("Text 2 (flips fonts)", value=s["text2"], key=f"outro_t2_{pid}")
    seconds = b.slider("Length (seconds)", 0.5, 4.0, float(s["seconds"]), 0.1, key=f"outro_len_{pid}",
                       help="How long the outro card stays. 1-2 s is usual.")
    sizes, cols = list(SIZES), list(COLOURS)
    size = b.radio("Text size", sizes, index=sizes.index(s["size"]) if s["size"] in sizes else 0,
                   key=f"outro_size_{pid}", horizontal=True)
    colours = b.radio("Colours", cols, index=cols.index(s["colours"]) if s["colours"] in cols else 0,
                      key=f"outro_col_{pid}", horizontal=True)
    new = dict(preset=preset, text1=text1, text2=text2, seconds=round(seconds, 2), size=size, colours=colours)
    if any(new[k] != s.get(k) for k in new):
        save(**new)
        s = settings(project)

    try:
        props_path, props, out_name, fmt, total = build(project_dir, project, A, s, pick, key)
    except Exception as exc:
        st.error(f"Could not prepare the video: {exc}")
        return

    # ---- 3. preview
    st.markdown("#### 3 · Preview")
    left, _ = st.columns([1, 1])
    with left:
        live(project_dir, project, A, pid, props, fmt, total)

    # ---- 4. render
    st.markdown("#### 4 · Render")
    st.caption(f"Makes {out_name}" + (" in one render (template, hook and outro together)." if pick == "carry"
                                      else ", your video with the outro added at the end."))
    if st.button("Render with the outro", type="primary", key=f"outro_render_{pid}"):
        try:
            with st.spinner("Rendering with Remotion..."):
                out = render(project_dir, props_path, out_name)
            st.session_state[f"outro_out_{pid}"] = str(out)
        except Exception as exc:
            st.error(str(exc))
    res = st.session_state.get(f"outro_out_{pid}")
    if res and Path(res).exists():
        v, dl = st.columns([1, 1])
        v.video(res)
        dl.success(f"Rendered: {Path(res).name}")
        dl.caption(f"Saved in: {Path(res).parent}")
        dl.download_button("Download video", Path(res).read_bytes(), file_name=Path(res).name,
                           key=f"outro_dl_{pid}")
