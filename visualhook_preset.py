"""Visual hook tab for Reel Studio AI: an opening effect put on top of a finished template video.

Two ways in:
  * Carried forward: a template page's "Carry forward" button sends its settings here without rendering. This
    page then renders the template and the hook together, once (Remotion composition "Finish").
  * A finished render: pick one of the videos rendered in the template steps and render a new copy with the
    hook (composition "VisualHook"). The template video itself is not changed.

Hook styles (more can be added to HOOKS):
  * Flash in (reference: visualhook.mp4): black frame, a warm flash, 2-3 blown-out frames that settle into the
    picture.

Rendered with the Remotion composition "VisualHook" in D:\\ClaudeAIVideoEdit\\remotion. Reuses talking1_preset
(Remotion runner, live preview server).
"""
import datetime
import json
from pathlib import Path

import streamlit as st

import talking1_preset as t1

KEY = "visualhook"
COMPOSITION = "VisualHook"
HOOKS = {
    "Flash in": {
        "style": "flash",
        "about": "A black frame, a warm flash and a few blown-out frames that settle into your video in about "
                 "0.2 s, like the visualhook reference.",
    },
}
SPEEDS = {"Like the reference": 1.0, "A bit slower": 1.5, "Slow": 2.0}
# flash sound, played on the flash frame (files in the app's sfx folder, copied into the project to render)
SFX_DIR = Path(__file__).resolve().parent / "sfx"
SOUNDS = {"Camera shutter": "flash_shutter.wav", "Whoosh": "flash_whoosh.wav", "No sound": None}
SOUND_FILE = "hook_sfx.wav"
# finished template renders in the project folder (previews and hooked copies left out)
SOURCES = ("talking1_*.mp4", "template2_*.mp4", "template3_*.mp4")
SKIP = ("talking1_source", "_preview", "_hook", "_still")
PRETTY = {"talking1": "Talking1", "template2": "Template 2", "template3": "Template 3"}


def settings(project):
    return {"hook": next(iter(HOOKS)), "speed": "Like the reference", "sound": next(iter(SOUNDS)),
            "sound_volume": 0.6, **(project.get(KEY) or {})}


def save_settings(project_dir, project, A, **changes):
    project[KEY] = {**(project.get(KEY) or {}), **changes}
    A["save_project"](project_dir, project)


def finished_videos(project_dir):
    vids = []
    for pat in SOURCES:
        for p in project_dir.glob(pat):
            if not any(s in p.stem for s in SKIP):
                vids.append(p)
    return sorted(vids, key=lambda p: p.stat().st_mtime, reverse=True)


def label(p):
    name = PRETTY.get(p.stem.split("_")[0], p.stem)
    when = datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime("%d %b %H:%M")
    return f"{name} · {p.name} · rendered {when}"


def sound_props(project_dir, s):
    """The flash sound ({src, volume}) copied into the project folder, or None for no sound."""
    f = SOUNDS.get(s.get("sound"))
    if not f or not (SFX_DIR / f).exists() or float(s.get("sound_volume", 0.6)) <= 0:
        return None
    path = project_dir / SOUND_FILE
    data = (SFX_DIR / f).read_bytes()
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)
    return {"src": SOUND_FILE, "volume": round(float(s.get("sound_volume", 0.6)), 2)}


def hook_props(project_dir, s):
    hook = HOOKS.get(s["hook"]) or next(iter(HOOKS.values()))
    props = {"style": hook["style"], "speed": SPEEDS.get(s["speed"], 1.0)}
    snd = sound_props(project_dir, s)
    if snd:
        props["sound"] = snd
    return props


def build_props(project_dir, s, src_name):
    props = {"src": src_name, **hook_props(project_dir, s)}
    path = project_dir / "visualhook_props.json"
    path.write_text(json.dumps(props, ensure_ascii=False, indent=1), encoding="utf-8")
    return path, props


# ---- carried-forward template (rendered here once, with the hook)
def _template_module(key):
    if key == "talking1":
        return t1
    if key == "template2":
        import template2_preset as m
        return m
    import template3_preset as m
    return m


def carried(project):
    key = (project.get("carry") or {}).get("template")
    return key if key in t1.CARRY_TEMPLATES else None


def build_finish(project_dir, project, A, key, s):
    """finish_props.json: the carried template's own props (its latest settings) + the hook."""
    mod = _template_module(key)
    src = t1.source_video(project_dir, project, A)
    base_props = json.loads(Path(mod.build_props(project_dir, project, A, src)).read_text(encoding="utf-8"))
    props = {"base": t1.CARRY_TEMPLATES[key][0], "baseProps": base_props, "hook": hook_props(project_dir, s)}
    path = project_dir / "finish_props.json"
    path.write_text(json.dumps(props, ensure_ascii=False, indent=1), encoding="utf-8")
    fmt = mod.settings(project)["format"]
    return path, props, src, fmt


def render_finish(project_dir, props_path, key, fmt):
    out = project_dir / f"{key}_{fmt.replace(':', 'x')}_hook.mp4"
    if out.exists():
        out.unlink()
    t1.run_remotion(["render", "src/index.ts", "Finish", str(out), f"--props={props_path}", "--crf=18"], project_dir)
    return out


def render(project_dir, props_path, src):
    out = project_dir / f"{src.stem}_hook.mp4"
    if out.exists():
        out.unlink()
    t1.run_remotion(["render", "src/index.ts", COMPOSITION, str(out), f"--props={props_path}", "--crf=18"], project_dir)
    return out


def _sound_url(project_dir, pid, port, hook):
    snd = hook.get("sound")
    if snd and not str(snd["src"]).startswith("http"):
        f = project_dir / snd["src"]
        if f.exists():
            name = f"p/{pid}_hook_{int(f.stat().st_mtime)}.wav"
            t1._LiveHandler.files[name] = str(f)
            hook["sound"] = {**snd, "src": f"http://127.0.0.1:{port}/{name}"}
    return hook


def live(project_dir, project, A, pid, props, src, info, finish=None):
    """The hook playing live (Remotion Player): on a finished video, or (finish = (fmt, total)) on the
    carried template drawn live."""
    try:
        with st.spinner("Preparing the live preview..."):
            bundle = t1.live_bundle()
            port = t1._live_server()
            video = t1.live_source(project_dir, project, A, "vh") if finish else None
    except Exception as exc:
        st.warning(f"Live preview is not available: {exc}")
        return
    t1._LiveHandler.files["live.js"] = str(bundle)
    if finish:
        fmt, total = finish
        props = json.loads(json.dumps(props))
        props["baseProps"] = t1.live_props(project_dir, pid, port, props["baseProps"], video)
        props["hook"] = _sound_url(project_dir, pid, port, props["hook"])
        comp, (w, h) = "Finish", t1.frame_size(fmt, project)
    else:
        vname = f"h/{pid}_{src.stem}_{int(src.stat().st_mtime)}.mp4"
        t1._LiveHandler.files[vname] = str(src)
        props = _sound_url(project_dir, pid, port, {**props, "src": f"http://127.0.0.1:{port}/{vname}"})
        comp, w, h = COMPOSITION, int(info.get("width") or 1080), int(info.get("height") or 1080)
        total = float(info.get("duration_seconds") or 1)
    seek_key = f"vh_seek_{pid}"
    opts = {"comp": comp, "props": props, "frames": max(1, int(total * 30)),
            "w": w // 2 * 2, "h": h // 2 * 2, "seek": 0.0, "seekId": st.session_state.get(seek_key, 0)}
    t1._live_component()(data={"bundle": f"http://127.0.0.1:{port}/live.js?v={int(bundle.stat().st_mtime)}",
                               "opts": opts}, key=f"vh_live_{pid}_{comp}")
    if st.button("▶ Play the hook again", key=f"vh_replay_{pid}", use_container_width=True):
        st.session_state[seek_key] = st.session_state.get(seek_key, 0) + 1
        st.rerun()
    st.caption("Live preview: the hook on your video, without rendering. Render below for the final file.")


# ------------------------------------------------------------------ the step (UI)
def step(project_dir, project, A):
    pid = project["project_id"]
    st.subheader("Visual hook")
    st.caption("An eye-catching opening for the first second of your video. Carry a template forward to here "
               "(its last step) and it renders once, with the hook; or put the hook on a video you already "
               "rendered (that video stays as it is; this makes a new copy).")

    def save(**ch):
        save_settings(project_dir, project, A, **ch)

    problem = t1.remotion_problem()
    if problem:
        st.error(problem)
        return
    key = carried(project)
    vids = finished_videos(project_dir)
    if not key and not vids:
        st.info("Nothing to put the hook on yet. In Talking1 style, Template 2 or Template 3, use **Carry forward "
                "to Visual hook** (their last step) to render it here once with the hook, or **Render** it first.")
        return
    s = settings(project)

    # ---- 1. video
    st.markdown("#### 1 · Video")
    CARRY = "carry"
    options = ([CARRY] if key else []) + [p.name for p in vids]
    names = {p.name: label(p) for p in vids}
    if key:
        names[CARRY] = (f"{t1.CARRY_TEMPLATES[key][1]} · carried forward (not rendered yet; renders once here, "
                        "with the hook)")
    cur = s.get("video") if s.get("video") in options else options[0]
    pick = st.selectbox("Put the hook on", options, index=options.index(cur), format_func=names.get,
                        key=f"vh_vid_{pid}", help="The carried-forward template (always with its page's latest "
                                                    "settings), then your finished videos, newest first.")
    if pick != s.get("video"):
        save(video=pick)
    is_carry = pick == CARRY
    if is_carry:
        st.caption(f"Change the words, colours or B-roll on the {t1.CARRY_TEMPLATES[key][1]} page; this page "
                   "always uses its latest settings.")

    # ---- 2. hook
    st.markdown("#### 2 · Hook")
    hooks = list(HOOKS)
    hook = st.radio("Hook style", hooks, index=hooks.index(s["hook"]) if s["hook"] in hooks else 0,
                    key=f"vh_hook_{pid}", horizontal=True)
    st.caption(HOOKS[hook]["about"])
    speeds = list(SPEEDS)
    speed = st.radio("Speed", speeds, index=speeds.index(s["speed"]) if s["speed"] in speeds else 0,
                     key=f"vh_speed_{pid}", horizontal=True)
    sounds = list(SOUNDS)
    a, b = st.columns([1, 1])
    sound = a.radio("Flash sound", sounds, index=sounds.index(s["sound"]) if s["sound"] in sounds else 0,
                    key=f"vh_sound_{pid}", horizontal=True, help="A sound on the flash, like the reference.")
    vol = b.slider("Sound volume", 0.1, 1.0, float(s.get("sound_volume", 0.6)), 0.05, key=f"vh_svol_{pid}",
                   disabled=SOUNDS.get(sound) is None)
    if (hook, speed, sound) != (s["hook"], s["speed"], s["sound"]) or abs(vol - float(s["sound_volume"])) > 1e-6:
        save(hook=hook, speed=speed, sound=sound, sound_volume=vol)
        s = settings(project)

    try:
        if is_carry:
            props_path, props, src_name, fmt = build_finish(project_dir, project, A, key, s)
            total = float(A["probe_video"](project_dir / src_name).get("duration_seconds") or 1)
            src, info = None, None
        else:
            src = project_dir / pick
            info = A["probe_video"](src)
            props_path, props = build_props(project_dir, s, src.name)
    except Exception as exc:
        st.error(f"Could not prepare the video: {exc}")
        return

    # ---- 3. preview
    st.markdown("#### 3 · Preview")
    left, _ = st.columns([1, 1])
    with left:
        live(project_dir, project, A, pid, props, src, info, finish=(fmt, total) if is_carry else None)

    # ---- 4. render
    st.markdown("#### 4 · Render")
    if is_carry:
        out_name = f"{key}_{fmt.replace(':', 'x')}_hook.mp4"
        st.caption(f"Renders {t1.CARRY_TEMPLATES[key][1]} with the hook in one go: {out_name}. About 2-4 minutes "
                   "for a 35 s video.")
    else:
        st.caption(f"Makes {src.stem}_hook.mp4 next to the template video. About 1-3 minutes for a 35 s video.")
    rc, cc = st.columns(2)
    outro_step = next((x for x in A.get("STEPS", []) if x.endswith("Outro")), None)
    if is_carry and outro_step and cc.button("➡ Carry forward to Outro (no render)", key=f"vh_carry_{pid}",
                                             use_container_width=True,
                                             help="Don't render now. The Outro page renders the template, this "
                                                  "hook and the outro in one go."):
        project.setdefault("outro", {})["with_hook"] = True
        project["outro"]["video"] = "carry"
        A["save_project"](project_dir, project)
        st.session_state["_goto"] = outro_step
        st.rerun()
    if rc.button("Render with the hook", type="primary", key=f"vh_render_{pid}", use_container_width=True):
        try:
            with st.spinner("Rendering with Remotion..."):
                out = (render_finish(project_dir, props_path, key, fmt) if is_carry
                       else render(project_dir, props_path, src))
            st.session_state[f"vh_out_{pid}"] = str(out)
        except Exception as exc:
            st.error(str(exc))
    res = st.session_state.get(f"vh_out_{pid}")
    if res and Path(res).exists():
        v, dl = st.columns([1, 1])
        v.video(res)
        dl.success(f"Rendered: {Path(res).name}")
        dl.caption(f"Saved in: {Path(res).parent}")
        dl.download_button("Download video", Path(res).read_bytes(), file_name=Path(res).name, key=f"vh_dl_{pid}")
