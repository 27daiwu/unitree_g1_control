#!/usr/bin/env python3
"""Play raw G1 q[29] NPZ data kinematically in MuJoCo."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
try:
    import tkinter as tk
except ImportError:
    tk = None
import mujoco
import mujoco.viewer
import glfw
from g1_piano.simulate.mujoco_player import (G1MujocoMapping, apply_q, initialize_data,
    clip_playback_range, interpolate_q, load_raw_npz, seek_seconds)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path); p.add_argument("--model", type=Path, default=Path("assets/robots/unitree_g1/xmls/scene_g1.xml"))
    p.add_argument("--joint-key", default="q"); p.add_argument("--time-key", default="timestamp_monotonic")
    p.add_argument("--sampling", choices=("nearest", "linear"), default="nearest"); p.add_argument("--start", type=float, default=0.0); p.add_argument("--end", type=float)
    p.add_argument("--speed", type=float, default=1.0); p.add_argument("--loop", action="store_true")
    args = p.parse_args()
    raw = load_raw_npz(args.npz, args.joint_key, args.time_key)
    try: start, end = clip_playback_range(raw.duration, args.start, args.end)
    except ValueError as exc: p.error(str(exc))
    model = mujoco.MjModel.from_xml_path(str(args.model)); data = mujoco.MjData(model); initialize_data(model, data)
    mapping = G1MujocoMapping.from_model(model); mapping.print()
    playback, playing, speed = start, True, args.speed; last = time.monotonic()
    state = {"playing": playing, "speed": speed, "playback": playback, "quit": False}
    root = status = scale = None
    def seek_from_gui(value: str) -> None:
        nonlocal playback
        playback = float(value)
    if tk is not None:
        try:
            root = tk.Tk(); root.title("G1 NPZ MuJoCo Player")
            status = tk.StringVar(); tk.Label(root, textvariable=status, width=42).pack(padx=8, pady=4)
            scale = tk.Scale(root, from_=start, to=end, orient=tk.HORIZONTAL, length=360,
                             resolution=0.001, command=seek_from_gui)
            scale.pack(padx=8); root.update()
        except tk.TclError:
            root = status = scale = None
    def on_key(key: int) -> None:
        nonlocal playback
        if key == glfw.KEY_SPACE: state["playing"] = not state["playing"]
        elif key == glfw.KEY_LEFT: playback = max(start, raw.times[max(0, interpolate_q(raw.times, raw.q, playback)[1] - 1)])
        elif key == glfw.KEY_RIGHT: playback = min(end, raw.times[min(len(raw.times) - 1, interpolate_q(raw.times, raw.q, playback)[1] + 1)])
        elif key == glfw.KEY_J: playback = seek_seconds(playback, -1.0, start, end)
        elif key == glfw.KEY_L: playback = seek_seconds(playback, 1.0, start, end)
        elif key in (glfw.KEY_HOME, glfw.KEY_R): playback = start
        elif key == glfw.KEY_END: playback = end
        elif key == glfw.KEY_LEFT_BRACKET: state["speed"] = max(0.25, state["speed"] / 2.0)
        elif key == glfw.KEY_RIGHT_BRACKET: state["speed"] = min(4.0, state["speed"] * 2.0)
        elif key == glfw.KEY_ESCAPE: state["quit"] = True
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            now = time.monotonic(); dt = now - last; last = now
            if state["quit"]: break
            if state["playing"]: playback += dt * state["speed"]
            if playback > end:
                if args.loop: playback = start
                else: playback = end; playing = False
            q_render, idx = interpolate_q(raw.times, raw.q, playback, args.sampling); apply_q(model, data, mapping, q_render); viewer.sync()
            if root is not None:
                state["playback"] = playback
                status.set(f"{'播放' if state['playing'] else '暂停'} | {playback:.3f}/{end:.3f}s | sample {idx}/{len(raw.q)-1} | {state['speed']:.2f}x")
                scale.set(playback); root.update()
            time.sleep(0.001)
    if root is not None: root.destroy()

if __name__ == "__main__": main()
