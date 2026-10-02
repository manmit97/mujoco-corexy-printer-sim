#!/usr/bin/env python3
"""
simulate_printer.py — MuJoCo CoreXY FDM Printer Digital Twin
=============================================================
Reads G-code, generates trapezoidal velocity profiles, drives the MJCF
model via position servos, and renders deposited material as capsule geoms
in the interactive viewer.

Usage:
    python simulate_printer.py                        # built-in demo path
    python simulate_printer.py demo_cube.gcode        # custom G-code file
    python simulate_printer.py --headless demo.gcode  # no viewer (CI test)

Requires: mujoco >= 3.0, numpy
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# #region agent log
_DEBUG_LOG_PATH = "/Users/manmitsingh/.cursor/debug-logs/debug-abcdc9.log"


DEBUG_RUN_ID = "post-fix"


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict, run_id: str | None = None):
    run_id = run_id or DEBUG_RUN_ID
    payload = {
        "sessionId": "abcdc9",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
# #endregion

import mujoco
import mujoco.viewer
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
MODEL_XML = Path(__file__).with_name("printer.xml")

# Printer physical limits (metres, m/s, m/s²)
BUILD_VOLUME_MM     = np.array([256.0, 256.0, 256.0])
MAX_VELOCITY_XY     = 0.500       # 500 mm/s
MAX_VELOCITY_Z      = 0.020       # 20 mm/s
MAX_ACCEL_XY        = 10.0        # 10 000 mm/s² = 10 m/s²
MAX_ACCEL_Z         = 0.500       # 500 mm/s²

# Coordinate transform: G-code origin is front-left corner of bed.
# In the MJCF the bed centre is at (0, 0) in XY.  Bed is 256×256 mm.
GCODE_ORIGIN_OFFSET = np.array([-0.128, -0.128, 0.0])  # metres

# Extrusion trace rendering (visual radius; slightly thicker than real 0.4 mm
# line width so deposited paths read clearly in the viewer)
FILAMENT_RADIUS  = 0.0008   # visual radius of deposited line (0.8 mm)
FILAMENT_RGBA    = np.array([0.88, 0.14, 0.14, 1.0])
# 1.75 mm filament from the spool into the extruder feed port
STRAND_RADIUS    = 0.0009
STRAND_RGBA      = np.array([0.90, 0.10, 0.10, 1.0])
MAX_TRACE_GEOMS  = 70000    # 100 mm / 0.04 mm cylinder is ~40k chords


# ═══════════════════════════════════════════════════════════════════════
#  G-CODE PARSER
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class GCodeMove:
    """A single parsed motion or command."""
    x: Optional[float] = None   # mm
    y: Optional[float] = None   # mm
    z: Optional[float] = None   # mm
    e: Optional[float] = None   # mm (extruder)
    f: Optional[float] = None   # mm/min (feedrate)
    cmd: str = ""               # e.g. "G1", "G28", "M104"
    line_no: int = 0
    # Temperature commands
    temp_hotend: Optional[float] = None
    temp_bed: Optional[float] = None


def parse_gcode(filepath: str | Path) -> list[GCodeMove]:
    """Lightweight G-code parser supporting G0/G1 moves + M104/M109/M140/M190."""
    moves: list[GCodeMove] = []
    with open(filepath, "r") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.split(";")[0].strip()      # strip comments
            if not line:
                continue
            tokens = line.upper().split()
            cmd = tokens[0]

            mv = GCodeMove(cmd=cmd, line_no=line_no)

            for tok in tokens[1:]:
                letter, value = tok[0], tok[1:]
                try:
                    val = float(value)
                except ValueError:
                    continue
                if letter == "X":
                    mv.x = val
                elif letter == "Y":
                    mv.y = val
                elif letter == "Z":
                    mv.z = val
                elif letter == "E":
                    mv.e = val
                elif letter == "F":
                    mv.f = val
                elif letter == "S":
                    if cmd in ("M104", "M109"):
                        mv.temp_hotend = val
                    elif cmd in ("M140", "M190"):
                        mv.temp_bed = val

            if cmd in ("G0", "G1", "G28", "M104", "M109", "M140", "M190"):
                moves.append(mv)
    return moves


# ═══════════════════════════════════════════════════════════════════════
#  TRAPEZOIDAL VELOCITY PROFILE
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class TrapezoidSegment:
    """Describes a trapezoidal (or triangular) velocity profile over a distance."""
    distance: float          # total distance (m)
    v_start: float = 0.0
    v_cruise: float = 0.0
    v_end: float = 0.0
    accel: float = 0.0
    t_accel: float = 0.0
    t_cruise: float = 0.0
    t_decel: float = 0.0

    @property
    def total_time(self) -> float:
        return self.t_accel + self.t_cruise + self.t_decel

    def position_at(self, t: float) -> float:
        """Return position along the profile at time t."""
        t = max(0.0, min(t, self.total_time))
        if t <= self.t_accel:
            # Acceleration phase
            return self.v_start * t + 0.5 * self.accel * t * t
        s_accel = self.v_start * self.t_accel + 0.5 * self.accel * self.t_accel**2
        t2 = t - self.t_accel
        if t2 <= self.t_cruise:
            # Cruise phase
            return s_accel + self.v_cruise * t2
        s_cruise = s_accel + self.v_cruise * self.t_cruise
        t3 = t2 - self.t_cruise
        # Deceleration phase
        return s_cruise + self.v_cruise * t3 - 0.5 * self.accel * t3 * t3


def plan_trapezoid(distance: float, v_max: float, accel: float) -> TrapezoidSegment:
    """Plan a trapezoidal velocity profile from rest to rest over *distance*."""
    if distance < 1e-9:
        return TrapezoidSegment(distance=0.0)

    # Time to accelerate to v_max
    t_acc = v_max / accel
    d_acc = 0.5 * accel * t_acc**2

    if 2 * d_acc > distance:
        # Triangular profile — can't reach v_max
        t_acc = math.sqrt(distance / accel)
        v_peak = accel * t_acc
        return TrapezoidSegment(
            distance=distance, v_start=0, v_cruise=v_peak, v_end=0,
            accel=accel, t_accel=t_acc, t_cruise=0.0, t_decel=t_acc,
        )
    else:
        d_cruise = distance - 2 * d_acc
        t_cruise = d_cruise / v_max
        return TrapezoidSegment(
            distance=distance, v_start=0, v_cruise=v_max, v_end=0,
            accel=accel, t_accel=t_acc, t_cruise=t_cruise, t_decel=t_acc,
        )


# ═══════════════════════════════════════════════════════════════════════
#  WAYPOINT SEQUENCER
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class Waypoint:
    """Absolute target position in MuJoCo joint-space (metres)."""
    x: float
    y: float
    z: float          # bed Z (negative = bed lowered)
    feedrate: float   # m/s
    extrude: bool     # True if E > 0 on this move
    accel: Optional[float] = None  # m/s²; None uses the axis default


@dataclass
class PrinterState:
    """Tracks the current modal G-code state."""
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    e_mm: float = 0.0
    feedrate_mmmin: float = 1500.0
    temp_hotend: float = 0.0
    temp_bed: float = 0.0


def gcode_to_waypoints(moves: list[GCodeMove]) -> list[Waypoint]:
    """Convert parsed G-code into a list of absolute waypoints in joint-space."""
    st = PrinterState()
    waypoints: list[Waypoint] = []

    for mv in moves:
        if mv.cmd == "G28":
            # Home — go to (0, 0, 0)
            st.x_mm, st.y_mm, st.z_mm = 0.0, 0.0, 0.0
            waypoints.append(Waypoint(x=0.0, y=0.0, z=0.0,
                                      feedrate=MAX_VELOCITY_XY, extrude=False))
            continue

        if mv.cmd in ("M104", "M109"):
            if mv.temp_hotend is not None:
                st.temp_hotend = mv.temp_hotend
            continue
        if mv.cmd in ("M140", "M190"):
            if mv.temp_bed is not None:
                st.temp_bed = mv.temp_bed
            continue

        if mv.cmd not in ("G0", "G1"):
            continue

        # Update feedrate
        if mv.f is not None:
            st.feedrate_mmmin = mv.f

        # Update position (absolute mode)
        if mv.x is not None:
            st.x_mm = mv.x
        if mv.y is not None:
            st.y_mm = mv.y
        if mv.z is not None:
            st.z_mm = mv.z
        has_extrusion = mv.e is not None and mv.e > 0.0
        if mv.e is not None:
            st.e_mm += mv.e

        # Convert mm → metres and apply origin offset to map G-code coords
        # into the MJCF joint space where (0,0) is bed centre.
        x_m = st.x_mm / 1000.0 + GCODE_ORIGIN_OFFSET[0]
        y_m = st.y_mm / 1000.0 + GCODE_ORIGIN_OFFSET[1]
        # Z in MJCF: bed moves DOWN, so nozzle-bed gap = z_mm means
        # bed_joint_pos = -(bed_home_z - z_mm*0.001).  We invert:
        # A Z of 0 mm in G-code means nozzle touching bed → bed at home (0).
        # Higher Z → bed descends → joint goes negative.
        z_joint = -st.z_mm / 1000.0   # negative = bed lower

        # Clamp to build volume
        x_m = np.clip(x_m, -0.128, 0.128)
        y_m = np.clip(y_m, -0.128, 0.128)
        z_joint = np.clip(z_joint, -0.246, 0.0)

        # Feedrate mm/min → m/s
        feed_ms = st.feedrate_mmmin / 60000.0
        # Separate XY / Z speed limits
        is_z_only = (mv.x is None and mv.y is None and mv.z is not None)
        if is_z_only:
            feed_ms = min(feed_ms, MAX_VELOCITY_Z)
        else:
            feed_ms = min(feed_ms, MAX_VELOCITY_XY)

        waypoints.append(Waypoint(x=x_m, y=y_m, z=z_joint,
                                  feedrate=feed_ms, extrude=has_extrusion))
    return waypoints


# ═══════════════════════════════════════════════════════════════════════
#  EXTRUSION TRACE RENDERER
# ═══════════════════════════════════════════════════════════════════════
class ExtrusionTrace:
    """Manages deposited-material capsule geoms in the MuJoCo scene."""

    def __init__(self, max_geoms: int = MAX_TRACE_GEOMS, min_step: float = 0.002):
        self.max_geoms = max_geoms
        self.min_step = min_step
        self._segments: list[tuple[np.ndarray, np.ndarray]] = []  # (p0, p1)
        self._z: list[float] = []
        self._last_pos: Optional[np.ndarray] = None

    def begin_segment(self, pos: np.ndarray):
        """Mark the start of a new extrusion move."""
        self._last_pos = pos.copy()

    def add_point(self, pos: np.ndarray):
        """Extend the current extrusion path."""
        if self._last_pos is None:
            self._last_pos = pos.copy()
            return
        dist = np.linalg.norm(pos - self._last_pos)
        if dist < self.min_step:  # coalesce micro-steps into visible beads
            return
        if len(self._segments) < self.max_geoms:
            self._segments.append((self._last_pos.copy(), pos.copy()))
            self._z.append(0.5 * (float(self._last_pos[2]) + float(pos[2])))
            # #region agent log
            if len(self._segments) in (1, 10, 100):
                _agent_log("B", "simulate_printer.py:add_point", "trace segment added", {
                    "n": len(self._segments),
                    "p0": self._last_pos.tolist(),
                    "p1": pos.tolist(),
                    "dist": float(dist),
                })
            # #endregion
        self._last_pos = pos.copy()

    def end_segment(self):
        self._last_pos = None

    def _visible_ids(self) -> np.ndarray:
        """Segment indices to draw.

        A 0.04 mm layer stack is far finer than the bead, so once the path is
        long only one ring per bead-width is drawn, plus the live top. Short
        paths (a single layer, a small G-code file) are drawn in full.
        """
        n = len(self._segments)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if n <= 2500 or len(self._z) != n:
            return np.arange(n, dtype=np.int64)
        z = np.asarray(self._z, dtype=np.float64)
        dz = float(z[-1]) - z
        pitch = 0.0012  # 1.2 mm, under the 1.6 mm visual bead
        phase = np.mod(dz, pitch)
        mask = (dz <= 0.0016) | (phase <= 8e-5)
        mask[-1] = True
        return np.flatnonzero(mask)

    def render(self, scene: mujoco.MjvScene, origin: np.ndarray, rot: np.ndarray):
        """Inject capsule geoms into the MuJoCo visualisation scene.

        Segments are stored in bed-local coordinates; *origin* / *rot* map
        them into the world so deposited material rides with the print bed.
        """
        n_avail = scene.maxgeom - scene.ngeom
        draw_ids = self._visible_ids()
        n_draw = min(len(draw_ids), n_avail)
        rgba = np.asarray(FILAMENT_RGBA, dtype=np.float32)
        world0 = None
        # #region agent log
        log_this = not hasattr(self, "_render_logged")
        # #endregion
        for i in draw_ids[:n_draw]:
            p0, p1 = self._segments[i]
            p0w = origin + rot @ p0
            p1w = origin + rot @ p1
            if world0 is None:
                world0 = p0w
            length = np.linalg.norm(p1w - p0w)
            if length < 1e-7:
                continue
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                rgba,
            )
            mujoco.mjv_connector(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                FILAMENT_RADIUS,
                p0w,
                p1w,
            )
            g.rgba[:] = FILAMENT_RGBA
            g.emission = 0.2
            g.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1
        # #region agent log
        if log_this:
            self._render_logged = True
            sample = None
            if self._segments:
                p0, p1 = self._segments[0]
                sample = {"p0_local": p0.tolist(), "p1_local": p1.tolist()}
            _agent_log("C", "simulate_printer.py:render", "user scene capacity", {
                "maxgeom": int(scene.maxgeom),
                "ngeom": int(scene.ngeom),
                "n_avail": int(n_avail),
                "n_segments": len(self._segments),
                "n_draw": int(n_draw),
                "sample": sample,
                "sample_world0": world0.tolist() if world0 is not None else None,
                "bed_origin": origin.tolist(),
            })
        # #endregion

    @property
    def count(self) -> int:
        return len(self._segments)


# ═══════════════════════════════════════════════════════════════════════
#  SIMULATION DRIVER
# ═══════════════════════════════════════════════════════════════════════
class PrinterSimulation:
    """Orchestrates model loading, waypoint execution, and rendering."""

    def __init__(self, model_path: str | Path, gcode_path: str | Path | None,
                 headless: bool = False, realtime_factor: float = 5.0):
        self.headless = headless
        self.realtime_factor = realtime_factor

        # ── Load model ──
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        # ── Actuator & sensor indices ──
        self.act_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_x")
        self.act_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_y")
        self.act_z = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_z")
        self.sens_nozzle = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "sens_nozzle_pos")
        self.site_nozzle = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "nozzle_tip_site")
        self.site_spool_exit = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "spool_exit_site")
        self.site_toolhead_inlet = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "toolhead_inlet_site")
        self.joint_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_z")
        self.joint_z_id_xml = self.joint_z_id
        self.joint_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_x")
        self.joint_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "joint_y")
        self.qadr_x = int(self.model.jnt_qposadr[self.joint_x_id])
        self.qadr_y = int(self.model.jnt_qposadr[self.joint_y_id])
        self.qadr_z = int(self.model.jnt_qposadr[self.joint_z_id])
        self.dof_x = int(self.model.jnt_dofadr[self.joint_x_id])
        self.dof_y = int(self.model.jnt_dofadr[self.joint_y_id])
        self.dof_z = int(self.model.jnt_dofadr[self.joint_z_id])
        self.body_bed = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bed_carriage")

        # #region agent log
        _agent_log("A", "simulate_printer.py:init", "joint lookup", {
            "joint_z_id_bed_z": int(self.joint_z_id),
            "joint_z_id_xml": int(self.joint_z_id_xml),
        })
        # #endregion
        if gcode_path is not None:
            gcmoves = parse_gcode(gcode_path)
            self.waypoints = gcode_to_waypoints(gcmoves)
            print(f"[GCode] Parsed {len(gcmoves)} commands → {len(self.waypoints)} waypoints")
        else:
            self.waypoints = self._demo_waypoints()
            print(f"[Demo] {len(self.waypoints)} waypoints")
        # #region agent log
        n_ext = sum(1 for w in self.waypoints if w.extrude)
        zs = [w.z for w in self.waypoints]
        _agent_log("A", "simulate_printer.py:waypoints", "waypoint extrusion stats", {
            "n_waypoints": len(self.waypoints),
            "n_extrude": n_ext,
            "z_min": float(min(zs) if zs else 0),
            "z_max": float(max(zs) if zs else 0),
            "gcode": str(gcode_path) if gcode_path else "demo",
        })
        # #endregion

        # ── State ──
        self.current_wp_idx = 0
        self.move_time = 0.0
        self.current_profile: Optional[TrapezoidSegment] = None
        self.move_start_pos = np.zeros(3)
        self.move_end_pos = np.zeros(3)
        # Chords on the cylinder are ~8 mm; keep one bead per chord.
        trace_step = 0.007 if gcode_path is None else 0.002
        self.trace = ExtrusionTrace(min_step=trace_step)
        self.is_extruding = False
        self._finished = False

        # Initialise simulation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    # ── Built-in demo path (used when no G-code file is given) ──────
    @staticmethod
    def _demo_waypoints() -> list[Waypoint]:
        """Single-wall cylinder, vase-mode spiral.

        Height 100 mm, layer pitch 0.04 mm (2500 layers). The head travels to
        the wall with extrusion off, then climbs one layer per revolution.
        Diameter is 40 mm so the tube sits clearly inside the 256 mm bed.
        """
        height = 0.100
        layer = 0.00004
        radius = 0.020
        n_layers = int(round(height / layer))
        n_seg = 16
        n_steps = (n_layers - 1) * n_seg
        feed = 0.40
        accel = 400.0
        print(f"[Demo] Cylinder Ø{radius * 2000:.0f} mm × {height * 1000:.0f} mm, "
              f"layer {layer * 1000:.2f} mm ({n_layers} layers)")

        wps: list[Waypoint] = [Waypoint(0, 0, 0, MAX_VELOCITY_XY, False)]
        for i in range(n_steps + 1):
            frac = i / n_steps
            z_height = layer + (height - layer) * frac
            ang = 2.0 * math.pi * i / n_seg
            x = radius * math.cos(ang)
            y = radius * math.sin(ang)
            z = -z_height
            if i == 0:
                wps.append(Waypoint(x, y, z, MAX_VELOCITY_XY, False))
            else:
                wps.append(Waypoint(x, y, z, feed, True, accel))
        # Park beside the finished tube. Leave the bed down so the 100 mm
        # wall stays visible under the nozzle.
        wps.append(Waypoint(0.06, 0.0, -height, MAX_VELOCITY_XY, False))
        return wps

    # ── Motion planning ────────────────────────────────────────────
    def _begin_move(self):
        """Start executing the next waypoint."""
        if self.current_wp_idx >= len(self.waypoints):
            self._finished = True
            return

        wp = self.waypoints[self.current_wp_idx]

        # Current joint positions
        self.move_start_pos = np.array([
            self.data.ctrl[self.act_x],
            self.data.ctrl[self.act_y],
            self.data.ctrl[self.act_z],
        ])
        self.move_end_pos = np.array([wp.x, wp.y, wp.z])

        delta = self.move_end_pos - self.move_start_pos
        distance = np.linalg.norm(delta)

        # Choose acceleration limit based on dominant axis
        if wp.accel is not None:
            accel = wp.accel
        elif abs(delta[2]) > 0 and abs(delta[0]) < 1e-6 and abs(delta[1]) < 1e-6:
            accel = MAX_ACCEL_Z
        else:
            accel = MAX_ACCEL_XY

        self.current_profile = plan_trapezoid(distance, wp.feedrate, accel)
        self.move_time = 0.0

        # Extrusion state
        self.is_extruding = wp.extrude
        if self.is_extruding:
            self.trace.begin_segment(self._deposit_pos())
        else:
            self.trace.end_segment()

    def _nozzle_world_pos(self) -> np.ndarray:
        """Get the nozzle tip position in world coordinates from the sensor."""
        adr = self.model.sensor_adr[self.sens_nozzle]
        return self.data.sensordata[adr:adr+3].copy()

    def _bed_pose(self) -> tuple[np.ndarray, np.ndarray]:
        origin = self.data.xpos[self.body_bed]
        rot = self.data.xmat[self.body_bed].reshape(3, 3)
        return origin, rot

    def _deposit_pos(self) -> np.ndarray:
        """Filament pose in the bed frame: nozzle XY, layer-height Z on the bed."""
        origin, rot = self._bed_pose()
        local = rot.T @ (self._nozzle_world_pos() - origin)
        print_height = 0.0
        if self.joint_z_id >= 0:
            print_height = -float(self.data.qpos[self.model.jnt_qposadr[self.joint_z_id]])
        # Bed surface is 4 mm above bed_carriage origin (print_bed top).
        local[2] = 0.004 + FILAMENT_RADIUS + print_height
        return local

    def _render_feed_filament(self, scene: mujoco.MjvScene) -> None:
        """1.75 mm strand from the spool into the extruder feed port.

        It leaves the spool rim, runs behind the frame at the toolhead's X,
        then drops in from the rear and stops at the port on the back of the
        extruder housing. The gantry beam covers the top of the housing, so
        the strand feeds the motor from behind that beam.
        """
        if self.site_spool_exit < 0 or self.site_toolhead_inlet < 0:
            return
        spool = self.data.site_xpos[self.site_spool_exit]
        inlet = self.data.site_xpos[self.site_toolhead_inlet]
        # Orange AMS spool → its feeder, back through the rear slot, then
        # down to the toolhead. The lower bend still enters the feed port
        # from behind the gantry beam.
        feeder = np.array([0.135, -0.050, 0.500])
        guide = np.array([0.135, 0.155, 0.470])
        drop = np.array([0.135, 0.165, 0.395])
        rail = np.array([float(inlet[0]), 0.168, 0.392])
        hover = np.array([float(inlet[0]), float(inlet[1]) + 0.055, 0.400])
        mouth = np.array([float(inlet[0]), float(inlet[1]) + 0.016, float(inlet[2])])
        # Bend stays behind the gantry beam and the housing, then runs
        # straight into the feed port.
        ctrl = np.array([float(inlet[0]), float(inlet[1]) + 0.038, 0.390])
        points = [spool, feeder, guide, drop, rail, hover]
        for t in np.linspace(0.0, 1.0, 6)[1:]:
            omt = 1.0 - t
            points.append(omt * omt * hover + 2.0 * omt * t * ctrl + t * t * mouth)
        points.append(inlet)

        rgba = np.asarray(STRAND_RGBA, dtype=np.float32)
        for p0, p1 in zip(points, points[1:]):
            if scene.ngeom >= scene.maxgeom:
                return
            if np.linalg.norm(p1 - p0) < 1e-6:
                continue
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                rgba,
            )
            mujoco.mjv_connector(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                STRAND_RADIUS,
                np.asarray(p0, dtype=np.float64),
                np.asarray(p1, dtype=np.float64),
            )
            g.rgba[:] = STRAND_RGBA
            g.emission = 0.15
            g.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1

    def _write_ctrl(self, target: np.ndarray) -> None:
        self.data.ctrl[self.act_x] = target[0]
        self.data.ctrl[self.act_y] = target[1]
        self.data.ctrl[self.act_z] = target[2]

    def _hold_command(self) -> None:
        """Put the carriages on the position command.

        Fast-forward steps the trajectory faster than the position servo can
        track. Planting qpos keeps the nozzle on the cylinder instead of
        lagging into a smaller, smeared path.
        """
        self.data.qpos[self.qadr_x] = self.data.ctrl[self.act_x]
        self.data.qpos[self.qadr_y] = self.data.ctrl[self.act_y]
        self.data.qpos[self.qadr_z] = self.data.ctrl[self.act_z]
        self.data.qvel[self.dof_x] = 0.0
        self.data.qvel[self.dof_y] = 0.0
        self.data.qvel[self.dof_z] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _step_motion(self):
        """Advance the current move by one timestep."""
        if self._finished or self.current_profile is None:
            return

        profile = self.current_profile
        self.move_time += self.dt * self.realtime_factor

        if self.move_time >= profile.total_time:
            # Move complete — snap to target
            self._write_ctrl(self.move_end_pos)
            self._hold_command()

            if self.is_extruding:
                self.trace.add_point(self._deposit_pos())

            self.current_wp_idx += 1
            self._begin_move()
            return

        # Interpolate along profile
        frac = profile.position_at(self.move_time) / max(profile.distance, 1e-12)
        frac = np.clip(frac, 0.0, 1.0)
        target = self.move_start_pos + frac * (self.move_end_pos - self.move_start_pos)

        self._write_ctrl(target)
        self._hold_command()

        # Record extrusion trace (bed-local so it rides with the printbed)
        if self.is_extruding:
            self.trace.add_point(self._deposit_pos())

        # #region agent log
        if self.current_wp_idx in (1, 5, 20) and not getattr(self, "_layer_logged", set()).intersection({self.current_wp_idx}):
            self._layer_logged = getattr(self, "_layer_logged", set())
            self._layer_logged.add(self.current_wp_idx)
            nozzle = self._nozzle_world_pos()
            bed_q = float(self.data.qpos[self.model.jnt_qposadr[self.joint_z_id_xml]]) if self.joint_z_id_xml >= 0 else None
            _agent_log("A", "simulate_printer.py:_step_motion", "nozzle vs bed", {
                "joint_z_id": int(self.joint_z_id),
                "bed_qpos": bed_q,
                "nozzle": nozzle.tolist(),
                "wp_idx": int(self.current_wp_idx),
                "is_extruding": bool(self.is_extruding),
                "trace_count": self.trace.count,
            })
        # #endregion

    # ── Main loop ──────────────────────────────────────────────────
    def run(self):
        """Execute the simulation with optional interactive viewer."""
        print("═" * 60)
        print("  MuJoCo CoreXY FDM Printer Simulation")
        print("  Model : printer.xml")
        print(f"  Waypts: {len(self.waypoints)}")
        print(f"  Mode  : {'Headless' if self.headless else 'Interactive Viewer'}")
        print(f"  Speed : {self.realtime_factor}× realtime")
        print("═" * 60)

        # Kick off first move
        self._begin_move()

        if self.headless:
            self._run_headless()
        else:
            self._run_viewer()

    def _run_headless(self):
        """Run without viewer — useful for CI or batch testing."""
        step_count = 0
        t_start = time.perf_counter()
        while not self._finished:
            self._step_motion()
            mujoco.mj_step(self.model, self.data)
            step_count += 1
            if step_count % 5000 == 0:
                wp = min(self.current_wp_idx, len(self.waypoints) - 1)
                pct = 100.0 * self.current_wp_idx / max(len(self.waypoints), 1)
                print(f"  [step {step_count:>8d}]  waypoint {wp}/{len(self.waypoints)}"
                      f"  ({pct:5.1f}%)  trace_segs={self.trace.count}")
        elapsed = time.perf_counter() - t_start
        print(f"\n✓ Headless simulation complete — {step_count} steps in {elapsed:.2f}s"
              f"  ({step_count/elapsed:.0f} steps/s)")
        print(f"  Extrusion trace segments: {self.trace.count}")
        # #region agent log
        sample = None
        if self.trace._segments:
            p0, p1 = self.trace._segments[0]
            sample = {"p0": p0.tolist(), "p1": p1.tolist()}
        scn = mujoco.MjvScene(self.model, maxgeom=max(2000, self.trace.count + 16))
        origin, rot = self._bed_pose()
        self.trace.render(scn, origin, rot)
        world0 = None
        if self.trace._segments:
            p0, _p1 = self.trace._segments[0]
            world0 = (origin + rot @ p0).tolist()
        _agent_log("A", "simulate_printer.py:_run_headless", "headless complete", {
            "steps": step_count,
            "trace_count": self.trace.count,
            "joint_z_id": int(self.joint_z_id),
            "sample_seg": sample,
            "render_ngeom": int(scn.ngeom),
            "bed_origin": origin.tolist(),
            "sample_world0": world0,
        })
        # #endregion

    def _run_viewer(self):
        """Run with interactive mujoco.viewer."""
        step_count = 0
        status_interval = 2.0     # seconds between status prints
        last_status = time.perf_counter()

        print("\n  Controls:")
        print("    • Mouse drag   — orbit / pan / zoom")
        print("    • Double-click — track body")
        print("    • Esc / ⌘Q     — quit")
        print("    • Space        — pause\n")

        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                # Configure camera for a nice default view
                viewer.cam.azimuth = 128
                viewer.cam.elevation = -18
                viewer.cam.distance = 1.15
                viewer.cam.lookat[:] = [0.0, 0.0, 0.30]
                # #region agent log
                _agent_log("C", "simulate_printer.py:_run_viewer", "viewer scene init", {
                    "maxgeom": int(viewer.user_scn.maxgeom),
                    "ngeom": int(viewer.user_scn.ngeom),
                })
                # #endregion

                while viewer.is_running():
                    t_loop_start = time.perf_counter()

                    # Advance simulation
                    self._step_motion()
                    mujoco.mj_step(self.model, self.data)
                    self._hold_command()
                    step_count += 1

                    # Inject extrusion trace and the spool-to-extruder strand.
                    viewer.user_scn.ngeom = 0
                    origin, rot = self._bed_pose()
                    self.trace.render(viewer.user_scn, origin, rot)
                    self._render_feed_filament(viewer.user_scn)

                    viewer.sync()

                    # Status readout
                    now = time.perf_counter()
                    if now - last_status > status_interval:
                        wp = min(self.current_wp_idx, len(self.waypoints) - 1)
                        pct = 100.0 * self.current_wp_idx / max(len(self.waypoints), 1)
                        nozzle = self._nozzle_world_pos()
                        print(f"  [wp {wp:>4d}/{len(self.waypoints)}]"
                              f"  {pct:5.1f}%"
                              f"  nozzle=({nozzle[0]:+.3f}, {nozzle[1]:+.3f}, {nozzle[2]:.4f})"
                              f"  traces={self.trace.count}"
                              f"  {'▓ EXTRUDING' if self.is_extruding else '░ travel'}")
                        last_status = now

                    if self._finished:
                        print("\n✓ Print complete! Viewer remains open — close window to exit.")
                        # Keep viewer alive after print finishes
                        while viewer.is_running():
                            viewer.user_scn.ngeom = 0
                            origin, rot = self._bed_pose()
                            self.trace.render(viewer.user_scn, origin, rot)
                            self._render_feed_filament(viewer.user_scn)
                            viewer.sync()
                            time.sleep(0.03)
                        break

                    # Pace to roughly realtime (accounting for sim speed factor)
                    elapsed = time.perf_counter() - t_loop_start
                    sleep_target = self.dt / self.realtime_factor - elapsed
                    if sleep_target > 0:
                        time.sleep(sleep_target)

        except KeyboardInterrupt:
            print("\n⏹  Interrupted by user.")
        except Exception as e:
            print(f"\n✗ Viewer error: {e}")
            raise

        print(f"  Total steps: {step_count},  trace segments: {self.trace.count}")


# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo CoreXY FDM 3D Printer Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simulate_printer.py                        # Built-in demo pattern
  python simulate_printer.py demo_cube.gcode        # Print G-code file
  python simulate_printer.py --headless test.gcode  # Headless (no GUI)
  python simulate_printer.py --speed 10             # 10× speed
        """,
    )
    parser.add_argument("gcode", nargs="?", default=None,
                        help="Path to a G-code file (optional; uses built-in demo if omitted)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without the interactive viewer")
    parser.add_argument("--speed", type=float, default=5.0,
                        help="Simulation speed multiplier (default: 5×)")
    parser.add_argument("--model", type=str, default=str(MODEL_XML),
                        help=f"Path to the MJCF printer model (default: {MODEL_XML.name})")
    args = parser.parse_args()

    # Validate paths
    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"✗ Model file not found: {model_path}")

    gcode_path = None
    if args.gcode:
        gcode_path = Path(args.gcode)
        if not gcode_path.exists():
            sys.exit(f"✗ G-code file not found: {gcode_path}")

    sim = PrinterSimulation(
        model_path=model_path,
        gcode_path=gcode_path,
        headless=args.headless,
        realtime_factor=args.speed,
    )
    sim.run()


if __name__ == "__main__":
    main()
