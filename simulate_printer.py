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
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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

# Extrusion trace rendering
FILAMENT_RADIUS  = 0.0003   # visual radius of deposited line (0.3 mm)
FILAMENT_RGBA    = np.array([0.1, 0.55, 0.92, 0.92])
MAX_TRACE_GEOMS  = 8000     # cap to avoid OOM on huge prints


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

    def __init__(self, max_geoms: int = MAX_TRACE_GEOMS):
        self.max_geoms = max_geoms
        self._segments: list[tuple[np.ndarray, np.ndarray]] = []  # (p0, p1)
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
        if dist < 0.0003:       # don't add micro-segments
            return
        if len(self._segments) < self.max_geoms:
            self._segments.append((self._last_pos.copy(), pos.copy()))
        self._last_pos = pos.copy()

    def end_segment(self):
        self._last_pos = None

    def render(self, scene: mujoco.MjvScene):
        """Inject capsule geoms into the MuJoCo visualisation scene."""
        n_avail = scene.maxgeom - scene.ngeom
        n_draw = min(len(self._segments), n_avail)
        for i in range(n_draw):
            p0, p1 = self._segments[i]
            g = scene.geoms[scene.ngeom]

            # Capsule from p0 to p1
            midpoint = 0.5 * (p0 + p1)
            diff = p1 - p0
            length = np.linalg.norm(diff)
            if length < 1e-7:
                continue

            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            g.size[0] = FILAMENT_RADIUS
            g.size[1] = length * 0.5   # half-length for capsule
            g.size[2] = 0.0

            # Position at midpoint
            g.pos[:] = midpoint

            # Orientation: align capsule Z-axis with the segment direction
            direction = diff / length
            # Build rotation matrix
            z_axis = direction
            # Choose a non-parallel vector for cross product
            up = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(z_axis, up)) > 0.99:
                up = np.array([1.0, 0.0, 0.0])
            x_axis = np.cross(up, z_axis)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)

            rot = np.zeros((3, 3))
            rot[0, :] = x_axis
            rot[1, :] = y_axis
            rot[2, :] = z_axis
            g.mat[:] = rot

            g.rgba[:] = FILAMENT_RGBA
            g.emission = 0.15
            g.category = mujoco.mjtCatBit.mjCAT_DECOR
            g.dataid = -1
            g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
            g.objid = -1
            g.texid = -1
            g.texuniform = 0
            g.texcoord = 0
            g.segid = -1
            g.modelrbound = 0

            scene.ngeom += 1

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
        self.joint_z_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bed_z")
        
        self.layer_geom_ids = []
        for i in range(1, 6):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"print_layer_{i}")
            if gid >= 0:
                self.layer_geom_ids.append(gid)

        # ── Parse G-code & generate waypoints ──
        if gcode_path is not None:
            gcmoves = parse_gcode(gcode_path)
            self.waypoints = gcode_to_waypoints(gcmoves)
            print(f"[GCode] Parsed {len(gcmoves)} commands → {len(self.waypoints)} waypoints")
        else:
            self.waypoints = self._demo_waypoints()
            print(f"[Demo] Using built-in demo path with {len(self.waypoints)} waypoints")

        # ── State ──
        self.current_wp_idx = 0
        self.move_time = 0.0
        self.current_profile: Optional[TrapezoidSegment] = None
        self.move_start_pos = np.zeros(3)
        self.move_end_pos = np.zeros(3)
        self.trace = ExtrusionTrace()
        self.is_extruding = False
        self._finished = False

        # Initialise simulation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    # ── Built-in demo path (used when no G-code file is given) ──────
    @staticmethod
    def _demo_waypoints() -> list[Waypoint]:
        """Generate a spirograph-like demo path to showcase the printer."""
        wps = []
        # Home
        wps.append(Waypoint(0, 0, 0, MAX_VELOCITY_XY, False))
        # Lower bed for first layer
        wps.append(Waypoint(0, 0, -0.002, MAX_VELOCITY_Z, False))

        # Draw a star pattern
        n_points = 7
        outer_r = 0.08   # 80 mm
        inner_r = 0.03   # 30 mm
        for i in range(n_points * 2 + 1):
            angle = i * math.pi / n_points
            r = outer_r if i % 2 == 0 else inner_r
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            wps.append(Waypoint(x, y, -0.002, 0.10, True))

        # Concentric circles (3 layers)
        for layer in range(3):
            z = -(0.002 + layer * 0.002)
            for r in [0.04, 0.06, 0.08, 0.10]:
                n_seg = 60
                for i in range(n_seg + 1):
                    angle = 2 * math.pi * i / n_seg
                    x = r * math.cos(angle)
                    y = r * math.sin(angle)
                    wps.append(Waypoint(x, y, z, 0.12, True))
                # Retract between circles
                wps.append(Waypoint(
                    r * math.cos(0), r * math.sin(0), z, 0.05, False))

        # Return home
        wps.append(Waypoint(0, 0, -0.01, MAX_VELOCITY_XY, False))
        wps.append(Waypoint(0, 0, 0, MAX_VELOCITY_Z, False))
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
        if abs(delta[2]) > 0 and abs(delta[0]) < 1e-6 and abs(delta[1]) < 1e-6:
            accel = MAX_ACCEL_Z
        else:
            accel = MAX_ACCEL_XY

        self.current_profile = plan_trapezoid(distance, wp.feedrate, accel)
        self.move_time = 0.0

        # Extrusion state
        self.is_extruding = wp.extrude
        if self.is_extruding:
            nozzle_pos = self._nozzle_world_pos()
            self.trace.begin_segment(nozzle_pos)
        else:
            self.trace.end_segment()

    def _nozzle_world_pos(self) -> np.ndarray:
        """Get the nozzle tip position in world coordinates from the sensor."""
        adr = self.model.sensor_adr[self.sens_nozzle]
        return self.data.sensordata[adr:adr+3].copy()

    def _step_motion(self):
        """Advance the current move by one timestep."""
        if self._finished or self.current_profile is None:
            return

        profile = self.current_profile
        self.move_time += self.dt * self.realtime_factor

        if self.move_time >= profile.total_time:
            # Move complete — snap to target
            self.data.ctrl[self.act_x] = self.move_end_pos[0]
            self.data.ctrl[self.act_y] = self.move_end_pos[1]
            self.data.ctrl[self.act_z] = self.move_end_pos[2]

            if self.is_extruding:
                nozzle_pos = self._nozzle_world_pos()
                self.trace.add_point(nozzle_pos)

            self.current_wp_idx += 1
            self._begin_move()
            return

        # Interpolate along profile
        frac = profile.position_at(self.move_time) / max(profile.distance, 1e-12)
        frac = np.clip(frac, 0.0, 1.0)
        target = self.move_start_pos + frac * (self.move_end_pos - self.move_start_pos)

        self.data.ctrl[self.act_x] = target[0]
        self.data.ctrl[self.act_y] = target[1]
        self.data.ctrl[self.act_z] = target[2]

        # Record extrusion trace
        if self.is_extruding:
            nozzle_pos = self._nozzle_world_pos()
            self.trace.add_point(nozzle_pos)

        # Update layer visibility based on Z-position of the bed
        if hasattr(self, 'joint_z_id') and self.joint_z_id >= 0:
            current_z = -self.data.qpos[self.model.jnt_qposadr[self.joint_z_id]]
            for i, gid in enumerate(self.layer_geom_ids):
                layer_start = i * 0.01
                layer_end = (i + 1) * 0.01
                if current_z <= layer_start:
                    alpha = 0.0
                elif current_z >= layer_end:
                    alpha = 1.0
                else:
                    alpha = (current_z - layer_start) / 0.01
                self.model.geom_rgba[gid, 3] = alpha

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
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
                viewer.cam.distance = 0.65
                viewer.cam.lookat[:] = [0.0, 0.0, 0.20]

                while viewer.is_running():
                    t_loop_start = time.perf_counter()

                    # Advance simulation
                    self._step_motion()
                    mujoco.mj_step(self.model, self.data)
                    step_count += 1

                    # Inject extrusion trace into the scene
                    viewer.user_scn.ngeom = 0   # clear previous user geoms
                    self.trace.render(viewer.user_scn)

                    # --- Render Dynamic Filament Strand ---
                    # Draw a capsule connecting spool exit to toolhead inlet
                    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom and self.site_spool_exit >= 0 and self.site_toolhead_inlet >= 0:
                        p0 = self.data.site_xpos[self.site_spool_exit]
                        p1 = self.data.site_xpos[self.site_toolhead_inlet]
                        diff = p1 - p0
                        length = np.linalg.norm(diff)
                        if length > 1e-7:
                            g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                            g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                            g.size[0] = 0.0015   # 1.5mm radius
                            g.size[1] = length * 0.5
                            g.size[2] = 0.0
                            g.pos[:] = 0.5 * (p0 + p1)
                            
                            z_axis = diff / length
                            up = np.array([0.0, 0.0, 1.0])
                            if abs(np.dot(z_axis, up)) > 0.99:
                                up = np.array([1.0, 0.0, 0.0])
                            x_axis = np.cross(up, z_axis)
                            x_axis /= np.linalg.norm(x_axis)
                            y_axis = np.cross(z_axis, x_axis)
                            
                            rot = np.zeros((3, 3))
                            rot[0, :] = x_axis
                            rot[1, :] = y_axis
                            rot[2, :] = z_axis
                            g.mat[:] = rot
                            
                            g.rgba[:] = np.array([0.9, 0.1, 0.1, 0.85]) # matching strand mat
                            g.emission = 0.2
                            g.category = mujoco.mjtCatBit.mjCAT_DECOR
                            g.dataid = -1
                            g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                            g.objid = -1
                            g.texid = -1
                            g.texuniform = 0
                            g.texcoord = 0
                            g.segid = -1
                            g.modelrbound = 0
                            
                            viewer.user_scn.ngeom += 1

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
                            self.trace.render(viewer.user_scn)
                            
                            # Keep rendering the dynamic strand while finished
                            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom and self.site_spool_exit >= 0 and self.site_toolhead_inlet >= 0:
                                p0 = self.data.site_xpos[self.site_spool_exit]
                                p1 = self.data.site_xpos[self.site_toolhead_inlet]
                                diff = p1 - p0
                                length = np.linalg.norm(diff)
                                if length > 1e-7:
                                    g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                                    g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                                    g.size[0] = 0.0015
                                    g.size[1] = length * 0.5
                                    g.size[2] = 0.0
                                    g.pos[:] = 0.5 * (p0 + p1)
                                    z_axis = diff / length
                                    up = np.array([0.0, 0.0, 1.0])
                                    if abs(np.dot(z_axis, up)) > 0.99:
                                        up = np.array([1.0, 0.0, 0.0])
                                    x_axis = np.cross(up, z_axis)
                                    x_axis /= np.linalg.norm(x_axis)
                                    y_axis = np.cross(z_axis, x_axis)
                                    rot = np.zeros((3, 3))
                                    rot[0, :] = x_axis
                                    rot[1, :] = y_axis
                                    rot[2, :] = z_axis
                                    g.mat[:] = rot
                                    g.rgba[:] = np.array([0.9, 0.1, 0.1, 0.85])
                                    g.emission = 0.2
                                    g.category = mujoco.mjtCatBit.mjCAT_DECOR
                                    g.dataid = -1
                                    g.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
                                    g.objid = -1
                                    g.texid = -1
                                    g.texuniform = 0
                                    g.texcoord = 0
                                    g.segid = -1
                                    g.modelrbound = 0
                                    viewer.user_scn.ngeom += 1

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
