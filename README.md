# MuJoCo CoreXY FDM 3D Printer Simulation

A Python-based simulation of a CoreXY FDM 3D printer using the MuJoCo physics engine. This tool parses real G-code files, calculates trapezoidal velocity profiles, and realistically simulates the printer's movements and material extrusion in a 3D environment.

## Features
- **G-Code Parsing:** Reads and interprets actual G-code files (e.g. `demo_cube.gcode`).
- **Physics Simulation:** Uses MuJoCo to accurately simulate the CoreXY kinematics, joints, and build plate movements.
- **Velocity Profiling:** Implements trapezoidal velocity planning to simulate realistic acceleration, cruising, and deceleration phases of the printhead.
- **Live Extrusion Rendering:** Dynamically generates visual geometries to display the filament being deposited on the build plate during the simulation.

## Requirements
- `mujoco` (Python bindings)
- `numpy`
- Python 3.12+ 

## Installation
Install the necessary dependencies using pip:
```bash
pip install mujoco numpy
```

## Usage
Run the simulator using the provided `mjpython` interpreter. **This is required on macOS** to ensure the 3D viewer window runs on the main application thread correctly.

```bash
mjpython simulate_printer.py
```
*(Note: If you are not on macOS, standard `python simulate_printer.py` may also work depending on your environment).*

## File Structure
- `simulate_printer.py`: The main simulation script containing the G-code parser, kinematic planner, and the MuJoCo simulation loop.
- `printer.xml`: The MuJoCo XML configuration (MJCF) detailing the 3D printer's mechanics, joints, physical properties, and visual assets.
- `demo_cube.gcode`: A sample G-code file used to demonstrate the simulation in action.
