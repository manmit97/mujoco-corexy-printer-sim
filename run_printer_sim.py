import os
import time
import numpy as np
try:
    import mujoco
    import mujoco.viewer
except ImportError:
    raise ImportError("Please install mujoco using: pip install mujoco")

# 1. MJCF XML Model with Spool, Red Filament, and Layered 5cm Test Cube
PRINTER_MJCF = """
<mujoco model="corexy_3d_printer_growth">
    <compiler angle="radian" inertiafromgeom="true"/>
    <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" iterations="50"/>

    <default>
        <joint damping="5.0" armature="0.01"/>
        <geom friction="0.6 0.005 0.0001" contype="1" conaffinity="1"/>
        <motor ctrlrange="-50 50" gear="1"/>
    </default>

    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.1 0.1" rgb2="0.2 0.2 0.2" width="512" height="512"/>
        <material name="workplate" texture="grid" texrepeat="5 5" specular="0.1" shininess="0.1"/>
        <material name="aluminum" rgba="0.8 0.8 0.8 1"/>
        <material name="dark_plastic" rgba="0.15 0.15 0.15 1"/>
        <material name="hotend_metal" rgba="0.9 0.5 0.1 1"/>
        <material name="red_filament" rgba="0.9 0.1 0.1 1" specular="0.5"/>
    </asset>

    <worldbody>
        <!-- Lighting & Camera -->
        <light pos="0 0 1.5" dir="0 0 -1" ambient="0.4 0.4 0.4"/>
        <camera name="printer_cam" pos="0.5 -0.6 0.4" xyaxes="1 0.8 0 -0.3 0.4 0.9"/>

        <!-- Printer Enclosure Frame -->
        <body name="frame" pos="0 0 0">
            <geom type="box" size="0.22 0.22 0.25" pos="0 0 0.25" material="dark_plastic" rgba="0.15 0.15 0.15 0.15" contype="0" conaffinity="0" group="2"/>
            
            <!-- Spool Holder & Filament Spool -->
            <body name="spool_holder" pos="-0.15 0 0.4">
                <geom type="cylinder" size="0.01 0.08" pos="0 0 0" euler="0 1.5708 0" material="aluminum"/>
                <geom type="cylinder" size="0.07 0.03" pos="0 0 0" euler="0 1.5708 0" material="red_filament"/>
            </body>

            <!-- Filament Feed Strand (Static visual link from spool to toolhead) -->
            <geom type="cylinder" size="0.0015 0.1" pos="-0.07 0 0.35" euler="0.3 0.5 0" material="red_filament" contype="0" conaffinity="0"/>

            <!-- Build Plate (Z Axis) -->
            <body name="bed" pos="0 0 0.05">
                <joint name="bed_z" type="slide" axis="0 0 1" range="0 0.2" damping="20.0"/>
                <geom type="box" size="0.11 0.11 0.005" material="aluminum" mass="0.5"/>
                <geom type="box" size="0.1 0.1 0.001" pos="0 0 0.005" material="workplate"/>

                <!-- 5 cm Test Cube - Pre-allocated layers for dynamic growth -->
                <!-- Layer 1 (0 to 1 cm) -->
                <body name="cube_layer_1" pos="0 0 0.01">
                    <geom name="geom_layer_1" type="box" size="0.025 0.025 0.005" material="red_filament" rgba="0.9 0.1 0.1 0"/>
                </body>
                <!-- Layer 2 (1 to 2 cm) -->
                <body name="cube_layer_2" pos="0 0 0.02">
                    <geom name="geom_layer_2" type="box" size="0.025 0.025 0.005" material="red_filament" rgba="0.9 0.1 0.1 0"/>
                </body>
                <!-- Layer 3 (2 to 3 cm) -->
                <body name="cube_layer_3" pos="0 0 0.03">
                    <geom name="geom_layer_3" type="box" size="0.025 0.025 0.005" material="red_filament" rgba="0.9 0.1 0.1 0"/>
                </body>
                <!-- Layer 4 (3 to 4 cm) -->
                <body name="cube_layer_4" pos="0 0 0.04">
                    <geom name="geom_layer_4" type="box" size="0.025 0.025 0.005" material="red_filament" rgba="0.9 0.1 0.1 0"/>
                </body>
                <!-- Layer 5 (4 to 5 cm) -->
                <body name="cube_layer_5" pos="0 0 0.05">
                    <geom name="geom_layer_5" type="box" size="0.025 0.025 0.005" material="red_filament" rgba="0.9 0.1 0.1 0"/>
                </body>
            </body>

            <!-- CoreXY Gantry - Y Beam Assembly -->
            <body name="gantry_y" pos="0 0 0.2">
                <joint name="gantry_y_joint" type="slide" axis="0 1 0" range="-0.11 0.11"/>
                <geom type="box" size="0.12 0.01 0.01" material="aluminum" mass="0.3"/>

                <!-- Toolhead X Carriage -->
                <body name="toolhead" pos="0 0 0">
                    <joint name="toolhead_x_joint" type="slide" axis="1 0 0" range="-0.11 0.11"/>
                    <geom type="box" size="0.02 0.02 0.03" pos="0 0 -0.015" material="dark_plastic" mass="0.2"/>
                    <!-- Nozzle Tip -->
                    <geom type="cylinder" size="0.002 0.01" pos="0 0 -0.035" material="hotend_metal"/>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <position name="pos_x" joint="toolhead_x_joint" kp="500" dampratio="1"/>
        <position name="pos_y" joint="gantry_y_joint" kp="500" dampratio="1"/>
        <position name="pos_z" joint="bed_z" kp="2000" dampratio="1"/>
    </actuator>
</mujoco>
"""

class GrowingPrintSimulator:
    def __init__(self, model_xml):
        print("Compiling MuJoCo model with spool and growth layers...")
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)
        
        self.act_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_x")
        self.act_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_y")
        self.act_z = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pos_z")
        
        # Collect geom IDs for the 5 cube layers to dynamically alter their opacity
        self.layer_geom_ids = []
        for i in range(1, 6):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"geom_layer_{i}")
            self.layer_geom_ids.append(gid)

        # Set initial positions
        self.data.ctrl[self.act_x] = 0.0
        self.data.ctrl[self.act_y] = 0.0
        self.data.ctrl[self.act_z] = 0.05
        mujoco.mj_forward(self.model, self.data)

    def run_simulation(self):
        waypoints = [
            (-0.025, -0.025, 0.06), # Approach layer 1
            (-0.025, -0.025, 0.06), # Print Layer 1
            (0.025, -0.025, 0.06),
            (0.025, 0.025, 0.06),
            (-0.025, 0.025, 0.06),
            (-0.025, -0.025, 0.07), # Approach layer 2
            (0.025, -0.025, 0.07), # Print Layer 2
            (0.025, 0.025, 0.07),
            (-0.025, 0.025, 0.07),
            (-0.025, -0.025, 0.08), # Approach layer 3
            (0.025, -0.025, 0.08), # Print Layer 3
            (0.025, 0.025, 0.08),
            (-0.025, 0.025, 0.08),
            (-0.025, -0.025, 0.09), # Layer 4
            (0.025, -0.025, 0.09),
            (0.025, 0.025, 0.09),
            (-0.025, 0.025, 0.09),
            (-0.025, -0.025, 0.10), # Layer 5
            (0.025, -0.025, 0.10),
            (0.025, 0.025, 0.10),
            (-0.025, 0.025, 0.10),
            (0.0, 0.0, 0.15)       # Finish / Head park
        ]

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            wp_index = 0
            target_x, target_y, target_z = waypoints[0]
            
            print("\nRunning Growing 3D Print Simulation...")
            print("Watch the red layers sequentially appear as the nozzle moves upwards!")

            while viewer.is_running():
                step_start = time.time()

                current_x = self.data.ctrl[self.act_x]
                current_y = self.data.ctrl[self.act_y]
                current_z = self.data.ctrl[self.act_z]

                # Move smoothly towards target waypoint
                alpha = 0.04
                self.data.ctrl[self.act_x] += (target_x - current_x) * alpha
                self.data.ctrl[self.act_y] += (target_y - current_y) * alpha
                self.data.ctrl[self.act_z] += (target_z - current_z) * alpha

                # Dynamically reveal layers based on simulation progress/height
                active_layers = min(5, max(1, int((target_z - 0.05) / 0.01)))
                for idx, gid in enumerate(self.layer_geom_ids):
                    if idx < active_layers:
                        # Make layer visible (Alpha = 1.0)
                        self.model.geom_rgba[gid][3] = 1.0
                    else:
                        # Make layer invisible (Alpha = 0.0)
                        self.model.geom_rgba[gid][3] = 0.0

                dist = np.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
                if dist < 0.003:
                    wp_index = (wp_index + 1) % len(waypoints)
                    target_x, target_y, target_z = waypoints[wp_index]

                mujoco.mj_step(self.model, self.data)
                viewer.sync()

                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

if __name__ == "__main__":
    printer_sim = GrowingPrintSimulator(PRINTER_MJCF)
    printer_sim.run_simulation()
