---
title: "URDF Fundamentals and Structure"
description: "Understanding URDF basics, XML structure, and fundamental robot modeling concepts"
sidebar_position: 1
keywords: [urdf, robot description, xml, links, joints, visualization]
---

# URDF Fundamentals and Structure

URDF (Unified Robot Description Format) is an XML-based format used to describe robot models in ROS. It defines the physical and visual properties of robots, including links, joints, and their relationships. Understanding URDF fundamentals is essential for creating accurate robot models for simulation and control.

## URDF XML Structure

### Basic URDF Document

```xml
<?xml version="1.0"?>
<robot name="simple_robot" xmlns:xacro="http://ros.org/wiki/xacro">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.6" radius="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.3 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="wheel_link">
    <visual>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder length="0.1" radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>
</robot>
```

## Link Elements

Links represent rigid bodies in the robot model. Each link contains visual, collision, and inertial properties.

### Link Structure

```xml
<link name="link_name">
  <!-- Visual properties for rendering -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Shape definition -->
    </geometry>
    <material name="material_name">
      <color rgba="1 1 1 1"/>
    </material>
  </visual>

  <!-- Collision properties for physics simulation -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Shape definition -->
    </geometry>
  </collision>

  <!-- Inertial properties for dynamics -->
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="1.0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
  </inertial>
</link>
```

### Visual and Collision Properties

```xml
<link name="visual_collision_example">
  <!-- Visual: How the link appears in RViz and simulators -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Multiple geometry types supported -->
      <!-- Box -->
      <box size="0.1 0.2 0.3"/>
      <!-- Cylinder -->
      <!-- <cylinder radius="0.1" length="0.2"/> -->
      <!-- Sphere -->
      <!-- <sphere radius="0.1"/> -->
      <!-- Mesh -->
      <!-- <mesh filename="package://my_robot/meshes/link.dae"/> -->
    </geometry>
    <material name="red">
      <color rgba="1 0 0 1"/>
      <!-- Or reference external material -->
      <!-- <texture filename="package://my_robot/materials/texture.png"/> -->
    </material>
  </visual>

  <!-- Collision: How the link interacts with physics -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Often simplified compared to visual for performance -->
      <box size="0.1 0.2 0.3"/>
    </geometry>
  </collision>
</link>
```

## Joint Elements

Joints define the relationship between links and specify how they can move relative to each other.

### Joint Types

```xml
<!-- Fixed joint (no movement) -->
<joint name="fixed_joint" type="fixed">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0.1 0 0" rpy="0 0 0"/>
</joint>

<!-- Revolute joint (rotational movement with limits) -->
<joint name="revolute_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
</joint>

<!-- Continuous joint (unlimited rotational movement) -->
<joint name="continuous_joint" type="continuous">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
</joint>

<!-- Prismatic joint (linear movement with limits) -->
<joint name="prismatic_joint" type="prismatic">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="100" velocity="1"/>
</joint>

<!-- Planar joint (movement in a plane) -->
<joint name="planar_joint" type="planar">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>

<!-- Floating joint (6DOF movement) -->
<joint name="floating_joint" type="floating">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
</joint>
```

## Inertial Properties

Inertial properties are crucial for dynamics simulation and control.

### Mass and Inertia Matrix

```xml
<link name="inertial_example">
  <inertial>
    <!-- Origin of the inertial reference frame relative to the link frame -->
    <origin xyz="0.01 0 0.02" rpy="0 0 0"/>

    <!-- Mass in kilograms -->
    <mass value="2.5"/>

    <!-- Inertia matrix (symmetric, only 6 values needed) -->
    <inertia
      ixx="0.01" ixy="0.0" ixz="0.001"
      iyy="0.02" iyz="0.002"
      izz="0.03"/>
  </inertial>
</link>
```

### Calculating Inertial Properties

For common shapes:

```xml
<!-- Solid cylinder (radius r, height h, mass m) -->
<!-- ixx = iyy = m*(3*r² + h²)/12, izz = m*r²/2 -->
<inertial>
  <mass value="1.0"/>
  <inertia ixx="0.0104" ixy="0.0" ixz="0.0" iyy="0.0104" iyz="0.0" izz="0.005"/>
</inertial>

<!-- Solid sphere (radius r, mass m) -->
<!-- ixx = iyy = izz = 2*m*r²/5 -->
<inertial>
  <mass value="1.0"/>
  <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.04"/>
</inertial>

<!-- Box (width w, depth d, height h, mass m) -->
<!-- ixx = m*(d² + h²)/12, iyy = m*(w² + h²)/12, izz = m*(w² + d²)/12 -->
<inertial>
  <mass value="1.0"/>
  <inertia ixx="0.0833" ixy="0.0" ixz="0.0" iyy="0.0278" iyz="0.0" izz="0.0556"/>
</inertial>
```

## Materials and Colors

Materials define the visual appearance of links.

### Material Definitions

```xml
<robot name="material_example">
  <!-- Define materials at the top level -->
  <material name="red">
    <color rgba="1 0 0 1"/>
  </material>

  <material name="blue">
    <color rgba="0 0 1 1"/>
  </material>

  <material name="green">
    <color rgba="0 1 0 1"/>
  </material>

  <!-- Use materials in links -->
  <link name="colored_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="red"/>
    </visual>
  </link>
</robot>
```

## URDF Validation and Tools

### Validation Commands

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Parse and display URDF info
urdf_to_graphiz /path/to/robot.urdf

# Display robot model in RViz
ros2 run rviz2 rviz2
```

### Python URDF Parsing

```python
import rclpy
from rclpy.node import Node
from urdf_parser_py.urdf import URDF
import os

class URDFValidatorNode(Node):
    def __init__(self):
        super().__init__('urdf_validator_node')

        # Validate URDF file
        self.validate_urdf('/path/to/robot.urdf')

    def validate_urdf(self, urdf_path):
        """Validate URDF file and extract information"""
        try:
            # Load URDF
            robot = URDF.from_xml_file(urdf_path)

            self.get_logger().info(f'Robot name: {robot.name}')
            self.get_logger().info(f'Number of links: {len(robot.links)}')
            self.get_logger().info(f'Number of joints: {len(robot.joints)}')

            # Print link information
            for link in robot.links:
                self.get_logger().info(f'Link: {link.name}')
                if link.visual:
                    self.get_logger().info(f'  Visual: {link.visual.geometry.type}')
                if link.collision:
                    self.get_logger().info(f'  Collision: {link.collision.geometry.type}')
                if link.inertial:
                    self.get_logger().info(f'  Mass: {link.inertial.mass}')

            # Print joint information
            for joint in robot.joints:
                self.get_logger().info(f'Joint: {joint.name}')
                self.get_logger().info(f'  Type: {joint.type}')
                self.get_logger().info(f'  Parent: {joint.parent}')
                self.get_logger().info(f'  Child: {joint.child}')

        except Exception as e:
            self.get_logger().error(f'URDF validation error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = URDFValidatorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices

### 1. Naming Conventions

```xml
<!-- Good naming conventions -->
<link name="base_link"/>           <!-- Base link should be named base_link -->
<link name="left_wheel_link"/>     <!-- Descriptive names with suffix _link -->
<link name="right_arm_link"/>      <!-- Include side/position in name -->

<joint name="base_to_left_wheel"/> <!-- Descriptive joint names -->
<joint name="torso_to_head"/>      <!-- Include parent and child in name -->
```

### 2. Coordinate Frame Conventions

```xml
<!-- Follow ROS coordinate frame conventions -->
<!-- X: forward, Y: left, Z: up -->
<origin xyz="0.1 0 0.2" rpy="0 0 1.57"/>
<!-- This moves 10cm forward, 20cm up, and rotates 90 degrees around Z -->
```

### 3. Unit Consistency

```xml
<!-- Always use consistent units (meters, radians, kilograms) -->
<link name="consistent_units">
  <visual>
    <geometry>
      <box size="0.1 0.2 0.3"/>  <!-- All in meters -->
    </geometry>
  </visual>
  <inertial>
    <mass value="2.5"/>          <!-- In kilograms -->
    <inertia ixx="0.01" ... />   <!-- In kg*m² -->
  </inertial>
</link>
```

## Common URDF Issues and Solutions

### 1. Missing Base Link

```xml
<!-- Always ensure one link has no parent (base link) -->
<robot name="robot_with_base">
  <!-- This is the base link (no joint connects to it as child) -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </visual>
  </link>

  <!-- This joint makes base_link the root -->
  <joint name="base_to_sensor" type="fixed">
    <parent link="base_link"/>
    <child link="sensor_link"/>
  </joint>

  <link name="sensor_link">
    <visual>
      <geometry>
        <cylinder radius="0.02" length="0.05"/>
      </geometry>
    </visual>
  </link>
</robot>
```

### 2. Invalid Joint Chains

```xml
<!-- Ensure all links are connected in a valid tree structure -->
<!-- No loops in the joint graph -->
<robot name="valid_tree">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>

  <joint name="base_to_link1" type="fixed">
    <parent link="base_link"/>
    <child link="link1"/>
  </joint>

  <joint name="link1_to_link2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
  </joint>
</robot>
```

URDF fundamentals provide the foundation for robot modeling in ROS, enabling accurate representation of robot geometry, kinematics, and dynamics for simulation, visualization, and control applications.