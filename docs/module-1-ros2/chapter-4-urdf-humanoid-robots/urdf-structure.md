---
title: "Understanding URDF XML Structure"
description: "Comprehensive guide to URDF XML format, links, joints, coordinate frames, and validation"
sidebar_position: 2
keywords: [urdf, xml, links, joints, coordinate frames, robot modeling]
---

# Understanding URDF XML Structure

The Unified Robot Description Format (URDF) is an XML-based format that describes robot models in ROS. Understanding its structure is fundamental to creating humanoid robot models. This section covers the essential components of URDF files, including links, joints, coordinate frames, and validation techniques.

## URDF File Format Overview

A URDF file is an XML document that describes a robot's physical and visual properties. The basic structure follows a tree-like hierarchy where the robot is composed of links connected by joints.

### Basic URDF Structure

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.5 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.15" length="0.6"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>
</robot>
```

### Key URDF Elements

1. **`<robot>`**: The root element that contains the entire robot description
2. **`<link>`**: Represents a rigid body with visual, collision, and inertial properties
3. **`<joint>`**: Defines the connection between two links with specific joint type and constraints
4. **`<material>`**: Defines visual materials (color, texture)
5. **`<gazebo>`**: Simulation-specific extensions (when using Gazebo)

## Links and Joints Explained

### Links

Links represent rigid bodies in the robot model. Each link can have multiple properties:

```xml
<link name="example_link">
  <!-- Visual properties for rendering -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <!-- Can be box, cylinder, sphere, or mesh -->
      <box size="0.1 0.1 0.1"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>

  <!-- Collision properties for physics simulation -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <box size="0.1 0.1 0.1"/>
    </geometry>
  </collision>

  <!-- Physical properties for dynamics -->
  <inertial>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <mass value="0.1"/>
    <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
  </inertial>
</link>
```

#### Link Properties

- **Visual**: Defines how the link appears in visualization tools
- **Collision**: Defines the collision geometry for physics simulation
- **Inertial**: Defines mass and inertial properties for dynamics simulation

### Joints

Joints define the connection between links and specify how they can move relative to each other:

```xml
<joint name="example_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  <dynamics damping="0.1" friction="0.0"/>
</joint>
```

#### Joint Types

- **`fixed`**: No movement allowed (welded connection)
- **`revolute`**: Single axis rotation with limits
- **`continuous`**: Single axis rotation without limits
- **`prismatic`**: Single axis translation with limits
- **`floating`**: 6DOF movement (rarely used)
- **`planar`**: Movement in a plane (rarely used)

## Coordinate Frames and Transforms

URDF uses a right-handed coordinate system where:
- X points forward
- Y points left
- Z points up

### Origin and Transformations

Each link and joint can have an origin that defines its position and orientation relative to its parent:

```xml
<!-- Origin with position and rotation -->
<origin xyz="0.1 0.2 0.3" rpy="0.1 0.2 0.3"/>

<!-- Alternative using quaternion -->
<origin xyz="0.1 0.2 0.3"
        rpy="0 0 1.57">  <!-- 90 degrees around Z axis -->
```

### Frame Conventions

In humanoid robots, common frame conventions include:

```xml
<!-- Torso frame - Z up, X forward -->
<link name="torso">
  <visual>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.1" length="0.6"/>
    </geometry>
  </visual>
</link>

<!-- Hip joint - connects torso to leg -->
<joint name="left_hip" type="revolute">
  <parent link="torso"/>
  <child link="left_thigh"/>
  <origin xyz="-0.05 -0.15 -0.3" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>  <!-- Rotation around Z axis -->
</joint>
```

## URDF Validation

Proper validation ensures your URDF files are correct and can be loaded by ROS tools.

### Basic Validation Commands

```bash
# Check if URDF is syntactically correct
check_urdf /path/to/your/robot.urdf

# Parse and display robot information
urdf_to_graphiz /path/to/your/robot.urdf
```

### Common Validation Issues

1. **Missing parent/child links**: Ensure all joint references exist
2. **Disconnected components**: All links must be connected through joints
3. **Invalid XML syntax**: Check for proper closing tags
4. **Missing inertial properties**: Required for dynamics simulation

### Validation Example Script

```python
#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import sys

def validate_urdf(file_path):
    """Basic URDF validation"""
    try:
        # Parse XML
        tree = ET.parse(file_path)
        root = tree.getroot()

        if root.tag != 'robot':
            print("Error: Root element must be 'robot'")
            return False

        robot_name = root.get('name')
        if not robot_name:
            print("Error: Robot must have a name")
            return False

        print(f"Robot name: {robot_name}")

        # Count links and joints
        links = root.findall('link')
        joints = root.findall('joint')

        print(f"Found {len(links)} links and {len(joints)} joints")

        # Check joint parent/child references
        link_names = {link.get('name') for link in links}

        for joint in joints:
            parent = joint.find('parent')
            child = joint.find('child')

            if parent is not None:
                parent_name = parent.get('link')
                if parent_name not in link_names:
                    print(f"Error: Joint {joint.get('name')} references non-existent parent {parent_name}")
                    return False

            if child is not None:
                child_name = child.get('link')
                if child_name not in link_names:
                    print(f"Error: Joint {joint.get('name')} references non-existent child {child_name}")
                    return False

        print("URDF validation passed!")
        return True

    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
        return False
    except Exception as e:
        print(f"Validation Error: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 validate_urdf.py <urdf_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    validate_urdf(file_path)
```

## Complete Simple URDF Example

Here's a complete, simple humanoid robot example:

```xml
<?xml version="1.0"?>
<robot name="simple_biped">
  <!-- Materials -->
  <material name="blue">
    <color rgba="0.0 0.0 1.0 1.0"/>
  </material>
  <material name="red">
    <color rgba="1.0 0.0 0.0 1.0"/>
  </material>
  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- Base link (pelvis) -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.25 0.15"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.25 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0.0 0.0 0.15"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.4"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="10.0" velocity="1.0"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>
</robot>
```

## Best Practices for URDF Structure

1. **Use consistent naming**: Follow a clear convention (e.g., `left_leg_upper`, `right_arm_lower`)
2. **Start simple**: Build your model incrementally from base to end effectors
3. **Validate frequently**: Use `check_urdf` regularly during development
4. **Document your model**: Add comments explaining complex joint arrangements
5. **Consider simulation**: Balance visual detail with collision performance

In the next section, we'll explore how to model complex humanoid kinematics, building on the structural foundation established here.