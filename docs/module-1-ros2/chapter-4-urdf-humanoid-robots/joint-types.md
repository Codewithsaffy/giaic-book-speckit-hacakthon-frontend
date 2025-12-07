---
title: "Joint Types and Constraints"
description: "Understanding revolute, prismatic, fixed, continuous joints, and joint limits for humanoid robots"
sidebar_position: 4
keywords: [joint types, revolute, prismatic, fixed, continuous, joint limits, constraints]
---

# Joint Types and Constraints

Joints in URDF define how links can move relative to each other. For humanoid robots, selecting the appropriate joint types and setting proper constraints is crucial for realistic movement and functionality. This section covers all joint types with specific examples for humanoid applications.

## Joint Type Overview

### 1. Revolute Joints (Most Common in Humanoids)

Revolute joints allow rotation around a single axis with defined limits. These are the most common joint type in humanoid robots, representing most human joints like elbows, knees, and shoulders.

```xml
<joint name="elbow_joint" type="revolute">
  <parent link="upper_arm"/>
  <child link="lower_arm"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>  <!-- Rotate around X axis -->
  <limit lower="0" upper="3.14" effort="15.0" velocity="2.0"/>
  <dynamics damping="0.1" friction="0.01"/>
</joint>
```

**Characteristics:**
- Single axis of rotation
- Limited by upper and lower bounds
- Most realistic for human joints
- Common applications: elbows, knees, wrists, finger joints

### 2. Continuous Joints

Continuous joints allow unlimited rotation around a single axis. These are useful for joints that can rotate continuously, like a neck turning around its axis.

```xml
<joint name="neck_rotation" type="continuous">
  <parent link="torso"/>
  <child link="head"/>
  <origin xyz="0 0 0.4" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>  <!-- Rotate around Z axis -->
  <dynamics damping="0.05" friction="0.01"/>
</joint>
```

**Characteristics:**
- Unlimited rotation around single axis
- No position limits (effort and velocity still limited)
- Less common in humanoids (humans have limited neck rotation)
- More common in mobile robot wheels

### 3. Prismatic Joints

Prismatic joints allow linear translation along a single axis. These are less common in humanoid robots but can be useful for telescoping mechanisms.

```xml
<joint name="telescoping_joint" type="prismatic">
  <parent link="base"/>
  <child link="extension"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>  <!-- Move along Z axis -->
  <limit lower="0.0" upper="0.2" effort="50.0" velocity="0.5"/>
  <dynamics damping="0.1" friction="0.05"/>
</joint>
```

**Characteristics:**
- Linear movement along single axis
- Limited by upper and lower bounds
- Rare in humanoids but useful for some mechanisms
- Common applications: sliding mechanisms, adjustable components

### 4. Fixed Joints

Fixed joints create rigid connections with no movement allowed. These are used to connect links that should move together as a single rigid body.

```xml
<joint name="head_to_camera" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0.05" rpy="0 0 0"/>
</joint>
```

**Characteristics:**
- No movement allowed
- Used to attach sensors, tools, or create compound links
- Most common joint type after revolute
- Essential for mounting accessories

### 5. Floating and Planar Joints

These joint types allow multiple degrees of freedom but are rarely used in humanoid robots as they're difficult to control and don't represent human joints well.

```xml
<!-- Floating joint (6DOF) - rarely used -->
<joint name="floating_joint" type="floating">
  <parent link="world"/>
  <child link="floating_link"/>
</joint>

<!-- Planar joint (3DOF in a plane) - rarely used -->
<joint name="planar_joint" type="planar">
  <parent link="base"/>
  <child link="planar_link"/>
</joint>
```

## Joint Limits and Dynamics

### Joint Limits

Joint limits define the range of motion for revolute and prismatic joints:

```xml
<joint name="shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="upper_arm"/>
  <origin xyz="-0.05 -0.15 0.2" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <!-- Joint limits -->
  <limit
    lower="-2.0"           <!-- -114 degrees -->
    upper="1.5"            <!-- 86 degrees -->
    effort="20.0"          <!-- Max effort (N or Nm) -->
    velocity="2.0"/>       <!-- Max velocity (rad/s or m/s) -->
  <dynamics damping="0.2" friction="0.02"/>
</joint>
```

### Common Joint Limit Values for Humanoids

**Upper Body Joints:**
- Shoulder pitch: -2.0 to 1.5 rad (-114° to 86°)
- Shoulder roll: -1.5 to 1.0 rad (-86° to 57°)
- Shoulder yaw: -1.57 to 1.57 rad (±90°)
- Elbow: 0 to 2.5 rad (0° to 143°)
- Wrist pitch: -1.0 to 1.0 rad (±57°)
- Wrist roll: -1.57 to 1.57 rad (±90°)

**Lower Body Joints:**
- Hip pitch: -1.57 to 0.7 rad (-90° to 40°)
- Hip roll: -0.5 to 0.5 rad (±29°)
- Hip yaw: -0.5 to 0.5 rad (±29°)
- Knee: 0 to 2.5 rad (0° to 143°)
- Ankle pitch: -0.5 to 0.5 rad (±29°)
- Ankle roll: -0.3 to 0.3 rad (±17°)

### Dynamics Properties

Dynamics properties affect how joints behave in simulation:

```xml
<joint name="realistic_joint" type="revolute">
  <parent link="link1"/>
  <child link="link2"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.0" upper="1.0" effort="10.0" velocity="1.0"/>
  <!-- Dynamics properties -->
  <dynamics
    damping="0.1"     <!-- Resistance to motion (like viscosity) -->
    friction="0.01"/> <!-- Static friction coefficient -->
</joint>
```

## Humanoid-Specific Joint Configurations

### Shoulder Complex (3 DOF)

Humanoid shoulders typically need 3 degrees of freedom to replicate human shoulder movement:

```xml
<!-- Shoulder complex for right arm -->
<joint name="right_shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="right_shoulder"/>
  <origin xyz="-0.05 -0.15 0.25" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-2.0" upper="1.5" effort="25.0" velocity="1.5"/>
</joint>

<joint name="right_shoulder_roll" type="revolute">
  <parent link="right_shoulder"/>
  <child link="right_upper_arm"/>
  <origin xyz="0 0 -0.1" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.5" upper="1.0" effort="20.0" velocity="1.5"/>
</joint>

<joint name="right_shoulder_yaw" type="revolute">
  <parent link="right_upper_arm"/>
  <child link="right_upper_arm_yaw"/>
  <origin xyz="0 0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="15.0" velocity="1.0"/>
</joint>
```

### Hip Complex (3 DOF)

Humanoid hips typically need 3 degrees of freedom for proper locomotion:

```xml
<!-- Hip complex for left leg -->
<joint name="left_hip_yaw" type="revolute">
  <parent link="base_link"/>
  <child link="left_thigh_yaw"/>
  <origin xyz="0.0 -0.1 -0.05" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-0.5" upper="0.5" effort="50.0" velocity="1.0"/>
</joint>

<joint name="left_hip_roll" type="revolute">
  <parent link="left_thigh_yaw"/>
  <child link="left_thigh_roll"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-0.5" upper="0.5" effort="50.0" velocity="1.0"/>
</joint>

<joint name="left_hip_pitch" type="revolute">
  <parent link="left_thigh_roll"/>
  <child link="left_thigh"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="0.7" effort="80.0" velocity="1.0"/>
</joint>
```

### Wrist Complex (2-3 DOF)

Wrist joints typically have 2-3 degrees of freedom:

```xml
<!-- Right wrist with 2 DOF -->
<joint name="right_wrist_pitch" type="revolute">
  <parent link="right_lower_arm"/>
  <child link="right_wrist"/>
  <origin xyz="0 0 -0.25" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-1.0" upper="1.0" effort="8.0" velocity="2.0"/>
</joint>

<joint name="right_wrist_yaw" type="revolute">
  <parent link="right_wrist"/>
  <child link="right_hand"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.0" upper="1.0" effort="6.0" velocity="2.0"/>
</joint>
```

## Advanced Joint Configurations

### Mimic Joints

Mimic joints move in relation to another joint, useful for symmetric movements:

```xml
<joint name="left_eyelid" type="revolute">
  <parent link="head"/>
  <child link="left_eyelid_link"/>
  <origin xyz="0.05 0.05 0.05" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="1.0" velocity="1.0"/>
</joint>

<joint name="right_eyelid" type="revolute">
  <parent link="head"/>
  <child link="right_eyelid_link"/>
  <origin xyz="0.05 -0.05 0.05" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="1.0" velocity="1.0"/>
  <!-- This joint mimics the left eyelid -->
  <mimic joint="left_eyelid" multiplier="1.0" offset="0.0"/>
</joint>
```

### Transmission Elements

For simulation and control integration, you may need to define transmissions:

```xml
<transmission name="right_elbow_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="right_elbow">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="right_elbow_motor">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Joint Validation and Testing

### Checking Joint Limits

Create a simple validation script to check joint limits:

```python
#!/usr/bin/env python3
import xml.etree.ElementTree as ET
import math

def validate_joint_limits(urdf_file):
    """Validate joint limits for humanoid robot"""
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    joints = root.findall('.//joint')

    for joint in joints:
        joint_type = joint.get('type')
        joint_name = joint.get('name')

        if joint_type in ['revolute', 'prismatic']:
            limit_elem = joint.find('limit')
            if limit_elem is not None:
                lower = float(limit_elem.get('lower', 0))
                upper = float(limit_elem.get('upper', 0))

                # Check if limits are reasonable
                if lower >= upper:
                    print(f"ERROR: Joint {joint_name} has invalid limits: {lower} >= {upper}")

                # Check if limits are within human-like ranges
                if joint_type == 'revolute':
                    # Convert to degrees for human-readable output
                    lower_deg = math.degrees(lower)
                    upper_deg = math.degrees(upper)

                    print(f"Joint {joint_name}: {lower_deg:.1f}° to {upper_deg:.1f}°")

                    # Example: Check if elbow has reasonable limits
                    if 'elbow' in joint_name.lower():
                        if lower < -0.1 or upper > 3.0:  # Too wide for elbow
                            print(f"  WARNING: Elbow joint {joint_name} has unusual limits")
            else:
                if joint_type != 'fixed':
                    print(f"WARNING: Joint {joint_name} has no limits defined")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python3 validate_joints.py <urdf_file>")
        sys.exit(1)

    validate_joint_limits(sys.argv[1])
```

## Complete Joint Configuration Example

Here's a complete example showing various joint types in a humanoid arm:

```xml
<?xml version="1.0"?>
<robot name="arm_with_all_joint_types">
  <material name="metal">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Shoulder: Revolute joint (pitch) -->
  <joint name="shoulder_pitch" type="revolute">
    <parent link="base_link"/>
    <child link="upper_arm"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="30.0" velocity="1.0"/>
  </joint>

  <link name="upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="metal"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <!-- Elbow: Revolute joint -->
  <joint name="elbow" type="revolute">
    <parent link="upper_arm"/>
    <child link="lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.5" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="lower_arm">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
      <material name="metal"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Wrist: Continuous joint (for unlimited rotation) -->
  <joint name="wrist_rotation" type="continuous">
    <parent link="lower_arm"/>
    <child link="wrist"/>
    <origin xyz="0 0 -0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
  </joint>

  <link name="wrist">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>

  <!-- Hand attachment: Fixed joint -->
  <joint name="wrist_to_hand" type="fixed">
    <parent link="wrist"/>
    <child link="hand"/>
    <origin xyz="0 0 -0.05" rpy="0 0 0"/>
  </joint>

  <link name="hand">
    <visual>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
      <material name="metal"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0005" iyz="0.0" izz="0.0005"/>
    </inertial>
  </link>
</robot>
```

## Best Practices for Joint Configuration

1. **Use realistic limits**: Base joint limits on human anatomical data
2. **Consider dynamics**: Add appropriate damping and friction values
3. **Plan for control**: Ensure joint configuration supports your control algorithms
4. **Validate early**: Check joint ranges before building complex models
5. **Document assumptions**: Note why specific joint types were chosen

In the next section, we'll explore visual and collision geometries, which work in conjunction with joints to create complete robot models.