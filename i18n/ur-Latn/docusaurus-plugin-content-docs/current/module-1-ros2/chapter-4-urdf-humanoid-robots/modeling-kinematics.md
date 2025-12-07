---
title: "Humanoid Kinematics Modeling"
description: "Humanoids ke liye kinematic chains banana, forward aur inverse kinematics basics, joint hierarchy"
sidebar_position: 3
keywords: [kinematics, humanoid, forward kinematics, inverse kinematics, joint hierarchy]
---

# Humanoid Kinematics Modeling

Humanoid robots complex kinematic structures rakhti hain jo human movement patterns ko mimic karti hain. Yeh section batata hai kaise in structures ko URDF mein model karna hai, jismein kinematic chains, forward aur inverse kinematics concepts, aur proper joint hierarchy organization included hain.

## Kinematic Chains for Humanoids

A kinematic chain rigid bodies (links) ki ek series hai jo joints ke dwara connected hoti hain. Humanoid robots mein, ye chains typically human anatomy ko follow karti hain:

- **Torso chain**: Base → Torso → Head
- **Arm chains**: Torso → Upper Arm → Lower Arm → Hand
- **Leg chains**: Base → Thigh → Shin → Foot

### Basic Humanoid Kinematic Structure

```xml
<?xml version="1.0"?>
<robot name="humanoid_kinematics">
  <!-- Materials -->
  <material name="skin">
    <color rgba="0.9 0.7 0.5 1.0"/>
  </material>
  <material name="metal">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>

  <!-- Base/Pelvis Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.25 0.1"/>
      </geometry>
      <material name="metal"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.25 0.1"/>
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
    <origin xyz="0.0 0.0 0.05"/>
  </joint>

  <link name="torso">
    <visual>
      <geometry>
        <cylinder radius="0.12" length="0.5"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.12" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="15.0"/>
      <inertia ixx="0.3" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.3"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="5.0" velocity="2.0"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>
</robot>
```

## Forward aur Inverse Kinematics Basics

### Forward Kinematics

Forward kinematics end effector (jaise hand ya foot) ke position aur orientation ko calculate karti hai diye gaye joint angles ke basis par. URDF mein, yeh ROS ke kinematics libraries ke dwara handle kiya jata hai, lekin model design ke liye concept ko samajhna important hai.

```xml
<!-- Example: Right arm forward kinematics chain -->
<joint name="right_shoulder_yaw" type="revolute">
  <parent link="torso"/>
  <child link="right_upper_arm"/>
  <origin xyz="-0.05 -0.15 0.2" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.57" upper="1.57" effort="20.0" velocity="2.0"/>
</joint>

<link name="right_upper_arm">
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
    <mass value="1.5"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>

<joint name="right_elbow" type="revolute">
  <parent link="right_upper_arm"/>
  <child link="right_lower_arm"/>
  <origin xyz="0.0 0.0 -0.3" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="0" upper="3.14" effort="15.0" velocity="2.0"/>
</joint>

<link name="right_lower_arm">
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
    <mass value="1.0"/>
    <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
  </inertial>
</link>
```

### Inverse Kinematics Considerations

Jab aap apna URDF design kar rahe hote hain, to consider karen kaise yeh inverse kinematics solvers ke sath kaam karega:

1. **Joint Limits**: Appropriate limits set karen jo physical capabilities ko match kare
2. **Chain Structure**: Proper parent-child relationships maintain karen
3. **Redundancy**: Humanoid robots mein often redundant degrees of freedom hote hain

## Joint Hierarchy (Torso → Legs → Feet, Torso → Arms → Hands)

### Leg Chain Example

```xml
<!-- Left Leg Chain -->
<joint name="left_hip_yaw" type="revolute">
  <parent link="base_link"/>
  <child link="left_thigh"/>
  <origin xyz="0.0 -0.125 -0.05" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <limit lower="-0.5" upper="0.5" effort="50.0" velocity="1.0"/>
</joint>

<link name="left_thigh">
  <visual>
    <geometry>
      <cylinder radius="0.07" length="0.4"/>
    </geometry>
    <material name="metal"/>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.07" length="0.4"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="3.0"/>
    <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
  </inertial>
</link>

<joint name="left_knee" type="revolute">
  <parent link="left_thigh"/>
  <child link="left_shin"/>
  <origin xyz="0.0 0.0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="0" upper="2.5" effort="40.0" velocity="1.0"/>
</joint>

<link name="left_shin">
  <visual>
    <geometry>
      <cylinder radius="0.06" length="0.4"/>
    </geometry>
    <material name="metal"/>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.06" length="0.4"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="2.5"/>
    <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.04"/>
  </inertial>
</link>

<joint name="left_ankle" type="revolute">
  <parent link="left_shin"/>
  <child link="left_foot"/>
  <origin xyz="0.0 0.0 -0.4" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.5" upper="0.5" effort="20.0" velocity="1.0"/>
</joint>

<link name="left_foot">
  <visual>
    <geometry>
      <box size="0.2 0.1 0.05"/>
    </geometry>
    <material name="metal"/>
  </visual>
  <collision>
    <geometry>
      <box size="0.2 0.1 0.05"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
  </inertial>
</link>
```

### Arm Chain Example

```xml
<!-- Right Arm Chain -->
<joint name="right_shoulder_pitch" type="revolute">
  <parent link="torso"/>
  <child link="right_shoulder"/>
  <origin xyz="-0.05 -0.15 0.2" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-1.57" upper="1.57" effort="20.0" velocity="2.0"/>
</joint>

<link name="right_shoulder">
  <visual>
    <geometry>
      <cylinder radius="0.06" length="0.1"/>
    </geometry>
    <material name="metal"/>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.06" length="0.1"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.5"/>
    <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
  </inertial>
</link>

<joint name="right_shoulder_roll" type="revolute">
  <parent link="right_shoulder"/>
  <child link="right_upper_arm"/>
  <origin xyz="0.0 0.0 -0.1" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-2.0" upper="1.0" effort="20.0" velocity="2.0"/>
</joint>

<joint name="right_elbow_pitch" type="revolute">
  <parent link="right_upper_arm"/>
  <child link="right_lower_arm"/>
  <origin xyz="0.0 0.0 -0.3" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-2.0" upper="0.5" effort="15.0" velocity="2.0"/>
</joint>

<link name="right_lower_arm">
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
    <mass value="1.0"/>
    <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
  </inertial>
</link>

<joint name="right_wrist_pitch" type="revolute">
  <parent link="right_lower_arm"/>
  <child link="right_hand"/>
  <origin xyz="0.0 0.0 -0.25" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="-1.0" upper="1.0" effort="5.0" velocity="3.0"/>
</joint>

<link name="right_hand">
  <visual>
    <geometry>
      <box size="0.1 0.08 0.05"/>
    </geometry>
    <material name="skin"/>
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
```

## Step-by-Step Humanoid Model Building

### Phase 1: Basic Skeleton

Core structure se shuru karen:

```xml
<?xml version="1.0"?>
<robot name="step_by_step_humanoid">
  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
  </joint>

  <link name="torso">
    <inertial>
      <mass value="20.0"/>
      <inertia ixx="2.0" ixy="0.0" ixz="0.0" iyy="2.0" iyz="0.0" izz="2.0"/>
    </inertial>
  </link>
</robot>
```

### Phase 2: Add Limbs

```xml
<!-- Add legs -->
<joint name="left_hip" type="revolute">
  <parent link="base_link"/>
  <child link="left_thigh"/>
  <origin xyz="0.0 -0.1 -0.1"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.0" upper="1.0" effort="100.0" velocity="1.0"/>
</joint>

<link name="left_thigh">
  <inertial>
    <mass value="5.0"/>
    <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.5"/>
  </inertial>
</link>

<joint name="right_hip" type="revolute">
  <parent link="base_link"/>
  <child link="right_thigh"/>
  <origin xyz="0.0 0.1 -0.1"/>
  <axis xyz="0 0 1"/>
  <limit lower="-1.0" upper="1.0" effort="100.0" velocity="1.0"/>
</joint>

<link name="right_thigh">
  <inertial>
    <mass value="5.0"/>
    <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.5"/>
  </inertial>
</link>
```

### Phase 3: Complete Structure

```xml
<!-- Add knees and ankles -->
<joint name="left_knee" type="revolute">
  <parent link="left_thigh"/>
  <child link="left_shin"/>
  <origin xyz="0.0 0.0 -0.4"/>
  <axis xyz="0 1 0"/>
  <limit lower="0" upper="2.0" effort="80.0" velocity="1.0"/>
</joint>

<link name="left_shin">
  <inertial>
    <mass value="4.0"/>
    <inertia ixx="0.4" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.4"/>
  </inertial>
</link>

<joint name="left_ankle" type="revolute">
  <parent link="left_shin"/>
  <child link="left_foot"/>
  <origin xyz="0.0 0.0 -0.4"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.5" upper="0.5" effort="40.0" velocity="1.0"/>
</joint>

<link name="left_foot">
  <inertial>
    <mass value="2.0"/>
    <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.2" iyz="0.0" izz="0.2"/>
  </inertial>
</link>
```

## Practical Example: Simple Bipedal Robot

Yahan ek complete, simple bipedal robot URDF hai jo proper kinematic structure ko demonstrate karta hai:

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

  <!-- Base/Pelvis -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.1"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.2 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0.0 0.0 0.05"/>
  </joint>

  <link name="torso">
    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.5"/>
      </geometry>
      <material name="white"/>
    </visual>
    <collision>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="15.0"/>
      <inertia ixx="0.3" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.3"/>
    </inertial>
  </link>

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.5"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="5.0" velocity="1.0"/>
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

  <!-- Left Leg -->
  <joint name="left_hip_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh"/>
    <origin xyz="0.0 -0.1 -0.05"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.3" upper="0.3" effort="100.0" velocity="1.0"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="left_knee" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0.0 0.0 -0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="80.0" velocity="1.0"/>
  </joint>

  <link name="left_shin">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="4.0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.04"/>
    </inertial>
  </link>

  <joint name="left_ankle" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0.0 0.0 -0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="40.0" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="blue"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Right Leg (symmetric to left) -->
  <joint name="right_hip_yaw" type="revolute">
    <parent link="base_link"/>
    <child link="right_thigh"/>
    <origin xyz="0.0 0.1 -0.05"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.3" upper="0.3" effort="100.0" velocity="1.0"/>
  </joint>

  <link name="right_thigh">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

  <joint name="right_knee" type="revolute">
    <parent link="right_thigh"/>
    <child link="right_shin"/>
    <origin xyz="0.0 0.0 -0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="80.0" velocity="1.0"/>
  </joint>

  <link name="right_shin">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="4.0"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.04"/>
    </inertial>
  </link>

  <joint name="right_ankle" type="revolute">
    <parent link="right_shin"/>
    <child link="right_foot"/>
    <origin xyz="0.0 0.0 -0.4"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="40.0" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
      <material name="red"/>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.15 0.08 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>
</robot>
```

## Visualization aur Debugging Kinematic Chains

### Checking the Kinematic Structure

ROS tools ka istemal karke apne kinematic chains ko visualize aur debug karen:

```bash
# URDF ko RViz mein visualize karen
ros2 run rviz2 rviz2

# Kinematic tree ka graph generate karen
urdf_to_graphiz your_robot.urdf
# Yeh connection structure ko show karne ke liye .dot aur .ps files create karta hai

# URDF ko errors ke liye check karen
check_urdf your_robot.urdf
```

### TF Tree Visualization

Kinematic structure transform (TF) tree create karti hai jo visualize kiya ja sakta hai:

```bash
# TF tree visualize karen
ros2 run tf2_tools view_frames
```

Yeh aapke robot ke complete transform hierarchy ko show karne wala PDF create karega.

Agla section mein, hum different joint types aur unke constraints ko detail mein explore karenge, yahan establish kiye gaye kinematic foundation par build karke.