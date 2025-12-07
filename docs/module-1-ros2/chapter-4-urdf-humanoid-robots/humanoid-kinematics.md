---
title: "Humanoid Robot Kinematics"
description: "Modeling complex humanoid joint structures and kinematic chains"
sidebar_position: 2
keywords: [urdf, humanoid, kinematics, joints, chains, limbs, mobility]
---

# Humanoid Robot Kinematics

Humanoid robot kinematics modeling involves creating complex joint structures that mimic human-like movement patterns. This section covers the design and implementation of kinematic chains for humanoid robots, including arms, legs, torso, and head mechanisms.

## Humanoid Kinematic Structure

### Basic Humanoid Skeleton

```xml
<?xml version="1.0"?>
<robot name="humanoid_robot">
  <!-- Base/Root link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <box size="0.3 0.2 0.1"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1.0"/>
      </material>
    </visual>
  </link>

  <!-- Torso -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.05"/>
  </joint>

  <link name="torso">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0 0 0.2"/>
      <inertia ixx="0.5" ixy="0.0" ixz="0.0" iyy="0.5" iyz="0.0" izz="0.2"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.2"/>
      <geometry>
        <box size="0.2 0.15 0.4"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
  </link>

  <!-- Head -->
  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.4"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10" velocity="2"/>
  </joint>

  <link name="head">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <sphere radius="0.08"/>
      </geometry>
      <material name="skin">
        <color rgba="1 0.8 0.6 1"/>
      </material>
    </visual>
  </link>
</robot>
```

## Upper Body Kinematics

### Left Arm Structure

```xml
<!-- Left Shoulder -->
<joint name="torso_to_left_shoulder" type="revolute">
  <parent link="torso"/>
  <child link="left_shoulder"/>
  <origin xyz="0.1 0.1 0.2" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
</joint>

<link name="left_shoulder">
  <inertial>
    <mass value="0.5"/>
    <origin xyz="0 0 0.05"/>
    <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.0005"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0.05"/>
    <geometry>
      <cylinder radius="0.03" length="0.1"/>
    </geometry>
    <material name="arm_color">
      <color rgba="0.7 0.7 0.7 1"/>
    </material>
  </visual>
</link>

<!-- Left Upper Arm -->
<joint name="left_shoulder_to_upper_arm" type="revolute">
  <parent link="left_shoulder"/>
  <child link="left_upper_arm"/>
  <origin xyz="0 0 0.1"/>
  <axis xyz="1 0 0"/>
  <limit lower="-2.36" upper="1.57" effort="50" velocity="2"/>
</joint>

<link name="left_upper_arm">
  <inertial>
    <mass value="1.0"/>
    <origin xyz="0 0 0.15"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.002"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0.15"/>
    <geometry>
      <cylinder radius="0.04" length="0.3"/>
    </geometry>
    <material name="arm_color"/>
  </visual>
</link>

<!-- Left Elbow -->
<joint name="left_upper_arm_to_forearm" type="revolute">
  <parent link="left_upper_arm"/>
  <child link="left_forearm"/>
  <origin xyz="0 0 0.3"/>
  <axis xyz="1 0 0"/>
  <limit lower="-2.5" upper="0" effort="30" velocity="2"/>
</joint>

<link name="left_forearm">
  <inertial>
    <mass value="0.8"/>
    <origin xyz="0 0 0.1"/>
    <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.001"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0.1"/>
    <geometry>
      <cylinder radius="0.03" length="0.2"/>
    </geometry>
    <material name="arm_color"/>
  </visual>
</link>

<!-- Left Hand/Wrist -->
<joint name="left_forearm_to_hand" type="revolute">
  <parent link="left_forearm"/>
  <child link="left_hand"/>
  <origin xyz="0 0 0.2"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="20" velocity="3"/>
</joint>

<link name="left_hand">
  <inertial>
    <mass value="0.3"/>
    <origin xyz="0 0 0.03"/>
    <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0005" iyz="0.0" izz="0.0002"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0.03"/>
    <geometry>
      <box size="0.08 0.06 0.06"/>
    </geometry>
    <material name="hand_color">
      <color rgba="0.9 0.9 0.9 1"/>
    </material>
  </visual>
</link>
```

### Right Arm (Mirror of Left Arm)

```xml
<!-- Right Shoulder -->
<joint name="torso_to_right_shoulder" type="revolute">
  <parent link="torso"/>
  <child link="right_shoulder"/>
  <origin xyz="0.1 -0.1 0.2" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>
</joint>

<link name="right_shoulder">
  <inertial>
    <mass value="0.5"/>
    <origin xyz="0 0 0.05"/>
    <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.0005"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0.05"/>
    <geometry>
      <cylinder radius="0.03" length="0.1"/>
    </geometry>
    <material name="arm_color"/>
  </visual>
</link>

<!-- Continue with right arm joints and links similar to left arm -->
<joint name="right_shoulder_to_upper_arm" type="revolute">
  <parent link="right_shoulder"/>
  <child link="right_upper_arm"/>
  <origin xyz="0 0 0.1"/>
  <axis xyz="1 0 0"/>
  <limit lower="-1.57" upper="2.36" effort="50" velocity="2"/>
</joint>

<link name="right_upper_arm">
  <inertial>
    <mass value="1.0"/>
    <origin xyz="0 0 0.15"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.002"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0.15"/>
    <geometry>
      <cylinder radius="0.04" length="0.3"/>
    </geometry>
    <material name="arm_color"/>
  </visual>
</link>

<joint name="right_upper_arm_to_forearm" type="revolute">
  <parent link="right_upper_arm"/>
  <child link="right_forearm"/>
  <origin xyz="0 0 0.3"/>
  <axis xyz="1 0 0"/>
  <limit lower="-2.5" upper="0" effort="30" velocity="2"/>
</joint>

<link name="right_forearm">
  <inertial>
    <mass value="0.8"/>
    <origin xyz="0 0 0.1"/>
    <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.001"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0.1"/>
    <geometry>
      <cylinder radius="0.03" length="0.2"/>
    </geometry>
    <material name="arm_color"/>
  </visual>
</link>

<joint name="right_forearm_to_hand" type="revolute">
  <parent link="right_forearm"/>
  <child link="right_hand"/>
  <origin xyz="0 0 0.2"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="20" velocity="3"/>
</joint>

<link name="right_hand">
  <inertial>
    <mass value="0.3"/>
    <origin xyz="0 0 0.03"/>
    <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0005" iyz="0.0" izz="0.0002"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0.03"/>
    <geometry>
      <box size="0.08 0.06 0.06"/>
    </geometry>
    <material name="hand_color"/>
  </visual>
</link>
```

## Lower Body Kinematics

### Left Leg Structure

```xml
<!-- Left Hip -->
<joint name="torso_to_left_hip" type="revolute">
  <parent link="torso"/>
  <child link="left_hip"/>
  <origin xyz="-0.05 0.08 -0.1" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.785" upper="0.785" effort="100" velocity="1"/>
</joint>

<link name="left_hip">
  <inertial>
    <mass value="1.0"/>
    <origin xyz="0 0 0.05"/>
    <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.002"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0.05"/>
    <geometry>
      <cylinder radius="0.05" length="0.1"/>
    </geometry>
    <material name="leg_color">
      <color rgba="0.4 0.4 0.8 1"/>
    </material>
  </visual>
</link>

<!-- Left Thigh -->
<joint name="left_hip_to_thigh" type="revolute">
  <parent link="left_hip"/>
  <child link="left_thigh"/>
  <origin xyz="0 0 0.1"/>
  <axis xyz="1 0 0"/>
  <limit lower="-0.52" upper="2.09" effort="100" velocity="1"/>
</joint>

<link name="left_thigh">
  <inertial>
    <mass value="2.0"/>
    <origin xyz="0 0 -0.2"/>
    <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
  </inertial>
  <visual>
    <origin xyz="0 0 -0.2"/>
    <geometry>
      <cylinder radius="0.06" length="0.4"/>
    </geometry>
    <material name="leg_color"/>
  </visual>
</link>

<!-- Left Knee -->
<joint name="left_thigh_to_shin" type="revolute">
  <parent link="left_thigh"/>
  <child link="left_shin"/>
  <origin xyz="0 0 -0.4"/>
  <axis xyz="1 0 0"/>
  <limit lower="-2.36" upper="0.2" effort="100" velocity="1"/>
</joint>

<link name="left_shin">
  <inertial>
    <mass value="1.5"/>
    <origin xyz="0 0 -0.15"/>
    <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.005"/>
  </inertial>
  <visual>
    <origin xyz="0 0 -0.15"/>
    <geometry>
      <cylinder radius="0.05" length="0.3"/>
    </geometry>
    <material name="leg_color"/>
  </visual>
</link>

<!-- Left Ankle -->
<joint name="left_shin_to_foot" type="revolute">
  <parent link="left_shin"/>
  <child link="left_foot"/>
  <origin xyz="0 0 -0.3"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.52" upper="0.52" effort="50" velocity="1"/>
</joint>

<link name="left_foot">
  <inertial>
    <mass value="0.8"/>
    <origin xyz="0.05 0 -0.02"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
  </inertial>
  <visual>
    <origin xyz="0.05 0 -0.02"/>
    <geometry>
      <box size="0.15 0.08 0.04"/>
    </geometry>
    <material name="foot_color">
      <color rgba="0.2 0.2 0.2 1"/>
    </material>
  </visual>
</link>
```

### Right Leg (Mirror of Left Leg)

```xml
<!-- Right Hip -->
<joint name="torso_to_right_hip" type="revolute">
  <parent link="torso"/>
  <child link="right_hip"/>
  <origin xyz="-0.05 -0.08 -0.1" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.785" upper="0.785" effort="100" velocity="1"/>
</joint>

<link name="right_hip">
  <inertial>
    <mass value="1.0"/>
    <origin xyz="0 0 0.05"/>
    <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.002"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0.05"/>
    <geometry>
      <cylinder radius="0.05" length="0.1"/>
    </geometry>
    <material name="leg_color"/>
  </visual>
</link>

<joint name="right_hip_to_thigh" type="revolute">
  <parent link="right_hip"/>
  <child link="right_thigh"/>
  <origin xyz="0 0 0.1"/>
  <axis xyz="1 0 0"/>
  <limit lower="-0.52" upper="2.09" effort="100" velocity="1"/>
</joint>

<link name="right_thigh">
  <inertial>
    <mass value="2.0"/>
    <origin xyz="0 0 -0.2"/>
    <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
  </inertial>
  <visual>
    <origin xyz="0 0 -0.2"/>
    <geometry>
      <cylinder radius="0.06" length="0.4"/>
    </geometry>
    <material name="leg_color"/>
  </visual>
</link>

<joint name="right_thigh_to_shin" type="revolute">
  <parent link="right_thigh"/>
  <child link="right_shin"/>
  <origin xyz="0 0 -0.4"/>
  <axis xyz="1 0 0"/>
  <limit lower="-2.36" upper="0.2" effort="100" velocity="1"/>
</joint>

<link name="right_shin">
  <inertial>
    <mass value="1.5"/>
    <origin xyz="0 0 -0.15"/>
    <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.005"/>
  </inertial>
  <visual>
    <origin xyz="0 0 -0.15"/>
    <geometry>
      <cylinder radius="0.05" length="0.3"/>
    </geometry>
    <material name="leg_color"/>
  </visual>
</link>

<joint name="right_shin_to_foot" type="revolute">
  <parent link="right_shin"/>
  <child link="right_foot"/>
  <origin xyz="0 0 -0.3"/>
  <axis xyz="0 1 0"/>
  <limit lower="-0.52" upper="0.52" effort="50" velocity="1"/>
</joint>

<link name="right_foot">
  <inertial>
    <mass value="0.8"/>
    <origin xyz="0.05 0 -0.02"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
  </inertial>
  <visual>
    <origin xyz="0.05 0 -0.02"/>
    <geometry>
      <box size="0.15 0.08 0.04"/>
    </geometry>
    <material name="foot_color"/>
  </visual>
</link>
```

## Joint Limit Considerations

### Human-like Joint Limits

```xml
<!-- Shoulder joint with human-like limits -->
<joint name="shoulder_joint" type="revolute">
  <parent link="torso"/>
  <child link="upper_arm"/>
  <origin xyz="0.1 0.1 0.2"/>
  <axis xyz="0 1 0"/>  <!-- Front-back movement -->
  <limit lower="-2.09" upper="1.57" effort="50" velocity="2"/>  <!-- -120° to 90° -->
</joint>

<joint name="shoulder_lift" type="revolute">
  <parent link="upper_arm"/>
  <child link="forearm"/>
  <origin xyz="0 0 0.3"/>
  <axis xyz="1 0 0"/>  <!-- Up-down movement -->
  <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>  <!-- -90° to 90° -->
</joint>

<!-- Elbow joint with realistic limits -->
<joint name="elbow_joint" type="revolute">
  <parent link="forearm"/>
  <child link="hand"/>
  <origin xyz="0 0 0.25"/>
  <axis xyz="1 0 0"/>  <!-- Bending motion -->
  <limit lower="0" upper="2.53" effort="30" velocity="2"/>  <!-- 0° to 145° (can't fully straighten) -->
</joint>

<!-- Hip joint with realistic limits -->
<joint name="hip_joint" type="revolute">
  <parent link="torso"/>
  <child link="thigh"/>
  <origin xyz="-0.05 0.08 -0.1"/>
  <axis xyz="0 1 0"/>  <!-- Side movement -->
  <limit lower="-0.785" upper="0.785" effort="100" velocity="1"/>  <!-- -45° to 45° -->
</joint>

<joint name="hip_lift" type="revolute">
  <parent link="thigh"/>
  <child link="shin"/>
  <origin xyz="0 0 -0.4"/>
  <axis xyz="1 0 0"/>  <!-- Front-back movement -->
  <limit lower="-0.52" upper="2.09" effort="100" velocity="1"/>  <!-- -30° to 120° -->
</joint>

<!-- Knee joint with realistic limits -->
<joint name="knee_joint" type="revolute">
  <parent link="shin"/>
  <child link="foot"/>
  <origin xyz="0 0 -0.3"/>
  <axis xyz="1 0 0"/>  <!-- Bending motion -->
  <limit lower="-2.36" upper="0.2" effort="100" velocity="1"/>  <!-- -135° to 11° (slightly bent when standing) -->
</joint>
```

## Kinematic Chain Analysis

### Forward Kinematics Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import numpy as np
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class KinematicAnalyzerNode(Node):
    def __init__(self):
        super().__init__('kinematic_analyzer')

        # Joint state subscriber
        self.joint_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Result publisher
        self.fk_publisher = self.create_publisher(
            String,
            'forward_kinematics_result',
            10
        )

        # TF broadcaster for visualization
        self.tf_broadcaster = TransformBroadcaster(self)

        self.joint_positions = {}

    def joint_state_callback(self, msg):
        """Process joint states and calculate forward kinematics"""
        # Update joint positions
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]

        # Calculate forward kinematics for specific chains
        left_arm_pose = self.calculate_left_arm_fk()
        right_arm_pose = self.calculate_right_arm_fk()

        # Publish results
        result_msg = String()
        result_msg.data = f"Left Arm: {left_arm_pose}, Right Arm: {right_arm_pose}"
        self.fk_publisher.publish(result_msg)

        # Broadcast transforms for visualization
        self.broadcast_transforms()

    def calculate_left_arm_fk(self):
        """Calculate forward kinematics for left arm"""
        # Get joint angles
        shoulder_yaw = self.joint_positions.get('left_shoulder_yaw', 0)
        shoulder_pitch = self.joint_positions.get('left_shoulder_pitch', 0)
        elbow_angle = self.joint_positions.get('left_elbow', 0)

        # Calculate transformation matrices (simplified)
        # This is a simplified example - full implementation would use DH parameters
        # or more sophisticated kinematic calculations

        # Example: Calculate end effector position
        upper_arm_length = 0.3
        forearm_length = 0.25

        # Calculate position based on joint angles
        x = upper_arm_length * np.cos(shoulder_pitch) + forearm_length * np.cos(shoulder_pitch + elbow_angle)
        y = upper_arm_length * np.sin(shoulder_pitch) + forearm_length * np.sin(shoulder_pitch + elbow_angle)
        z = 0  # Simplified 2D case

        return [x, y, z]

    def calculate_right_arm_fk(self):
        """Calculate forward kinematics for right arm"""
        # Similar to left arm but with right arm joints
        pass

    def broadcast_transforms(self):
        """Broadcast transforms for RViz visualization"""
        # Create and broadcast transforms for each link
        # This allows visualization in RViz
        pass

def main(args=None):
    rclpy.init(args=args)
    node = KinematicAnalyzerNode()

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

## Inverse Kinematics Considerations

### IK Solver Integration

```xml
<!-- Add transmission elements for IK solvers -->
<transmission name="left_arm_transmission" type="transmission_interface/SimpleTransmission">
  <joint name="left_shoulder_yaw">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_shoulder_yaw_motor">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>

<!-- Add ROS control interface -->
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/humanoid</robotNamespace>
  </plugin>
</gazebo>
```

## Balance and Stability Considerations

### Center of Mass Optimization

```xml
<!-- Position links to optimize center of mass -->
<link name="torso">
  <inertial>
    <!-- Keep center of mass low and centered -->
    <mass value="8.0"/>
    <origin xyz="0 0 0.1"/>  <!-- Lower origin for stability -->
    <inertia ixx="0.8" ixy="0.0" ixz="0.0" iyy="0.8" iyz="0.0" izz="0.4"/>
  </inertial>
  <visual>
    <origin xyz="0 0 0.1"/>
    <geometry>
      <box size="0.25 0.2 0.4"/>
    </geometry>
    <material name="torso_color">
      <color rgba="0.6 0.6 0.8 1"/>
    </material>
  </visual>
</link>
```

## Best Practices for Humanoid Kinematics

### 1. Joint Configuration

```xml
<!-- Use appropriate joint types for human-like movement -->
<!-- Avoid over-constraining or under-constraining joints -->
<joint name="wrist_joint" type="revolute">  <!-- Not continuous for realistic wrist -->
  <parent link="forearm"/>
  <child link="hand"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.05" upper="1.05" effort="10" velocity="3"/>  <!-- -60° to 60° -->
</joint>
```

### 2. Mass Distribution

```xml
<!-- Distribute mass realistically -->
<!-- Heavier components lower in the robot for stability -->
<link name="torso">
  <inertial>
    <mass value="12.0"/>  <!-- Torso carries most mass -->
    <origin xyz="0 0 0.15"/>
    <inertia ixx="1.2" ixy="0.0" ixz="0.0" iyy="1.2" iyz="0.0" izz="0.6"/>
  </inertial>
</link>

<link name="head">
  <inertial>
    <mass value="2.0"/>   <!-- Head is lighter -->
    <origin xyz="0 0 0.05"/>
    <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
  </inertial>
</link>
```

### 3. Safety Margins

```xml
<!-- Include safety margins in joint limits -->
<joint name="knee_joint" type="revolute">
  <parent link="thigh"/>
  <child link="shin"/>
  <axis xyz="1 0 0"/>
  <!-- Don't allow full extension to prevent mechanical stress -->
  <limit lower="-2.36" upper="0.1" effort="100" velocity="1"/>  <!-- -135° to 5.7° -->
</joint>
```

Humanoid robot kinematics require careful attention to joint types, limits, and physical properties to create realistic and functional robot models that can be used for simulation, control, and actual robotic applications.