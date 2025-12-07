---
title: "Humanoid-Specific URDF Design"
description: "Design considerations and best practices for humanoid robot URDF models"
sidebar_position: 5
keywords: [urdf, humanoid, balance, stability, design, modeling, robotics]
---

# Humanoid-Specific URDF Design

Humanoid robot URDF design requires special considerations for balance, stability, and human-like movement patterns. This section covers the unique aspects of designing URDF models specifically for humanoid robots.

## Balance and Stability Considerations

### Center of Mass Optimization

```xml
<?xml version="1.0"?>
<robot name="balanced_humanoid" xmlns:xacro="http://ros.org/wiki/xacro">

  <!-- Define properties for balance optimization -->
  <xacro:property name="total_robot_mass" value="30.0" />
  <xacro:property name="torso_mass_ratio" value="0.4" />  <!-- 40% of total mass in torso -->
  <xacro:property name="head_mass_ratio" value="0.05" />   <!-- 5% of total mass in head -->
  <xacro:property name="limb_mass_ratio" value="0.125" /> <!-- 12.5% per limb (4 limbs) -->

  <!-- Base link - represents the robot's center reference -->
  <link name="base_link">
    <inertial>
      <!-- Keep base link light - it's just a reference -->
      <mass value="0.001"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- Torso - carries most of the robot's mass for stability -->
  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.4"/>  <!-- Position torso at appropriate height -->
  </joint>

  <link name="torso">
    <inertial>
      <!-- Heavy torso lowers center of mass for stability -->
      <mass value="${total_robot_mass * torso_mass_ratio}"/>
      <origin xyz="0 0 0.1"/>  <!-- Keep CoM low -->
      <inertia ixx="0.8" ixy="0.0" ixz="0.0" iyy="0.8" iyz="0.0" izz="0.4"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <box size="0.3 0.25 0.4"/>  <!-- Wide base for stability -->
      </geometry>
      <material name="torso_color">
        <color rgba="0.6 0.6 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.1"/>
      <geometry>
        <box size="0.3 0.25 0.4"/>
      </geometry>
    </collision>
  </link>

  <!-- Head - positioned to maintain balance -->
  <joint name="torso_to_head" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.35"/>  <!-- Position head appropriately -->
    <axis xyz="0 1 0"/>      <!-- Yaw movement for head -->
    <limit lower="-0.785" upper="0.785" effort="10" velocity="1"/>  <!-- Limited movement for stability -->
  </joint>

  <link name="head">
    <inertial>
      <mass value="${total_robot_mass * head_mass_ratio}"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="head_color">
        <color rgba="1 0.8 0.6 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
  </link>

</robot>
```

### Wide Stance Configuration

```xml
<!-- Hip structure for stable stance -->
<joint name="torso_to_hips" type="fixed">
  <parent link="torso"/>
  <child link="hips"/>
  <origin xyz="0 0 -0.2"/>
</joint>

<link name="hips">
  <inertial>
    <mass value="2.0"/>
    <origin xyz="0 0 0"/>
    <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.05"/>
  </inertial>
  <visual>
    <geometry>
      <box size="0.25 0.2 0.1"/>  <!-- Wide hip structure -->
    </geometry>
    <material name="pelvis_color">
      <color rgba="0.5 0.5 0.7 1"/>
    </material>
  </visual>
</link>

<!-- Wide leg spacing for stability -->
<joint name="hips_to_left_leg" type="fixed">
  <parent link="hips"/>
  <child link="left_leg_root"/>
  <origin xyz="0 0.12 -0.05"/>  <!-- Wider stance -->
</joint>

<joint name="hips_to_right_leg" type="fixed">
  <parent link="hips"/>
  <child link="right_leg_root"/>
  <origin xyz="0 -0.12 -0.05"/>  <!-- Wider stance -->
</joint>
```

## Humanoid Movement Constraints

### Realistic Joint Limitations

```xml
<!-- Shoulder joint with human-like constraints -->
<joint name="left_shoulder_yaw" type="revolute">
  <parent link="torso"/>
  <child link="left_shoulder"/>
  <origin xyz="0.15 0.12 0.2"/>
  <axis xyz="0 1 0"/>  <!-- Yaw movement -->
  <limit lower="-0.785" upper="1.57" effort="50" velocity="2"/>  <!-- -45° to 90° -->
</joint>

<joint name="left_shoulder_pitch" type="revolute">
  <parent link="left_shoulder"/>
  <child link="left_upper_arm"/>
  <origin xyz="0 0 0.05"/>
  <axis xyz="1 0 0"/>  <!-- Pitch movement -->
  <limit lower="-1.57" upper="1.57" effort="50" velocity="2"/>  <!-- -90° to 90° -->
</joint>

<joint name="left_shoulder_roll" type="revolute">
  <parent link="left_upper_arm"/>
  <child link="left_forearm"/>
  <origin xyz="0 0 0.25"/>
  <axis xyz="0 0 1"/>  <!-- Roll movement -->
  <limit lower="-2.09" upper="1.57" effort="40" velocity="2"/>  <!-- -120° to 90° -->
</joint>

<joint name="left_elbow" type="revolute">
  <parent link="left_forearm"/>
  <child link="left_hand"/>
  <origin xyz="0 0 0.25"/>
  <axis xyz="1 0 0"/>  <!-- Bending motion -->
  <limit lower="0" upper="2.35" effort="30" velocity="2"/>  <!-- 0° to 135° -->
</joint>

<!-- Hip joint with realistic constraints -->
<joint name="hips_to_left_thigh" type="revolute">
  <parent link="left_leg_root"/>
  <child link="left_thigh"/>
  <origin xyz="0 0 -0.05"/>
  <axis xyz="0 1 0"/>  <!-- Yaw (abduction/adduction) -->
  <limit lower="-0.52" upper="0.52" effort="100" velocity="1"/>  <!-- -30° to 30° -->
</joint>

<joint name="left_hip_pitch" type="revolute">
  <parent link="left_thigh"/>
  <child link="left_shin"/>
  <origin xyz="0 0 -0.35"/>
  <axis xyz="1 0 0"/>  <!-- Pitch (forward/back) -->
  <limit lower="-0.698" upper="2.094" effort="150" velocity="1"/>  <!-- -40° to 120° -->
</joint>

<joint name="left_knee" type="revolute">
  <parent link="left_shin"/>
  <child link="left_foot"/>
  <origin xyz="0 0 -0.35"/>
  <axis xyz="1 0 0"/>  <!-- Bending motion -->
  <limit lower="-2.53" upper="0.26" effort="150" velocity="1"/>  <!-- -145° to 15° -->
</joint>

<joint name="left_ankle" type="revolute">
  <parent link="left_foot"/>
  <child link="left_toes"/>
  <origin xyz="0.1 0 -0.05"/>
  <axis xyz="0 1 0"/>  <!-- Pitch (dorsiflexion/plantarflexion) -->
  <limit lower="-0.52" upper="0.52" effort="50" velocity="1"/>  <!-- -30° to 30° -->
</joint>
```

## Multi-Limb Coordination

### Synchronized Limb Definitions

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="coordinated_humanoid">

  <!-- Define limb macro for consistent structure -->
  <xacro:macro name="humanoid_arm" params="side parent xyz_rpy:=0 0 0 shoulder_limit:=1.57 elbow_limit:=2.0">
    <!-- Shoulder -->
    <joint name="${side}_shoulder_yaw" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_shoulder"/>
      <origin xyz="${'0.15 0.12 0.2' if side == 'left' else '0.15 -0.12 0.2'}"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.785" upper="1.57" effort="50" velocity="2"/>
    </joint>

    <link name="${side}_shoulder">
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.0005"/>
      </inertial>
      <visual>
        <geometry>
          <cylinder radius="0.03" length="0.1"/>
        </geometry>
        <material name="arm_material">
          <color rgba="0.7 0.7 0.7 1"/>
        </material>
      </visual>
    </link>

    <!-- Upper arm -->
    <joint name="${side}_shoulder_pitch" type="revolute">
      <parent link="${side}_shoulder"/>
      <child link="${side}_upper_arm"/>
      <origin xyz="0 0 0.05"/>
      <axis xyz="1 0 0"/>
      <limit lower="-${shoulder_limit}" upper="${shoulder_limit}" effort="50" velocity="2"/>
    </joint>

    <link name="${side}_upper_arm">
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.002"/>
      </inertial>
      <visual>
        <geometry>
          <cylinder radius="0.04" length="0.25"/>
        </geometry>
        <material name="arm_material"/>
      </visual>
    </link>

    <!-- Forearm -->
    <joint name="${side}_elbow" type="revolute">
      <parent link="${side}_upper_arm"/>
      <child link="${side}_forearm"/>
      <origin xyz="0 0 0.25"/>
      <axis xyz="1 0 0"/>
      <limit lower="0" upper="${elbow_limit}" effort="30" velocity="2"/>
    </joint>

    <link name="${side}_forearm">
      <inertial>
        <mass value="0.8"/>
        <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.001"/>
      </inertial>
      <visual>
        <geometry>
          <cylinder radius="0.03" length="0.25"/>
        </geometry>
        <material name="arm_material"/>
      </visual>
    </link>

    <!-- Hand -->
    <joint name="${side}_wrist" type="revolute">
      <parent link="${side}_forearm"/>
      <child link="${side}_hand"/>
      <origin xyz="0 0 0.25"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.785" upper="0.785" effort="20" velocity="3"/>
    </joint>

    <link name="${side}_hand">
      <inertial>
        <mass value="0.3"/>
        <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0005" iyz="0.0" izz="0.0002"/>
      </inertial>
      <visual>
        <geometry>
          <box size="0.08 0.06 0.04"/>
        </geometry>
        <material name="hand_material">
          <color rgba="0.9 0.9 0.9 1"/>
        </material>
      </visual>
    </link>
  </xacro:macro>

  <!-- Define legs with similar macro approach -->
  <xacro:macro name="humanoid_leg" params="side parent">
    <joint name="${side}_hip_yaw" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_hip"/>
      <origin xyz="0 ${'0.12' if side == 'left' else '-0.12'} -0.05"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.52" upper="0.52" effort="100" velocity="1"/>
    </joint>

    <link name="${side}_hip">
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.002"/>
      </inertial>
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.1"/>
        </geometry>
        <material name="leg_material">
          <color rgba="0.4 0.4 0.8 1"/>
        </material>
      </visual>
    </link>

    <joint name="${side}_hip_pitch" type="revolute">
      <parent link="${side}_hip"/>
      <child link="${side}_thigh"/>
      <origin xyz="0 0 -0.05"/>
      <axis xyz="1 0 0"/>
      <limit lower="-0.698" upper="2.094" effort="150" velocity="1"/>
    </joint>

    <link name="${side}_thigh">
      <inertial>
        <mass value="2.5"/>
        <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
      </inertial>
      <visual>
        <geometry>
          <cylinder radius="0.06" length="0.35"/>
        </geometry>
        <material name="leg_material"/>
      </visual>
    </link>

    <joint name="${side}_knee" type="revolute">
      <parent link="${side}_thigh"/>
      <child link="${side}_shin"/>
      <origin xyz="0 0 -0.35"/>
      <axis xyz="1 0 0"/>
      <limit lower="-2.53" upper="0.26" effort="150" velocity="1"/>
    </joint>

    <link name="${side}_shin">
      <inertial>
        <mass value="2.0"/>
        <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.005"/>
      </inertial>
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.35"/>
        </geometry>
        <material name="leg_material"/>
      </visual>
    </link>

    <joint name="${side}_ankle" type="revolute">
      <parent link="${side}_shin"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 -0.35"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.52" upper="0.52" effort="80" velocity="1"/>
    </joint>

    <link name="${side}_foot">
      <inertial>
        <mass value="1.0"/>
        <origin xyz="0.05 0 -0.02"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
      </inertial>
      <visual>
        <origin xyz="0.05 0 -0.02"/>
        <geometry>
          <box size="0.18 0.08 0.04"/>
        </geometry>
        <material name="foot_material">
          <color rgba="0.2 0.2 0.2 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0.05 0 -0.02"/>
        <geometry>
          <box size="0.18 0.08 0.04"/>
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Main body structure -->
  <link name="base_link"/>

  <link name="torso">
    <inertial>
      <mass value="12.0"/>
      <origin xyz="0 0 0.15"/>
      <inertia ixx="1.2" ixy="0.0" ixz="0.0" iyy="1.2" iyz="0.0" izz="0.6"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.15"/>
      <geometry>
        <box size="0.3 0.25 0.4"/>
      </geometry>
      <material name="torso_color">
        <color rgba="0.6 0.6 0.8 1"/>
      </material>
    </visual>
  </link>

  <joint name="base_to_torso" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.4"/>
  </joint>

  <!-- Use macros to create limbs -->
  <xacro:humanoid_arm side="left" parent="torso"/>
  <xacro:humanoid_arm side="right" parent="torso"/>
  <xacro:humanoid_leg side="left" parent="torso"/>
  <xacro:humanoid_leg side="right" parent="torso"/>

</robot>
```

## Humanoid-Specific Sensors

### Balance and Orientation Sensors

```xml
<!-- IMU for balance and orientation sensing -->
<link name="imu_link">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
  </inertial>
  <visual>
    <geometry>
      <box size="0.02 0.02 0.01"/>
    </geometry>
    <material name="sensor_material">
      <color rgba="1 0 0 1"/>
    </material>
  </visual>
</link>

<joint name="torso_to_imu" type="fixed">
  <parent link="torso"/>
  <child link="imu_link"/>
  <origin xyz="0 0 0.1"/>  <!-- Center of torso for best orientation sensing -->
</joint>

<!-- Force/Torque sensors in feet for balance -->
<link name="left_foot_sensor">
  <inertial>
    <mass value="0.01"/>
    <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
  </inertial>
</link>

<joint name="left_foot_to_sensor" type="fixed">
  <parent link="left_foot"/>
  <child link="left_foot_sensor"/>
  <origin xyz="0.05 0 -0.02"/>  <!-- Center of foot -->
</joint>

<!-- Camera for vision-based navigation -->
<link name="camera_link">
  <inertial>
    <mass value="0.1"/>
    <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
  </inertial>
  <visual>
    <geometry>
      <box size="0.05 0.03 0.03"/>
    </geometry>
    <material name="camera_material">
      <color rgba="0.8 0.8 0.8 1"/>
    </material>
  </visual>
</link>

<joint name="head_to_camera" type="fixed">
  <parent link="head"/>
  <child link="camera_link"/>
  <origin xyz="0.05 0 0.05" rpy="0 0 0"/>  <!-- Looking forward from head -->
</joint>

<!-- Gazebo sensor definitions -->
<gazebo reference="imu_link">
  <sensor name="imu_sensor" type="imu">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <imu>
      <angular_velocity>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>2e-4</stddev>
          </noise>
        </z>
      </angular_velocity>
      <linear_acceleration>
        <x>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </x>
        <y>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </y>
        <z>
          <noise type="gaussian">
            <mean>0.0</mean>
            <stddev>1.7e-2</stddev>
          </noise>
        </z>
      </linear_acceleration>
    </imu>
    <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
      <topicName>imu/data</topicName>
      <bodyName>imu_link</bodyName>
      <frameName>imu_link</frameName>
      <serviceName>imu/service</serviceName>
      <gaussianNoise>0.0</gaussianNoise>
      <updateRate>100.0</updateRate>
    </plugin>
  </sensor>
</gazebo>

<gazebo reference="left_foot_sensor">
  <sensor name="ft_sensor" type="force_torque">
    <always_on>true</always_on>
    <update_rate>100</update_rate>
    <force_torque>
      <frame>child</frame>
      <measure_direction>child_to_parent</measure_direction>
    </force_torque>
    <plugin name="ft_plugin" filename="libgazebo_ros_ft_sensor.so">
      <topicName>left_foot/force_torque</topicName>
      <bodyName>left_foot_sensor</bodyName>
    </plugin>
  </sensor>
</gazebo>

<gazebo reference="camera_link">
  <sensor type="camera" name="camera">
    <always_on>true</always_on>
    <update_rate>30.0</update_rate>
    <camera name="head_camera">
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <format>R8G8B8</format>
        <width>640</width>
        <height>480</height>
      </image>
      <clip>
        <near>0.1</near>
        <far>300</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>100.0</max_depth>
    </plugin>
  </sensor>
</gazebo>
```

## Walking Pattern Considerations

### Foot Design for Stable Walking

```xml
<!-- Enhanced foot design for walking stability -->
<link name="enhanced_foot">
  <inertial>
    <mass value="1.2"/>
    <origin xyz="0.06 0 -0.025"/>
    <inertia ixx="0.015" ixy="0.0" ixz="0.001" iyy="0.025" iyz="0.001" izz="0.012"/>
  </inertial>
  <visual>
    <origin xyz="0.06 0 -0.025"/>
    <geometry>
      <!-- Foot with rounded heel and toe for natural rolling motion -->
      <mesh filename="package://humanoid_description/meshes/foot.dae"/>
    </geometry>
    <material name="foot_material">
      <color rgba="0.1 0.1 0.1 1"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0.06 0 -0.025"/>
    <geometry>
      <!-- Simplified collision geometry -->
      <box size="0.2 0.1 0.05"/>
    </geometry>
  </collision>
</link>

<!-- Ankle joint with walking-specific constraints -->
<joint name="ankle_pitch" type="revolute">
  <parent link="shin"/>
  <child link="enhanced_foot"/>
  <origin xyz="0 0 -0.35"/>
  <axis xyz="0 1 0"/>  <!-- Pitch axis for walking -->
  <limit lower="-0.35" upper="0.35" effort="80" velocity="2"/>  <!-- Limited for stable walking -->
  <dynamics damping="5.0" friction="1.0"/>  <!-- Damping for shock absorption -->
</joint>

<!-- Toe joint for additional walking flexibility -->
<joint name="toe_joint" type="revolute">
  <parent link="enhanced_foot"/>
  <child link="toe"/>
  <origin xyz="0.09 0 0"/>
  <axis xyz="0 1 0"/>  <!-- Toe pitch for push-off -->
  <limit lower="0" upper="0.52" effort="30" velocity="2"/>  <!-- Forward only movement -->
</joint>

<link name="toe">
  <inertial>
    <mass value="0.2"/>
    <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
  </inertial>
  <visual>
    <geometry>
      <cylinder radius="0.02" length="0.04"/>
    </geometry>
    <material name="foot_material"/>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.02" length="0.04"/>
    </geometry>
  </collision>
</link>
```

## Control-Specific Design Elements

### Transmission Definitions for Humanoid Control

```xml
<!-- Complex transmissions for coordinated movement -->
<transmission name="left_arm_transmission" type="transmission_interface/SimpleTransmission">
  <joint name="left_shoulder_yaw">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
  </joint>
  <joint name="left_shoulder_pitch">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
  </joint>
  <joint name="left_elbow">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_arm_controller">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>

<!-- Balance control specific transmissions -->
<transmission name="balance_control_transmission" type="transmission_interface/DifferentialTransmission">
  <leftJoint name="left_ankle">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
  </leftJoint>
  <rightJoint name="right_ankle">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
  </rightJoint>
  <actuator name="balance_actuator">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Gazebo-Specific Humanoid Configurations

### Physics Optimization for Humanoid Simulation

```xml
<!-- Gazebo physics configuration for humanoid stability -->
<gazebo reference="torso">
  <material>Gazebo/Blue</material>
  <mu1>0.8</mu1>  <!-- Higher friction for better ground contact -->
  <mu2>0.8</mu2>
  <kp>1000000.0</kp>  <!-- High stiffness for stable contact -->
  <kd>100.0</kd>
  <max_contacts>10</max_contacts>
  <self_collide>false</self_collide>
</gazebo>

<gazebo reference="left_foot">
  <material>Gazebo/Black</material>
  <mu1>1.0</mu1>  <!-- Maximum friction for feet -->
  <mu2>1.0</mu2>
  <kp>10000000.0</kp>  <!-- Very high stiffness for feet -->
  <kd>1000.0</kd>
  <max_contacts>20</max_contacts>  <!-- More contacts for stable stance -->
  <self_collide>false</self_collide>
</gazebo>

<!-- Humanoid-specific Gazebo plugins -->
<gazebo>
  <plugin name="balance_controller" filename="libbalance_controller.so">
    <robotNamespace>/humanoid_robot</robotNamespace>
    <imuTopic>/imu/data</imuTopic>
    <ftSensors>left_foot/force_torque,right_foot/force_torque</ftSensors>
    <updateRate>500</updateRate>
    <kp>50.0</kp>
    <ki>5.0</ki>
    <kd>10.0</kd>
  </plugin>
</gazebo>

<gazebo>
  <plugin name="walking_controller" filename="libwalking_controller.so">
    <robotNamespace>/humanoid_robot</robotNamespace>
    <imuTopic>/imu/data</imuTopic>
    <leftFootFrame>left_foot</leftFootFrame>
    <rightFootFrame>right_foot</rightFootFrame>
    <updateRate>100</updateRate>
  </plugin>
</gazebo>
```

## Best Practices for Humanoid URDF Design

### 1. Mass Distribution Guidelines

```xml
<!-- Proper mass distribution for humanoid stability -->
<link name="optimized_torso">
  <inertial>
    <!-- Heavy torso for low center of mass -->
    <mass value="15.0"/>
    <!-- Keep CoM low and centered -->
    <origin xyz="0 0 0.1"/>
    <inertia ixx="1.5" ixy="0.0" ixz="0.0" iyy="1.5" iyz="0.0" izz="0.8"/>
  </inertial>
</link>

<link name="light_head">
  <inertial>
    <!-- Light head to avoid top-heaviness -->
    <mass value="1.5"/>
    <inertia ixx="0.03" ixy="0.0" ixz="0.0" iyy="0.03" iyz="0.0" izz="0.03"/>
  </inertial>
</link>
```

### 2. Joint Limit Safety Margins

```xml
<!-- Conservative joint limits with safety margins -->
<joint name="knee_safe" type="revolute">
  <parent link="thigh"/>
  <child link="shin"/>
  <axis xyz="1 0 0"/>
  <!-- Don't allow full extension to prevent mechanical stress -->
  <limit lower="-2.4" upper="0.2" effort="150" velocity="1"/>
  <!-- Add safety limits -->
  <safety_controller k_position="20" k_velocity="50"
                   soft_lower_limit="-2.3" soft_upper_limit="0.15"/>
</joint>
```

### 3. Sensor Integration Planning

```xml
<!-- Plan for sensor integration from the start -->
<!-- IMU in torso for orientation -->
<joint name="torso_to_imu_mount" type="fixed">
  <parent link="torso"/>
  <child link="imu_mount"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>

<!-- Force sensors in feet -->
<joint name="left_foot_to_sensor_mount" type="fixed">
  <parent link="left_foot"/>
  <child link="left_ft_sensor_mount"/>
  <origin xyz="0.0 0.0 -0.025"/>  <!-- Center of foot sole -->
</joint>
```

## Validation for Humanoid Applications

### Balance Validation Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, WrenchStamped
from geometry_msgs.msg import Vector3
import numpy as np

class BalanceValidatorNode(Node):
    def __init__(self):
        super().__init__('balance_validator')

        # Subscribe to sensor data
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        self.left_ft_subscriber = self.create_subscription(
            WrenchStamped,
            '/left_foot/force_torque',
            self.ft_callback,
            10
        )

        self.right_ft_subscriber = self.create_subscription(
            WrenchStamped,
            '/right_foot/force_torque',
            self.ft_callback,
            10
        )

        # Timer for balance validation
        self.balance_timer = self.create_timer(0.1, self.validate_balance)

        self.imu_data = None
        self.left_force = None
        self.right_force = None

        self.get_logger().info('Balance Validator initialized')

    def imu_callback(self, msg):
        """Process IMU data for balance validation"""
        self.imu_data = msg

    def ft_callback(self, msg):
        """Process force/torque data"""
        # This would distinguish between left and right based on topic
        pass

    def validate_balance(self):
        """Validate robot balance based on sensor data"""
        if not self.imu_data:
            return

        # Check orientation for balance
        orientation = self.imu_data.orientation
        roll, pitch, yaw = self.quaternion_to_euler(
            orientation.x, orientation.y, orientation.z, orientation.w
        )

        # Check if tilt is within safe limits
        max_tilt = 0.35  # ~20 degrees
        if abs(roll) > max_tilt or abs(pitch) > max_tilt:
            self.get_logger().warn(f'Balance warning: Roll={roll:.2f}, Pitch={pitch:.2f}')

        # Check zero moment point (ZMP) if force/torque data available
        if self.left_force and self.right_force:
            self.validate_zmp()

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        # Simplified conversion - use tf_transformations in practice
        import math
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return roll, pitch, yaw

    def validate_zmp(self):
        """Validate Zero Moment Point for walking stability"""
        # Implementation would calculate ZMP based on foot forces
        pass

def main(args=None):
    rclpy.init(args=args)
    node = BalanceValidatorNode()

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

Humanoid-specific URDF design requires careful attention to balance, stability, human-like movement patterns, and sensor integration to create effective and stable humanoid robot models suitable for both simulation and real-world applications.