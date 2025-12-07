---
title: "Advanced URDF Features"
description: "Advanced URDF capabilities including transmissions, Gazebo integration, aur Xacro macros ko samajhna"
sidebar_position: 3
keywords: [urdf, xacro, transmissions, gazebo, macros, advanced, simulation]
---

# Advanced URDF Features

Advanced URDF features basic robot description capabilities ko extend karti hain jismein simulation-specific elements, Xacro ke through parameterization, aur control interface definitions included hain. Yeh features complex simulation aur control scenarios ke liye suitable more sophisticated robot models ko enable karti hain.

## Xacro Parameterization

Xacro (XML Macros) URDF files mein parameterization aur macro definitions ko allow karta hai, jisse woh maintainable aur reusable ho jate hain.

### Basic Xacro Structure

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="xacro_robot">

  <!-- Properties define karen -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="robot_name" value="my_robot" />
  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />

  <!-- Macros define karen -->
  <xacro:macro name="wheel" params="prefix parent *origin">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <xacro:insert_block name="origin"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

  <!-- Macro ka istemal karen -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </visual>
  </link>

  <!-- Left wheel banayein -->
  <xacro:wheel prefix="left" parent="base_link">
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
  </xacro:wheel>

  <!-- Right wheel banayein -->
  <xacro:wheel prefix="right" parent="base_link">
    <origin xyz="0 -0.2 0" rpy="0 0 0"/>
  </xacro:wheel>

</robot>
```

### Conditional Xacro Macros

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="conditional_robot">

  <!-- Properties -->
  <xacro:property name="has_camera" value="true" />
  <xacro:property name="camera_type" value="rgbd" />

  <!-- Conditional sensor macro -->
  <xacro:macro name="sensor_mount" params="sensor_enabled sensor_type *origin">
    <xacro:if value="${sensor_enabled}">
      <link name="sensor_link">
        <visual>
          <geometry>
            <box size="0.05 0.05 0.05"/>
          </geometry>
          <material name="sensor_material">
            <color rgba="0.8 0.8 0.8 1"/>
          </material>
        </visual>
      </link>

      <joint name="sensor_joint" type="fixed">
        <parent link="base_link"/>
        <child link="sensor_link"/>
        <xacro:insert_block name="origin"/>
      </joint>

      <!-- Sensor-specific elements type ke adhar par -->
      <xacro:if value="${sensor_type == 'rgbd'}">
        <gazebo reference="sensor_link">
          <sensor type="depth" name="camera">
            <always_on>true</always_on>
            <update_rate>30.0</update_rate>
            <camera name="head">
              <horizontal_fov>1.3962634</horizontal_fov>
              <image>
                <format>R8G8B8</format>
                <width>640</width>
                <height>480</height>
              </image>
              <clip>
                <near>0.1</near>
                <far>100</far>
              </clip>
            </camera>
            <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
              <frame_name>sensor_link</frame_name>
            </plugin>
          </sensor>
        </gazebo>
      </xacro:if>
    </xacro:if>
  </xacro:macro>

  <!-- Robot body -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.2" length="0.2"/>
      </geometry>
    </visual>
  </link>

  <!-- Mount sensor conditionally -->
  <xacro:sensor_mount sensor_enabled="${has_camera}" sensor_type="${camera_type}">
    <origin xyz="0.15 0 0.1" rpy="0 0 0"/>
  </xacro:sensor_mount>

</robot>
```

### Mathematical Expressions in Xacro

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="math_robot">

  <!-- Parameters define karen -->
  <xacro:property name="base_width" value="0.5" />
  <xacro:property name="base_depth" value="0.3" />
  <xacro:property name="base_height" value="0.2" />
  <xacro:property name="wheel_offset_x" value="${base_width/2 - 0.05}" />
  <xacro:property name="wheel_offset_y" value="${base_depth/2 + 0.05}" />
  <xacro:property name="wheel_radius" value="0.1" />

  <!-- Inertial properties calculate karen -->
  <xacro:property name="base_mass" value="5.0" />
  <xacro:property name="base_ixx" value="${base_mass * (base_depth*base_depth + base_height*base_height) / 12.0}" />
  <xacro:property name="base_iyy" value="${base_mass * (base_width*base_width + base_height*base_height) / 12.0}" />
  <xacro:property name="base_izz" value="${base_mass * (base_width*base_width + base_depth*base_depth) / 12.0}" />

  <link name="base_link">
    <visual>
      <geometry>
        <box size="${base_width} ${base_depth} ${base_height}"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="${base_width} ${base_depth} ${base_height}"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${base_mass}"/>
      <inertia ixx="${base_ixx}" ixy="0.0" ixz="0.0"
               iyy="${base_iyy}" iyz="0.0" izz="${base_izz}"/>
    </inertial>
  </link>

  <!-- Calculated values ka istemal karke wheel positions -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="${wheel_offset_x} ${wheel_offset_y} 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="${wheel_radius}" length="0.05"/>
      </geometry>
    </visual>
  </link>

</robot>
```

## Transmissions aur Actuators

Transmissions ROS control aur physical joints ke darmiyan interface ko define karta hai, jo robot actuators ke precise control ko enable karta hai.

### Simple Transmission

```xml
<?xml version="1.0"?>
<robot name="transmission_robot" xmlns:xacro="http://ros.org/wiki/xacro">

  <link name="base_link"/>
  <link name="joint_link"/>

  <joint name="example_joint" type="revolute">
    <parent link="base_link"/>
    <child link="joint_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1"/>
  </joint>

  <!-- Position control ke liye simple transmission -->
  <transmission name="joint_transmission" type="transmission_interface/SimpleTransmission">
    <joint name="example_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <!-- Alternative interfaces:
           hardware_interface/PositionJointInterface
           hardware_interface/VelocityJointInterface
           hardware_interface/EffortJointInterface -->
    </joint>
    <actuator name="joint_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

</robot>
```

### Differential Transmission

```xml
<!-- Skid-steer robots ke liye differential transmission -->
<transmission name="differential_transmission"
              type="transmission_interface/DifferentialTransmission">
  <leftJoint name="left_wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </leftJoint>
  <rightJoint name="right_wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </rightJoint>
  <actuator name="left_wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
  <actuator name="right_wheel_motor">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Four-Bar Linkage Transmission

```xml
<!-- Grippers jaise mechanisms ke liye complex transmission -->
<transmission name="gripper_transmission"
              type="transmission_interface/FourBarLinkageTransmission">
  <joint name="gripper_left_joint">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <joint name="gripper_right_joint">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="gripper_motor">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Gazebo Integration

Gazebo-specific elements URDF ko physics simulation, sensors, aur visualization ke liye extend karta hai.

### Basic Gazebo Integration

```xml
<?xml version="1.0"?>
<robot name="gazebo_robot">

  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.4"/>
    </inertial>
  </link>

  <!-- Gazebo-specific properties -->
  <gazebo reference="base_link">
    <material>Gazebo/Blue</material>
    <mu1>0.2</mu1>
    <mu2>0.2</mu2>
    <self_collide>false</self_collide>
    <gravity>true</gravity>
    <max_contacts>10</max_contacts>
  </gazebo>

  <!-- ROS control ke liye Gazebo plugin -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/my_robot</robotNamespace>
      <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    </plugin>
  </gazebo>

</robot>
```

### Sensor Integration in Gazebo

```xml
<!-- Camera sensor -->
<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <always_on>true</always_on>
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <format>R8G8B8</format>
        <width>800</width>
        <height>600</height>
      </image>
      <clip>
        <near>0.02</near>
        <far>300</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
      <min_depth>0.1</min_depth>
      <max_depth>10.0</max_depth>
    </plugin>
  </sensor>
</gazebo>

<!-- Laser scanner -->
<gazebo reference="laser_link">
  <sensor type="ray" name="laser_scan">
    <always_on>true</always_on>
    <update_rate>40</update_rate>
    <ray>
      <scan>
        <horizontal>
          <samples>720</samples>
          <resolution>1</resolution>
          <min_angle>-1.570796</min_angle>
          <max_angle>1.570796</max_angle>
        </horizontal>
      </scan>
      <range>
        <min>0.10</min>
        <max>30.0</max>
        <resolution>0.01</resolution>
      </range>
    </ray>
    <plugin name="laser_controller" filename="libgazebo_ros_laser.so">
      <frame_name>laser_link</frame_name>
      <topic_name>scan</topic_name>
    </plugin>
  </sensor>
</gazebo>

<!-- IMU sensor -->
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
  </sensor>
</gazebo>
```

### Physics Properties

```xml
<!-- Specific links ke liye custom physics properties -->
<gazebo reference="wheel_link">
  <mu1>1.0</mu1>  <!-- Friction coefficient 1 -->
  <mu2>1.0</mu2>  <!-- Friction coefficient 2 -->
  <kp>1000000.0</kp>  <!-- Contact stiffness -->
  <kd>100.0</kd>     <!-- Contact damping -->
  <min_depth>0.001</min_depth>  <!-- Penetration depth -->
  <max_vel>1.0</max_vel>        <!-- Maximum contact correction velocity -->
  <fdir1>1 0 0</fdir1>          <!-- Friction direction -->
  <material>Gazebo/Black</material>
</gazebo>
```

## Advanced Xacro Macros for Humanoid Robots

### Humanoid Limb Macro

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="macro_humanoid">

  <!-- Define humanoid limb macro -->
  <xacro:macro name="humanoid_arm" params="side parent_link position shoulder_limits:=1.57 elbow_limits:=1.57">
    <!-- Shoulder joint -->
    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent_link}"/>
      <child link="${side}_shoulder_link"/>
      <origin xyz="${position}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-shoulder_limits}" upper="${shoulder_limits}" effort="50" velocity="2"/>
    </joint>

    <link name="${side}_shoulder_link">
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
    <joint name="${side}_upper_arm_joint" type="revolute">
      <parent link="${side}_shoulder_link"/>
      <child link="${side}_upper_arm_link"/>
      <origin xyz="0 0 0.1"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-elbow_limits}" upper="${elbow_limits}" effort="50" velocity="2"/>
    </joint>

    <link name="${side}_upper_arm_link">
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.002"/>
      </inertial>
      <visual>
        <geometry>
          <cylinder radius="0.04" length="0.3"/>
        </geometry>
        <material name="arm_material"/>
      </visual>
    </link>

    <!-- Lower arm -->
    <joint name="${side}_lower_arm_joint" type="revolute">
      <parent link="${side}_upper_arm_link"/>
      <child link="${side}_lower_arm_link"/>
      <origin xyz="0 0 0.3"/>
      <axis xyz="1 0 0"/>
      <limit lower="${-elbow_limits}" upper="0" effort="30" velocity="2"/>
    </joint>

    <link name="${side}_lower_arm_link">
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
    <joint name="${side}_hand_joint" type="revolute">
      <parent link="${side}_lower_arm_link"/>
      <child link="${side}_hand_link"/>
      <origin xyz="0 0 0.25"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.785" upper="0.785" effort="20" velocity="3"/>
    </joint>

    <link name="${side}_hand_link">
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

    <!-- ROS control ke liye transmission add karen -->
    <transmission name="${side}_arm_transmission" type="transmission_interface/SimpleTransmission">
      <joint name="${side}_shoulder_joint">
        <hardwareInterface>PositionJointInterface</hardwareInterface>
      </joint>
      <actuator name="${side}_shoulder_actuator">
        <hardwareInterface>PositionJointInterface</hardwareInterface>
        <mechanicalReduction>1</mechanicalReduction>
      </actuator>
    </transmission>
  </xacro:macro>

  <!-- Macros ka istemal karke dono arms banayein -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
    </visual>
  </link>

  <xacro:humanoid_arm side="left" parent_link="torso" position="0.15 0.1 0.1"/>
  <xacro:humanoid_arm side="right" parent_link="torso" position="0.15 -0.1 0.1"/>

</robot>
```

## Gazebo Controllers aur Plugins

### ROS Control Plugin Configuration

```xml
<!-- ROS control ke sath complete robot -->
<gazebo>
  <plugin name="ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/humanoid_robot</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    <legacyModeNS>true</legacyModeNS>
  </plugin>
</gazebo>

<!-- Joint state publisher -->
<gazebo>
  <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
    <robotNamespace>/humanoid_robot</robotNamespace>
    <jointName>left_shoulder_joint, right_shoulder_joint, left_elbow_joint, right_elbow_joint</jointName>
    <updateRate>30</updateRate>
  </plugin>
</gazebo>
```

### Custom Gazebo Plugins

```xml
<!-- Specialized behavior ke liye custom plugin -->
<gazebo>
  <plugin name="balance_controller" filename="libbalance_controller.so">
    <robotNamespace>/humanoid_robot</robotNamespace>
    <controlTopic>/balance_control</controlTopic>
    <imuTopic>/imu/data</imuTopic>
    <updateRate>100</updateRate>
    <kp>10.0</kp>
    <ki>1.0</ki>
    <kd>5.0</kd>
  </plugin>
</gazebo>
```

## Advanced Material Definitions

### Texture aur Material Properties

```xml
<!-- Textures ke sath materials define karen -->
<material name="carbon_fiber">
  <color rgba="0.2 0.2 0.2 1"/>
  <texture filename="package://my_robot/materials/carbon_fiber.png"/>
</material>

<material name="rubber_tire">
  <color rgba="0.1 0.1 0.1 1"/>
  <texture filename="package://my_robot/materials/rubber_tire.png"/>
</material>

<!-- Gazebo material properties -->
<gazebo reference="wheel_link">
  <material>humanoid_robot/rubber_tire</material>
  <turnGravityOff>false</turnGravityOff>
</gazebo>
```

## URDF Organization Best Practices

### Modular URDF Structure

```xml
<!-- Main robot file -->
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro" name="modular_robot">

  <!-- Doosre files include karen -->
  <xacro:include filename="$(find my_robot_description)/urdf/macros.xacro"/>
  <xacro:include filename="$(find my_robot_description)/urdf/base.urdf.xacro"/>
  <xacro:include filename="$(find my_robot_description)/urdf/sensors.urdf.xacro"/>
  <xacro:include filename="$(find my_robot_description)/urdf/transmissions.urdf.xacro"/>

  <!-- Included components ka istemal karen -->
  <xacro:robot_base/>
  <xacro:robot_sensors/>
  <xacro:robot_transmissions/>

</robot>
```

### Macros File (macros.xacro)

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://ros.org/wiki/xacro">

  <!-- Common properties -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>

  <!-- Wheel macro -->
  <xacro:macro name="wheel" params="prefix parent xyz:=0 0 0 rpy:=0 0 0 radius width">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${radius}" length="${width}"/>
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${radius}" length="${width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.02"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 1 0"/>
    </joint>
  </xacro:macro>

</robot>
```

## Validation aur Debugging

### Xacro Validation

```bash
# Check karen kya xacro syntax sahi hai
xacro --check-order my_robot.urdf.xacro

# Xacro ko URDF mein process karen
xacro my_robot.urdf.xacro > my_robot.urdf

# URDF validate karen
check_urdf my_robot.urdf
```

### Python Validation Tools

```python
import rclpy
from rclpy.node import Node
from urdf_parser_py.urdf import URDF
from xacro import process_file
import os

class URDFValidatorNode(Node):
    def __init__(self):
        super().__init__('urdf_validator')

        # Xacro aur URDF validate karen
        self.validate_xacro_urdf('path/to/robot.urdf.xacro')

    def validate_xacro_urdf(self, xacro_path):
        """Xacro file aur converted URDF validate karen"
        try:
            # Xacro ko URDF mein process karen
            urdf_string = process_file(xacro_path).toprettyxml(indent='  ')

            # URDF parse karen
            robot = URDF.from_xml_string(urdf_string)

            self.get_logger().info(f'Robot name: {robot.name}')
            self.get_logger().info(f'Links: {len(robot.links)}')
            self.get_logger().info(f'Joints: {len(robot.joints)}')

            # Joint connections validate karen
            self.validate_kinematic_chain(robot)

        except Exception as e:
            self.get_logger().error(f'Validation error: {str(e)}')

    def validate_kinematic_chain(self, robot):
        """Validate kya robot ke paas proper kinematic tree hai"
        # Check karen kya exactly ek root link hai (koi parent nahi)
        child_links = set()
        for joint in robot.joints:
            child_links.add(joint.child)

        root_links = []
        for link in robot.links:
            if link.name not in child_links:
                root_links.append(link.name)

        if len(root_links) != 1:
            self.get_logger().warn(f'Expected 1 root link, found {len(root_links)}: {root_links}')
        else:
            self.get_logger().info(f'Valid kinematic chain with root: {root_links[0]}')

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

Advanced URDF features including Xacro parameterization, transmissions, aur Gazebo integration complex robotic applications ke liye sophisticated, reusable, aur simulation-ready robot models banane ke liye zaruri tools provide karti hain.