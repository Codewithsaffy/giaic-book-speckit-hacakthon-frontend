---
title: "URDF for Control Integration"
description: "Integrating URDF models with ROS 2 control systems and robot state publishing"
sidebar_position: 4
keywords: [urdf, control, robot state, joint state, tf, ros control, integration]
---

# URDF for Control Integration

URDF models must be properly integrated with ROS 2 control systems to enable effective robot control and state estimation. This section covers the integration of URDF with robot state publishing, joint state management, and ROS Control frameworks.

## Robot State Publisher

The Robot State Publisher is crucial for transforming URDF joint states into TF transforms, enabling spatial awareness and visualization.

### Basic Robot State Publisher Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header
import math

class RobotStatePublisher(Node):
    def __init__(self):
        super().__init__('robot_state_publisher')

        # Initialize joint state storage
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_efforts = {}

        # Subscribe to joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Initialize TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing transforms
        self.timer = self.create_timer(0.05, self.publish_transforms)  # 20 Hz

        self.get_logger().info('Robot State Publisher initialized')

    def joint_state_callback(self, msg):
        """Update joint states from JointState message"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.joint_efforts[name] = msg.effort[i]

    def publish_transforms(self):
        """Publish TF transforms based on current joint states"""
        # This method should be implemented with actual kinematic calculations
        # For now, we'll publish a simple example

        # Example: Publish base_link to sensor_link transform
        if 'sensor_joint' in self.joint_positions:
            t = TransformStamped()

            # Header
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'base_link'
            t.child_frame_id = 'sensor_link'

            # Transform (example values)
            t.transform.translation.x = 0.1
            t.transform.translation.y = 0.0
            t.transform.translation.z = 0.15
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

            # Publish transform
            self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = RobotStatePublisher()

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

### Advanced Robot State Publisher with URDF Parsing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from urdf_parser_py.urdf import URDF
import tf_transformations
import numpy as np

class AdvancedRobotStatePublisher(Node):
    def __init__(self):
        super().__init__('advanced_robot_state_publisher')

        # Declare parameters
        self.declare_parameter('robot_description', '')
        self.declare_parameter('publish_frequency', 50.0)

        # Load robot description
        robot_description = self.get_parameter('robot_description').value
        if robot_description:
            self.robot = URDF.from_xml_string(robot_description)
        else:
            # Load from parameter server
            robot_description_param = self.get_parameter_or('robot_description', '')
            if robot_description_param.value:
                self.robot = URDF.from_xml_string(robot_description_param.value)
            else:
                self.get_logger().error('No robot description provided')
                return

        # Initialize joint state storage
        self.joint_positions = {joint.name: 0.0 for joint in self.robot.joints}
        self.joint_velocities = {joint.name: 0.0 for joint in self.robot.joints}
        self.joint_efforts = {joint.name: 0.0 for joint in self.robot.joints}

        # Subscribe to joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Initialize TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Calculate publish frequency
        publish_freq = self.get_parameter('publish_frequency').value
        self.timer = self.create_timer(1.0/publish_freq, self.publish_transforms)

        self.get_logger().info(f'Advanced Robot State Publisher initialized with {len(self.robot.joints)} joints')

    def joint_state_callback(self, msg):
        """Update joint states from JointState message"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.joint_positions[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_velocities[name] = msg.velocity[i]
            if i < len(msg.effort):
                self.joint_efforts[name] = msg.effort[i]

    def publish_transforms(self):
        """Publish TF transforms for all joints in the robot chain"""
        transforms = []

        # Process each joint in the robot
        for joint in self.robot.joints:
            if joint.type == 'fixed':
                # Fixed joints - publish static transform
                t = self.create_fixed_transform(joint)
                transforms.append(t)
            elif joint.type in ['revolute', 'continuous', 'prismatic']:
                # Moveable joints - use current joint position
                t = self.create_moving_transform(joint)
                transforms.append(t)

        # Broadcast all transforms
        self.tf_broadcaster.sendTransform(transforms)

    def create_fixed_transform(self, joint):
        """Create transform for fixed joints"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = joint.parent
        t.child_frame_id = joint.child

        # Extract origin from joint
        origin = joint.origin
        t.transform.translation.x = origin.xyz[0] if origin else 0.0
        t.transform.translation.y = origin.xyz[1] if origin else 0.0
        t.transform.translation.z = origin.xyz[2] if origin else 0.0

        # Convert RPY to quaternion
        rpy = origin.rpy if origin else [0.0, 0.0, 0.0]
        quaternion = tf_transformations.quaternion_from_euler(rpy[0], rpy[1], rpy[2])
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        return t

    def create_moving_transform(self, joint):
        """Create transform for moveable joints"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = joint.parent
        t.child_frame_id = joint.child

        # Extract origin from joint
        origin = joint.origin
        t.transform.translation.x = origin.xyz[0] if origin else 0.0
        t.transform.translation.y = origin.xyz[1] if origin else 0.0
        t.transform.translation.z = origin.xyz[2] if origin else 0.0

        # Apply joint transformation based on joint type and current position
        current_pos = self.joint_positions.get(joint.name, 0.0)

        if joint.type in ['revolute', 'continuous']:
            # Revolute joints rotate around their axis
            axis = joint.axis if joint.axis else [0, 0, 1]
            quaternion = tf_transformations.quaternion_about_axis(current_pos, axis)
            t.transform.rotation.x = quaternion[0]
            t.transform.rotation.y = quaternion[1]
            t.transform.rotation.z = quaternion[2]
            t.transform.rotation.w = quaternion[3]
        elif joint.type == 'prismatic':
            # Prismatic joints translate along their axis
            axis = joint.axis if joint.axis else [1, 0, 0]
            t.transform.translation.x += axis[0] * current_pos
            t.transform.translation.y += axis[1] * current_pos
            t.transform.translation.z += axis[2] * current_pos
            t.transform.rotation.w = 1.0  # No rotation for prismatic

        return t

def main(args=None):
    rclpy.init(args=args)
    node = AdvancedRobotStatePublisher()

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

## Joint State Publisher

The Joint State Publisher provides simulated joint states for visualization and testing when real hardware is not available.

### Joint State Publisher Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import math
import time

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Declare parameters
        self.declare_parameter('publish_rate', 50.0)
        self.declare_parameter('joint_names', [])
        self.declare_parameter('initial_positions', [])

        # Get parameters
        self.publish_rate = self.get_parameter('publish_rate').value
        self.joint_names = self.get_parameter('joint_names').value
        initial_positions = self.get_parameter('initial_positions').value

        # Initialize joint positions
        if initial_positions:
            self.joint_positions = dict(zip(self.joint_names, initial_positions))
        else:
            self.joint_positions = {name: 0.0 for name in self.joint_names}

        # Create publisher
        self.joint_publisher = self.create_publisher(
            JointState,
            'joint_states',
            10
        )

        # Timer for publishing joint states
        self.timer = self.create_timer(1.0/self.publish_rate, self.publish_joint_states)

        # Initialize time for dynamic joint movements
        self.start_time = time.time()

        self.get_logger().info(f'Joint State Publisher initialized with {len(self.joint_names)} joints')

    def publish_joint_states(self):
        """Publish joint states message"""
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Update dynamic joint positions (example: oscillating motion)
        current_time = time.time() - self.start_time
        for joint_name in self.joint_names:
            # Add some dynamic motion for demonstration
            if 'oscillating' in joint_name.lower():
                self.joint_positions[joint_name] = 0.5 * math.sin(current_time)
            elif 'rotating' in joint_name.lower():
                self.joint_positions[joint_name] = current_time * 0.1

        # Set joint names and positions
        msg.name = self.joint_names
        msg.position = [self.joint_positions[name] for name in self.joint_names]

        # Set velocities and efforts (optional)
        msg.velocity = [0.0] * len(self.joint_names)
        msg.effort = [0.0] * len(self.joint_names)

        # Publish the message
        self.joint_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    # Example usage with parameters
    node = JointStatePublisher()

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

## ROS Control Integration

### ROS Control Hardware Interface

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectoryPoint
import numpy as np

class HardwareInterface(Node):
    def __init__(self):
        super().__init__('hardware_interface')

        # Define joint names and properties
        self.joint_names = [
            'left_shoulder_joint', 'left_elbow_joint', 'right_shoulder_joint', 'right_elbow_joint'
        ]

        # Initialize joint positions, velocities, and efforts
        self.joint_positions = np.zeros(len(self.joint_names))
        self.joint_velocities = np.zeros(len(self.joint_names))
        self.joint_efforts = np.zeros(len(self.joint_names))

        # Command storage
        self.joint_commands = np.zeros(len(self.joint_names))
        self.command_timestamp = None

        # Publishers and subscribers
        self.joint_state_publisher = self.create_publisher(
            JointState,
            'joint_states',
            QoSProfile(depth=1)
        )

        self.command_subscriber = self.create_subscription(
            JointTrajectoryPoint,
            'joint_commands',
            self.command_callback,
            QoSProfile(depth=1)
        )

        # Timer for publishing joint states
        self.publish_timer = self.create_timer(0.01, self.publish_joint_states)  # 100 Hz

        # Simulate hardware update timer
        self.update_timer = self.create_timer(0.001, self.update_hardware)  # 1000 Hz

        self.get_logger().info('Hardware Interface initialized')

    def command_callback(self, msg):
        """Receive joint commands"""
        if len(msg.positions) == len(self.joint_names):
            self.joint_commands = np.array(msg.positions)
            self.command_timestamp = self.get_clock().now()
            self.get_logger().debug(f'Received commands: {self.joint_commands}')

    def update_hardware(self):
        """Simulate hardware update (in real hardware, this would interface with actual motors)"""
        # Simple PD controller simulation
        kp = 10.0  # Proportional gain
        kd = 1.0   # Derivative gain

        for i in range(len(self.joint_names)):
            # Calculate error
            error = self.joint_commands[i] - self.joint_positions[i]

            # Simple PD control (in real hardware, this would be more sophisticated)
            command_effort = kp * error - kd * self.joint_velocities[i]

            # Update joint state with simulated dynamics
            # This is a very simplified model - real hardware would have proper dynamics
            acceleration = command_effort  # Simplified: F = ma, assuming m=1
            self.joint_velocities[i] += acceleration * 0.001  # dt = 0.001s
            self.joint_positions[i] += self.joint_velocities[i] * 0.001

    def publish_joint_states(self):
        """Publish current joint states"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = self.joint_positions.tolist()
        msg.velocity = self.joint_velocities.tolist()
        msg.effort = self.joint_efforts.tolist()

        self.joint_state_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = HardwareInterface()

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

### Joint Trajectory Controller Interface

```python
import rclpy
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from rclpy.action import ActionServer, GoalResponse, CancelResponse
import time

class JointTrajectoryController(Node):
    def __init__(self):
        super().__init__('joint_trajectory_controller')

        # Initialize joint information
        self.joint_names = ['joint1', 'joint2', 'joint3']
        self.current_positions = [0.0, 0.0, 0.0]
        self.current_velocities = [0.0, 0.0, 0.0]
        self.current_accelerations = [0.0, 0.0, 0.0]

        # Action server for trajectory execution
        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            'joint_trajectory_controller/follow_joint_trajectory',
            execute_callback=self.execute_trajectory,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Publisher for current trajectory state
        self.state_publisher = self.create_publisher(
            JointTrajectory,
            'joint_trajectory_controller/state',
            10
        )

        self.get_logger().info('Joint Trajectory Controller initialized')

    def goal_callback(self, goal_request):
        """Accept or reject trajectory goals"""
        # Check if trajectory is valid
        if len(goal_request.trajectory.joint_names) != len(self.joint_names):
            self.get_logger().error('Joint name count mismatch')
            return GoalResponse.REJECT

        # Check if all joint names match
        if set(goal_request.trajectory.joint_names) != set(self.joint_names):
            self.get_logger().error('Joint names do not match')
            return GoalResponse.REJECT

        self.get_logger().info('Trajectory goal accepted')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject trajectory cancellation"""
        self.get_logger().info('Trajectory cancellation requested')
        return CancelResponse.ACCEPT

    def execute_trajectory(self, goal_handle):
        """Execute the joint trajectory"""
        self.get_logger().info('Executing joint trajectory')

        trajectory = goal_handle.request.trajectory
        points = trajectory.points
        joint_names = trajectory.joint_names

        # Execute each point in the trajectory
        for i, point in enumerate(points):
            if goal_handle.is_canceling():
                goal_handle.canceled()
                return FollowJointTrajectory.Result()

            # Move to the specified joint positions
            self.move_to_point(point, joint_names)

            # Publish feedback
            feedback = FollowJointTrajectory.Feedback()
            feedback.joint_names = joint_names
            feedback.actual.positions = self.current_positions
            feedback.actual.velocities = self.current_velocities
            feedback.desired = point
            feedback.error.positions = [
                actual - desired
                for actual, desired in zip(self.current_positions, point.positions)
            ]

            goal_handle.publish_feedback(feedback)

        # Complete successfully
        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    def move_to_point(self, point, joint_names):
        """Move joints to specified point with interpolation"""
        # In a real implementation, this would send commands to hardware
        # For simulation, we'll just update the positions

        # Find indices of joints in our controller
        for i, joint_name in enumerate(joint_names):
            try:
                controller_idx = self.joint_names.index(joint_name)
                self.current_positions[controller_idx] = point.positions[i]

                if len(point.velocities) > i:
                    self.current_velocities[controller_idx] = point.velocities[i]

                if len(point.accelerations) > i:
                    self.current_accelerations[controller_idx] = point.accelerations[i]

            except ValueError:
                self.get_logger().warn(f'Joint {joint_name} not found in controller')

        # Simulate movement time
        if point.time_from_start.sec > 0 or point.time_from_start.nanosec > 0:
            duration = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
            time.sleep(duration)

def main(args=None):
    rclpy.init(args=args)
    node = JointTrajectoryController()

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

## Control Configuration Files

### Controller Manager Configuration

```yaml
# config/controller_manager.yaml
controller_manager:
  ros__parameters:
    update_rate: 100  # Hz
    use_sim_time: true

    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster

    joint_trajectory_controller:
      type: joint_trajectory_controller/JointTrajectoryController

joint_trajectory_controller:
  ros__parameters:
    type: joint_trajectory_controller/JointTrajectoryController
    joints:
      - left_shoulder_joint
      - left_elbow_joint
      - right_shoulder_joint
      - right_elbow_joint

    interface_name: position

    # Command interfaces
    command_interfaces:
      - position

    # State interfaces
    state_interfaces:
      - position
      - velocity
```

### Joint State Controller Configuration

```yaml
# config/joint_state_broadcaster.yaml
joint_state_broadcaster:
  ros__parameters:
    type: joint_state_broadcaster/JointStateBroadcaster
    # Publish rate is handled by controller manager update_rate
```

## Launch Files for Control Integration

### Control System Launch

```python
# launch/control_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_description_path = LaunchConfiguration('robot_description_path')

    # Robot State Publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': PathJoinSubstitution([
                FindPackageShare('my_robot_description'),
                'urdf',
                'my_robot.urdf.xacro'
            ])}
        ]
    )

    # Controller Manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('my_robot_control'),
                'config',
                'controller_manager.yaml'
            ]),
            {'use_sim_time': use_sim_time}
        ],
        output='both'
    )

    # Joint State Broadcaster spawner
    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Joint Trajectory Controller spawner
    joint_trajectory_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller'],
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Delay spawning of joint trajectory controller until joint state broadcaster is running
    delay_joint_trajectory_controller_spawner_after_joint_state_broadcaster_spawner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[joint_trajectory_controller_spawner],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'robot_description_path',
            default_value='urdf/my_robot.urdf.xacro',
            description='Path to robot description file'
        ),
        robot_state_publisher,
        controller_manager,
        joint_state_broadcaster_spawner,
        delay_joint_trajectory_controller_spawner_after_joint_state_broadcaster_spawner,
    ])
```

## Control Integration Best Practices

### 1. Proper URDF Joint Definitions for Control

```xml
<!-- Ensure joints have proper limits and interfaces for control -->
<joint name="controlled_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>
  <!-- Proper limits for the physical joint -->
  <limit lower="-1.57" upper="1.57" effort="100" velocity="2"/>
  <!-- Safety limits -->
  <safety_controller k_position="10" k_velocity="10" soft_lower_limit="-1.5" soft_upper_limit="1.5"/>
</joint>

<!-- Add transmission for ROS Control -->
<transmission name="joint_transmission" type="transmission_interface/SimpleTransmission">
  <joint name="controlled_joint">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="joint_motor">
    <hardwareInterface>PositionJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### 2. Control Loop Frequency Considerations

```python
class OptimizedController(Node):
    def __init__(self):
        super().__init__('optimized_controller')

        # Use appropriate update rates based on control requirements
        # Position control: 100-200 Hz
        # Trajectory following: 50-100 Hz
        # High-frequency control (e.g., impedance): 1000+ Hz

        self.control_loop = self.create_timer(0.01, self.control_callback)  # 100 Hz
        self.state_publisher = self.create_timer(0.02, self.publish_state)  # 50 Hz
```

### 3. Safety and Limits Management

```python
class SafeController(Node):
    def __init__(self):
        super().__init__('safe_controller')

        # Define safety limits
        self.position_limits = {
            'joint1': (-1.57, 1.57),
            'joint2': (-2.0, 2.0),
        }
        self.velocity_limits = {joint: 2.0 for joint in self.position_limits.keys()}
        self.effort_limits = {joint: 100.0 for joint in self.position_limits.keys()}

    def apply_limits(self, commands):
        """Apply safety limits to joint commands"""
        limited_commands = []
        for i, (joint_name, cmd) in enumerate(zip(self.joint_names, commands)):
            # Position limits
            if joint_name in self.position_limits:
                min_pos, max_pos = self.position_limits[joint_name]
                cmd = max(min_pos, min(max_pos, cmd))

            limited_commands.append(cmd)

        return limited_commands
```

## Testing and Validation

### Control System Test Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time

class ControlTestNode(Node):
    def __init__(self):
        super().__init__('control_test_node')

        # Publishers
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10
        )

        # Subscriber for joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_state_callback,
            10
        )

        # Timer to send test trajectories
        self.test_timer = self.create_timer(5.0, self.send_test_trajectory)

        self.joint_states = None
        self.get_logger().info('Control Test Node initialized')

    def joint_state_callback(self, msg):
        """Receive and store joint states"""
        self.joint_states = msg

    def send_test_trajectory(self):
        """Send a simple test trajectory"""
        if not self.joint_states or not self.joint_states.name:
            self.get_logger().warn('No joint states received yet')
            return

        # Create a simple trajectory
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_states.name[:2]  # Use first 2 joints for test

        # Create points for the trajectory
        point1 = JointTrajectoryPoint()
        point1.positions = [0.0, 0.0]
        point1.velocities = [0.0, 0.0]
        point1.time_from_start = Duration(sec=0, nanosec=0)

        point2 = JointTrajectoryPoint()
        point2.positions = [0.5, 0.5]  # Move to 0.5 rad for both joints
        point2.velocities = [0.0, 0.0]
        point2.time_from_start = Duration(sec=2, nanosec=0)  # Reach in 2 seconds

        trajectory.points = [point1, point2]

        self.trajectory_publisher.publish(trajectory)
        self.get_logger().info(f'Sent test trajectory for joints: {trajectory.joint_names}')

def main(args=None):
    rclpy.init(args=args)
    node = ControlTestNode()

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

URDF control integration enables effective robot control by connecting the robot's physical model with ROS 2's control frameworks, providing the necessary infrastructure for joint state management, trajectory execution, and real-time control.