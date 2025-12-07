---
title: "Parameters aur Launch Systems"
description: "ROS 2 parameters ke through configuration management aur humanoid robotics ke liye launch files ke sath robot systems ko organize karna"
sidebar_position: 5
keywords: ["ROS 2 parameters", "launch files", "configuration", "humanoid robotics", "system management"]
---

# Parameters aur Launch Systems

Configuration management aur system orchestration humanoid robotics applications ke liye critical hain. ROS 2 powerful mechanisms provide karta hai parameters ko manage karne aur complex robot systems ko launch karne ke liye through its parameters system aur launch framework.

## Parameters: Configuration Management

ROS 2 mein parameters aapko nodes ka behavior configure karne ki allow karti hain bina code ko recompile kiye. Yeh humanoid robotics ke liye essential hai jahan different robots ke different physical characteristics, calibration values, ya operational parameters ho sakti hain.

### Parameter Declaration aur Usage

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_system_default

class HumanoidConfigurableNode(Node):
    def __init__(self):
        super().__init__('humanoid_configurable_node')

        # Default values aur descriptions ke sath parameters declare karen
        self.declare_parameter('control_frequency', 100,
                              'Control loop frequency in Hz')
        self.declare_parameter('max_joint_velocity', 2.0,
                              'Maximum joint velocity in rad/s')
        self.declare_parameter('balance_threshold', 0.05,
                              'Balance threshold in meters')
        self.declare_parameter('robot_name', 'default_robot',
                              'Name of the robot instance')
        self.declare_parameter('foot_dimensions', [0.2, 0.1, 0.05],
                              'Foot dimensions [length, width, height] in meters')
        self.declare_parameter('gait_parameters',
                              {'step_height': 0.05, 'step_duration': 1.0},
                              'Gait parameters dictionary')

        # Parameter values ko access karen
        self.control_freq = self.get_parameter('control_frequency').value
        self.max_velocity = self.get_parameter('max_joint_velocity').value
        self.balance_threshold = self.get_parameter('balance_threshold').value
        self.robot_name = self.get_parameter('robot_name').value
        self.foot_dims = self.get_parameter('foot_dimensions').value
        self.gait_params = self.get_parameter('gait_parameters').value

        self.get_logger().info(f'Initialized {self.robot_name} with control freq: {self.control_freq}Hz')

        # Dynamic reconfiguration ke liye parameter callback set karen
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        """Parameter changes ko handle karen"""
        for param in params:
            if param.name == 'control_frequency' and param.type_ == Parameter.Type.INTEGER:
                if param.value >= 10 and param.value <= 1000:  # Valid range
                    self.control_freq = param.value
                    self.get_logger().info(f'Control frequency updated to {param.value}Hz')
                else:
                    return rclpy.node.SetParametersResult(
                        successful=False,
                        reason='Control frequency must be between 10 and 1000 Hz'
                    )
            elif param.name == 'max_joint_velocity' and param.type_ == Parameter.Type.DOUBLE:
                if param.value > 0.0:
                    self.max_velocity = param.value
                else:
                    return rclpy.node.SetParametersResult(
                        successful=False,
                        reason='Max joint velocity must be positive'
                    )

        return rclpy.node.SetParametersResult(successful=True)
```

### Parameter Types aur Validation

ROS 2 various parameter types ko support karta hai with built-in validation:

```python
class ParameterValidationNode(Node):
    def __init__(self):
        super().__init__('parameter_validation_node')

        # Integer parameters
        self.declare_parameter('thread_priority', 50,
                              descriptor={'integer_range': [{'from_value': 1, 'to_value': 99, 'step': 1}]})

        # Range ke sath double parameters
        self.declare_parameter('control_gain', 1.0,
                              descriptor={'floating_point_range': [{'from_value': 0.0, 'to_value': 10.0, 'step': 0.1}]})

        # Boolean parameters
        self.declare_parameter('enable_logging', True)

        # Allowed values ke sath string parameters
        self.declare_parameter('control_mode', 'position',
                              descriptor={'description': 'Control mode: position, velocity, or effort'})

        # Array parameters
        self.declare_parameter('joint_limits', [1.5, 1.5, 2.0, 2.0])

    def validate_parameters(self):
        """Additional validation logic"""
        control_mode = self.get_parameter('control_mode').value
        if control_mode not in ['position', 'velocity', 'effort']:
            self.get_logger().error(f'Invalid control mode: {control_mode}')
            return False
        return True
```

## Launch Systems: System Orchestration

Launch files aapko multiple nodes ko specific configurations ke sath simultaneously start karne ki allow karti hain. Humanoid robotics ke liye, yeh complete robot system ko bring up karne ke liye essential hai.

### Basic Launch File

```python
# launch/humanoid_system.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import TextSubstitution

def generate_launch_description():
    # Launch arguments declare karen
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_name = LaunchConfiguration('robot_name', default='humanoid_robot')
    config_file = LaunchConfiguration('config_file',
                                    default=PathJoinSubstitution([
                                        FindPackageShare('humanoid_bringup'),
                                        'config',
                                        'humanoid_config.yaml'
                                    ]))

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'rate': 50}  # 50Hz update rate
        ],
        remappings=[
            ('/joint_states', '/robot/joint_states')
        ]
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'publish_frequency': 50.0}
        ],
        arguments=[PathJoinSubstitution([
            FindPackageShare('humanoid_description'),
            'urdf',
            'humanoid.urdf.xacro'
        ])],
        remappings=[
            ('/joint_states', '/robot/joint_states')
        ]
    )

    # Joint controller manager
    controller_manager = Node(
        package='controller_manager',
        executable='ros2_control_node',
        name='controller_manager',
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('/joint_states', '/robot/joint_states')
        ]
    )

    # Balance controller
    balance_controller = Node(
        package='humanoid_control',
        executable='balance_controller',
        name='balance_controller',
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time},
            {'control_frequency': 200},  # 200Hz for balance control
            {'robot_name': robot_name}
        ]
    )

    # Gait generator
    gait_generator = Node(
        package='humanoid_locomotion',
        executable='gait_generator',
        name='gait_generator',
        parameters=[
            config_file,
            {'use_sim_time': use_sim_time},
            {'step_height': 0.05},
            {'step_duration': 1.0}
        ]
    )

    # Launch description
    ld = LaunchDescription()

    # Launch arguments add karen
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    ))

    ld.add_action(DeclareLaunchArgument(
        'robot_name',
        default_value='humanoid_robot',
        description='Name of the robot'
    ))

    ld.add_action(DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_bringup'),
            'config',
            'humanoid_config.yaml'
        ]),
        description='Path to configuration file'
    ))

    # Launch description mein nodes add karen
    ld.add_action(joint_state_publisher)
    ld.add_action(robot_state_publisher)
    ld.add_action(controller_manager)
    ld.add_action(balance_controller)
    ld.add_action(gait_generator)

    return ld
```

### Advanced Launch Configuration

```python
# launch/humanoid_with_rviz.launch.py
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
    SetEnvironmentVariable
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time')
    robot_name = LaunchConfiguration('robot_name')
    rviz_config = LaunchConfiguration('rviz_config')
    enable_rviz = LaunchConfiguration('enable_rviz', default='true')

    # Robot namespace ke neechhe nodes ko group karen
    robot_group = GroupAction(
        actions=[
            PushRosNamespace(robot_name),

            # Basic robot system launch include karen
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    FindPackageShare('humanoid_bringup'),
                    '/launch/humanoid_system.launch.py'
                ]),
                launch_arguments={
                    'use_sim_time': use_sim_time,
                    'robot_name': robot_name
                }.items()
            ),
        ]
    )

    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(enable_rviz),
        parameters=[
            {'use_sim_time': use_sim_time}
        ]
    )

    # Diagnostic aggregator
    diagnostic_aggregator = Node(
        package='diagnostic_aggregator',
        executable='aggregator_node',
        name='diagnostic_aggregator',
        parameters=[
            PathJoinSubstitution([
                FindPackageShare('humanoid_bringup'),
                'config',
                'diagnostics.yaml'
            ])
        ]
    )

    # Launch description
    ld = LaunchDescription()

    # Launch arguments
    ld.add_action(DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time if true'
    ))

    ld.add_action(DeclareLaunchArgument(
        'robot_name',
        default_value='humanoid_robot',
        description='Name of the robot'
    ))

    ld.add_action(DeclareLaunchArgument(
        'rviz_config',
        default_value=PathJoinSubstitution([
            FindPackageShare('humanoid_bringup'),
            'rviz',
            'humanoid_view.rviz'
        ]),
        description='RViz config file'
    ))

    # Launch description mein actions add karen
    ld.add_action(robot_group)
    ld.add_action(rviz_node)
    ld.add_action(diagnostic_aggregator)

    return ld
```

## YAML Configuration Files

YAML files complex humanoid systems ke liye structured configuration provide karti hain:

```yaml
# config/humanoid_config.yaml
/**:
  ros__parameters:
    # Robot physical parameters
    robot_name: "humanoid_robot"
    robot_mass: 30.0  # kg
    foot_dimensions:
      length: 0.2
      width: 0.1
      height: 0.05

    # Control parameters
    control_frequency: 200
    balance_threshold: 0.05
    max_joint_velocity: 2.0
    max_joint_acceleration: 5.0

    # Gait parameters
    gait:
      step_height: 0.05
      step_duration: 1.0
      step_width: 0.3
      nominal_height: 0.8

    # Joint mapping
    joint_names:
      - left_hip_yaw
      - left_hip_roll
      - left_hip_pitch
      - left_knee_pitch
      - left_ankle_pitch
      - left_ankle_roll
      - right_hip_yaw
      - right_hip_roll
      - right_hip_pitch
      - right_knee_pitch
      - right_ankle_pitch
      - right_ankle_roll

    # Joint limits
    joint_limits:
      hip_yaw:
        min: -1.0
        max: 1.0
      hip_roll:
        min: -0.5
        max: 0.5
      hip_pitch:
        min: -1.5
        max: 1.5
      knee_pitch:
        min: -0.5
        max: 2.0
      ankle_pitch:
        min: -0.8
        max: 0.8
      ankle_roll:
        min: -0.5
        max: 0.5

    # Sensor parameters
    imu_frame_id: "imu_link"
    camera_frame_id: "camera_link"
    base_frame_id: "base_link"
    world_frame_id: "world"
```

## Parameter Management Best Practices

### Dynamic Parameter Updates

```python
import rclpy
from rclpy.parameter import Parameter
from rclpy.node import Node

class DynamicParameterNode(Node):
    def __init__(self):
        super().__init__('dynamic_parameter_node')

        # Parameters declare karen
        self.declare_parameter('walking_speed', 0.5)
        self.declare_parameter('step_height', 0.05)
        self.declare_parameter('enable_balance', True)

        # Parameter change publisher set karen
        self.param_change_pub = self.create_publisher(
            Parameter, '/parameter_updates', 10)

        # Parameter changes ko check karne ke liye timer
        self.param_timer = self.create_timer(1.0, self.check_parameters)

        # Comparison ke liye previous values store karen
        self.prev_speed = self.get_parameter('walking_speed').value
        self.prev_height = self.get_parameter('step_height').value
        self.prev_balance = self.get_parameter('enable_balance').value

    def check_parameters(self):
        """Parameter changes ko check karen aur accordingly react karen"""
        current_speed = self.get_parameter('walking_speed').value
        current_height = self.get_parameter('step_height').value
        current_balance = self.get_parameter('enable_balance').value

        # Changes detect karen aur react karen
        if current_speed != self.prev_speed:
            self.get_logger().info(f'Walking speed changed to {current_speed}')
            self.handle_speed_change(current_speed)
            self.prev_speed = current_speed

        if current_height != self.prev_height:
            self.get_logger().info(f'Step height changed to {current_height}')
            self.handle_height_change(current_height)
            self.prev_height = current_height

        if current_balance != self.prev_balance:
            self.get_logger().info(f'Balance control enabled: {current_balance}')
            self.handle_balance_change(current_balance)
            self.prev_balance = current_balance

    def handle_speed_change(self, new_speed):
        """Walking speed parameter change ko handle karen"""
        # New speed ke adhar par control algorithms adjust karen
        pass

    def handle_height_change(self, new_height):
        """Step height parameter change ko handle karen"""
        # Gait generation parameters update karen
        pass

    def handle_balance_change(self, enabled):
        """Balance control enable/disable ko handle karen"""
        # Balance control system enable/disable karen
        pass
```

### Parameter Validation aur Constraints

```python
class ParameterConstraintNode(Node):
    def __init__(self):
        super().__init__('parameter_constraint_node')

        # Constraints ke sath parameters declare karen
        self.declare_parameter('control_loop_frequency', 100,
                              descriptor={'integer_range': [{'from_value': 10, 'to_value': 1000, 'step': 1}]})

        self.declare_parameter('safety_margin', 0.1,
                              descriptor={'floating_point_range': [{'from_value': 0.01, 'to_value': 0.5, 'step': 0.01}]})

        self.declare_parameter('debug_level', 'info',
                              descriptor={'description': 'Log level: debug, info, warn, error'})

    def add_on_set_parameters_callback(self, param_callback):
        """Override karen custom validation ke liye"""
        def callback(params):
            for param in params:
                if param.name == 'control_loop_frequency':
                    if param.value < 10 or param.value > 1000:
                        return SetParametersResult(
                            successful=False,
                            reason='Control loop frequency must be between 10 and 1000 Hz'
                        )
                elif param.name == 'safety_margin':
                    if param.value < 0.01 or param.value > 0.5:
                        return SetParametersResult(
                            successful=False,
                            reason='Safety margin must be between 0.01 and 0.5'
                        )
            return SetParametersResult(successful=True)

        super().add_on_set_parameters_callback(callback)
```

## Launch File Best Practices for Humanoid Robotics

:::tip Launch File Best Practices
- Nodes ke liye descriptive names use karen purpose identify karne ke liye
- Nodes ko logical groups mein organize karen (sensors, controllers, perception)
- Configuration flexibility ke liye launch arguments use karen
- Proper error handling aur logging implement karen
- Multi-robot systems mein naming conflicts se bachne ke liye namespaces use karen
- System health monitoring ke liye diagnostic nodes include karen
:::

:::note Parameter Management Tips
- Complex parameter structures ke liye YAML files use karen
- Node initialization ke dauran parameters validate karen
- Dynamic reconfiguration ke liye parameter callbacks implement karen
- Validation ke liye appropriate parameter types use karen
- Parameter meanings aur valid ranges document karen
- Safety ke liye parameter constraints consider karen
:::

## Command Line Parameter Operations

Aap command line se bhi parameters ke sath kaam kar sakte hain:

```bash
# Kisi node ke saare parameters list karen
ros2 param list /humanoid_controller

# Kisi specific parameter ko get karen
ros2 param get /humanoid_controller control_frequency

# Koi parameter set karen
ros2 param set /humanoid_controller control_frequency 150

# File se parameters load karen
ros2 param load /humanoid_controller config/robot_params.yaml

# Current parameters ko file mein save karen
ros2 param dump /humanoid_controller --output config/current_params.yaml
```

## Key Takeaways

- Parameters code recompilation ke bina flexible configuration provide karti hain
- Launch files complex robot systems ke coordinated startup ko enable karti hain
- Parameter validation safe configuration values ko ensure karti hain
- Dynamic parameter updates runtime configuration changes ko allow karti hain
- YAML configuration files structured parameter organization provide karti hain
- Proper launch file organization humanoid robotics systems ke liye essential hai
- Namespacing multi-robot deployments mein naming conflicts ko prevent karti hai

Parameters aur launch systems ko samajhna different robots, environments, aur operational requirements ke liye adapt karne wale humanoid robotics applications deploy aur configure karne ke liye crucial hai while maintaining safety aur reliability.