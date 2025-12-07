---
title: "Topics and Messaging"
description: "Deep dive into ROS 2 topics, messaging patterns, and QoS for humanoid robotics communication"
sidebar_position: 3
keywords: ["ROS 2 topics", "publishers", "subscribers", "QoS", "humanoid robotics", "messaging"]
---

# Topics and Messaging

Topics form the backbone of ROS 2's publish-subscribe communication model, enabling asynchronous data exchange between nodes. For humanoid robotics, where multiple sensors continuously generate data streams, understanding topics and messaging patterns is essential for building responsive and efficient systems.

## Publish-Subscribe Architecture

The publish-subscribe pattern allows nodes to communicate without direct dependencies. Publishers send messages to topics, and subscribers receive messages from topics they're interested in. This decoupling is particularly valuable in humanoid robotics where:

- Multiple sensors publish data simultaneously
- Different subsystems consume sensor data at different rates
- Components can be added or removed without affecting others

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Publisher for joint states
        self.joint_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10  # queue size
        )

        # Timer to publish joint states at regular intervals
        self.timer = self.create_timer(0.01, self.publish_joint_states)  # 100Hz

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Example joint names and positions for a humanoid
        msg.name = [
            'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch',
            'left_knee_pitch', 'left_ankle_pitch', 'left_ankle_roll',
            'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch',
            'right_knee_pitch', 'right_ankle_pitch', 'right_ankle_roll'
        ]

        # Current joint positions (example values)
        msg.position = [0.0] * 12
        msg.velocity = [0.0] * 12
        msg.effort = [0.0] * 12

        self.joint_pub.publish(msg)
```

## Quality of Service (QoS) Profiles

QoS profiles allow you to configure how messages are delivered, which is critical for humanoid robotics applications with different reliability and timing requirements:

### Reliability Settings

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class QoSDemonstrationNode(Node):
    def __init__(self):
        super().__init__('qos_demo_node')

        # Reliable communication for critical data
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Best effort for sensor data where occasional drops are acceptable
        best_effort_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Publishers with different QoS
        self.critical_pub = self.create_publisher(
            JointState, '/critical_joint_commands', reliable_qos)

        self.sensor_pub = self.create_publisher(
            Image, '/camera/image_raw', best_effort_qos)
```

### Common QoS Configurations for Humanoid Robotics

:::tip QoS Guidelines for Humanoid Robotics
- **Joint commands**: RELIABLE, low latency, small history
- **Sensor data**: BEST_EFFORT, larger history for buffering
- **Emergency stops**: RELIABLE, TRANSIENT_LOCAL durability
- **State information**: RELIABLE, medium history
:::

## Subscriber Implementation

Subscribers process incoming messages from topics. In humanoid robotics, subscribers often handle sensor fusion and control feedback:

```python
class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')

        # Subscribe to joint states
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10  # queue size
        )

        # Store latest joint states
        self.latest_joint_states = None
        self.joint_names = []
        self.joint_positions = []
        self.joint_velocities = []
        self.joint_efforts = []

    def joint_state_callback(self, msg):
        """Process incoming joint state messages"""
        self.latest_joint_states = msg
        self.joint_names = msg.name
        self.joint_positions = list(msg.position)
        self.joint_velocities = list(msg.velocity)
        self.joint_efforts = list(msg.effort)

        # Perform real-time processing
        self.process_joint_data()

    def process_joint_data(self):
        """Real-time joint data processing"""
        if not self.joint_positions:
            return

        # Example: Check for joint limits
        for i, (name, pos) in enumerate(zip(self.joint_names, self.joint_positions)):
            if abs(pos) > 3.14:  # Example limit
                self.get_logger().warn(f'Joint {name} position limit exceeded: {pos}')
```

## Advanced Messaging Patterns

### Message Filtering and Synchronization

For humanoid robotics, you often need to synchronize data from multiple sensors:

```python
from message_filters import ApproximateTimeSynchronizer, Subscriber
import message_filters

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Create subscribers for different sensors
        self.imu_sub = message_filters.Subscriber(
            self, Imu, '/imu/data')
        self.joint_sub = message_filters.Subscriber(
            self, JointState, '/joint_states')

        # Synchronize messages with approximate time sync
        self.ts = ApproximateTimeSynchronizer(
            [self.imu_sub, self.joint_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )
        self.ts.registerCallback(self.sensor_callback)

    def sensor_callback(self, imu_msg, joint_msg):
        """Process synchronized sensor data"""
        # Fuse IMU and joint data for balance control
        self.fuse_sensor_data(imu_msg, joint_msg)
```

### Custom Message Types

For humanoid-specific data, you might need custom messages:

```python
# Custom message: HumanoidState.msg
# Header header
# float64[] joint_positions
# float64[] joint_velocities
# geometry_msgs/Point center_of_mass
# bool in_balance
# float64 balance_margin

from std_msgs.msg import Header
from geometry_msgs.msg import Point

class HumanoidStatePublisher(Node):
    def __init__(self):
        super().__init__('humanoid_state_publisher')

        # Custom message publisher
        self.state_pub = self.create_publisher(
            HumanoidState, '/humanoid/state', 10)

    def publish_humanoid_state(self, joint_positions, com_position, in_balance):
        msg = HumanoidState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.joint_positions = joint_positions
        msg.center_of_mass = Point(
            x=com_position[0],
            y=com_position[1],
            z=com_position[2]
        )
        msg.in_balance = in_balance
        msg.balance_margin = self.calculate_balance_margin()

        self.state_pub.publish(msg)
```

## Performance Considerations

### Memory Management

For real-time humanoid applications, efficient memory usage is critical:

```python
class EfficientMessagingNode(Node):
    def __init__(self):
        super().__init__('efficient_messaging_node')

        # Pre-allocate message objects to reduce allocation overhead
        self.preallocated_msg = JointState()
        self.joint_names = [
            'left_hip', 'right_hip', 'left_knee', 'right_knee'
        ]

        self.publisher = self.create_publisher(
            JointState, '/efficient_joint_states', 10)

        # Timer for efficient publishing
        self.timer = self.create_timer(0.01, self.efficient_publish)

    def efficient_publish(self):
        """Efficiently publish pre-allocated messages"""
        # Reuse message object
        msg = self.preallocated_msg
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # Update only the data that changes
        current_positions = self.get_current_joint_positions()
        msg.position = current_positions
        msg.velocity = self.get_current_velocities()
        msg.effort = self.get_current_efforts()

        self.publisher.publish(msg)
```

### Bandwidth Optimization

Humanoid robots often have limited communication bandwidth:

```python
class BandwidthOptimizedNode(Node):
    def __init__(self):
        super().__init__('bandwidth_optimized_node')

        # Use smaller data types when possible
        self.compressed_pub = self.create_publisher(
            CompressedImage, '/camera/image_compressed', 5)

        # Throttle high-frequency data
        self.throttled_timer = self.create_timer(
            0.05, self.publish_throttled_data)  # 20Hz instead of 100Hz

    def publish_throttled_data(self):
        """Publish data at reduced frequency to save bandwidth"""
        # Only publish essential data at lower rate
        essential_data = self.get_essential_data()
        self.publish_essential_info(essential_data)
```

## Error Handling and Diagnostics

Robust error handling is essential for humanoid robotics:

```python
import threading
from rclpy.qos import qos_profile_system_default

class RobustMessagingNode(Node):
    def __init__(self):
        super().__init__('robust_messaging_node')

        # Multiple publishers with different profiles
        self.main_pub = self.create_publisher(
            JointState, '/joint_states', 10)

        # Diagnostic publisher
        self.diag_pub = self.create_publisher(
            DiagnosticArray, '/diagnostics', 1)

        # Thread-safe data storage
        self.data_lock = threading.RLock()
        self.published_count = 0
        self.error_count = 0

    def safe_publish(self, msg):
        """Safely publish messages with error handling"""
        try:
            with self.data_lock:
                self.main_pub.publish(msg)
                self.published_count += 1
        except Exception as e:
            with self.data_lock:
                self.error_count += 1
                self.get_logger().error(f'Publish error: {e}')

            # Publish diagnostic information
            self.publish_diagnostics()

    def publish_diagnostics(self):
        """Publish diagnostic information"""
        diag_msg = DiagnosticArray()
        diag_msg.header.stamp = self.get_clock().now().to_msg()

        # Add diagnostic status
        status = DiagnosticStatus()
        status.name = 'Messaging Node'
        status.level = DiagnosticStatus.OK if self.error_count == 0 else DiagnosticStatus.WARN
        status.message = f'Published: {self.published_count}, Errors: {self.error_count}'

        diag_msg.status.append(status)
        self.diag_pub.publish(diag_msg)
```

## Best Practices for Humanoid Robotics

:::note Best Practices
- Use appropriate QoS settings for different data types
- Implement message throttling for high-frequency sensors
- Pre-allocate messages for real-time performance
- Monitor message rates and bandwidth usage
- Implement robust error handling and diagnostics
- Consider message compression for large data like images
:::

## Key Takeaways

- Topics enable asynchronous, decoupled communication between nodes
- QoS profiles provide control over reliability, latency, and durability
- Proper message filtering and synchronization are crucial for sensor fusion
- Memory management and bandwidth optimization are essential for humanoid systems
- Error handling and diagnostics ensure robust operation
- Performance considerations must be addressed for real-time applications

Understanding topics and messaging patterns is fundamental to creating efficient and reliable humanoid robotics systems that can handle the complex data flows required for safe and responsive robot operation.