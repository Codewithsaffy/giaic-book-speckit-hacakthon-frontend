---
title: "Topics aur Messaging"
description: "Humanoid robotics communication ke liye ROS 2 topics, messaging patterns, aur QoS mein deep dive"
sidebar_position: 3
keywords: ["ROS 2 topics", "publishers", "subscribers", "QoS", "humanoid robotics", "messaging"]
---

# Topics aur Messaging

Topics ROS 2 ke publish-subscribe communication model ka backbone hain, jo nodes ke beech asynchronous data exchange ko enable karta hai. Humanoid robotics ke liye, jahan multiple sensors continuously data streams generate karte hain, topics aur messaging patterns ko samajhna responsive aur efficient systems banana ke liye essential hai.

## Publish-Subscribe Architecture

Publish-subscribe pattern nodes ko bina direct dependencies ke communicate karne ki allow karti hai. Publishers topics mein messages bhejte hain, aur subscribers un topics se messages receive karte hain jo unhe interested hain. Yeh decoupling humanoid robotics mein particularly valuable hai jahan:

- Multiple sensors simultaneously data publish karte hain
- Different subsystems sensor data ko different rates par consume karte hain
- Components ko add ya remove kiya ja sakta hai bina others ko affect kiye

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')

        # Joint states ke liye publisher
        self.joint_pub = self.create_publisher(
            JointState,
            '/joint_states',
            10  # queue size
        )

        # Timer joint states ko regular intervals par publish karne ke liye
        self.timer = self.create_timer(0.01, self.publish_joint_states)  # 100Hz

    def publish_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        # Example joint names aur positions for a humanoid
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

QoS profiles aapko yeh configure karne ki allow karti hai kaise messages delivered hote hain, jo different reliability aur timing requirements wale humanoid robotics applications ke liye critical hai:

### Reliability Settings

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class QoSDemonstrationNode(Node):
    def __init__(self):
        super().__init__('qos_demo_node')

        # Critical data ke liye reliable communication
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Best effort for sensor data jahan occasional drops acceptable hain
        best_effort_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Different QoS ke sath publishers
        self.critical_pub = self.create_publisher(
            JointState, '/critical_joint_commands', reliable_qos)

        self.sensor_pub = self.create_publisher(
            Image, '/camera/image_raw', best_effort_qos)
```

### Common QoS Configurations for Humanoid Robotics

:::tip QoS Guidelines for Humanoid Robotics
- **Joint commands**: RELIABLE, low latency, small history
- **Sensor data**: BEST_EFFORT, larger history buffering ke liye
- **Emergency stops**: RELIABLE, TRANSIENT_LOCAL durability
- **State information**: RELIABLE, medium history
:::

## Subscriber Implementation

Subscribers topics se incoming messages ko process karte hain. Humanoid robotics mein, subscribers often sensor fusion aur control feedback handle karte hain:

```python
class JointStateSubscriber(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')

        # Joint states ko subscribe karen
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
        """Incoming joint state messages ko process karen"""
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

### Message Filtering aur Synchronization

Humanoid robotics ke liye, aapko often multiple sensors se data synchronize karne ki zarurat hoti hai:

```python
from message_filters import ApproximateTimeSynchronizer, Subscriber
import message_filters

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Different sensors ke liye subscribers create karen
        self.imu_sub = message_filters.Subscriber(
            self, Imu, '/imu/data')
        self.joint_sub = message_filters.Subscriber(
            self, JointState, '/joint_states')

        # Messages ko approximate time sync ke sath synchronize karen
        self.ts = ApproximateTimeSynchronizer(
            [self.imu_sub, self.joint_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )
        self.ts.registerCallback(self.sensor_callback)

    def sensor_callback(self, imu_msg, joint_msg):
        """Synchronized sensor data ko process karen"""
        # Fuse IMU aur joint data balance control ke liye
        self.fuse_sensor_data(imu_msg, joint_msg)
```

### Custom Message Types

Humanoid-specific data ke liye, aapko custom messages ki zarurat pad sakti hai:

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

Real-time humanoid applications ke liye, efficient memory usage critical hai:

```python
class EfficientMessagingNode(Node):
    def __init__(self):
        super().__init__('efficient_messaging_node')

        # Allocation overhead reduce karne ke liye message objects pre-allocate karen
        self.preallocated_msg = JointState()
        self.joint_names = [
            'left_hip', 'right_hip', 'left_knee', 'right_knee'
        ]

        self.publisher = self.create_publisher(
            JointState, '/efficient_joint_states', 10)

        # Efficient publishing ke liye timer
        self.timer = self.create_timer(0.01, self.efficient_publish)

    def efficient_publish(self):
        """Pre-allocated messages ko efficiently publish karen"""
        # Message object reuse karen
        msg = self.preallocated_msg
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names

        # Sirf wahi data update karen jo changes hota hai
        current_positions = self.get_current_joint_positions()
        msg.position = current_positions
        msg.velocity = self.get_current_velocities()
        msg.effort = self.get_current_efforts()

        self.publisher.publish(msg)
```

### Bandwidth Optimization

Humanoid robots often limited communication bandwidth ke sath hote hain:

```python
class BandwidthOptimizedNode(Node):
    def __init__(self):
        super().__init__('bandwidth_optimized_node')

        # Possible hone par smaller data types ka istemal karen
        self.compressed_pub = self.create_publisher(
            CompressedImage, '/camera/image_compressed', 5)

        # High-frequency data ko throttle karen
        self.throttled_timer = self.create_timer(
            0.05, self.publish_throttled_data)  # 20Hz instead of 100Hz

    def publish_throttled_data(self):
        """Bandwidth bachane ke liye reduced frequency par data publish karen"""
        # Lower rate par sirf essential data publish karen
        essential_data = self.get_essential_data()
        self.publish_essential_info(essential_data)
```

## Error Handling aur Diagnostics

Humanoid robotics ke liye robust error handling essential hai:

```python
import threading
from rclpy.qos import qos_profile_system_default

class RobustMessagingNode(Node):
    def __init__(self):
        super().__init__('robust_messaging_node')

        # Different profiles ke sath multiple publishers
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
        """Error handling ke sath messages safely publish karen"""
        try:
            with self.data_lock:
                self.main_pub.publish(msg)
                self.published_count += 1
        except Exception as e:
            with self.data_lock:
                self.error_count += 1
                self.get_logger().error(f'Publish error: {e}')

            # Diagnostic information publish karen
            self.publish_diagnostics()

    def publish_diagnostics(self):
        """Diagnostic information publish karen"""
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

## Humanoid Robotics ke liye Best Practices

:::note Best Practices
- Different data types ke liye appropriate QoS settings ka istemal karen
- High-frequency sensors ke liye message throttling implement karen
- Real-time performance ke liye messages pre-allocate karen
- Message rates aur bandwidth usage ko monitor karen
- Robust error handling aur diagnostics implement karen
- Images jaise large data ke liye message compression consider karen
:::

## Key Takeaways

- Topics nodes ke beech asynchronous, decoupled communication enable karte hain
- QoS profiles reliability, latency, aur durability ke upar control provide karte hain
- Sensor fusion ke liye proper message filtering aur synchronization crucial hai
- Memory management aur bandwidth optimization humanoid systems ke liye essential hai
- Error handling aur diagnostics robust operation ensure karte hain
- Real-time applications ke liye performance considerations ko address kiya jane chahiye

Topics aur messaging patterns ko samajhna efficient aur reliable humanoid robotics systems banana ke liye fundamental hai jo safe aur responsive robot operation ke liye required complex data flows ko handle kar sakti hain.