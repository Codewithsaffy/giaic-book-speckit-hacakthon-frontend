---
title: "Latency aur Synchronization Handle Karna"
description: "AI-ROS integration ke liye timing constraints, time synchronization, aur message filtering manage karna"
sidebar_position: 5
keywords: [latency, synchronization, time sync, message filters, real-time, ros2, ai]
---

# Latency aur Synchronization Handle Karna

AI-ROS integration mein latency aur synchronization manage karna critical hai, khaas kar real-time robotic applications mein jahan proper operation ke liye timing constraints ko meet karna zaruri hai. Yeh section ROS 2 mein time synchronization, sensor fusion ke liye message filters, delayed data handle karna, aur real-time constraints implement karna cover karta hai.

## AI-ROS Systems mein Latency ko Samajhna

AI-ROS systems mein latency multiple sources se aati hai aur isko carefully manage kiya jana chahiye:

### Latency ke Sources

1. **Sensor Latency**: Physical event se sensor measurement tak ka time
2. **Transport Latency**: Data ko ROS network ke through travel karne mein time
3. **Processing Latency**: AI model ke data ko process karne mein time
4. **Control Latency**: Decision se actuator response tak ka time

### Latency Measure Karna

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
import time

class LatencyMeasurementNode(Node):
    def __init__(self):
        super().__init__('latency_measurement')

        # Different stages ke liye publishers
        self.start_pub = self.create_publisher(Header, 'latency_start', 10)
        self.process_pub = self.create_publisher(Header, 'latency_process', 10)
        self.end_pub = self.create_publisher(Header, 'latency_end', 10)

        # Latency measurement ke liye subscriber
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.measure_latency, 10
        )

        self.get_logger().info('Latency Measurement Node initialized')

    def measure_latency(self, msg):
        """End-to-end latency measure karen"""
        # Sensor timestamp record karen (image header se)
        sensor_time = msg.header.stamp
        sensor_timestamp = sensor_time.sec + sensor_time.nanosec / 1e9

        # ROS receive time record karen
        receive_time = self.get_clock().now().nanoseconds / 1e9

        # AI processing simulate karen
        start_process = time.time()
        self.simulate_ai_processing(msg)
        process_time = time.time() - start_process

        # Various latencies calculate karen
        transport_latency = receive_time - sensor_timestamp
        processing_latency = process_time
        total_latency = (self.get_clock().now().nanoseconds / 1e9) - sensor_timestamp

        self.get_logger().info(
            f'Latency breakdown - Transport: {transport_latency:.3f}s, '
            f'Processing: {processing_latency:.3f}s, '
            f'Total: {total_latency:.3f}s'
        )

        # Latency measurements publish karen
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = f'total_latency_{total_latency:.3f}'
        self.end_pub.publish(header)

    def simulate_ai_processing(self, image_msg):
        """AI processing time simulate karen"""
        # Processing delay simulate karen
        time.sleep(0.05)  # 50ms processing time

def main(args=None):
    rclpy.init(args=args)
    node = LatencyMeasurementNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Latency Measurement Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## ROS 2 mein Time Synchronization

ROS 2 nodes ke darmiyan time synchronization ke liye several mechanisms provide karta hai:

### ROS Time ka Istemal

```python
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from builtin_interfaces.msg import Time as TimeMsg
from sensor_msgs.msg import Image
from std_msgs.msg import Header

class TimeSyncNode(Node):
    def __init__(self):
        super().__init__('time_sync_node')

        # Clock type parameter create karen
        self.use_sim_time = self.declare_parameter('use_sim_time', False).get_parameter_value().bool_value

        # Publishers aur subscribers
        self.image_pub = self.create_publisher(Image, 'synced_image', 10)
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.time_sync_callback, 10
        )

        # Time synchronization ke liye timer
        self.time_sync_timer = self.create_timer(1.0, self.log_time_sync)

        self.get_logger().info('Time Synchronization Node initialized')

    def time_sync_callback(self, msg):
        """Time-synchronized messages process karen"""
        # Current ROS time get karen
        ros_time = self.get_clock().now()

        # Sensor aur ROS time ke darmiyan time difference calculate karen
        sensor_time = Time.from_msg(msg.header.stamp)
        time_diff = ros_time.nanoseconds - sensor_time.nanoseconds

        self.get_logger().info(f'Time difference: {time_diff / 1e9:.6f}s')

        # Check karen kya time acceptable tolerance ke andhar hai
        tolerance = 0.1  # 100ms tolerance
        if abs(time_diff) / 1e9 > tolerance:
            self.get_logger().warn(f'Large time difference detected: {time_diff / 1e9:.6f}s')

        # Process message agar time tolerance ke andhar hai
        if abs(time_diff) / 1e9 <= tolerance:
            self.process_time_synced_data(msg)

    def process_time_synced_data(self, msg):
        """Time synchronization requirements ko meet karne wale data ko process karen"""
        self.get_logger().info(f'Processing time-synced data from {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')

    def log_time_sync(self):
        """Current time synchronization status log karen"""
        current_time = self.get_clock().now()
        self.get_logger().info(f'Current ROS time: {current_time.nanoseconds / 1e9:.6f}s')

def main(args=None):
    rclpy.init(args=args)
    node = TimeSyncNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Time Synchronization Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Custom Time Synchronization

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from builtin_interfaces.msg import Time
import time

class CustomTimeSyncNode(Node):
    def __init__(self):
        super().__init__('custom_time_sync')

        # Time synchronization variables
        self.time_offset = 0.0  # Local aur reference time ke darmiyan offset
        self.sync_accuracy = 0.0  # Synchronization ki accuracy
        self.last_sync_time = 0.0

        # Time sync ke liye publishers aur subscribers
        self.time_request_pub = self.create_publisher(Float64, 'time_sync_request', 10)
        self.time_response_sub = self.create_subscription(
            Float64, 'time_sync_response', self.handle_time_response, 10
        )

        # Periodic synchronization ke liye timer
        self.sync_timer = self.create_timer(5.0, self.request_time_sync)

        self.get_logger().info('Custom Time Synchronization Node initialized')

    def request_time_sync(self):
        """Time synchronization request karen"""
        # Reference node ko current local time bhejen
        local_time_msg = Float64()
        local_time_msg.data = self.get_clock().now().nanoseconds / 1e9
        self.time_request_pub.publish(local_time_msg)

        self.get_logger().info(f'Requested time sync at {local_time_msg.data:.6f}')

    def handle_time_response(self, msg):
        """Time synchronization response handle karen"""
        # Round-trip time calculate karen
        current_time = self.get_clock().now().nanoseconds / 1e9
        round_trip_time = current_time - msg.data

        # One-way delay estimate karen
        one_way_delay = round_trip_time / 2.0

        # Time offset calculate karen
        reference_time = msg.data + one_way_delay
        local_time = current_time
        self.time_offset = reference_time - local_time

        # Sync accuracy update karen
        self.sync_accuracy = abs(round_trip_time)
        self.last_sync_time = current_time

        self.get_logger().info(
            f'Time sync - Offset: {self.time_offset:.6f}s, '
            f'Accuracy: {self.sync_accuracy:.6f}s'
        )

    def get_synced_time(self):
        """Synchronization offset ke liye adjusted time get karen"""
        local_time = self.get_clock().now().nanoseconds / 1e9
        return local_time + self.time_offset

def main(args=None):
    rclpy.init(args=args)
    node = CustomTimeSyncNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Custom Time Synchronization Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Sensor Fusion ke liye Message Filters

Message filters different timestamps wale multiple sensors se messages ko synchronize karne mein help karta hai:

### Approximate Time Synchronization

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class MessageFilterNode(Node):
    def __init__(self):
        super().__init__('message_filter_node')

        # Sensor data ke liye QoS profile create karen
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Different sensor types ke liye subscribers create karen
        self.image_sub = Subscriber(self, Image, 'camera/image_raw', qos_profile=sensor_qos)
        self.imu_sub = Subscriber(self, Imu, 'imu/data', qos_profile=sensor_qos)
        self.laser_sub = Subscriber(self, LaserScan, 'scan', qos_profile=sensor_qos)

        # Approximate time synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.imu_sub, self.laser_sub],
            queue_size=10,      # Har subscriber ke liye maximum queue size
            slop=0.1,           # Seconds mein time tolerance
            allow_headerless=False  # Header timestamps ki requirement
        )
        self.sync.registerCallback(self.synchronized_callback)

        self.get_logger().info('Message Filter Node initialized')

    def synchronized_callback(self, image_msg, imu_msg, laser_msg):
        """Synchronized messages process karen"""
        # Saare messages ke approximately same timestamp hain
        sync_time = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec / 1e9

        # Actual time differences calculate karen
        image_time = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec / 1e9
        imu_time = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec / 1e9
        laser_time = laser_msg.header.stamp.sec + laser_msg.header.stamp.nanosec / 1e9

        time_diffs = [
            abs(image_time - sync_time),
            abs(imu_time - sync_time),
            abs(laser_time - sync_time)
        ]

        max_time_diff = max(time_diffs)

        self.get_logger().info(
            f'Synchronized at {sync_time:.3f}s, '
            f'Max time diff: {max_time_diff:.3f}s'
        )

        # Synchronized data ke sath sensor fusion perform karen
        fused_data = self.perform_sensor_fusion(image_msg, imu_msg, laser_msg)

        # Synchronization quality validate karen
        if max_time_diff > 0.1:  # 100ms threshold
            self.get_logger().warn(f'Poor synchronization quality: {max_time_diff:.3f}s')

    def perform_sensor_fusion(self, image_msg, imu_msg, laser_msg):
        """Synchronized data ke sath sensor fusion perform karen"""
        # Image features extract karen
        # (Real implementation mein, yeh image par AI models run karega)
        image_features = "image_features_placeholder"

        # IMU data extract karen
        imu_data = {
            'orientation': [imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w],
            'angular_velocity': [imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z],
            'linear_acceleration': [imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z]
        }

        # Laser data extract karen
        laser_ranges = list(laser_msg.ranges)

        # Sensor data fuse karen
        fused_result = {
            'timestamp': image_msg.header.stamp,
            'image_features': image_features,
            'imu_orientation': imu_data['orientation'],
            'obstacle_distances': [r for r in laser_ranges if r < laser_msg.range_max]
        }

        return fused_result

def main(args=None):
    rclpy.init(args=args)
    node = MessageFilterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Message Filter Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Exact Time Synchronization

```python
from message_filters import TimeSynchronizer

class ExactTimeSyncNode(Node):
    def __init__(self):
        super().__init__('exact_time_sync_node')

        # Subscribers create karen
        self.image_sub = Subscriber(self, Image, 'camera/image_raw')
        self.imu_sub = Subscriber(self, Imu, 'imu/data')

        # Exact time synchronizer (exact timestamp matches ki requirement hoti hai)
        self.exact_sync = TimeSynchronizer(
            [self.image_sub, self.imu_sub],
            queue_size=10
        )
        self.exact_sync.registerCallback(self.exact_sync_callback)

        self.get_logger().info('Exact Time Synchronization Node initialized')

    def exact_sync_callback(self, image_msg, imu_msg):
        """Exactly synchronized messages process karen"""
        # Messages ke identical timestamps hain
        timestamp = image_msg.header.stamp
        self.get_logger().info(f'Exact sync at {timestamp.sec}.{timestamp.nanosec}')

        # Synchronized data process karen
        result = self.process_exact_sync_data(image_msg, imu_msg)
        self.get_logger().info(f'Exact sync processing result: {result}')

    def process_exact_sync_data(self, image_msg, imu_msg):
        """Exactly synchronized data process karen"""
        # Exact synchronization ke liye implementation
        return "exact_sync_result"
```

## Delayed Data Handle Karna

AI-ROS systems mein processing time ki wajah se delayed data common hai. Proper handling zaruri hai:

### Delay Compensation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from builtin_interfaces.msg import Time
import numpy as np
from collections import deque

class DelayCompensationNode(Node):
    def __init__(self):
        super().__init__('delay_compensation_node')

        # Delay compensation ke liye state history
        self.state_history = deque(maxlen=100)  # Last 100 states store karen
        self.processing_delay = 0.05  # 50ms average processing delay

        # Publishers aur subscribers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.delayed_image_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.state_update_callback, 10
        )

        self.get_logger().info('Delay Compensation Node initialized')

    def state_update_callback(self, msg):
        """Robot state history update karen"""
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        # State ko timestamp ke sath store karen
        state = {
            'timestamp': current_time,
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

        self.state_history.append(state)

    def delayed_image_callback(self, msg):
        """Delayed image ko compensation ke sath process karen"""
        # Image timestamp reflect karta hai jab yeh capture kiya gaya tha
        image_capture_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        # Calculate karen kya decision execute kiya jana chahiye
        decision_execution_time = self.get_clock().now().nanoseconds / 1e9

        # Estimate state at image capture time using history
        estimated_past_state = self.estimate_state_at_time(image_capture_time)

        # Image process karen aur estimated past state ke adhar par decision len
        ai_decision = self.make_decision_with_past_state(msg, estimated_past_state)

        # Additional processing delay ka account rakhen
        compensated_execution_time = decision_execution_time + self.processing_delay

        # Decision execute karen
        self.execute_decision(ai_decision, compensated_execution_time)

    def estimate_state_at_time(self, target_time):
        """History ka istemal karke robot state estimate karen"""
        if not self.state_history:
            return None

        # History mein closest state find karen
        closest_state = min(
            self.state_history,
            key=lambda s: abs(s['timestamp'] - target_time)
        )

        # Agar closest state reasonably close hai to return karen
        time_diff = abs(closest_state['timestamp'] - target_time)
        if time_diff < 0.1:  # 100ms tolerance
            return closest_state

        # Agar close enough nahi hai to states ke beech interpolate karen
        return self.interpolate_state(target_time)

    def interpolate_state(self, target_time):
        """Target time par robot state interpolate karen"""
        # Interpolation ke liye two closest states find karen
        states = list(self.state_history)
        if len(states) < 2:
            return states[-1] if states else None

        # Target time se pehle aur baad ke states find karen
        before_states = [s for s in states if s['timestamp'] <= target_time]
        after_states = [s for s in states if s['timestamp'] >= target_time]

        if not before_states or not after_states:
            return states[-1]  # Return most recent agar interpolation possible nahi hai

        before_state = before_states[-1]
        after_state = after_states[0]

        # Time difference ke liye linear interpolation
        time_range = after_state['timestamp'] - before_state['timestamp']
        if time_range == 0:
            return before_state

        ratio = (target_time - before_state['timestamp']) / time_range

        # Orientation interpolate karen (simplified)
        interpolated_state = {
            'timestamp': target_time,
            'orientation': [
                before_state['orientation'][i] * (1 - ratio) + after_state['orientation'][i] * ratio
                for i in range(4)
            ],
            'angular_velocity': [
                before_state['angular_velocity'][i] * (1 - ratio) + after_state['angular_velocity'][i] * ratio
                for i in range(3)
            ],
            'linear_acceleration': [
                before_state['linear_acceleration'][i] * (1 - ratio) + after_state['linear_acceleration'][i] * ratio
                for i in range(3)
            ]
        }

        return interpolated_state

    def make_decision_with_past_state(self, image_msg, past_state):
        """Past state ko consider karke AI decision len"""
        if past_state is None:
            self.get_logger().warn('Could not estimate past state for delay compensation')
            return {'command': 'stop', 'confidence': 0.0}

        # Image par AI processing perform karen
        # (Real implementation mein, yeh aapka AI model run karega)
        ai_result = self.process_image_for_decision(image_msg)

        # Past state ke adhar par decision adjust karen
        adjusted_decision = {
            'command': ai_result.get('command', 'stop'),
            'confidence': ai_result.get('confidence', 0.0),
            'state_at_capture': past_state,
            'compensation_applied': True
        }

        return adjusted_decision

    def execute_decision(self, decision, execution_time):
        """Timing consideration ke sath AI decision execute karen"""
        cmd = Twist()

        if decision['command'] == 'forward':
            cmd.linear.x = 0.5  # 0.5 m/s se forward move karen
        elif decision['command'] == 'turn_left':
            cmd.angular.z = 0.5  # 0.5 rad/s se left turn karen
        elif decision['command'] == 'turn_right':
            cmd.angular.z = -0.5  # 0.5 rad/s se right turn karen
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)
        self.get_logger().info(f'Executed decision at {execution_time:.3f}s: {decision["command"]}')

    def process_image_for_decision(self, image_msg):
        """Navigation decision len ke liye image process karen"""
        # Actual AI processing ke liye placeholder
        return {'command': 'forward', 'confidence': 0.8}

def main(args=None):
    rclpy.init(args=args)
    node = DelayCompensationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Delay Compensation Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Real-time Constraints Implementation

Real-time constraints implement karne se AI-ROS systems timing requirements ko meet karti hain:

### Real-time Scheduling

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import threading
import time
import signal
import os

class RealTimeConstraintNode(Node):
    def __init__(self):
        super().__init__('real_time_constraint_node')

        # Real-time parameters
        self.processing_deadline = 0.1  # 100ms deadline
        self.control_period = 0.05      # 50ms control period
        self.missed_deadlines = 0
        self.total_deadlines = 0

        # Publishers aur subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.real_time_callback, 1
        )
        self.deadline_status_pub = self.create_publisher(Bool, 'deadline_status', 10)

        # Control loop ke liye timer
        self.control_timer = self.create_timer(self.control_period, self.control_loop)

        # Agar possible ho to real-time priority enable karen
        self.enable_real_time_priority()

        self.get_logger().info('Real-time Constraint Node initialized')

    def enable_real_time_priority(self):
        """Process ke liye real-time priority enable karne ki koshish karen"""
        try:
            # Real-time scheduling set karne ki koshish karen (appropriate permissions ki zarurat hoti hai)
            import sched
            # Note: Yeh ek simplified example hai; real real-time setup ko
            # system configuration aur proper permissions ki zarurat hoti hai
            self.get_logger().info('Real-time priority enabled')
        except Exception as e:
            self.get_logger().info(f'Could not enable real-time priority: {e}')

    def real_time_callback(self, msg):
        """Real-time constraints ke sath message process karen"""
        # Start time record karen
        start_time = self.get_clock().now().nanoseconds / 1e9

        try:
            # AI processing perform karen
            result = self.real_time_ai_processing(msg)

            # Processing time calculate karen
            end_time = self.get_clock().now().nanoseconds / 1e9
            processing_time = end_time - start_time

            # Deadline check karen
            deadline_met = processing_time <= self.processing_deadline
            self.total_deadlines += 1

            if not deadline_met:
                self.missed_deadlines += 1
                self.get_logger().warn(
                    f'Deadline missed: {processing_time:.3f}s > {self.processing_deadline:.3f}s'
                )
            else:
                self.get_logger().info(
                    f'Deadline met: {processing_time:.3f}s <= {self.processing_deadline:.3f}s'
                )

            # Deadline status publish karen
            status_msg = Bool()
            status_msg.data = deadline_met
            self.deadline_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Real-time processing error: {e}')

    def real_time_ai_processing(self, image_msg):
        """Real-time considerations ke sath AI processing perform karen"""
        # Processing deadline set karen using signal for timeout (Unix-like systems)
        def timeout_handler(signum, frame):
            raise TimeoutError("AI processing exceeded deadline")

        # Sirf Unix-like systems par timeout set karen
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.processing_deadline))

        try:
            # AI processing perform karen (placeholder)
            # Real implementation mein, yeh aapka AI model run karega
            time.sleep(min(0.05, self.processing_deadline * 0.8))  # Processing simulate karen
            return "real_time_result"
        finally:
            # Alarm cancel karen agar set kiya gaya ho
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def control_loop(self):
        """Timing constraints ko respect karne wala control loop"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Deadline statistics calculate karen
        if self.total_deadlines > 0:
            deadline_rate = 1.0 - (self.missed_deadlines / self.total_deadlines)
            self.get_logger().info(f'Deadline success rate: {deadline_rate:.2%}')

        # Yahan control logic implement karen
        self.perform_control_action()

    def perform_control_action(self):
        """Timing constraints ke andhar control action perform karen"""
        # Control logic ke liye placeholder
        pass

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeConstraintNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Real-time Constraint Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Adaptive Real-time Management

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
import time
import statistics

class AdaptiveRealTimeNode(Node):
    def __init__(self):
        super().__init__('adaptive_real_time_node')

        # Adaptive parameters
        self.target_processing_rate = 30  # 30 FPS target
        self.current_processing_rate = 30
        self.processing_times = []  # Processing times track karen
        self.max_processing_times = 100  # Last 100 measurements rakhna

        # Publishers aur subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.adaptive_callback, 1
        )
        self.rate_pub = self.create_publisher(Float64, 'current_rate', 10)

        # Rate adaptation ke liye timer
        self.adaptation_timer = self.create_timer(2.0, self.adapt_rate)

        self.get_logger().info('Adaptive Real-time Node initialized')

    def adaptive_callback(self, msg):
        """Adaptive timing ke sath process karen"""
        start_time = time.time()

        try:
            # Processing perform karen
            result = self.adaptive_ai_processing(msg)

            # Processing time record karen
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Sirf recent measurements rakhna
            if len(self.processing_times) > self.max_processing_times:
                self.processing_times.pop(0)

            # Performance log karen
            self.get_logger().info(
                f'Processing time: {processing_time:.3f}s, '
                f'Current rate: {self.current_processing_rate:.1f} FPS'
            )

        except Exception as e:
            self.get_logger().error(f'Adaptive processing error: {e}')

    def adaptive_ai_processing(self, image_msg):
        """Adaptive complexity ke sath AI processing perform karen"""
        # Current performance calculate karen
        if self.processing_times:
            avg_processing_time = statistics.mean(self.processing_times)
            target_interval = 1.0 / self.current_processing_rate

            # Performance ke adhar par processing complexity adjust karen
            if avg_processing_time > target_interval * 0.8:
                # Bohot slow hai, complexity reduce karen
                return self.simple_ai_processing(image_msg)
            elif avg_processing_time < target_interval * 0.5:
                # Fast enough hai, complexity increase kar sakte hain
                return self.complex_ai_processing(image_msg)
            else:
                # Performance achhi hai, current complexity maintain karen
                return self.normal_ai_processing(image_msg)
        else:
            # First processing, normal complexity ka istemal karen
            return self.normal_ai_processing(image_msg)

    def simple_ai_processing(self, image_msg):
        """Better performance ke liye simple AI processing"""
        # Chota model ya kam operations ka istemal karen
        time.sleep(0.02)  # Faster processing simulate karen
        return "simple_result"

    def normal_ai_processing(self, image_msg):
        """Normal AI processing"""
        # Standard model aur operations ka istemal karen
        time.sleep(0.033)  # Normal processing simulate karen (~30 FPS)
        return "normal_result"

    def complex_ai_processing(self, image_msg):
        """Better accuracy ke liye complex AI processing"""
        # Bada model ya zyada operations ka istemal karen
        time.sleep(0.05)  # Slower processing simulate karen
        return "complex_result"

    def adapt_rate(self):
        """Performance ke adhar par processing rate adapt karen"""
        if not self.processing_times:
            return

        avg_processing_time = statistics.mean(self.processing_times)
        max_acceptable_rate = 1.0 / avg_processing_time

        # Rate ko conservatively adjust karen (90% of maximum)
        target_rate = min(self.target_processing_rate, max_acceptable_rate * 0.9)

        # Current rate ko gradually update karen
        rate_diff = target_rate - self.current_processing_rate
        self.current_processing_rate += rate_diff * 0.1  # 10% adjustment per cycle

        # Current rate publish karen
        rate_msg = Float64()
        rate_msg.data = self.current_processing_rate
        self.rate_pub.publish(rate_msg)

        self.get_logger().info(f'Adapted processing rate to: {self.current_processing_rate:.2f} FPS')

def main(args=None):
    rclpy.init(args=args)
    node = AdaptiveRealTimeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Adaptive Real-time Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Practical Example: Multi-sensor AI Fusion with Synchronization

Yeh ek complete example hai saare concepts ko combine karke multi-sensor AI fusion ke liye:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan
from geometry_msgs.msg import Twist
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
from collections import deque
import time

class MultiSensorFusionNode(Node):
    def __init__(self):
        super().__init__('multi_sensor_fusion')

        # State estimation
        self.state_history = deque(maxlen=50)
        self.processing_delay = 0.04  # 40ms processing delay

        # QoS profiles create karen
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Subscribers create karen
        self.image_sub = Subscriber(self, Image, 'camera/image_raw', qos_profile=sensor_qos)
        self.imu_sub = Subscriber(self, Imu, 'imu/data', qos_profile=sensor_qos)
        self.laser_sub = Subscriber(self, LaserScan, 'scan', qos_profile=sensor_qos)

        # Sensors synchronize karen
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.imu_sub, self.laser_sub],
            queue_size=10,
            slop=0.05  # 50ms tolerance
        )
        self.sync.registerCallback(self.fusion_callback)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.get_logger().info('Multi-sensor Fusion Node initialized')

    def fusion_callback(self, image_msg, imu_msg, laser_msg):
        """Synchronized multi-sensor data process karen"""
        # State history update karen
        self.update_state_history(imu_msg)

        # Estimate state at image capture time (delay compensation)
        image_capture_time = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec / 1e9
        estimated_state = self.estimate_state_at_time(image_capture_time)

        # AI fusion perform karen
        fusion_result = self.perform_sensor_fusion(
            image_msg, imu_msg, laser_msg, estimated_state
        )

        # Control command generate karen
        control_cmd = self.generate_control_command(fusion_result)

        # Command publish karen
        self.cmd_pub.publish(control_cmd)

        self.get_logger().info(f'Published fusion-based command: linear={control_cmd.linear.x}, angular={control_cmd.angular.z}')

    def update_state_history(self, imu_msg):
        """Delay compensation ke liye robot state history update karen"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        state = {
            'timestamp': current_time,
            'orientation': [imu_msg.orientation.x, imu_msg.orientation.y,
                           imu_msg.orientation.z, imu_msg.orientation.w],
            'angular_velocity': [imu_msg.angular_velocity.x, imu_msg.angular_velocity.y,
                                imu_msg.angular_velocity.z],
            'linear_acceleration': [imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y,
                                   imu_msg.linear_acceleration.z]
        }

        self.state_history.append(state)

    def estimate_state_at_time(self, target_time):
        """History ka istemal karke state estimate karen"""
        if not self.state_history:
            return None

        # Closest state find karen
        closest_state = min(
            self.state_history,
            key=lambda s: abs(s['timestamp'] - target_time)
        )

        return closest_state

    def perform_sensor_fusion(self, image_msg, imu_msg, laser_msg, estimated_state):
        """AI-based sensor fusion perform karen"""
        # Visual features ke liye image process karen
        visual_features = self.extract_visual_features(image_msg)

        # Orientation ke liye IMU process karen
        orientation = np.array(estimated_state['orientation'] if estimated_state else
                              [imu_msg.orientation.x, imu_msg.orientation.y,
                               imu_msg.orientation.z, imu_msg.orientation.w])

        # Obstacle detection ke liye laser process karen
        obstacles = self.detect_obstacles(laser_msg)

        # Saare sensor data fuse karen
        fusion_result = {
            'visual_features': visual_features,
            'orientation': orientation,
            'obstacles': obstacles,
            'timestamp': image_msg.header.stamp
        }

        return fusion_result

    def extract_visual_features(self, image_msg):
        """Image se features extract karen (placeholder)"""
        # Real implementation mein, yeh ek CNN ya doosra AI model run karega
        return {'features': [1.0, 2.0, 3.0], 'objects': []}

    def detect_obstacles(self, laser_msg):
        """Laser scan se obstacles detect karen"""
        obstacles = []
        for i, range_val in enumerate(laser_msg.ranges):
            if 0 < range_val < laser_msg.range_min * 0.8:  # Obstacle detected
                angle = laser_msg.angle_min + i * laser_msg.angle_increment
                obstacles.append({'range': range_val, 'angle': angle})
        return obstacles

    def generate_control_command(self, fusion_result):
        """Fusion results ke adhar par control command generate karen"""
        cmd = Twist()

        # Fused data ke adhar par simple navigation logic
        if fusion_result['obstacles']:
            # Obstacles avoid karen
            closest_obstacle = min(fusion_result['obstacles'], key=lambda o: o['range'])
            if closest_obstacle['range'] < 1.0:  # 1 meter threshold
                cmd.angular.z = 0.5 if closest_obstacle['angle'] > 0 else -0.5
            else:
                cmd.linear.x = 0.5  # Forward move karen
        else:
            cmd.linear.x = 0.5  # Obstacles nahi hone par forward move karen

        return cmd

def main(args=None):
    rclpy.init(args=args)
    node = MultiSensorFusionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Multi-sensor Fusion Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

Yeh section AI-ROS integration mein latency aur synchronization handle karne ko cover karta hai, jismein time synchronization, sensor fusion ke liye message filters, delay compensation, aur real-time constraint management included hain. Yeh techniques robust AI-ROS systems banane ke liye essential hain jo real-time environments mein effectively operate kar sakti hain.