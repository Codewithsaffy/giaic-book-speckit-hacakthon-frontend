---
title: "Handling Latency and Synchronization"
description: "Managing timing constraints, time synchronization, and message filtering for AI-ROS integration"
sidebar_position: 5
keywords: [latency, synchronization, time sync, message filters, real-time, ros2, ai]
---

# Handling Latency and Synchronization

Managing latency and synchronization is critical for AI-ROS integration, especially in real-time robotic applications where timing constraints must be met for proper operation. This section covers time synchronization in ROS 2, message filters for sensor fusion, handling delayed data, and implementing real-time constraints.

## Understanding Latency in AI-ROS Systems

Latency in AI-ROS systems comes from multiple sources and must be carefully managed:

### Sources of Latency

1. **Sensor Latency**: Time from physical event to sensor measurement
2. **Transport Latency**: Time for data to travel through ROS network
3. **Processing Latency**: Time for AI model to process data
4. **Control Latency**: Time from decision to actuator response

### Measuring Latency

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

        # Publishers for different stages
        self.start_pub = self.create_publisher(Header, 'latency_start', 10)
        self.process_pub = self.create_publisher(Header, 'latency_process', 10)
        self.end_pub = self.create_publisher(Header, 'latency_end', 10)

        # Subscriber for latency measurement
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.measure_latency, 10
        )

        self.get_logger().info('Latency Measurement Node initialized')

    def measure_latency(self, msg):
        """Measure end-to-end latency"""
        # Record sensor timestamp (from image header)
        sensor_time = msg.header.stamp
        sensor_timestamp = sensor_time.sec + sensor_time.nanosec / 1e9

        # Record ROS receive time
        receive_time = self.get_clock().now().nanoseconds / 1e9

        # Simulate AI processing
        start_process = time.time()
        self.simulate_ai_processing(msg)
        process_time = time.time() - start_process

        # Calculate various latencies
        transport_latency = receive_time - sensor_timestamp
        processing_latency = process_time
        total_latency = (self.get_clock().now().nanoseconds / 1e9) - sensor_timestamp

        self.get_logger().info(
            f'Latency breakdown - Transport: {transport_latency:.3f}s, '
            f'Processing: {processing_latency:.3f}s, '
            f'Total: {total_latency:.3f}s'
        )

        # Publish latency measurements
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = f'total_latency_{total_latency:.3f}'
        self.end_pub.publish(header)

    def simulate_ai_processing(self, image_msg):
        """Simulate AI processing time"""
        # Simulate processing delay
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

## Time Synchronization in ROS 2

ROS 2 provides several mechanisms for time synchronization across nodes:

### Using ROS Time

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

        # Create clock type parameter
        self.use_sim_time = self.declare_parameter('use_sim_time', False).get_parameter_value().bool_value

        # Publishers and subscribers
        self.image_pub = self.create_publisher(Image, 'synced_image', 10)
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.time_sync_callback, 10
        )

        # Timer for time synchronization
        self.time_sync_timer = self.create_timer(1.0, self.log_time_sync)

        self.get_logger().info('Time Synchronization Node initialized')

    def time_sync_callback(self, msg):
        """Process time-synchronized messages"""
        # Get current ROS time
        ros_time = self.get_clock().now()

        # Calculate time difference between sensor and ROS time
        sensor_time = Time.from_msg(msg.header.stamp)
        time_diff = ros_time.nanoseconds - sensor_time.nanoseconds

        self.get_logger().info(f'Time difference: {time_diff / 1e9:.6f}s')

        # Check if time is within acceptable tolerance
        tolerance = 0.1  # 100ms tolerance
        if abs(time_diff) / 1e9 > tolerance:
            self.get_logger().warn(f'Large time difference detected: {time_diff / 1e9:.6f}s')

        # Process message if within time tolerance
        if abs(time_diff) / 1e9 <= tolerance:
            self.process_time_synced_data(msg)

    def process_time_synced_data(self, msg):
        """Process data that meets time synchronization requirements"""
        self.get_logger().info(f'Processing time-synced data from {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')

    def log_time_sync(self):
        """Log current time synchronization status"""
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
        self.time_offset = 0.0  # Offset between local and reference time
        self.sync_accuracy = 0.0  # Accuracy of synchronization
        self.last_sync_time = 0.0

        # Publishers and subscribers for time sync
        self.time_request_pub = self.create_publisher(Float64, 'time_sync_request', 10)
        self.time_response_sub = self.create_subscription(
            Float64, 'time_sync_response', self.handle_time_response, 10
        )

        # Timer for periodic synchronization
        self.sync_timer = self.create_timer(5.0, self.request_time_sync)

        self.get_logger().info('Custom Time Synchronization Node initialized')

    def request_time_sync(self):
        """Request time synchronization"""
        # Send current local time to reference node
        local_time_msg = Float64()
        local_time_msg.data = self.get_clock().now().nanoseconds / 1e9
        self.time_request_pub.publish(local_time_msg)

        self.get_logger().info(f'Requested time sync at {local_time_msg.data:.6f}')

    def handle_time_response(self, msg):
        """Handle time synchronization response"""
        # Calculate round-trip time
        current_time = self.get_clock().now().nanoseconds / 1e9
        round_trip_time = current_time - msg.data

        # Estimate one-way delay
        one_way_delay = round_trip_time / 2.0

        # Calculate time offset
        reference_time = msg.data + one_way_delay
        local_time = current_time
        self.time_offset = reference_time - local_time

        # Update sync accuracy
        self.sync_accuracy = abs(round_trip_time)
        self.last_sync_time = current_time

        self.get_logger().info(
            f'Time sync - Offset: {self.time_offset:.6f}s, '
            f'Accuracy: {self.sync_accuracy:.6f}s'
        )

    def get_synced_time(self):
        """Get time adjusted for synchronization offset"""
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

## Message Filters for Sensor Fusion

Message filters help synchronize messages from multiple sensors with different timestamps:

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

        # Create QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Create subscribers for different sensor types
        self.image_sub = Subscriber(self, Image, 'camera/image_raw', qos_profile=sensor_qos)
        self.imu_sub = Subscriber(self, Imu, 'imu/data', qos_profile=sensor_qos)
        self.laser_sub = Subscriber(self, LaserScan, 'scan', qos_profile=sensor_qos)

        # Approximate time synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.imu_sub, self.laser_sub],
            queue_size=10,      # Maximum queue size for each subscriber
            slop=0.1,           # Time tolerance in seconds
            allow_headerless=False  # Require header timestamps
        )
        self.sync.registerCallback(self.synchronized_callback)

        self.get_logger().info('Message Filter Node initialized')

    def synchronized_callback(self, image_msg, imu_msg, laser_msg):
        """Process synchronized messages"""
        # All messages have approximately the same timestamp
        sync_time = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec / 1e9

        # Calculate actual time differences
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

        # Perform sensor fusion with synchronized data
        fused_data = self.perform_sensor_fusion(image_msg, imu_msg, laser_msg)

        # Validate synchronization quality
        if max_time_diff > 0.1:  # 100ms threshold
            self.get_logger().warn(f'Poor synchronization quality: {max_time_diff:.3f}s')

    def perform_sensor_fusion(self, image_msg, imu_msg, laser_msg):
        """Perform sensor fusion with synchronized data"""
        # Extract image features
        # (In real implementation, this would run AI models on the image)
        image_features = "image_features_placeholder"

        # Extract IMU data
        imu_data = {
            'orientation': [imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w],
            'angular_velocity': [imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z],
            'linear_acceleration': [imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z]
        }

        # Extract laser data
        laser_ranges = list(laser_msg.ranges)

        # Fuse sensor data
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

        # Create subscribers
        self.image_sub = Subscriber(self, Image, 'camera/image_raw')
        self.imu_sub = Subscriber(self, Imu, 'imu/data')

        # Exact time synchronizer (requires exact timestamp matches)
        self.exact_sync = TimeSynchronizer(
            [self.image_sub, self.imu_sub],
            queue_size=10
        )
        self.exact_sync.registerCallback(self.exact_sync_callback)

        self.get_logger().info('Exact Time Synchronization Node initialized')

    def exact_sync_callback(self, image_msg, imu_msg):
        """Process exactly synchronized messages"""
        # Messages have identical timestamps
        timestamp = image_msg.header.stamp
        self.get_logger().info(f'Exact sync at {timestamp.sec}.{timestamp.nanosec}')

        # Process synchronized data
        result = self.process_exact_sync_data(image_msg, imu_msg)
        self.get_logger().info(f'Exact sync processing result: {result}')

    def process_exact_sync_data(self, image_msg, imu_msg):
        """Process exactly synchronized data"""
        # Implementation for exact synchronization
        return "exact_sync_result"
```

## Handling Delayed Data

Delayed data is common in AI-ROS systems due to processing time. Proper handling is essential:

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

        # State history for delay compensation
        self.state_history = deque(maxlen=100)  # Store last 100 states
        self.processing_delay = 0.05  # 50ms average processing delay

        # Publishers and subscribers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.delayed_image_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.state_update_callback, 10
        )

        self.get_logger().info('Delay Compensation Node initialized')

    def state_update_callback(self, msg):
        """Update robot state history"""
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        # Store state with timestamp
        state = {
            'timestamp': current_time,
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

        self.state_history.append(state)

    def delayed_image_callback(self, msg):
        """Process delayed image with compensation"""
        # Image timestamp reflects when it was captured
        image_capture_time = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        # Calculate when the decision should be executed
        decision_execution_time = self.get_clock().now().nanoseconds / 1e9

        # Estimate state at image capture time using history
        estimated_past_state = self.estimate_state_at_time(image_capture_time)

        # Process image and make decision based on estimated past state
        ai_decision = self.make_decision_with_past_state(msg, estimated_past_state)

        # Account for additional processing delay
        compensated_execution_time = decision_execution_time + self.processing_delay

        # Execute decision
        self.execute_decision(ai_decision, compensated_execution_time)

    def estimate_state_at_time(self, target_time):
        """Estimate robot state at a past time using history"""
        if not self.state_history:
            return None

        # Find closest state in history
        closest_state = min(
            self.state_history,
            key=lambda s: abs(s['timestamp'] - target_time)
        )

        # If the closest state is reasonably close, return it
        time_diff = abs(closest_state['timestamp'] - target_time)
        if time_diff < 0.1:  # 100ms tolerance
            return closest_state

        # If not close enough, interpolate between states
        return self.interpolate_state(target_time)

    def interpolate_state(self, target_time):
        """Interpolate robot state at target time"""
        # Find two closest states for interpolation
        states = list(self.state_history)
        if len(states) < 2:
            return states[-1] if states else None

        # Find states before and after target time
        before_states = [s for s in states if s['timestamp'] <= target_time]
        after_states = [s for s in states if s['timestamp'] >= target_time]

        if not before_states or not after_states:
            return states[-1]  # Return most recent if interpolation not possible

        before_state = before_states[-1]
        after_state = after_states[0]

        # Linear interpolation for time difference
        time_range = after_state['timestamp'] - before_state['timestamp']
        if time_range == 0:
            return before_state

        ratio = (target_time - before_state['timestamp']) / time_range

        # Interpolate orientation (simplified)
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
        """Make AI decision considering past state"""
        if past_state is None:
            self.get_logger().warn('Could not estimate past state for delay compensation')
            return {'command': 'stop', 'confidence': 0.0}

        # Perform AI processing on image
        # (In real implementation, this would run your AI model)
        ai_result = self.process_image_for_decision(image_msg)

        # Adjust decision based on past state
        adjusted_decision = {
            'command': ai_result.get('command', 'stop'),
            'confidence': ai_result.get('confidence', 0.0),
            'state_at_capture': past_state,
            'compensation_applied': True
        }

        return adjusted_decision

    def execute_decision(self, decision, execution_time):
        """Execute AI decision with timing consideration"""
        cmd = Twist()

        if decision['command'] == 'forward':
            cmd.linear.x = 0.5  # Move forward at 0.5 m/s
        elif decision['command'] == 'turn_left':
            cmd.angular.z = 0.5  # Turn left at 0.5 rad/s
        elif decision['command'] == 'turn_right':
            cmd.angular.z = -0.5  # Turn right at 0.5 rad/s
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0

        self.cmd_pub.publish(cmd)
        self.get_logger().info(f'Executed decision at {execution_time:.3f}s: {decision["command"]}')

    def process_image_for_decision(self, image_msg):
        """Process image to make navigation decision"""
        # Placeholder for actual AI processing
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

Implementing real-time constraints ensures AI-ROS systems meet timing requirements:

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

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.real_time_callback, 1
        )
        self.deadline_status_pub = self.create_publisher(Bool, 'deadline_status', 10)

        # Timer for control loop
        self.control_timer = self.create_timer(self.control_period, self.control_loop)

        # Enable real-time priority if possible
        self.enable_real_time_priority()

        self.get_logger().info('Real-time Constraint Node initialized')

    def enable_real_time_priority(self):
        """Attempt to enable real-time priority for the process"""
        try:
            # Try to set real-time scheduling (requires appropriate permissions)
            import sched
            # Note: This is a simplified example; real real-time setup requires
            # system configuration and proper permissions
            self.get_logger().info('Real-time priority enabled')
        except Exception as e:
            self.get_logger().info(f'Could not enable real-time priority: {e}')

    def real_time_callback(self, msg):
        """Process message with real-time constraints"""
        # Record start time
        start_time = self.get_clock().now().nanoseconds / 1e9

        try:
            # Perform AI processing
            result = self.real_time_ai_processing(msg)

            # Calculate processing time
            end_time = self.get_clock().now().nanoseconds / 1e9
            processing_time = end_time - start_time

            # Check deadline
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

            # Publish deadline status
            status_msg = Bool()
            status_msg.data = deadline_met
            self.deadline_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Real-time processing error: {e}')

    def real_time_ai_processing(self, image_msg):
        """Perform AI processing with real-time considerations"""
        # Set processing deadline using signal for timeout (Unix-like systems)
        def timeout_handler(signum, frame):
            raise TimeoutError("AI processing exceeded deadline")

        # Only set timeout on Unix-like systems
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.processing_deadline))

        try:
            # Perform AI processing (placeholder)
            # In real implementation, this would run your AI model
            time.sleep(min(0.05, self.processing_deadline * 0.8))  # Simulate processing
            return "real_time_result"
        finally:
            # Cancel alarm if set
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def control_loop(self):
        """Control loop that respects timing constraints"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        # Calculate deadline statistics
        if self.total_deadlines > 0:
            deadline_rate = 1.0 - (self.missed_deadlines / self.total_deadlines)
            self.get_logger().info(f'Deadline success rate: {deadline_rate:.2%}')

        # Implement control logic here
        self.perform_control_action()

    def perform_control_action(self):
        """Perform control action within timing constraints"""
        # Placeholder for control logic
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
        self.processing_times = []  # Track processing times
        self.max_processing_times = 100  # Keep last 100 measurements

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.adaptive_callback, 1
        )
        self.rate_pub = self.create_publisher(Float64, 'current_rate', 10)

        # Timer for rate adaptation
        self.adaptation_timer = self.create_timer(2.0, self.adapt_rate)

        self.get_logger().info('Adaptive Real-time Node initialized')

    def adaptive_callback(self, msg):
        """Process with adaptive timing"""
        start_time = time.time()

        try:
            # Perform processing
            result = self.adaptive_ai_processing(msg)

            # Record processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            # Keep only recent measurements
            if len(self.processing_times) > self.max_processing_times:
                self.processing_times.pop(0)

            # Log performance
            self.get_logger().info(
                f'Processing time: {processing_time:.3f}s, '
                f'Current rate: {self.current_processing_rate:.1f} FPS'
            )

        except Exception as e:
            self.get_logger().error(f'Adaptive processing error: {e}')

    def adaptive_ai_processing(self, image_msg):
        """Perform AI processing with adaptive complexity"""
        # Calculate current performance
        if self.processing_times:
            avg_processing_time = statistics.mean(self.processing_times)
            target_interval = 1.0 / self.current_processing_rate

            # Adjust processing complexity based on performance
            if avg_processing_time > target_interval * 0.8:
                # Too slow, reduce complexity
                return self.simple_ai_processing(image_msg)
            elif avg_processing_time < target_interval * 0.5:
                # Fast enough, can increase complexity
                return self.complex_ai_processing(image_msg)
            else:
                # Performance is good, maintain current complexity
                return self.normal_ai_processing(image_msg)
        else:
            # First processing, use normal complexity
            return self.normal_ai_processing(image_msg)

    def simple_ai_processing(self, image_msg):
        """Simple AI processing for better performance"""
        # Use smaller model or fewer operations
        time.sleep(0.02)  # Simulate faster processing
        return "simple_result"

    def normal_ai_processing(self, image_msg):
        """Normal AI processing"""
        # Use standard model and operations
        time.sleep(0.033)  # Simulate normal processing (~30 FPS)
        return "normal_result"

    def complex_ai_processing(self, image_msg):
        """Complex AI processing for better accuracy"""
        # Use larger model or more operations
        time.sleep(0.05)  # Simulate slower processing
        return "complex_result"

    def adapt_rate(self):
        """Adapt processing rate based on performance"""
        if not self.processing_times:
            return

        avg_processing_time = statistics.mean(self.processing_times)
        max_acceptable_rate = 1.0 / avg_processing_time

        # Adjust rate conservatively (90% of maximum)
        target_rate = min(self.target_processing_rate, max_acceptable_rate * 0.9)

        # Update current rate gradually
        rate_diff = target_rate - self.current_processing_rate
        self.current_processing_rate += rate_diff * 0.1  # 10% adjustment per cycle

        # Publish current rate
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

Here's a complete example combining all concepts for multi-sensor AI fusion:

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

        # Create QoS profiles
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )

        # Create subscribers
        self.image_sub = Subscriber(self, Image, 'camera/image_raw', qos_profile=sensor_qos)
        self.imu_sub = Subscriber(self, Imu, 'imu/data', qos_profile=sensor_qos)
        self.laser_sub = Subscriber(self, LaserScan, 'scan', qos_profile=sensor_qos)

        # Synchronize sensors
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
        """Process synchronized multi-sensor data"""
        # Update state history
        self.update_state_history(imu_msg)

        # Estimate state at image capture time (delay compensation)
        image_capture_time = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec / 1e9
        estimated_state = self.estimate_state_at_time(image_capture_time)

        # Perform AI fusion
        fusion_result = self.perform_sensor_fusion(
            image_msg, imu_msg, laser_msg, estimated_state
        )

        # Generate control command
        control_cmd = self.generate_control_command(fusion_result)

        # Publish command
        self.cmd_pub.publish(control_cmd)

        self.get_logger().info(f'Published fusion-based command: linear={control_cmd.linear.x}, angular={control_cmd.angular.z}')

    def update_state_history(self, imu_msg):
        """Update robot state history for delay compensation"""
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
        """Estimate state at target time using history"""
        if not self.state_history:
            return None

        # Find closest state
        closest_state = min(
            self.state_history,
            key=lambda s: abs(s['timestamp'] - target_time)
        )

        return closest_state

    def perform_sensor_fusion(self, image_msg, imu_msg, laser_msg, estimated_state):
        """Perform AI-based sensor fusion"""
        # Process image for visual features
        visual_features = self.extract_visual_features(image_msg)

        # Process IMU for orientation
        orientation = np.array(estimated_state['orientation'] if estimated_state else
                              [imu_msg.orientation.x, imu_msg.orientation.y,
                               imu_msg.orientation.z, imu_msg.orientation.w])

        # Process laser for obstacle detection
        obstacles = self.detect_obstacles(laser_msg)

        # Fuse all sensor data
        fusion_result = {
            'visual_features': visual_features,
            'orientation': orientation,
            'obstacles': obstacles,
            'timestamp': image_msg.header.stamp
        }

        return fusion_result

    def extract_visual_features(self, image_msg):
        """Extract features from image (placeholder)"""
        # In real implementation, this would run a CNN or other AI model
        return {'features': [1.0, 2.0, 3.0], 'objects': []}

    def detect_obstacles(self, laser_msg):
        """Detect obstacles from laser scan"""
        obstacles = []
        for i, range_val in enumerate(laser_msg.ranges):
            if 0 < range_val < laser_msg.range_min * 0.8:  # Obstacle detected
                angle = laser_msg.angle_min + i * laser_msg.angle_increment
                obstacles.append({'range': range_val, 'angle': angle})
        return obstacles

    def generate_control_command(self, fusion_result):
        """Generate control command based on fusion results"""
        cmd = Twist()

        # Simple navigation logic based on fused data
        if fusion_result['obstacles']:
            # Avoid obstacles
            closest_obstacle = min(fusion_result['obstacles'], key=lambda o: o['range'])
            if closest_obstacle['range'] < 1.0:  # 1 meter threshold
                cmd.angular.z = 0.5 if closest_obstacle['angle'] > 0 else -0.5
            else:
                cmd.linear.x = 0.5  # Move forward
        else:
            cmd.linear.x = 0.5  # Move forward if no obstacles

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

This section covered handling latency and synchronization in AI-ROS integration, including time synchronization, message filters for sensor fusion, delay compensation, and real-time constraint management. These techniques are essential for building robust AI-ROS systems that can operate effectively in real-time environments.