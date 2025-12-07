---
title: "Python ROS 2 Client Libraries"
description: "Using rclpy for Python-based ROS 2 development and AI integration"
sidebar_position: 1
keywords: [ros2, python, rclpy, client libraries, ai integration]
---

# Python ROS 2 Client Libraries

The Python ROS 2 client library (rclpy) provides Python developers with the tools to create ROS 2 nodes, handle messages, and integrate AI systems. This library enables seamless integration between Python's rich AI ecosystem and ROS 2's distributed architecture.

## rclpy Architecture

rclpy is a Python binding for the ROS 2 client library (rcl) and provides:
- Node creation and management
- Publisher and subscriber functionality
- Service and action client/server implementations
- Parameter management
- Time and timer utilities

## Basic Node Structure

### Minimal Python Node

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Hello from minimal node!')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node with Lifecycle Management

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile

class LifecycleNode(Node):
    def __init__(self):
        super().__init__('lifecycle_node')

        # Create a QoS profile
        qos_profile = QoSProfile(depth=10)

        # Initialize components
        self.setup_publishers_subscribers()
        self.setup_services_actions()
        self.setup_parameters()

        self.get_logger().info('Lifecycle node initialized')

    def setup_publishers_subscribers(self):
        """Setup publishers and subscribers"""
        self.publisher = self.create_publisher(
            String,
            'lifecycle_topic',
            10
        )

        self.subscriber = self.create_subscription(
            String,
            'lifecycle_input',
            self.subscription_callback,
            10
        )

    def setup_services_actions(self):
        """Setup services and actions"""
        self.service = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.service_callback
        )

    def setup_parameters(self):
        """Setup parameters"""
        self.declare_parameter('processing_rate', 1.0)
        self.processing_rate = self.get_parameter('processing_rate').value

    def subscription_callback(self, msg):
        self.get_logger().info(f'Received: {msg.data}')

    def service_callback(self, request, response):
        response.sum = request.a + request.b
        return response

def main(args=None):
    rclpy.init(args=args)
    node = LifecycleNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Message Handling in Python

### Publisher Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image
import time

class DataPublisher(Node):
    def __init__(self):
        super().__init__('data_publisher')

        # Multiple publishers for different data types
        self.string_publisher = self.create_publisher(String, 'string_topic', 10)
        self.int_publisher = self.create_publisher(Int32, 'int_topic', 10)

        # Timer for periodic publishing
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.counter = 0

    def timer_callback(self):
        # Publish string message
        string_msg = String()
        string_msg.data = f'Hello World: {self.counter}'
        self.string_publisher.publish(string_msg)

        # Publish integer message
        int_msg = Int32()
        int_msg.data = self.counter
        self.int_publisher.publish(int_msg)

        self.counter += 1

def main(args=None):
    rclpy.init(args=args)
    publisher = DataPublisher()

    try:
        rclpy.spin(publisher)
    except KeyboardInterrupt:
        pass
    finally:
        publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
from sensor_msgs.msg import Image

class DataSubscriber(Node):
    def __init__(self):
        super().__init__('data_subscriber')

        # Subscribe to multiple topics
        self.string_subscription = self.create_subscription(
            String,
            'string_topic',
            self.string_callback,
            10
        )

        self.int_subscription = self.create_subscription(
            Int32,
            'int_topic',
            self.int_callback,
            10
        )

    def string_callback(self, msg):
        self.get_logger().info(f'String received: {msg.data}')

    def int_callback(self, msg):
        self.get_logger().info(f'Integer received: {msg.data}')

def main(args=None):
    rclpy.init(args=args)
    subscriber = DataSubscriber()

    try:
        rclpy.spin(subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Considerations

### Threading and Concurrency

```python
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import threading
import time

class ThreadingNode(Node):
    def __init__(self):
        super().__init__('threading_node')

        # Create callback groups for concurrent execution
        self.group1 = MutuallyExclusiveCallbackGroup()
        self.group2 = MutuallyExclusiveCallbackGroup()

        # Publishers with different callback groups
        self.pub1 = self.create_publisher(String, 'topic1', 10, callback_group=self.group1)
        self.pub2 = self.create_publisher(String, 'topic2', 10, callback_group=self.group2)

        # Timers with different callback groups
        self.timer1 = self.create_timer(1.0, self.timer1_callback, callback_group=self.group1)
        self.timer2 = self.create_timer(0.5, self.timer2_callback, callback_group=self.group2)

    def timer1_callback(self):
        msg = String()
        msg.data = f'Timer 1: {time.time()}'
        self.pub1.publish(msg)

    def timer2_callback(self):
        msg = String()
        msg.data = f'Timer 2: {time.time()}'
        self.pub2.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ThreadingNode()

    # Use multi-threaded executor for concurrent callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Integration with Python AI Libraries

### Basic AI Integration Setup

```python
import rclpy
from rclpy.node import Node
import numpy as np
# Import your AI library (e.g., tensorflow, pytorch, sklearn)

class AIIntegrationNode(Node):
    def __init__(self):
        super().__init__('ai_integration_node')

        # Initialize AI model here
        # self.model = self.load_model()

        # Setup ROS 2 components
        self.subscription = self.create_subscription(
            String,  # Replace with appropriate message type
            'input_topic',
            self.ai_processing_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,  # Replace with appropriate output message type
            'output_topic',
            10
        )

    def load_model(self):
        """Load and initialize AI model"""
        # Example: Load a simple model
        # model = tensorflow.keras.models.load_model('path/to/model')
        # return model
        pass

    def ai_processing_callback(self, msg):
        """Process incoming data with AI model"""
        try:
            # Convert ROS message to format suitable for AI model
            input_data = self.process_input(msg)

            # Run inference
            # result = self.model.predict(input_data)

            # Convert result back to ROS message
            # output_msg = self.create_output_message(result)

            # Publish result
            # self.publisher.publish(output_msg)

            self.get_logger().info(f'Processed: {msg.data}')

        except Exception as e:
            self.get_logger().error(f'AI processing error: {str(e)}')

    def process_input(self, msg):
        """Convert ROS message to AI model input format"""
        # Implement input processing logic
        pass

    def create_output_message(self, result):
        """Convert AI model output to ROS message"""
        # Implement output conversion logic
        pass

def main(args=None):
    rclpy.init(args=args)
    node = AIIntegrationNode()

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

## Best Practices for Python-ROS Integration

### Error Handling and Logging

```python
import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
import traceback

class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')

        # Set up error handling
        self.setup_parameters_safely()
        self.setup_ros_components_safely()

    def setup_parameters_safely(self):
        """Safely declare and get parameters"""
        try:
            self.declare_parameter('model_path', '/default/path')
            self.model_path = self.get_parameter('model_path').value
        except ParameterNotDeclaredException:
            self.get_logger().error('Parameter declaration failed')
            self.model_path = '/default/path'

    def setup_ros_components_safely(self):
        """Safely setup ROS components"""
        try:
            self.publisher = self.create_publisher(String, 'output', 10)
            self.subscription = self.create_subscription(
                String, 'input', self.safe_callback, 10
            )
        except Exception as e:
            self.get_logger().error(f'ROS component setup failed: {str(e)}')

    def safe_callback(self, msg):
        """Safely handle incoming messages"""
        try:
            # Process message
            result = self.process_message(msg)
            if result is not None:
                self.publisher.publish(result)
        except Exception as e:
            self.get_logger().error(f'Callback error: {str(e)}')
            self.get_logger().debug(traceback.format_exc())

    def process_message(self, msg):
        """Process message with error handling"""
        try:
            # Your processing logic here
            output_msg = String()
            output_msg.data = f'Processed: {msg.data}'
            return output_msg
        except Exception as e:
            self.get_logger().error(f'Processing error: {str(e)}')
            return None

def main(args=None):
    rclpy.init(args=args)
    node = RobustNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    except Exception as e:
        node.get_logger().error(f'Unexpected error: {str(e)}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quality of Service Configuration

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

class QoSNode(Node):
    def __init__(self):
        super().__init__('qos_node')

        # Different QoS profiles for different use cases
        # For real-time sensor data
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # For important control commands
        control_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # For configuration parameters
        config_qos = QoSProfile(
            history=HistoryPolicy.KEEP_ALL,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Apply QoS to publishers and subscribers
        self.sensor_pub = self.create_publisher(Image, 'sensor_data', sensor_qos)
        self.control_pub = self.create_publisher(String, 'control_cmd', control_qos)
        self.config_pub = self.create_publisher(String, 'config', config_qos)
```

rclpy provides a comprehensive Python interface to ROS 2, enabling developers to leverage Python's extensive ecosystem for AI and robotics applications while maintaining the robust communication infrastructure that ROS 2 provides.