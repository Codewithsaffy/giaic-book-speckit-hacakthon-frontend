---
title: "Python ROS 2 Client Libraries"
description: "Python-based ROS 2 development aur AI integration ke liye rclpy ka istemal"
sidebar_position: 1
keywords: [ros2, python, rclpy, client libraries, ai integration]
---

# Python ROS 2 Client Libraries

Python ROS 2 client library (rclpy) Python developers ko ROS 2 nodes banane, messages handle karne, aur AI systems integrate karne ke tools provide karta hai. Yeh library Python ke rich AI ecosystem aur ROS 2 ke distributed architecture ke darmiyan seamless integration ko enable karta hai.

## rclpy Architecture

rclpy ROS 2 client library (rcl) ka Python binding hai aur provide karta hai:
- Node creation aur management
- Publisher aur subscriber functionality
- Service aur action client/server implementations
- Parameter management
- Time aur timer utilities

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

        # QoS profile create karen
        qos_profile = QoSProfile(depth=10)

        # Components initialize karen
        self.setup_publishers_subscribers()
        self.setup_services_actions()
        self.setup_parameters()

        self.get_logger().info('Lifecycle node initialized')

    def setup_publishers_subscribers(self):
        """Publishers aur subscribers setup karen"
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
        """Services aur actions setup karen"
        self.service = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.service_callback
        )

    def setup_parameters(self):
        """Parameters setup karen"
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

        # Different data types ke liye multiple publishers
        self.string_publisher = self.create_publisher(String, 'string_topic', 10)
        self.int_publisher = self.create_publisher(Int32, 'int_topic', 10)

        # Periodic publishing ke liye timer
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

        # Multiple topics ko subscribe karen
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

### Threading aur Concurrency

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

        # Concurrent execution ke liye callback groups create karen
        self.group1 = MutuallyExclusiveCallbackGroup()
        self.group2 = MutuallyExclusiveCallbackGroup()

        # Different callback groups ke sath publishers
        self.pub1 = self.create_publisher(String, 'topic1', 10, callback_group=self.group1)
        self.pub2 = self.create_publisher(String, 'topic2', 10, callback_group=self.group2)

        # Different callback groups ke sath timers
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

    # Concurrent callbacks ke liye multi-threaded executor ka istemal karen
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
# Aapka AI library import karen (e.g., tensorflow, pytorch, sklearn)

class AIIntegrationNode(Node):
    def __init__(self):
        super().__init__('ai_integration_node')

        # Yahan AI model initialize karen
        # self.model = self.load_model()

        # ROS 2 components setup karen
        self.subscription = self.create_subscription(
            String,  # Appropriate message type se replace karen
            'input_topic',
            self.ai_processing_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,  # Appropriate output message type se replace karen
            'output_topic',
            10
        )

    def load_model(self):
        """AI model load aur initialize karen"
        # Example: Simple model load karen
        # model = tensorflow.keras.models.load_model('path/to/model')
        # return model
        pass

    def ai_processing_callback(self, msg):
        """AI model ke sath incoming data process karen"
        try:
            # ROS message ko AI model ke format mein convert karen
            input_data = self.process_input(msg)

            # Inference run karen
            # result = self.model.predict(input_data)

            # Result ko wapas se ROS message mein convert karen
            # output_msg = self.create_output_message(result)

            # Result publish karen
            # self.publisher.publish(output_msg)

            self.get_logger().info(f'Processed: {msg.data}')

        except Exception as e:
            self.get_logger().error(f'AI processing error: {str(e)}')

    def process_input(self, msg):
        """ROS message ko AI model input format mein convert karen"
        # Input processing logic implement karen
        pass

    def create_output_message(self, result):
        """AI model output ko ROS message mein convert karen"
        # Output conversion logic implement karen
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

### Error Handling aur Logging

```python
import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
import traceback

class RobustNode(Node):
    def __init__(self):
        super().__init__('robust_node')

        # Error handling setup karen
        self.setup_parameters_safely()
        self.setup_ros_components_safely()

    def setup_parameters_safely(self):
        """Safely parameters declare aur get karen"
        try:
            self.declare_parameter('model_path', '/default/path')
            self.model_path = self.get_parameter('model_path').value
        except ParameterNotDeclaredException:
            self.get_logger().error('Parameter declaration failed')
            self.model_path = '/default/path'

    def setup_ros_components_safely(self):
        """Safely ROS components setup karen"
        try:
            self.publisher = self.create_publisher(String, 'output', 10)
            self.subscription = self.create_subscription(
                String, 'input', self.safe_callback, 10
            )
        except Exception as e:
            self.get_logger().error(f'ROS component setup failed: {str(e)}')

    def safe_callback(self, msg):
        """Safely incoming messages handle karen"
        try:
            # Message process karen
            result = self.process_message(msg)
            if result is not None:
                self.publisher.publish(result)
        except Exception as e:
            self.get_logger().error(f'Callback error: {str(e)}')
            self.get_logger().debug(traceback.format_exc())

    def process_message(self, msg):
        """Error handling ke sath message process karen"
        try:
            # Yahan aapka processing logic hai
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

        # Different use cases ke liye different QoS profiles
        # Real-time sensor data ke liye
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Important control commands ke liye
        control_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # Configuration parameters ke liye
        config_qos = QoSProfile(
            history=HistoryPolicy.KEEP_ALL,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        # Publishers aur subscribers mein QoS apply karen
        self.sensor_pub = self.create_publisher(Image, 'sensor_data', sensor_qos)
        self.control_pub = self.create_publisher(String, 'control_cmd', control_qos)
        self.config_pub = self.create_publisher(String, 'config', config_qos)
```

rclpy ROS 2 ke liye comprehensive Python interface provide karta hai, jo developers ko AI aur robotics applications ke liye Python ke extensive ecosystem ka leverage lena enable karta hai jabki ROS 2 jo robust communication infrastructure maintain karta hai.