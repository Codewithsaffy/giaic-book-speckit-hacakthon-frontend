---
title: "Nodes and Lifecycle Management"
description: "Understanding ROS 2 nodes and their lifecycle management for humanoid robotics applications"
sidebar_position: 2
keywords: ["ROS 2 nodes", "lifecycle", "humanoid robotics", "node management"]
---

# Nodes and Lifecycle Management

Nodes form the backbone of any ROS 2 system, serving as the fundamental computational units that execute specific tasks. In humanoid robotics, effective node management is crucial for coordinating complex behaviors across multiple subsystems.

## Understanding ROS 2 Nodes

A ROS 2 node is an executable process that performs computation. Nodes are organized into packages and communicate with other nodes through topics, services, actions, and parameters. In humanoid robotics, nodes typically represent:

- **Sensor drivers**: Processing data from cameras, IMUs, force sensors
- **Control systems**: Managing joint controllers, balance algorithms
- **Perception modules**: Object detection, person tracking, environment mapping
- **Planning systems**: Motion planning, pathfinding, task scheduling
- **Behavior managers**: State machines, high-level decision making

## Creating a Basic Node

Let's examine the structure of a typical ROS 2 node:

```python
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Initialize parameters
        self.declare_parameter('control_frequency', 100)
        self.control_freq = self.get_parameter('control_frequency').value

        # Set up publishers, subscribers, services
        self.get_logger().info(f'Humanoid Controller initialized at {self.control_freq}Hz')

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Interrupted by user')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Node Lifecycle States

ROS 2 provides a lifecycle management system that allows nodes to transition through well-defined states:

- **Unconfigured**: Initial state after creation
- **Inactive**: Configured but not active
- **Active**: Fully operational
- **Finalized**: Cleaned up and ready for destruction

This is particularly important for humanoid robots where components need to be brought up in a specific order for safety reasons.

```python
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn

class LifecycleHumanoidNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_humanoid_node')
        self.get_logger().info('Lifecycle Humanoid Node created, currently unconfigured')

    def on_configure(self, state):
        self.get_logger().info('Configuring lifecycle node')
        # Initialize resources
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating lifecycle node')
        # Start timers, publishers, subscribers
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating lifecycle node')
        # Stop active components
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up lifecycle node')
        # Release resources
        return TransitionCallbackReturn.SUCCESS
```

## Best Practices for Humanoid Robotics

### Modular Design Principles

:::tip Design Guidelines
- **Single Responsibility**: Each node should handle one primary function
- **Loose Coupling**: Minimize dependencies between nodes
- **Clear Interfaces**: Define precise input/output contracts
- **Error Isolation**: Ensure failures in one node don't cascade
:::

### Resource Management

Humanoid robots often operate with limited computational resources, making efficient resource management critical:

```python
import rclpy
from rclpy.node import Node
import psutil  # For monitoring system resources

class ResourceAwareController(Node):
    def __init__(self):
        super().__init__('resource_aware_controller')

        # Monitor CPU and memory usage
        self.cpu_threshold = 80.0  # percentage
        self.memory_threshold = 85.0  # percentage

        # Create timer to check resource usage
        self.resource_timer = self.create_timer(1.0, self.check_resources)

    def check_resources(self):
        cpu_percent = psutil.cpu_percent(interval=None)
        memory_percent = psutil.virtual_memory().percent

        if cpu_percent > self.cpu_threshold:
            self.get_logger().warn(f'High CPU usage: {cpu_percent}%')

        if memory_percent > self.memory_threshold:
            self.get_logger().warn(f'High memory usage: {memory_percent}%')
```

### Safety Considerations

For humanoid robots, safety is paramount. Implement safety checks in your nodes:

```python
class SafetyController(Node):
    def __init__(self):
        super().__init__('safety_controller')

        # Emergency stop publisher
        self.emergency_stop_pub = self.create_publisher(
            Bool, '/emergency_stop', 1)

        # Joint position limits
        self.joint_limits = {
            'hip_pitch': (-1.5, 1.5),
            'knee_pitch': (-0.5, 2.0),
            'ankle_pitch': (-0.8, 0.8)
        }

    def validate_joint_positions(self, joint_msg):
        """Check if joint positions are within safe limits"""
        for i, joint_name in enumerate(joint_msg.name):
            if joint_name in self.joint_limits:
                min_val, max_val = self.joint_limits[joint_name]
                if not (min_val <= joint_msg.position[i] <= max_val):
                    self.get_logger().error(
                        f'Safety violation: {joint_name} out of range')
                    self.trigger_emergency_stop()
                    return False
        return True

    def trigger_emergency_stop(self):
        """Publish emergency stop command"""
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)
```

## Node Communication Patterns

Effective communication between nodes is essential for humanoid robotics:

### Publisher-Subscriber Pattern

```python
# Publisher example
class SensorDataPublisher(Node):
    def __init__(self):
        super().__init__('sensor_data_publisher')
        self.sensor_pub = self.create_publisher(SensorData, '/sensor_data', 10)

    def publish_sensor_data(self, data):
        msg = SensorData()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.data = data
        self.sensor_pub.publish(msg)

# Subscriber example
class SensorDataSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_data_subscriber')
        self.subscription = self.create_subscription(
            SensorData,
            '/sensor_data',
            self.sensor_callback,
            10)

    def sensor_callback(self, msg):
        # Process sensor data
        self.process_sensor_data(msg.data)
```

### Service-Based Communication

```python
# Service server
class ControlServiceServer(Node):
    def __init__(self):
        super().__init__('control_service_server')
        self.srv = self.create_service(
            SetJointPositions,
            '/set_joint_positions',
            self.set_joint_positions_callback)

    def set_joint_positions_callback(self, request, response):
        try:
            # Execute joint position command
            self.execute_joint_command(request.positions)
            response.success = True
            response.message = "Joint positions set successfully"
        except Exception as e:
            response.success = False
            response.message = str(e)

        return response
```

## Performance Optimization

For real-time humanoid applications, performance is critical:

:::note Performance Tips
- Use appropriate QoS profiles for different data types
- Implement efficient data structures for sensor processing
- Consider multi-threading for compute-intensive tasks
- Profile your nodes to identify bottlenecks
:::

## Key Takeaways

- Nodes are the fundamental building blocks of ROS 2 systems
- Lifecycle management provides controlled startup and shutdown
- Modular design enhances maintainability and safety
- Resource management is critical for embedded humanoid systems
- Safety considerations must be integrated into node design
- Effective communication patterns enable coordinated robot behavior

Understanding node architecture and lifecycle management is essential for building robust humanoid robotics applications that can operate safely and efficiently in real-world environments.