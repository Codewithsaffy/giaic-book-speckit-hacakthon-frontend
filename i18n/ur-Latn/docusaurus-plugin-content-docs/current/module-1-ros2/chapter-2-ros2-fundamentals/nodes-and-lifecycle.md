---
title: "Nodes aur Lifecycle Management"
description: "Humanoid robotics applications ke liye ROS 2 nodes aur unke lifecycle management ko samajhna"
sidebar_position: 2
keywords: ["ROS 2 nodes", "lifecycle", "humanoid robotics", "node management"]
---

# Nodes aur Lifecycle Management

Nodes kisi bhi ROS 2 system ka backbone hain, jo specific tasks execute karne wale fundamental computational units ka kaam karte hain. Humanoid robotics mein, effective node management multiple subsystems ke across complex behaviors ko coordinate karne ke liye crucial hai.

## ROS 2 Nodes ko Samajhna

ROS 2 node executable process hai jo computation perform karta hai. Nodes packages mein organized hote hain aur topics, services, actions, aur parameters ke through other nodes ke sath communicate karte hain. Humanoid robotics mein, nodes typically represent karte hain:

- **Sensor drivers**: Cameras, IMUs, force sensors se data processing
- **Control systems**: Joint controllers, balance algorithms manage karna
- **Perception modules**: Object detection, person tracking, environment mapping
- **Planning systems**: Motion planning, pathfinding, task scheduling
- **Behavior managers**: State machines, high-level decision making

## Basic Node Banana

Aao typical ROS 2 node ke structure ko examine karte hain:

```python
import rclpy
from rclpy.node import Node

class HumanoidController(Node):
    def __init__(self):
        super().__init__('humanoid_controller')

        # Parameters initialize karen
        self.declare_parameter('control_frequency', 100)
        self.control_freq = self.get_parameter('control_frequency').value

        # Publishers, subscribers, services set up karen
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

ROS 2 lifecycle management system provide karta hai jo nodes ko well-defined states mein transition karne ki allow karta hai:

- **Unconfigured**: Creation ke baad initial state
- **Inactive**: Configured lekin active nahi
- **Active**: Fully operational
- **Finalized**: Cleaned up aur destruction ke liye ready

Yeh humanoid robots ke liye particularly important hai jahan safety reasons ke liye components ko specific order mein bring up kiya jana chahiye.

```python
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn

class LifecycleHumanoidNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_humanoid_node')
        self.get_logger().info('Lifecycle Humanoid Node created, currently unconfigured')

    def on_configure(self, state):
        self.get_logger().info('Configuring lifecycle node')
        # Resources initialize karen
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating lifecycle node')
        # Timers, publishers, subscribers start karen
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating lifecycle node')
        # Active components stop karen
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up lifecycle node')
        # Resources release karen
        return TransitionCallbackReturn.SUCCESS
```

## Humanoid Robotics ke liye Best Practices

### Modular Design Principles

:::tip Design Guidelines
- **Single Responsibility**: Har node ko ek primary function handle karna chahiye
- **Loose Coupling**: Nodes ke beech dependencies minimize karen
- **Clear Interfaces**: Precise input/output contracts define karen
- **Error Isolation**: Ensure karen ek node mein failures cascade nahi hoti
:::

### Resource Management

Humanoid robots often limited computational resources ke sath operate karte hain, jo efficient resource management ko critical banata hai:

```python
import rclpy
from rclpy.node import Node
import psutil  # For monitoring system resources

class ResourceAwareController(Node):
    def __init__(self):
        super().__init__('resource_aware_controller')

        # CPU aur memory usage monitor karen
        self.cpu_threshold = 80.0  # percentage
        self.memory_threshold = 85.0  # percentage

        # Resource usage check karne ke liye timer create karen
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

Humanoid robots ke liye, safety paramount hai. Apne nodes mein safety checks implement karen:

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
        """Check karen kya joint positions safe limits ke andhar hain"""
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

Humanoid robotics ke liye nodes ke beech effective communication essential hai:

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
        # Sensor data process karen
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
            # Joint position command execute karen
            self.execute_joint_command(request.positions)
            response.success = True
            response.message = "Joint positions set successfully"
        except Exception as e:
            response.success = False
            response.message = str(e)

        return response
```

## Performance Optimization

Real-time humanoid applications ke liye, performance critical hai:

:::note Performance Tips
- Different data types ke liye appropriate QoS profiles ka istemal karen
- Sensor processing ke liye efficient data structures implement karen
- Compute-intensive tasks ke liye multi-threading consider karen
- Bottlenecks identify karne ke liye apne nodes ko profile karen
:::

## Key Takeaways

- Nodes ROS 2 systems ke fundamental building blocks hain
- Lifecycle management controlled startup aur shutdown provide karta hai
- Modular design maintainability aur safety enhance karta hai
- Resource management embedded humanoid systems ke liye critical hai
- Safety considerations node design mein integrated kiye jane chahiye
- Effective communication patterns coordinated robot behavior enable karte hain

Node architecture aur lifecycle management ko samajhna real-world environments mein safely aur efficiently operate karne wale robust humanoid robotics applications banana ke liye essential hai.