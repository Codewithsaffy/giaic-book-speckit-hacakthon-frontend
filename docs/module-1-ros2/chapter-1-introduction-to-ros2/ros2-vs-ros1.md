---
title: "ROS 2 vs ROS 1"
description: "Comparing ROS 2 and ROS 1: key improvements, migration considerations, and when to use each version."
sidebar_position: 3
keywords: ["ROS 2 vs ROS 1", "comparison", "migration", "DDS", "middleware"]
---

# ROS 2 vs ROS 1

Understanding the differences between ROS 1 and ROS 2 is crucial for making informed decisions about which version to use for your robotics projects. This comparison will help you understand the evolution of the ROS ecosystem and the improvements that make ROS 2 the preferred choice for modern robotics applications.

## Key Differences Comparison Table

| Feature | ROS 1 | ROS 2 |
|---------|-------|-------|
| **Architecture** | Master-based centralized | DDS-based decentralized |
| **Middleware** | Custom TCP/UDP implementation | DDS (Data Distribution Service) |
| **Real-time Support** | Limited | Native support with QoS policies |
| **Multi-robot Systems** | Complex setup required | Native support, easier configuration |
| **Security** | No built-in security | Built-in authentication and encryption |
| **Platforms** | Linux/macOS primarily | Linux, Windows, macOS, embedded systems |
| **Programming Languages** | C++, Python primarily | C++, Python, Java, Rust, C# |
| **Quality of Service** | Basic | Advanced QoS policies |
| **Deployment** | Research-focused | Production-ready |
| **Communication** | TCPROS/UDPROS | DDS with configurable policies |
| **Lifecycle Management** | Manual | Built-in node lifecycle |
| **Time Handling** | Simulation time only | Real and simulation time |

## Key Improvements in ROS 2

### 1. **DDS Middleware Integration**
ROS 2's adoption of DDS (Data Distribution Service) as the underlying communication middleware brings significant advantages:

- **Decentralized Architecture**: No single point of failure with the master
- **Better Network Resilience**: Robust communication in unreliable networks
- **Interoperability**: Can communicate with other DDS-based systems
- **Scalability**: Better performance in multi-robot scenarios

### 2. **Real-time Capabilities**
ROS 2 introduces real-time support that was missing in ROS 1:

```cpp
// Example of real-time configuration in ROS 2
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/static_single_threaded_executor.hpp>

// Configure real-time QoS settings
rclcpp::QoS qos_profile(10);
qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
qos_profile.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
```

### 3. **Enhanced Security Features**
ROS 2 includes built-in security mechanisms:

- **Authentication**: Verify identity of nodes and participants
- **Encryption**: Encrypt data in transit and at rest
- **Access Control**: Control who can publish/subscribe to topics
- **Secure Discovery**: Protected participant discovery

### 4. **Quality of Service (QoS) Policies**
ROS 2's QoS system provides fine-grained control over communication behavior:

```yaml
# QoS Policy Configuration Examples
Reliability:
  - RELIABLE: All messages guaranteed delivery
  - BEST_EFFORT: Messages may be lost, but faster delivery

Durability:
  - TRANSIENT_LOCAL: Historical data available to late joiners
  - VOLATILE: Only future data available

Deadline: Maximum time between consecutive messages

Liveliness: How to determine if a participant is alive
```

### 5. **Improved Multi-robot Support**
ROS 2 handles multi-robot systems more effectively:

- **Namespace Isolation**: Better organization of topics/services
- **Domain IDs**: Isolate communication between robot groups
- **Built-in Discovery**: Automatic detection of robots on network

## Migration Considerations

### When to Migrate from ROS 1 to ROS 2

#### Migrate Now If:
- **Production Deployment**: You're moving from research to production
- **Security Requirements**: Your application needs authentication and encryption
- **Real-time Needs**: You require deterministic timing guarantees
- **Multi-robot Systems**: You're working with multiple robots
- **Cross-platform**: You need to run on Windows or embedded systems
- **Long-term Support**: You want the latest features and ongoing development

#### Stay with ROS 1 If:
- **Legacy Systems**: You have large existing ROS 1 codebases
- **Stable Applications**: Your ROS 1 system works well and doesn't need new features
- **Resource Constraints**: You have limited time/budget for migration
- **Package Dependencies**: Critical packages aren't available in ROS 2 yet

### Migration Strategy

1. **Assessment Phase**:
   - Inventory all existing packages and dependencies
   - Identify ROS 1-specific features used
   - Plan timeline and resource allocation

2. **Parallel Development**:
   - Start new features in ROS 2
   - Gradually port existing functionality
   - Use ROS 1 bridge for temporary compatibility

3. **Testing and Validation**:
   - Verify functionality after migration
   - Test performance in target environment
   - Ensure security requirements are met

## When to Use ROS 2

### Choose ROS 2 for:

#### **Production Robotics**
- Industrial automation systems
- Commercial robot deployment
- Safety-critical applications
- Long-term maintenance requirements

#### **Advanced Robotics Applications**
- Multi-robot coordination
- Complex sensor fusion
- Real-time control systems
- Distributed robotic systems

#### **Cross-Platform Development**
- Applications requiring Windows support
- Embedded systems with limited resources
- Mobile robotics with diverse hardware

#### **Security-Critical Systems**
- Applications requiring authentication
- Systems handling sensitive data
- Compliance with security standards

### Scenarios Where ROS 2 Excels:

1. **Humanoid Robots**: The distributed architecture and real-time capabilities are perfect for coordinating complex multi-joint systems
2. **Autonomous Vehicles**: Security features and reliable communication for safety-critical applications
3. **Industrial Automation**: Production-ready features and long-term support
4. **Research with Production Goals**: Future-proofing research that may transition to deployment

## Practical Migration Example

Here's a simple example showing the difference between ROS 1 and ROS 2 code:

### ROS 1 Publisher Example:
```python
#!/usr/bin/env python
import rospy
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('chatter', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)  # 10hz
    while not rospy.is_shutdown():
        hello_str = "hello world %s" % rospy.get_time()
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
```

### ROS 2 Publisher Example:
```python
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Key Takeaways

- ROS 2 addresses fundamental limitations of ROS 1 with DDS-based architecture
- Real-time capabilities, security, and multi-robot support make ROS 2 production-ready
- QoS policies provide fine-grained control over communication behavior
- Migration from ROS 1 requires careful planning but offers significant benefits
- ROS 2 is the recommended choice for new robotics projects, especially for humanoid robots