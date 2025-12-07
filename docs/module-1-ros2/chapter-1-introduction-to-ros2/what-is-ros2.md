---
title: "What is ROS 2?"
description: "Understanding the core concepts of ROS 2, its architecture, DDS middleware, and real-time capabilities for robotics development."
sidebar_position: 2
keywords: ["ROS 2 architecture", "DDS middleware", "real-time", "robotics framework", "humanoid robots"]
---

# What is ROS 2?

ROS 2 (Robot Operating System 2) is the next-generation framework for robotics development that builds upon the success of ROS 1 while addressing its limitations. Unlike ROS 1, which was primarily designed for research environments, ROS 2 is engineered for production use with enhanced reliability, scalability, and security features.

## Core Concepts

ROS 2 is not an actual operating system but rather a collection of software frameworks and tools that provide services designed for robotics applications. It includes hardware abstraction, device drivers, libraries, visualizers, message-passing functionality, and package management.

At its heart, ROS 2 enables:

- **Communication**: Nodes communicate through topics, services, and actions
- **Coordination**: Distributed systems coordination with fault tolerance
- **Reusability**: Standardized interfaces and packages
- **Visualization**: Tools for monitoring and debugging

## ROS 2 Architecture

The architecture of ROS 2 is fundamentally different from ROS 1, primarily due to its underlying middleware layer. Here's a breakdown of the key architectural components:

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
├─────────────────────────────────────────────────────────┤
│                    Client Libraries                      │
│            (rclcpp, rclpy, rcljava, etc.)               │
├─────────────────────────────────────────────────────────┤
│                   ROS Client Library                     │
│                        (rcl)                            │
├─────────────────────────────────────────────────────────┤
│                 Middleware Interface                     │
│                    (rmw)                                │
├─────────────────────────────────────────────────┬───────┤
│                DDS Implementation               │ Other │
│        (Fast DDS, Cyclone DDS, RTI Connext)     │ RMW   │
└─────────────────────────────────────────────────┴───────┘
```

### Key Architectural Components:

1. **Application Layer**: Your robot applications and nodes
2. **Client Libraries**: Language-specific libraries (rclcpp for C++, rclpy for Python)
3. **ROS Client Library (rcl)**: Core ROS abstractions implemented in C
4. **Middleware Interface (rmw)**: Abstraction layer for different middleware implementations
5. **DDS Implementation**: The underlying Data Distribution Service implementation

## DDS Middleware Concept

DDS (Data Distribution Service) is the cornerstone of ROS 2's architecture. DDS is an OMG (Object Management Group) standard for real-time, scalable, dependable data-coupling connectivity.

### DDS Characteristics:

- **Publish/Subscribe Model**: Nodes publish data to topics, and other nodes subscribe to receive that data
- **Quality of Service (QoS)**: Configurable policies for reliability, durability, deadline, liveliness, etc.
- **Discovery**: Automatic peer discovery and connection establishment
- **Data-Centric**: Focuses on data rather than communication endpoints

### QoS Policies in DDS:

```yaml
Reliability:
  - RELIABLE: All messages are delivered (like TCP)
  - BEST_EFFORT: Messages may be lost (like UDP)

Durability:
  - TRANSIENT_LOCAL: Historical data available to late joiners
  - VOLATILE: Only future data available

Deadline:
  - Period within which data should be updated

Liveliness:
  - How to detect if a participant is alive
```

The QoS system allows ROS 2 to handle diverse communication requirements, from best-effort sensor data to reliable command messages.

## Real-time Capabilities

ROS 2 is designed with real-time systems in mind, offering several features that make it suitable for time-critical applications:

### Real-time Features:

1. **Deterministic Communication**: Predictable message delivery times
2. **Thread Safety**: Properly designed for concurrent execution
3. **Memory Management**: Pre-allocated memory pools to avoid allocation delays
4. **Priority-based Scheduling**: Support for real-time scheduling policies

### Real-time Configuration Example:

```bash
# Set real-time capabilities for ROS 2 processes
sudo apt install linux-image-rt-amd64  # Real-time kernel
ulimit -r 99                           # Set real-time priority limit
```

### Real-time Considerations:

- Use real-time kernel patches for deterministic timing
- Configure appropriate QoS policies for time-sensitive data
- Implement proper memory pre-allocation strategies
- Monitor system performance and latency

## Why ROS 2 for Humanoid Robots

Humanoid robots have unique requirements that make ROS 2 particularly suitable:

### 1. **Complex Sensor Fusion**
Humanoid robots typically have numerous sensors (IMUs, cameras, force sensors, joint encoders) that need to be synchronized and processed in real-time. ROS 2's QoS system ensures reliable data delivery with appropriate timing constraints.

### 2. **Distributed Control Architecture**
Humanoid robots often have distributed controllers for different subsystems (legs, arms, torso, head). ROS 2's decentralized architecture supports this naturally.

### 3. **Safety and Reliability**
With safety-critical applications like humanoid robots, the reliability and fault-tolerance features of ROS 2 are crucial for preventing dangerous situations.

### 4. **Multi-Robot Coordination**
In scenarios where multiple humanoid robots need to coordinate, ROS 2's robust networking and discovery mechanisms excel.

### 5. **Real-time Performance**
Humanoid locomotion requires precise timing for balance control and motion planning, which ROS 2's real-time capabilities support well.

## Practical Example: Node Communication

Here's a simple example of how nodes communicate in ROS 2:

```python
# Publisher node example
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
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1
```

## Key Takeaways

- ROS 2 uses a layered architecture with DDS as the underlying middleware
- DDS provides publish/subscribe communication with configurable QoS policies
- Real-time capabilities make ROS 2 suitable for time-critical applications
- The distributed architecture is ideal for complex robotic systems like humanoid robots
- QoS policies allow fine-tuning communication behavior for different use cases