---
title: "Installation and Setup"
description: "Complete guide to installing ROS 2 on Ubuntu 22.04, setting up your workspace, and verifying the installation."
sidebar_position: 4
keywords: ["ROS 2 installation", "Ubuntu 22.04", "setup", "workspace", "hello world"]
---

# Installation and Setup

This guide will walk you through the complete process of installing ROS 2 on Ubuntu 22.04, setting up your development workspace, and verifying that everything is working correctly. We'll cover the installation process step-by-step and provide troubleshooting tips for common issues.

## System Requirements

Before installing ROS 2, ensure your system meets the following requirements:

- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Architecture**: 64-bit x86 (AMD64)
- **RAM**: Minimum 4GB recommended (8GB+ for development)
- **Disk Space**: At least 5GB of free space
- **Internet Connection**: Required for installation and updates

## Installing ROS 2 on Ubuntu 22.04

### 1. Set up the ROS 2 Repository

First, update your system's package list and install the necessary tools:

```bash
# Update package list
sudo apt update

# Install required packages
sudo apt install software-properties-common

# Add the ROS 2 GPG key
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add the ROS 2 repository to your sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 2. Install ROS 2 Distribution

For this guide, we'll install the latest stable ROS 2 distribution (Humble Hawksbill):

```bash
# Update package list
sudo apt update

# Install ROS 2 development tools
sudo apt install ros-humble-desktop

# Install additional tools
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### 3. Initialize rosdep

```bash
# Initialize rosdep
sudo rosdep init

# Update rosdep
rosdep update
```

### 4. Environment Setup

Add ROS 2 to your environment by sourcing the setup script:

```bash
# Source the ROS 2 setup script
source /opt/ros/humble/setup.bash

# Add to your bashrc to make it permanent
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Setting Up Your ROS 2 Workspace

### 1. Create a Workspace Directory

```bash
# Create workspace directory
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### 2. Build the Workspace

```bash
# Build the workspace (even though it's empty)
colcon build
```

### 3. Source Your Workspace

```bash
# Source your workspace
source install/setup.bash

# Add to bashrc to make it permanent
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Verify Installation with Hello World Node

Let's create a simple "Hello World" node to verify that your ROS 2 installation is working correctly.

### 1. Create a Simple Publisher Package

```bash
cd ~/ros2_ws/src

# Create a new package
ros2 pkg create --build-type ament_python hello_ros2 --dependencies rclpy std_msgs

# Navigate to the package
cd hello_ros2
```

### 2. Create the Publisher Node

Create the publisher script at `hello_ros2/hello_ros2/publisher.py`:

```python
#!/usr/bin/env python3
# hello_ros2/hello_ros2/publisher.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HelloWorldPublisher(Node):
    def __init__(self):
        super().__init__('hello_world_publisher')
        self.publisher_ = self.create_publisher(String, 'hello_world', 10)
        timer_period = 1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    hello_world_publisher = HelloWorldPublisher()
    rclpy.spin(hello_world_publisher)
    hello_world_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 3. Create a Subscriber Node

Create the subscriber script at `hello_ros2/hello_ros2/subscriber.py`:

```python
#!/usr/bin/env python3
# hello_ros2/hello_ros2/subscriber.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class HelloWorldSubscriber(Node):
    def __init__(self):
        super().__init__('hello_world_subscriber')
        self.subscription = self.create_subscription(
            String,
            'hello_world',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    hello_world_subscriber = HelloWorldSubscriber()
    rclpy.spin(hello_world_subscriber)
    hello_world_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 4. Update the Package Setup

Update the `setup.py` file in the package directory:

```python
# hello_ros2/setup.py
from setuptools import find_packages, setup

package_name = 'hello_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Simple Hello World ROS 2 package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher = hello_ros2.publisher:main',
            'subscriber = hello_ros2.subscriber:main',
        ],
    },
)
```

### 5. Update Package.xml

Make sure the `package.xml` file includes the necessary dependencies:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>hello_ros2</name>
  <version>0.0.0</version>
  <description>Simple Hello World ROS 2 package</description>
  <maintainer email="your.email@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>std_msgs</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

### 6. Build and Run the Hello World Example

```bash
# Navigate back to the workspace root
cd ~/ros2_ws

# Build the package
colcon build --packages-select hello_ros2

# Source the workspace
source install/setup.bash

# Run the publisher in one terminal
ros2 run hello_ros2 publisher
```

In another terminal, run the subscriber:

```bash
# Source the workspace in the new terminal
source ~/ros2_ws/install/setup.bash

# Run the subscriber
ros2 run hello_ros2 subscriber
```

You should see the publisher sending "Hello World" messages and the subscriber receiving them. This confirms that your ROS 2 installation is working correctly.

## Troubleshooting Common Issues

### 1. Permission Issues

If you encounter permission errors, ensure you've added the ROS repository correctly:

```bash
# Check if the key is properly installed
apt-key list | grep -i ros

# If missing, reinstall the key
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
```

### 2. Missing Dependencies

If you encounter missing dependencies during installation:

```bash
# Update package lists
sudo apt update

# Fix broken dependencies
sudo apt --fix-broken install

# Try installing again
sudo apt install ros-humble-desktop
```

### 3. Environment Setup Issues

If ROS 2 commands are not recognized:

```bash
# Manually source the setup script
source /opt/ros/humble/setup.bash

# Check if environment variables are set
echo $ROS_DISTRO
# Should output: humble

# Check if ROS packages are found
ros2 --help
```

### 4. Workspace Build Issues

If you encounter build errors:

```bash
# Clean the workspace
rm -rf build/ install/ log/

# Rebuild
colcon build --packages-select hello_ros2
```

### 5. Network Configuration Issues

For multi-machine communication, you might need to configure your network:

```bash
# Set the ROS domain ID (default is 0)
export ROS_DOMAIN_ID=42

# Set the RMW implementation (if needed)
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### 6. Python Path Issues

If Python packages are not found:

```bash
# Ensure Python path is set correctly
echo $PYTHONPATH | grep ros

# Add to your bashrc if missing
echo "export PYTHONPATH=\$PYTHONPATH:/opt/ros/humble/lib/python3.10/site-packages" >> ~/.bashrc
source ~/.bashrc
```

## Best Practices

### 1. Workspace Organization
- Keep separate workspaces for different projects
- Use descriptive names for packages
- Follow ROS 2 naming conventions

### 2. Virtual Environments
Consider using virtual environments for Python-based ROS 2 development:

```bash
# Create a virtual environment
python3 -m venv ~/ros2_env

# Activate it
source ~/ros2_env/bin/activate

# Install ROS 2 Python packages in the virtual environment
pip install rclpy std_msgs
```

### 3. Regular Updates
Keep your ROS 2 installation up to date:

```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update ROS 2 packages
sudo apt update && sudo apt upgrade ros-humble-*
```

## Key Takeaways

- ROS 2 installation on Ubuntu 22.04 requires setting up the ROS 2 repository and installing the desktop package
- Workspaces provide isolated environments for your ROS 2 projects
- The hello world example demonstrates basic publisher-subscriber communication
- Proper environment setup is crucial for ROS 2 to function correctly
- Troubleshooting common issues involves checking dependencies, permissions, and environment variables
- Following best practices ensures a smooth development experience