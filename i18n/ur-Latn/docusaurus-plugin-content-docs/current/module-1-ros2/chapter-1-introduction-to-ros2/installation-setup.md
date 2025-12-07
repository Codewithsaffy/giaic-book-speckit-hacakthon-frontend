---
title: "Installation aur Setup"
description: "Ubuntu 22.04 par ROS 2 ko install karne, apna workspace set karna, aur installation verify karne ka complete guide."
sidebar_position: 4
keywords: ["ROS 2 installation", "Ubuntu 22.04", "setup", "workspace", "hello world"]
---

# Installation aur Setup

Yeh guide aapko Ubuntu 22.04 par ROS 2 ko install karne ke complete process, apna development workspace set karne, aur verify karne mein guide karega kya sab kuchh sahi se kaam kar raha hai. Hum installation process ko step-by-step cover karenge aur common issues ke liye troubleshooting tips provide karenge.

## System Requirements

ROS 2 install karne se pehle, ensure karen aapka system following requirements ko meet karta hai:

- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Architecture**: 64-bit x86 (AMD64)
- **RAM**: Minimum 4GB recommended (8GB+ for development)
- **Disk Space**: Kam se kam 5GB free space
- **Internet Connection**: Installation aur updates ke liye required hai

## Ubuntu 22.04 par ROS 2 ko Install karna

### 1. ROS 2 Repository ko Set karna

Sabse pehle, apne system ke package list ko update karen aur necessary tools install karen:

```bash
# Package list update karen
sudo apt update

# Required packages install karen
sudo apt install software-properties-common

# ROS 2 GPG key add karen
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# ROS 2 repository ko apne sources list mein add karen
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 2. ROS 2 Distribution ko Install karna

Yeh guide mein, hum latest stable ROS 2 distribution (Humble Hawksbill) install karenge:

```bash
# Package list update karen
sudo apt update

# ROS 2 development tools install karen
sudo apt install ros-humble-desktop

# Additional tools install karen
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### 3. rosdep ko Initialize karna

```bash
# rosdep initialize karen
sudo rosdep init

# rosdep update karen
rosdep update
```

### 4. Environment Setup

Setup script ko source karke ROS 2 ko apne environment mein add karen:

```bash
# ROS 2 setup script source karen
source /opt/ros/humble/setup.bash

# Isko permanent banane ke liye apne bashrc mein add karen
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

## Apna ROS 2 Workspace ko Set karna

### 1. Workspace Directory banana

```bash
# Workspace directory banayein
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### 2. Workspace ko Build karna

```bash
# Workspace ko build karen (even though yeh empty hai)
colcon build
```

### 3. Apna Workspace source karna

```bash
# Apna workspace source karen
source install/setup.bash

# Isko permanent banane ke liye bashrc mein add karen
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
```

## Hello World Node ke sath Installation Verify karna

Hum verify karne ke liye simple "Hello World" node banayenge kya aapki ROS 2 installation sahi se kaam kar rahi hai.

### 1. Simple Publisher Package banana

```bash
cd ~/ros2_ws/src

# New package banayein
ros2 pkg create --build-type ament_python hello_ros2 --dependencies rclpy std_msgs

# Package mein navigate karen
cd hello_ros2
```

### 2. Publisher Node banana

Publisher script ko `hello_ros2/hello_ros2/publisher.py` par banayein:

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

### 3. Subscriber Node banana

Subscriber script ko `hello_ros2/hello_ros2/subscriber.py` par banayein:

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

### 4. Package Setup ko Update karna

Package directory mein `setup.py` file ko update karen:

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

### 5. Package.xml ko Update karna

Ensure karen `package.xml` file necessary dependencies ko include karta hai:

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

### 6. Hello World Example ko Build aur Run karna

```bash
# Workspace root par wapas navigate karen
cd ~/ros2_ws

# Package ko build karen
colcon build --packages-select hello_ros2

# Workspace source karen
source install/setup.bash

# Ek terminal mein publisher run karen
ros2 run hello_ros2 publisher
```

Ek aur terminal mein, subscriber run karen:

```bash
# New terminal mein workspace source karen
source ~/ros2_ws/install/setup.bash

# Subscriber run karen
ros2 run hello_ros2 subscriber
```

Aapko publisher ko "Hello World" messages bhejte aur subscriber ko inhe receive karte dekhna chahiye. Yeh confirm karta hai kya aapki ROS 2 installation sahi se kaam kar rahi hai.

## Common Issues ko Troubleshoot karna

### 1. Permission Issues

Agar aapko permission errors milte hain, ensure karen aapne ROS repository sahi se add kiya hai:

```bash
# Check karen kya key sahi se install hai
apt-key list | grep -i ros

# Agar missing hai, key reinstall karen
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
```

### 2. Missing Dependencies

Agar installation ke dauran aapko missing dependencies milte hain:

```bash
# Package lists update karen
sudo apt update

# Broken dependencies fix karen
sudo apt --fix-broken install

# Phir se install karne ki koshish karen
sudo apt install ros-humble-desktop
```

### 3. Environment Setup Issues

Agar ROS 2 commands recognized nahi hote:

```bash
# Manually setup script source karen
source /opt/ros/humble/setup.bash

# Check karen kya environment variables set hain
echo $ROS_DISTRO
# Should output: humble

# Check karen kya ROS packages found hote hain
ros2 --help
```

### 4. Workspace Build Issues

Agar aapko build errors milte hain:

```bash
# Workspace clean karen
rm -rf build/ install/ log/

# Phir se build karen
colcon build --packages-select hello_ros2
```

### 5. Network Configuration Issues

Multi-machine communication ke liye, aapko apna network configure karna pad sakta hai:

```bash
# ROS domain ID set karen (default 0 hai)
export ROS_DOMAIN_ID=42

# RMW implementation set karen (if needed)
export RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
```

### 6. Python Path Issues

Agar Python packages found nahi hote:

```bash
# Ensure karen Python path sahi se set hai
echo $PYTHONPATH | grep ros

# Missing hone par apne bashrc mein add karen
echo "export PYTHONPATH=\$PYTHONPATH:/opt/ros/humble/lib/python3.10/site-packages" >> ~/.bashrc
source ~/.bashrc
```

## Best Practices

### 1. Workspace Organization
- Different projects ke liye separate workspaces rakhna
- Packages ke liye descriptive names ka istemal karna
- ROS 2 naming conventions follow karna

### 2. Virtual Environments
Python-based ROS 2 development ke liye virtual environments ka istemal consider karna:

```bash
# Virtual environment banayein
python3 -m venv ~/ros2_env

# Activate karen
source ~/ros2_env/bin/activate

# Virtual environment mein ROS 2 Python packages install karen
pip install rclpy std_msgs
```

### 3. Regular Updates
Apni ROS 2 installation ko updated rakhein:

```bash
# System packages update karen
sudo apt update && sudo apt upgrade

# ROS 2 packages update karen
sudo apt update && sudo apt upgrade ros-humble-*
```

## Key Takeaways

- Ubuntu 22.04 par ROS 2 installation ROS 2 repository ko set karna aur desktop package install karna require karta hai
- Workspaces aapke ROS 2 projects ke liye isolated environments provide karte hain
- Hello world example basic publisher-subscriber communication ko demonstrate karta hai
- ROS 2 sahi se kaam karne ke liye proper environment setup crucial hai
- Common issues ko troubleshoot karna dependencies, permissions, aur environment variables check karne mein involve hota hai
- Best practices follow karne se smooth development experience ensure hoti hai