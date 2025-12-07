---
sidebar_position: 4
title: "Isaac Installation and Setup"
description: "Step-by-step guide to installing and setting up NVIDIA Isaac platform"
---

# Isaac Installation and Setup

Installing and configuring the NVIDIA Isaac platform requires careful attention to system requirements, software dependencies, and configuration settings. This guide provides a comprehensive step-by-step process for setting up Isaac on your system.

## Prerequisites Verification

### Hardware Verification

Before beginning installation, verify your system meets the requirements:

```bash
# Check GPU availability and CUDA support
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check GPU compute capability (should be 7.5 or higher)
nvidia-ml-py3 -c "import pynvml; pynvml.nvmlInit(); handle = pynvml.nvmlDeviceGetHandleByIndex(0); print(pynvml.nvmlDeviceGetName(handle).decode('utf-8'))"
```

### System Dependencies

Ensure your system has the required dependencies:

```bash
# For Ubuntu 20.04/22.04
sudo apt update
sudo apt install -y build-essential cmake git python3-dev python3-pip
sudo apt install -y libeigen3-dev libopencv-dev libboost-all-dev
sudo apt install -y libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
```

## Isaac Sim Installation

### Prerequisites for Isaac Sim

Isaac Sim has specific requirements:

```bash
# Install NVIDIA drivers (if not already installed)
sudo apt install nvidia-driver-535

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install -y cuda-toolkit-12-0
```

### Installing Isaac Sim via Omniverse Launcher

1. **Download Omniverse Launcher**:
   - Visit the NVIDIA Omniverse website
   - Download the Omniverse Launcher for your operating system
   - Install and run the launcher

2. **Install Isaac Sim**:
   - Open Omniverse Launcher
   - Navigate to the "Isaac Sim" application
   - Click "Install" to download and install Isaac Sim
   - Accept the license agreement

3. **Configure Isaac Sim**:
   - Launch Isaac Sim from the launcher
   - Configure initial settings (workspace, assets location)
   - Verify GPU acceleration is enabled

### Alternative: Isaac Sim via Docker

For containerized deployment:

```bash
# Pull Isaac Sim Docker image
docker pull nvcr.io/nvidia/isaac-sim:4.0.0

# Run Isaac Sim container
docker run --gpus all -it --rm \
  --network=host \
  --env "DISPLAY" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="${PWD}:/workspace" \
  --volume="/home/${USER}:/workspace/user" \
  --volume="/etc/timezone:/etc/timezone:ro" \
  --volume="/etc/localtime:/etc/localtime:ro" \
  nvcr.io/nvidia/isaac-sim:4.0.0
```

## Isaac ROS Installation

### Setting up ROS Environment

Isaac ROS requires a ROS/ROS2 installation:

```bash
# Install ROS2 Humble Hawksbill (for Ubuntu 22.04)
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install -y ros-humble-desktop
sudo apt install -y python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
```

### Installing Isaac ROS from Source

```bash
# Create ROS2 workspace
mkdir -p ~/isaac_ros_ws/src
cd ~/isaac_ros_ws

# Install vcs tool
pip3 install -U vcstool

# Clone Isaac ROS repositories
wget https://raw.githubusercontent.com/NVIDIA-ISAAC-ROS/.github/main/repositories/isaac_ros.repos
vcs import src < isaac_ros.repos

# Install dependencies
sudo apt update
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Build Isaac ROS packages
colcon build --symlink-install --packages-select \
  isaac_ros_visual_slam \
  isaac_ros_point_cloud_localizer \
  isaac_ros_apriltag \
  isaac_ros_compressed_image_transport \
  isaac_ros_image_pipeline
```

### Isaac ROS via Debian Packages

Alternative installation method:

```bash
# Add NVIDIA repository
sudo apt update && sudo apt install wget gnupg lsb-release
sudo sh -c 'echo "deb https://packages.isaac-ros.nvidia.com/$(lsb_release -cs) main" > /etc/apt/sources.list.d/nvidia-isaac-ros.list'
wget -O - https://packages.isaac-ros.nvidia.com/gpg | sudo apt-key add -

# Install Isaac ROS packages
sudo apt update
sudo apt install -y \
  ros-humble-isaac-ros-visual-slam \
  ros-humble-isaac-ros-point-cloud-localizer \
  ros-humble-isaac-ros-apriltag \
  ros-humble-isaac-ros-compressed-image-transport
```

## CUDA and GPU Optimization

### CUDA Environment Setup

Configure CUDA environment variables:

```bash
# Add to ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### GPU Memory Optimization

Configure GPU memory settings for optimal Isaac performance:

```bash
# Check current GPU memory usage
nvidia-smi -q -d MEMORY

# Configure persistence mode for better performance
sudo nvidia-smi -pm 1

# Set GPU to maximum performance mode
sudo nvidia-smi -ac 10000,1800  # Adjust based on your GPU
```

## Isaac Navigation Installation

### Building Navigation Stack

```bash
# Navigate to workspace
cd ~/isaac_ros_ws

# Clone navigation packages
git clone -b ros2 https://github.com/ros-planning/navigation2.git src/navigation2
git clone -b main https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_navigation.git src/isaac_ros_navigation

# Install navigation dependencies
sudo apt update
rosdep install --from-paths src --ignore-src -r -y

# Build navigation packages
colcon build --symlink-install --packages-select \
  nav2_common \
  nav2_costmap_2d \
  nav2_planner \
  nav2_controller \
  nav2_behavior_tree \
  nav2_msgs \
  isaac_ros_navigation
```

## Isaac Manipulation Installation

### Installing Manipulation Packages

```bash
# Clone manipulation packages
cd ~/isaac_ros_ws/src
git clone -b main https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_manipulation.git

# Install dependencies
sudo apt update
rosdep install --from-paths isaac_ros_manipulation --ignore-src -r -y

# Build manipulation packages
cd ~/isaac_ros_ws
colcon build --symlink-install --packages-select \
  isaac_ros_apriltag \
  isaac_ros_pose_estimation \
  isaac_ros_bi3d
```

## Environment Configuration

### Setting up Environment Variables

Create a setup script for Isaac:

```bash
# Create setup script
cat > ~/isaac_setup.sh << 'EOF'
#!/bin/bash

# Source ROS2
source /opt/ros/humble/setup.bash

# Source Isaac ROS workspace
source ~/isaac_ros_ws/install/setup.bash

# CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Isaac specific environment variables
export ISAAC_ROS_WS=~/isaac_ros_ws
export OMNI_USER_DATA_PATH=~/isaac_sim_data

# GPU optimization settings
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
EOF

chmod +x ~/isaac_setup.sh
```

### Adding to Shell Profile

```bash
# Add to ~/.bashrc
echo 'source ~/isaac_setup.sh' >> ~/.bashrc
source ~/.bashrc
```

## Verification and Testing

### Testing Isaac Sim

Launch Isaac Sim to verify installation:

```bash
# If installed via Omniverse Launcher
# Launch from the Omniverse Launcher

# If installed via Docker
xhost +local:docker
docker run --gpus all -it --rm \
  --network=host \
  --env "DISPLAY" \
  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="${PWD}:/workspace" \
  nvcr.io/nvidia/isaac-sim:4.0.0
```

### Testing Isaac ROS

Test basic Isaac ROS functionality:

```bash
# Source environment
source ~/isaac_setup.sh

# Run a simple Isaac ROS node
ros2 run isaac_ros_apriltag apriltag_node

# Test with sample image
ros2 launch isaac_ros_apriltag isaac_ros_apriltag.launch.py
```

### GPU Acceleration Verification

Verify GPU acceleration is working:

```bash
# Monitor GPU usage during Isaac operations
watch -n 1 nvidia-smi

# Test CUDA functionality
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

## Troubleshooting Common Issues

### CUDA Installation Issues

```bash
# If CUDA installation fails
sudo apt remove --purge nvidia-*
sudo apt autoremove
sudo apt update
sudo apt install nvidia-driver-535 nvidia-utils-535
reboot

# After reboot, install CUDA toolkit again
sudo apt install cuda-toolkit-12-0
```

### Isaac Sim Launch Issues

```bash
# Check OpenGL support
glxinfo | grep -i nvidia
glxinfo | grep -i opengl

# Verify display settings
echo $DISPLAY

# Check X11 forwarding if using remote display
xhost +local:docker
```

### Isaac ROS Build Issues

```bash
# Clean workspace and rebuild
cd ~/isaac_ros_ws
rm -rf build install log
colcon build --symlink-install

# Check for missing dependencies
rosdep install --from-paths src --ignore-src -r -y --rosdistro humble
```

## Post-Installation Configuration

### Performance Tuning

Configure system for optimal Isaac performance:

```bash
# Increase shared memory size
echo 'tmpfs /dev/shm tmpfs defaults,size=8G 0 0' | sudo tee -a /etc/fstab
sudo mount -o remount,size=8G /dev/shm

# Configure swappiness
echo 'vm.swappiness=1' | sudo tee -a /etc/sysctl.conf

# Increase file descriptor limits
echo '* soft nofile 65536' | sudo tee -a /etc/security/limits.conf
echo '* hard nofile 65536' | sudo tee -a /etc/security/limits.conf
```

### Creating Isaac Project Template

Create a template for new Isaac projects:

```bash
# Create project structure
mkdir -p ~/isaac_projects/my_robot/config
mkdir -p ~/isaac_projects/my_robot/launch
mkdir -p ~/isaac_projects/my_robot/models

# Create basic launch file template
cat > ~/isaac_projects/my_robot/launch/isaac_app.launch.py << 'EOF'
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='isaac_ros_apriltag',
            executable='apriltag_node',
            name='apriltag',
            parameters=[{
                'family': 'tag36h11',
                'max_tags': 64,
                'tag_size': 0.032
            }]
        )
    ])
EOF
```

The Isaac installation and setup process requires careful attention to dependencies and configuration. Proper installation ensures optimal performance of GPU-accelerated robotics applications. The next sections will cover specific Isaac components in detail.