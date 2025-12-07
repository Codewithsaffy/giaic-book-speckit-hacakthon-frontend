---
sidebar_position: 2
title: "Nav2 Integration with Isaac"
description: "Integrating GPU-accelerated navigation with the ROS 2 Navigation Stack"
---

# Nav2 Integration with Isaac

The ROS 2 Navigation Stack (Nav2) provides a comprehensive framework for robot navigation, and Isaac extends this with GPU acceleration for enhanced performance. This integration enables humanoid robots to leverage both the robust navigation capabilities of Nav2 and the computational power of GPU acceleration for complex navigation tasks.

## Understanding Nav2 Architecture

### Core Nav2 Components

Nav2 consists of several key components that work together:

#### Navigation System Architecture
```
[Planner Server] → [Controller Server] → [Recovery Server]
       ↓                ↓                   ↓
   Global Planner    Local Controller   Recovery Behaviors
```

#### Key Services and Actions
- **Navigation Action**: Main navigation interface (NavigateToPose)
- **Task Planning**: High-level task planning (ComputePathToPose)
- **Control Interface**: Low-level control (FollowPath)
- **Recovery System**: Failure recovery (Spin, Backup, Wait)

### Nav2 Plugins System

Nav2 uses a plugin-based architecture for flexibility:

#### Planner Plugins
- **GlobalPlanner**: Interface for global path planning
- **LocalPlanner**: Interface for local path following
- **Controller**: Interface for trajectory following
- **Smoother**: Interface for path smoothing

#### Example Plugin Configuration
```yaml
# nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: true
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.5
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.1
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: true
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "odom"
    default_bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
      - "bt_navigator/navigate_to_pose"
      - "bt_navigator/spin"
      - "bt_navigator/back_up"
      - "bt_navigator/wait"
      - "bt_navigator/clear_costmap_service"
      - "bt_navigator/costmap_to_costmap"

controller_server:
  ros__parameters:
    use_sim_time: true
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 10.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    # Controller parameters
    FollowPath:
      plugin: "nav2_rotation_shim_controller::RotationShimController"
      progress_checker_plugin: "progress_checker"
      goal_checker_plugin: "goal_checker"
      RotateWithConstantVelocity:
        plugin: "nav2_controller::SimpleRotate"
        desired_linear_vel: 0.0
        max_angular_accel: 1.0
        max_angular_vel: 1.0
        min_angular_vel: 0.0
        tolerance: 0.1
```

## Isaac Nav2 Extensions

### GPU-Accelerated Navigation Components

Isaac extends Nav2 with GPU acceleration:

#### Isaac Nav2 Planner Server
```cpp
#include <nav2_core/global_planner.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.h>
#include <nav2_util/lifecycle_node.hpp>
#include <nav2_msgs/msg/costmap.hpp>
#include <cuda_runtime.h>

class IsaacGPUPlanner : public nav2_core::GlobalPlanner {
public:
    IsaacGPUPlanner() = default;
    ~IsaacGPUPlanner() override = default;

    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
        std::string name,
        std::shared_ptr<tf2_ros::Buffer> tf,
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override {

        node_ = parent.lock();
        name_ = name;
        tf_ = tf;
        costmap_ros_ = costmap_ros;
        costmap_ = costmap_ros_->getCostmap();

        // Initialize GPU context
        cudaSetDevice(0);
        cudaFree(0); // Initialize context

        // Allocate GPU memory for planning
        initialize_gpu_planning();
    }

    void cleanup() override {
        // Clean up GPU memory
        cleanup_gpu_planning();
    }

    void activate() override {
        RCLCPP_INFO(node_->get_logger(), "Activating IsaacGPUPlanner");
    }

    void deactivate() override {
        RCLCPP_INFO(node_->get_logger(), "Deactivating IsaacGPUPlanner");
    }

    nav_msgs::msg::Path createPlan(
        const geometry_msgs::msg::PoseStamped& start,
        const geometry_msgs::msg::PoseStamped& goal) override {

        nav_msgs::msg::Path path;

        // Check if goal is valid
        if (!costmap_ros_->getCostmap()->isInBounds(
                static_cast<unsigned int>(goal.pose.position.x / costmap_->getResolution()),
                static_cast<unsigned int>(goal.pose.position.y / costmap_->getResolution()))) {
            RCLCPP_WARN(node_->get_logger(), "Goal is out of bounds, cannot create plan");
            return path;
        }

        // Convert costmap to GPU format
        convert_costmap_to_gpu();

        // Perform GPU-accelerated path planning
        std::vector<geometry_msgs::msg::PoseStamped> poses =
            plan_path_gpu(start, goal);

        // Build path message
        path.header.frame_id = costmap_ros_->getGlobalFrameID();
        path.header.stamp = node_->now();
        path.poses = poses;

        return path;
    }

private:
    void initialize_gpu_planning() {
        // Allocate GPU memory for costmap
        int width = costmap_->getSizeInCellsX();
        int height = costmap_->getSizeInCellsY();
        size_t costmap_size = width * height * sizeof(unsigned char);

        cudaMalloc(&d_costmap_, costmap_size);
        cudaMalloc(&d_path_, width * height * sizeof(int)); // For path reconstruction

        // Initialize CUDA streams
        cudaStreamCreate(&planning_stream_);
    }

    void cleanup_gpu_planning() {
        if (d_costmap_) cudaFree(d_costmap_);
        if (d_path_) cudaFree(d_path_);
        if (planning_stream_) cudaStreamDestroy(planning_stream_);
    }

    void convert_costmap_to_gpu() {
        int width = costmap_->getSizeInCellsX();
        int height = costmap_->getSizeInCellsY();

        const unsigned char* costmap_data = costmap_->getCharMap();
        cudaMemcpyAsync(d_costmap_, costmap_data,
                       width * height * sizeof(unsigned char),
                       cudaMemcpyHostToDevice, planning_stream_);
    }

    std::vector<geometry_msgs::msg::PoseStamped> plan_path_gpu(
        const geometry_msgs::msg::PoseStamped& start,
        const geometry_msgs::msg::PoseStamped& goal) {

        std::vector<geometry_msgs::msg::PoseStamped> path;

        // Convert start and goal to grid coordinates
        unsigned int start_x, start_y, goal_x, goal_y;
        worldToMap(start_x, start_y, start.pose.position.x, start.pose.position.y);
        worldToMap(goal_x, goal_y, goal.pose.position.x, goal.pose.position.y);

        // Launch GPU A* planner
        dim3 block_size(16, 16);
        dim3 grid_size((costmap_->getSizeInCellsX() + block_size.x - 1) / block_size.x,
                      (costmap_->getSizeInCellsY() + block_size.y - 1) / block_size.y);

        // Execute A* algorithm on GPU
        execute_astar_gpu_kernel<<<grid_size, block_size, 0, planning_stream_>>>(
            d_costmap_, d_path_,
            costmap_->getSizeInCellsX(), costmap_->getSizeInCellsY(),
            start_x, start_y, goal_x, goal_y
        );

        // Copy path back to CPU and reconstruct
        path = reconstruct_path_gpu(start, goal);

        return path;
    }

    void worldToMap(unsigned int& mx, unsigned int& my, double wx, double wy) {
        mx = static_cast<unsigned int>(
            (wx - costmap_->getOriginX()) / costmap_->getResolution());
        my = static_cast<unsigned int>(
            (wy - costmap_->getOriginY()) / costmap_->getResolution());
    }

    std::vector<geometry_msgs::msg::PoseStamped> reconstruct_path_gpu(
        const geometry_msgs::msg::PoseStamped& start,
        const geometry_msgs::msg::PoseStamped& goal) {

        std::vector<geometry_msgs::msg::PoseStamped> path;

        // This would involve copying the path data back from GPU
        // and reconstructing the path using the parent pointers
        // Implementation would depend on the specific GPU path planning algorithm

        return path; // Placeholder
    }

    // GPU kernel for A* path planning
    void execute_astar_gpu_kernel(
        const unsigned char* costmap,
        int* path_parent,
        int width, int height,
        unsigned int start_x, unsigned int start_y,
        unsigned int goal_x, unsigned int goal_y
    );

    rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
    std::string name_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    nav2_costmap_2d::Costmap2D* costmap_;

    unsigned char* d_costmap_;
    int* d_path_;
    cudaStream_t planning_stream_;
};
```

### GPU-Accelerated Local Controller

#### Isaac Local Controller
```cpp
#include <nav2_core/controller.hpp>
#include <nav2_costmap_2d/costmap_2d_ros.h>
#include <nav2_util/lifecycle_node.hpp>
#include <nav2_msgs/msg/velocity_smoother_status.hpp>

class IsaacGPULocalController : public nav2_core::Controller {
public:
    IsaacGPULocalController() = default;
    ~IsaacGPULocalController() override = default;

    void configure(
        const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent,
        std::string name,
        std::shared_ptr<tf2_ros::Buffer> tf,
        std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override {

        node_ = parent.lock();
        name_ = name;
        tf_ = tf;
        costmap_ros_ = costmap_ros;
        costmap_ = costmap_ros_->getCostmap();

        // Initialize GPU context
        cudaSetDevice(0);
        cudaFree(0);

        // Initialize trajectory optimization on GPU
        initialize_gpu_trajectory_optimization();
    }

    void cleanup() override {
        cleanup_gpu_trajectory_optimization();
    }

    void activate() override {}
    void deactivate() override {}

    geometry_msgs::msg::TwistStamped computeVelocityCommands(
        const geometry_msgs::msg::PoseStamped& pose,
        const geometry_msgs::msg::Twist& velocity,
        nav2_core::GoalChecker* goal_checker) override {

        geometry_msgs::msg::TwistStamped cmd_vel;
        cmd_vel.header.stamp = node_->now();
        cmd_vel.header.frame_id = "base_link";

        // Get current path from global planner
        if (current_path_.poses.empty()) {
            cmd_vel.twist.linear.x = 0.0;
            cmd_vel.twist.angular.z = 0.0;
            return cmd_vel;
        }

        // Perform GPU-accelerated trajectory optimization
        cmd_vel.twist = optimize_trajectory_gpu(pose, velocity);

        return cmd_vel;
    }

private:
    void initialize_gpu_trajectory_optimization() {
        // Allocate GPU memory for trajectory optimization
        cudaMalloc(&d_trajectory_, max_trajectory_points_ * sizeof(TrajectoryPoint));
        cudaMalloc(&d_obstacles_, max_obstacles_ * sizeof(Obstacle));
        cudaStreamCreate(&control_stream_);
    }

    void cleanup_gpu_trajectory_optimization() {
        if (d_trajectory_) cudaFree(d_trajectory_);
        if (d_obstacles_) cudaFree(d_obstacles_);
        if (control_stream_) cudaStreamDestroy(control_stream_);
    }

    geometry_msgs::msg::Twist optimize_trajectory_gpu(
        const geometry_msgs::msg::PoseStamped& pose,
        const geometry_msgs::msg::Twist& velocity) {

        geometry_msgs::msg::Twist cmd_vel;

        // Update obstacle data on GPU
        update_obstacles_gpu();

        // Perform trajectory optimization on GPU
        TrajectoryPoint optimal_point = optimize_dwb_gpu(pose, velocity);

        // Convert to velocity command
        cmd_vel.linear.x = optimal_point.linear_vel;
        cmd_vel.angular.z = optimal_point.angular_vel;

        return cmd_vel;
    }

    void update_obstacles_gpu() {
        // Get current obstacles from costmap
        // Copy to GPU memory for trajectory optimization
    }

    TrajectoryPoint optimize_dwb_gpu(
        const geometry_msgs::msg::PoseStamped& pose,
        const geometry_msgs::msg::Twist& velocity) {

        // Launch Dynamic Window Approach on GPU
        dim3 block_size(256);
        dim3 grid_size((num_velocity_samples_ + block_size.x - 1) / block_size.x);

        // Evaluate all possible velocities on GPU
        evaluate_velocities_gpu_kernel<<<grid_size, block_size, 0, control_stream_>>>(
            d_trajectory_, d_obstacles_, num_obstacles_,
            pose.pose.position.x, pose.pose.position.y, pose.pose.orientation.z,
            velocity.linear.x, velocity.angular.z
        );

        // Find optimal velocity
        return find_optimal_velocity_gpu();
    }

    struct TrajectoryPoint {
        float x, y, theta;
        float linear_vel, angular_vel;
        float cost;
    };

    struct Obstacle {
        float x, y, radius;
    };

    rclcpp_lifecycle::LifecycleNode::SharedPtr node_;
    std::string name_;
    std::shared_ptr<tf2_ros::Buffer> tf_;
    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
    nav2_costmap_2d::Costmap2D* costmap_;

    TrajectoryPoint* d_trajectory_;
    Obstacle* d_obstacles_;
    cudaStream_t control_stream_;

    int max_trajectory_points_ = 1000;
    int max_obstacles_ = 100;
    int num_velocity_samples_ = 1000;
};
```

## Isaac Nav2 Launch Configuration

### Launch Files and Configuration

#### Isaac Nav2 Launch File
```python
# isaac_nav2_bringup/launch/isaac_nav2.launch.py
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Get the launch directory
    bringup_dir = get_package_share_directory('isaac_nav2_bringup')

    # Create launch configuration variables
    namespace = LaunchConfiguration('namespace')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    params_file = LaunchConfiguration('params_file')
    default_nav2_params = os.path.join(bringup_dir, 'params', 'isaac_nav2_params.yaml')
    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites={},
        convert_types=True)

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'namespace', default_value='',
            description='Top-level namespace'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='Use simulation (Gazebo) clock if true'),
        DeclareLaunchArgument(
            'autostart', default_value='true',
            description='Automatically startup the nav2 stack'),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_nav2_params,
            description='Full path to the ROS2 parameters file to use'),

        # Isaac GPU Navigation Planner Server
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            namespace=namespace,
            parameters=[configured_params],
            remappings=[('cmd_vel', 'cmd_vel_nav')],
            output='screen'),

        # Isaac GPU Local Controller Server
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            namespace=namespace,
            parameters=[configured_params],
            remappings=[('cmd_vel', 'cmd_vel_nav'),
                       ('odom', 'odom')],
            output='screen'),

        # Isaac GPU Navigator Server
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            namespace=namespace,
            parameters=[configured_params],
            remappings=[('cmd_vel', 'cmd_vel_nav')],
            output='screen'),

        # Isaac GPU Velocity Smoother
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            name='velocity_smoother',
            namespace=namespace,
            parameters=[configured_params],
            remappings=[('cmd_vel', 'cmd_vel_nav'),
                       ('cmd_vel_smoothed', 'cmd_vel')],
            output='screen'),

        # Isaac GPU Lifecycle Manager
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            namespace=namespace,
            parameters=[{'use_sim_time': use_sim_time},
                       {'autostart': autostart},
                       {'node_names': ['planner_server',
                                     'controller_server',
                                     'bt_navigator',
                                     'velocity_smoother']}],
            output='screen'),
    ])
```

### Isaac-Optimized Parameters

#### GPU-Accelerated Navigation Parameters
```yaml
# isaac_nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: true
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    laser_likelihood_max_dist: 2.0
    laser_max_range: 100.0
    laser_min_range: -1.0
    laser_model_type: "likelihood_field"
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.5
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.1
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

bt_navigator:
  ros__parameters:
    use_sim_time: true
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "odom"
    default_bt_xml_filename: "navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
      - "nav2_compute_path_to_pose_action_bt_node"
      - "nav2_follow_path_action_bt_node"
      - "nav2_back_up_action_bt_node"
      - "nav2_spin_action_bt_node"
      - "nav2_wait_action_bt_node"
      - "nav2_clear_costmap_service_bt_node"
      - "nav2_is_stuck_condition_bt_node"
      - "nav2_goal_reached_condition_bt_node"
      - "nav2_goal_updated_condition_bt_node"
      - "nav2_initial_pose_received_condition_bt_node"
      - "nav2_reinitialize_global_localization_service_bt_node"
      - "nav2_rate_controller_bt_node"
      - "nav2_distance_controller_bt_node"
      - "nav2_speed_controller_bt_node"
      - "nav2_truncate_path_action_bt_node"
      - "nav2_goal_updater_node_bt_node"
      - "nav2_recovery_node_bt_node"
      - "nav2_pipeline_sequence_bt_node"
      - "nav2_round_robin_node_bt_node"
      - "nav2_transform_available_condition_bt_node"
      - "nav2_time_expired_condition_bt_node"
      - "nav2_path_expiring_timer_condition"
      - "nav2_distance_traveled_condition_bt_node"
      - "nav2_single_trigger_bt_node"
      - "nav2_is_battery_low_condition_bt_node"
      - "nav2_navigate_through_poses_action_bt_node"
      - "nav2_navigate_to_pose_action_bt_node"
      - "nav2_remove_passed_goals_action_bt_node"
      - "nav2_planner_selector_bt_node"
      - "nav2_controller_selector_bt_node"
      - "nav2_goal_checker_selector_bt_node"

controller_server:
  ros__parameters:
    use_sim_time: true
    controller_frequency: 50.0  # Higher frequency for GPU acceleration
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.001
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Progress checker parameters
    progress_checker:
      plugin: "nav2_controller::SimpleProgressChecker"
      required_movement_radius: 0.5
      movement_time_allowance: 30.0

    # Goal checker parameters
    goal_checker:
      plugin: "nav2_controller::SimpleGoalChecker"
      xy_goal_tolerance: 0.25
      yaw_goal_tolerance: 0.25
      stateful: True

    # Isaac GPU-enhanced controller
    FollowPath:
      plugin: "isaac_nav2_controller::GPUPathFollower"
      max_linear_speed: 1.0
      min_linear_speed: 0.1
      max_angular_speed: 1.0
      min_angular_speed: 0.1
      speed_scaling_factor: 1.0
      max_acceleration: 2.0
      max_deceleration: 2.0
      goal_dist_tolerance: 0.25
      goal_yaw_tolerance: 0.25
      xy_goal_tolerance: 0.25
      trans_stopped_velocity: 0.25
      short_circuit_trajectory: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 20.0
      publish_frequency: 10.0
      global_frame: "odom"
      robot_base_frame: "base_link"
      use_sim_time: true
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.3
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: "/scan"
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      always_send_full_costmap: True

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: "map"
      robot_base_frame: "base_link"
      use_sim_time: true
      robot_radius: 0.3
      resolution: 0.05
      track_unknown_space: true
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: "/scan"
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      always_send_full_costmap: True

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: true
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "isaac_nav2_planner::GPUAStarPlanner"  # Isaac GPU planner
      tolerance: 0.5
      use_astar: true
      allow_unknown: true
      max_iterations: 1000000  # Increased for GPU capability
      max_on_approach_iterations: 1000
      max_planning_retries: 5
      smooth_path: true

smoother_server:
  ros__parameters:
    use_sim_time: true
    smoother_plugins: ["simple_smoother"]
    simple_smoother:
      plugin: "nav2_smoother::SimpleSmoother"
      tolerance: 1.0e-10
      max_its: 1000
      do_refinement: True

behavior_server:
  ros__parameters:
    costmap_topic: "local_costmap/costmap_raw"
    footprint_topic: "local_costmap/published_footprint"
    cycle_frequency: 10.0
    behavior_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_behaviors::Spin"
      spin_dist: 1.57
    backup:
      plugin: "nav2_behaviors::BackUp"
      backup_dist: 0.15
      backup_speed: 0.025
    wait:
      plugin: "nav2_behaviors::Wait"
      wait_duration: 1.0
```

## Isaac Nav2 Behavior Trees

### GPU-Enhanced Behavior Trees

#### Custom Behavior Tree for GPU Navigation
```xml
<!-- navigate_w_gpu_replanning_and_recovery.xml -->
<root main_tree_to_execute="MainTree">
    <BehaviorTree ID="MainTree">
        <RecoveryNode number_of_retries="6" name="NavigateRecovery">
            <PipelineSequence name="NavigateWithReplanning">
                <RateController hz="1.0">
                    <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
                </RateController>
                <RecoveryNode number_of_retries="1" name="FollowPathRecovery">
                    <FollowPath path="{path}" controller_id="FollowPath"/>
                    <ReactiveFallback name="FollowPathFallback">
                        <GoalReached goal="{goal}"/>
                        <ClearEntireCostmap name="ClearLocalCostmap-Context" service_name="local_costmap/clear_entirely_local_costmap"/>
                    </ReactiveFallback>
                </RecoveryNode>
            </PipelineSequence>
            <ReactiveFallback name="RecoveryFallback">
                <GoalReached goal="{goal}"/>
                <ClearEntireCostmap name="ClearGlobalCostmap-Context" service_name="global_costmap/clear_entirely_global_costmap"/>
                <ClearEntireCostmap name="ClearLocalCostmap-Context" service_name="local_costmap/clear_entirely_local_costmap"/>
            </ReactiveFallback>
        </RecoveryNode>
    </BehaviorTree>
</root>
```

### Isaac-Specific Behavior Tree Nodes

#### GPU-Accelerated Behavior Tree Nodes
```cpp
#include <behaviortree_cpp_v3/action_node.h>
#include <nav2_behavior_tree/bt_service_node.h>
#include <nav2_msgs/srv/compute_path_to_pose.hpp>

// GPU-accelerated path computation node
class ComputeGPUPathToPose : public nav2_behavior_tree::BtServiceNode<nav2_msgs::srv::ComputePathToPose> {
public:
    ComputeGPUPathToPose(
        const std::string & service_node_name,
        const BT::NodeConfiguration & conf)
    : nav2_behavior_tree::BtServiceNode<nav2_msgs::srv::ComputePathToPose>(
        service_node_name, conf) {

        // Initialize GPU context for path computation
        cudaSetDevice(0);
        cudaFree(0);
    }

    void on_tick() override {
        // Get goal from blackboard
        geometry_msgs::msg::PoseStamped goal;
        if (!getInput("goal", goal)) {
            RCLCPP_ERROR(node_->get_logger(), "Failed to get goal from blackboard");
            return;
        }

        // Set service request
        request_.start = start_;
        request_.goal = goal;
        request_.planner_id = get_planner_id();

        // GPU-accelerated path computation can be triggered here
        RCLCPP_INFO(node_->get_logger(), "Computing GPU-accelerated path to (%.2f, %.2f)",
                   goal.pose.position.x, goal.pose.position.y);
    }

    BT::NodeStatus on_success() override {
        // Process GPU-accelerated path result
        if (result_.result->path.poses.empty()) {
            RCLCPP_WARN(node_->get_logger(), "GPU path planner returned empty path");
            return BT::NodeStatus::FAILURE;
        }

        // Set path to blackboard
        setOutput("path", result_.result->path);
        RCLCPP_INFO(node_->get_logger(), "GPU path computed with %zu poses",
                   result_.result->path.poses.size());

        return BT::NodeStatus::SUCCESS;
    }

private:
    geometry_msgs::msg::PoseStamped start_;
};
```

## Performance Optimization

### GPU Resource Management

#### Isaac Nav2 GPU Resource Manager
```cpp
class IsaacNav2GPUManager {
public:
    static IsaacNav2GPUManager& get_instance() {
        static IsaacNav2GPUManager instance;
        return instance;
    }

    void initialize_gpu_resources() {
        // Set GPU device
        cudaSetDevice(0);
        cudaFree(0); // Initialize context

        // Get GPU properties
        cudaGetDeviceProperties(&device_prop_, 0);

        RCLCPP_INFO(rclcpp::get_logger("isaac_nav2_gpu_manager"),
                   "Using GPU: %s with %d SMs, %zu memory",
                   device_prop_.name, device_prop_.multiProcessorCount,
                   device_prop_.totalGlobalMem);

        // Initialize streams for different navigation components
        cudaStreamCreate(&planning_stream_);
        cudaStreamCreate(&control_stream_);
        cudaStreamCreate(&smoothing_stream_);
    }

    cudaStream_t get_planning_stream() { return planning_stream_; }
    cudaStream_t get_control_stream() { return control_stream_; }
    cudaStream_t get_smoothing_stream() { return smoothing_stream_; }

    int get_max_threads_per_block() { return device_prop_.maxThreadsPerBlock; }
    int get_max_shared_memory() { return device_prop_.sharedMemPerBlock; }

private:
    IsaacNav2GPUManager() = default;
    ~IsaacNav2GPUManager() {
        if (planning_stream_) cudaStreamDestroy(planning_stream_);
        if (control_stream_) cudaStreamDestroy(control_stream_);
        if (smoothing_stream_) cudaStreamDestroy(smoothing_stream_);
    }

    cudaDeviceProp device_prop_;
    cudaStream_t planning_stream_;
    cudaStream_t control_stream_;
    cudaStream_t smoothing_stream_;
};
```

The integration of Isaac with Nav2 provides GPU acceleration to the standard ROS 2 navigation stack, significantly improving performance for complex navigation tasks. This combination leverages the reliability and flexibility of Nav2 while adding the computational power of GPU acceleration for humanoid robots and other demanding applications.