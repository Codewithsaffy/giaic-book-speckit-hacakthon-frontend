---
sidebar_position: 4
title: "Obstacle Avoidance for Humanoid Robots"
description: "GPU-accelerated obstacle detection and avoidance for humanoid navigation"
---

# Obstacle Avoidance for Humanoid Robots

Obstacle avoidance for humanoid robots presents unique challenges compared to wheeled robots. Humanoid robots must not only avoid obstacles but also consider their bipedal locomotion, balance requirements, and the ability to step over or around obstacles in 3D space. Isaac provides GPU-accelerated obstacle avoidance capabilities that enable real-time processing of complex sensor data for safe humanoid navigation.

## Humanoid-Specific Obstacle Challenges

### 3D Navigation Considerations

Humanoid robots navigate in 3D space with unique capabilities:

#### Step-Over Capabilities
- **Step Height Limits**: Maximum obstacle height that can be stepped over
- **Step Length Planning**: Planning step length to clear obstacles
- **Balance During Stepping**: Maintaining balance while stepping over obstacles
- **Foot Placement**: Precise foot placement for obstacle clearance

#### Dynamic Obstacle Considerations
- **Moving Obstacles**: Humans and other moving objects
- **Predictive Avoidance**: Predicting future positions of moving obstacles
- **Social Navigation**: Following social norms around humans
- **Group Navigation**: Navigating through groups of people

### Balance-Aware Obstacle Avoidance

#### Stability Constraints
- **ZMP Maintenance**: Ensuring Zero Moment Point stays within support polygon
- **CoM Trajectory**: Adjusting Center of Mass trajectory for obstacle avoidance
- **Step Timing**: Adjusting step timing to accommodate obstacle avoidance
- **Recovery Planning**: Planning for potential balance recovery during avoidance

## Isaac Obstacle Detection

### GPU-Accelerated Sensor Processing

#### LiDAR Processing
```cpp
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

class GPULiDARDetector {
public:
    GPULiDARDetector(int max_points = 100000)
        : max_points_(max_points) {

        // Allocate GPU memory for LiDAR processing
        cudaMalloc(&d_ranges_, max_points_ * sizeof(float));
        cudaMalloc(&d_angles_, max_points_ * sizeof(float));
        cudaMalloc(&d_cartesian_, max_points_ * sizeof(float3));
        cudaMalloc(&d_obstacles_, max_points_ * sizeof(ObstaclePoint));
        cudaMalloc(&d_clusters_, max_clusters_ * sizeof(ObstacleCluster));

        // Initialize CUDA streams
        cudaStreamCreate(&detection_stream_);
    }

    void process_lidar_scan(const std::vector<float>& ranges,
                           const std::vector<float>& angles,
                           std::vector<ObstacleCluster>& obstacles) {
        // Copy LiDAR data to GPU
        cudaMemcpyAsync(d_ranges_, ranges.data(), ranges.size() * sizeof(float),
                       cudaMemcpyHostToDevice, detection_stream_);
        cudaMemcpyAsync(d_angles_, angles.data(), angles.size() * sizeof(float),
                       cudaMemcpyHostToDevice, detection_stream_);

        // Convert polar to Cartesian coordinates
        convert_polar_to_cartesian_gpu(ranges.size());

        // Detect obstacles on GPU
        detect_obstacles_gpu(ranges.size());

        // Cluster obstacles on GPU
        cluster_obstacles_gpu();

        // Copy results back to CPU
        int num_clusters = get_num_clusters_gpu();
        std::vector<ObstacleCluster> gpu_clusters(num_clusters);
        cudaMemcpyAsync(gpu_clusters.data(), d_clusters_,
                       num_clusters * sizeof(ObstacleCluster),
                       cudaMemcpyDeviceToHost, detection_stream_);
        cudaStreamSynchronize(detection_stream_);

        obstacles = gpu_clusters;
    }

private:
    void convert_polar_to_cartesian_gpu(int num_points) {
        dim3 block_size(256);
        dim3 grid_size((num_points + block_size.x - 1) / block_size.x);

        polar_to_cartesian_kernel<<<grid_size, block_size, 0, detection_stream_>>>(
            d_ranges_, d_angles_, d_cartesian_, num_points
        );
    }

    void detect_obstacles_gpu(int num_points) {
        dim3 block_size(256);
        dim3 grid_size((num_points + block_size.x - 1) / block_size.x);

        detect_obstacles_kernel<<<grid_size, block_size, 0, detection_stream_>>>(
            d_cartesian_, d_obstacles_, num_points, min_obstacle_range_, max_obstacle_range_
        );
    }

    void cluster_obstacles_gpu() {
        // Launch clustering algorithm on GPU (e.g., DBSCAN)
        dim3 block_size(256);
        dim3 grid_size((max_points_ + block_size.x - 1) / block_size.x);

        cluster_obstacles_kernel<<<grid_size, block_size, 0, detection_stream_>>>(
            d_obstacles_, d_clusters_, max_points_, cluster_distance_
        );
    }

    int get_num_clusters_gpu() {
        // This would involve atomic operations or other methods to count clusters
        return 0; // Simplified
    }

    float* d_ranges_;
    float* d_angles_;
    float3* d_cartesian_;
    ObstaclePoint* d_obstacles_;
    ObstacleCluster* d_clusters_;
    cudaStream_t detection_stream_;

    int max_points_;
    int max_clusters_ = 1000;
    float min_obstacle_range_ = 0.1f;
    float max_obstacle_range_ = 10.0f;
    float cluster_distance_ = 0.3f;
};

// CUDA kernel for polar to Cartesian conversion
__global__ void polar_to_cartesian_kernel(
    const float* ranges,
    const float* angles,
    float3* cartesian,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        float range = ranges[idx];
        float angle = angles[idx];

        cartesian[idx].x = range * cosf(angle);
        cartesian[idx].y = range * sinf(angle);
        cartesian[idx].z = 0.0f; // Assuming 2D LiDAR
    }
}

// CUDA kernel for obstacle detection
__global__ void detect_obstacles_kernel(
    const float3* points,
    ObstaclePoint* obstacles,
    int num_points,
    float min_range,
    float max_range
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        float3 pt = points[idx];

        // Check if point is within obstacle range
        float distance = sqrtf(pt.x * pt.x + pt.y * pt.y);
        if (distance >= min_range && distance <= max_range) {
            obstacles[idx].position = pt;
            obstacles[idx].is_obstacle = true;
            obstacles[idx].confidence = 1.0f; // For detected points
        } else {
            obstacles[idx].is_obstacle = false;
        }
    }
}

// CUDA kernel for obstacle clustering
__global__ void cluster_obstacles_kernel(
    const ObstaclePoint* points,
    ObstacleCluster* clusters,
    int num_points,
    float cluster_distance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points && points[idx].is_obstacle) {
        // Simple clustering - find nearest neighbor and assign to same cluster
        // In practice, this would implement a proper clustering algorithm like DBSCAN
        int cluster_id = idx; // Simplified - each point is its own cluster initially

        // Find nearby points and merge clusters (simplified)
        for (int j = 0; j < num_points; j++) {
            if (j != idx && points[j].is_obstacle) {
                float3 diff = make_float3(
                    points[idx].position.x - points[j].position.x,
                    points[idx].position.y - points[j].position.y,
                    points[idx].position.z - points[j].position.z
                );
                float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

                if (dist < cluster_distance) {
                    // Merge clusters (simplified approach)
                    cluster_id = min(cluster_id, j);
                }
            }
        }

        // Store cluster assignment
        // This is a simplified implementation - full clustering would be more complex
    }
}

struct ObstaclePoint {
    float3 position;
    bool is_obstacle;
    float confidence;
};

struct ObstacleCluster {
    float3 center;
    float3 dimensions;
    int num_points;
    float min_height;
    float max_height;
    ObstacleType type; // static, dynamic, human, etc.
};

enum class ObstacleType {
    STATIC,
    DYNAMIC,
    HUMAN,
    UNKNOWN
};
```

### Vision-Based Obstacle Detection

#### GPU-Accelerated Vision Processing
```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <NvInfer.h>

class GPUVisionObstacleDetector {
public:
    GPUVisionObstacleDetector(const std::string& model_path) {
        // Initialize TensorRT engine for object detection
        initialize_tensorrt(model_path);

        // Initialize CUDA streams
        cudaStreamCreate(&vision_stream_);
    }

    void detect_obstacles_vision(const cv::Mat& image,
                                std::vector<DetectedObstacle>& obstacles) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_image;
        gpu_image.upload(image, vision_stream_);

        // Convert to RGB if needed
        cv::cuda::GpuMat gpu_rgb;
        if (image.channels() == 3) {
            cv::cuda::cvtColor(gpu_image, gpu_rgb, cv::COLOR_BGR2RGB, 0, vision_stream_);
        } else {
            gpu_rgb = gpu_image;
        }

        // Preprocess image for neural network
        cv::cuda::GpuMat gpu_input = preprocess_image_gpu(gpu_rgb);

        // Perform inference
        std::vector<Detection> detections = perform_inference_gpu(gpu_input);

        // Post-process detections to obstacles
        obstacles = detections_to_obstacles(detections, image.size());
    }

private:
    void initialize_tensorrt(const std::string& model_path) {
        // Load TensorRT engine
        std::ifstream engine_file(model_path, std::ios::binary);
        std::vector<char> engine_data((std::istreambuf_iterator<char>(engine_file)),
                                      std::istreambuf_iterator<char>());

        runtime_ = nvinfer1::createInferRuntime(logger_);
        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
        context_ = engine_->createExecutionContext();

        // Allocate I/O buffers
        allocate_io_buffers();
    }

    cv::cuda::GpuMat preprocess_image_gpu(const cv::cuda::GpuMat& image) {
        cv::cuda::GpuMat resized, normalized;

        // Resize to model input size
        cv::cuda::resize(image, resized, cv::Size(input_width_, input_height_));

        // Normalize pixel values
        resized.convertTo(normalized, CV_32F, 1.0/255.0);

        return normalized;
    }

    std::vector<Detection> perform_inference_gpu(const cv::cuda::GpuMat& input) {
        // Copy input to GPU buffer
        cv::Mat input_cpu;
        input.download(input_cpu);
        cudaMemcpyAsync(d_input_, input_cpu.data, input_size_,
                       cudaMemcpyHostToDevice, vision_stream_);

        // Perform inference
        void* bindings[] = {d_input_, d_output_};
        context_->executeV2(bindings);

        // Copy output back
        cudaMemcpyAsync(h_output_, d_output_, output_size_,
                       cudaMemcpyDeviceToHost, vision_stream_);
        cudaStreamSynchronize(vision_stream_);

        // Parse detection results
        return parse_detections();
    }

    std::vector<DetectedObstacle> detections_to_obstacles(
        const std::vector<Detection>& detections,
        cv::Size image_size) {

        std::vector<DetectedObstacle> obstacles;
        for (const auto& detection : detections) {
            if (detection.confidence > confidence_threshold_) {
                DetectedObstacle obstacle;
                obstacle.type = detection.class_id;
                obstacle.confidence = detection.confidence;

                // Convert normalized coordinates to image coordinates
                obstacle.bbox.x = detection.bbox.x * image_size.width;
                obstacle.bbox.y = detection.bbox.y * image_size.height;
                obstacle.bbox.width = detection.bbox.width * image_size.width;
                obstacle.bbox.height = detection.bbox.height * image_size.height;

                obstacles.push_back(obstacle);
            }
        }

        return obstacles;
    }

    void allocate_io_buffers() {
        // Get binding information and allocate buffers
        int num_bindings = engine_->getNbBindings();
        for (int i = 0; i < num_bindings; i++) {
            auto dims = engine_->getBindingDimensions(i);
            size_t size = 1;
            for (int j = 0; j < dims.nbDims; j++) {
                size *= dims.d[j];
            }

            if (engine_->bindingIsInput(i)) {
                input_size_ = size * sizeof(float);
                cudaMalloc(&d_input_, input_size_);
                h_input_ = malloc(input_size_);
            } else {
                output_size_ = size * sizeof(float);
                cudaMalloc(&d_output_, output_size_);
                h_output_ = malloc(output_size_);
            }
        }
    }

    std::vector<Detection> parse_detections() {
        // Parse the TensorRT output into Detection structures
        // Implementation depends on the specific model output format
        return std::vector<Detection>();
    }

    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    Logger logger_;

    void* d_input_;
    void* d_output_;
    void* h_input_;
    void* h_output_;
    size_t input_size_;
    size_t output_size_;

    int input_width_ = 640;
    int input_height_ = 640;
    float confidence_threshold_ = 0.5f;

    cudaStream_t vision_stream_;
};

struct Detection {
    int class_id;
    float confidence;
    cv::Rect2f bbox;  // Normalized coordinates (0-1)
    float* features;  // Optional feature vector
};

struct DetectedObstacle {
    ObstacleType type;
    float confidence;
    cv::Rect2f bbox;  // Bounding box in image coordinates
    cv::Point2f center;  // Center of obstacle in world coordinates
    float height;    // Estimated height of obstacle
    bool is_dynamic; // Whether obstacle is moving
};
```

## GPU-Accelerated Path Planning with Obstacles

### Dynamic Path Planning

#### Real-time Replanning
```cpp
class GPUObstacleAvoidancePlanner {
public:
    GPUObstacleAvoidancePlanner(float robot_radius = 0.3f)
        : robot_radius_(robot_radius) {

        // Allocate GPU memory for costmap and path planning
        cudaMalloc(&d_costmap_, costmap_width_ * costmap_height_ * sizeof(unsigned char));
        cudaMalloc(&d_dynamic_obstacles_, max_dynamic_obstacles_ * sizeof(DynamicObstacle));
        cudaMalloc(&d_path_, max_path_points_ * sizeof(float2));
        cudaMalloc(&d_temp_path_, max_path_points_ * sizeof(float2));

        // Initialize path planning algorithms
        initialize_path_planners();
    }

    bool plan_path_around_obstacles(
        const float2& start,
        const float2& goal,
        const std::vector<ObstacleCluster>& static_obstacles,
        const std::vector<DynamicObstacle>& dynamic_obstacles,
        std::vector<float2>& path) {

        // Update costmap with obstacles on GPU
        update_costmap_gpu(static_obstacles, dynamic_obstacles);

        // Plan path using GPU-accelerated algorithm
        bool success = plan_path_gpu(start, goal);

        if (success) {
            // Extract path from GPU memory
            extract_path_gpu(path);
        }

        return success;
    }

    bool replan_path_around_dynamic_obstacles(
        const std::vector<DynamicObstacle>& new_obstacles,
        const std::vector<float2>& current_path,
        std::vector<float2>& replanned_path) {

        // Update dynamic obstacles on GPU
        update_dynamic_obstacles_gpu(new_obstacles);

        // Check current path for collisions
        bool path_blocked = check_path_collision_gpu(current_path);

        if (path_blocked) {
            // Find first safe point in current path
            float2 new_start = find_safe_replan_point_gpu(current_path);

            // Replan from safe point to goal
            float2 goal = current_path.back();
            bool success = plan_path_gpu(new_start, goal);

            if (success) {
                extract_path_gpu(replanned_path);
                return true;
            }
        }

        // Current path is still valid
        replanned_path = current_path;
        return true;
    }

private:
    void update_costmap_gpu(const std::vector<ObstacleCluster>& static_obstacles,
                           const std::vector<DynamicObstacle>& dynamic_obstacles) {
        // Clear costmap
        cudaMemsetAsync(d_costmap_, 0, costmap_width_ * costmap_height_ * sizeof(unsigned char),
                       planning_stream_);

        // Add static obstacles to costmap
        for (const auto& obstacle : static_obstacles) {
            add_obstacle_to_costmap_gpu(obstacle, static_cost_);
        }

        // Add dynamic obstacles to costmap
        cudaMemcpyAsync(d_dynamic_obstacles_, dynamic_obstacles.data(),
                       dynamic_obstacles.size() * sizeof(DynamicObstacle),
                       cudaMemcpyHostToDevice, planning_stream_);

        for (size_t i = 0; i < dynamic_obstacles.size(); i++) {
            add_dynamic_obstacle_to_costmap_gpu(i, dynamic_cost_);
        }
    }

    void add_obstacle_to_costmap_gpu(const ObstacleCluster& obstacle, unsigned char cost) {
        dim3 block_size(16, 16);
        dim3 grid_size((obstacle.dimensions.x / costmap_resolution_ + block_size.x - 1) / block_size.x,
                      (obstacle.dimensions.y / costmap_resolution_ + block_size.y - 1) / block_size.y);

        add_obstacle_to_costmap_kernel<<<grid_size, block_size, 0, planning_stream_>>>(
            d_costmap_, obstacle.center, obstacle.dimensions, cost,
            costmap_width_, costmap_height_, costmap_resolution_
        );
    }

    bool plan_path_gpu(const float2& start, const float2& goal) {
        // Launch GPU path planning algorithm (A* or Dijkstra)
        dim3 block_size(256);
        dim3 grid_size((costmap_width_ * costmap_height_ + block_size.x - 1) / block_size.x);

        // Initialize path planning
        initialize_path_planning_kernel<<<1, 1, 0, planning_stream_>>>(
            d_path_, start, goal, costmap_width_, costmap_height_
        );

        // Execute path planning iterations
        for (int iter = 0; iter < max_iterations_; iter++) {
            execute_path_planning_iteration_kernel<<<grid_size, block_size, 0, planning_stream_>>>(
                d_costmap_, d_path_, iter, costmap_width_, costmap_height_
            );

            // Check if goal is reached (simplified)
            if (is_goal_reached_gpu(goal)) {
                return true;
            }
        }

        return false; // Planning failed
    }

    bool check_path_collision_gpu(const std::vector<float2>& path) {
        // Check if any point in path collides with obstacles
        int num_points = path.size();
        float2* d_path_points;
        cudaMalloc(&d_path_points, num_points * sizeof(float2));
        cudaMemcpyAsync(d_path_points, path.data(), num_points * sizeof(float2),
                       cudaMemcpyHostToDevice, planning_stream_);

        bool* d_collision;
        cudaMalloc(&d_collision, sizeof(bool));
        cudaMemsetAsync(d_collision, false, sizeof(bool), planning_stream_);

        dim3 block_size(256);
        dim3 grid_size((num_points + block_size.x - 1) / block_size.x);

        check_path_collision_kernel<<<grid_size, block_size, 0, planning_stream_>>>(
            d_path_points, d_collision, d_costmap_,
            num_points, costmap_resolution_, robot_radius_
        );

        bool collision;
        cudaMemcpyAsync(&collision, d_collision, sizeof(bool),
                       cudaMemcpyDeviceToHost, planning_stream_);
        cudaStreamSynchronize(planning_stream_);

        cudaFree(d_path_points);
        cudaFree(d_collision);

        return collision;
    }

    void initialize_path_planners() {
        // Initialize different path planning algorithms on GPU
        // Could include A*, Dijkstra, RRT, etc.
    }

    bool is_goal_reached_gpu(const float2& goal) {
        // Check if goal has been reached in path planning
        // This would involve checking GPU memory for goal state
        return false; // Simplified
    }

    void extract_path_gpu(std::vector<float2>& path) {
        // Extract planned path from GPU memory
        // Implementation would copy path from GPU and reconstruct it
    }

    unsigned char* d_costmap_;
    DynamicObstacle* d_dynamic_obstacles_;
    float2* d_path_;
    float2* d_temp_path_;
    cudaStream_t planning_stream_;

    int costmap_width_ = 200;
    int costmap_height_ = 200;
    float costmap_resolution_ = 0.05f;
    int max_dynamic_obstacles_ = 100;
    int max_path_points_ = 1000;
    int max_iterations_ = 10000;
    unsigned char static_cost_ = 254;
    unsigned char dynamic_cost_ = 200;
    float robot_radius_;
};

// CUDA kernel for adding obstacle to costmap
__global__ void add_obstacle_to_costmap_kernel(
    unsigned char* costmap,
    float2 obstacle_center,
    float2 obstacle_size,
    unsigned char cost,
    int width, int height,
    float resolution
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float world_x = x * resolution;
        float world_y = y * resolution;

        // Check if cell is within obstacle bounds
        float half_size_x = obstacle_size.x * 0.5f;
        float half_size_y = obstacle_size.y * 0.5f;

        if (fabsf(world_x - obstacle_center.x) <= half_size_x &&
            fabsf(world_y - obstacle_center.y) <= half_size_y) {
            int idx = y * width + x;
            costmap[idx] = max(costmap[idx], cost);
        }
    }
}

// CUDA kernel for checking path collision
__global__ void check_path_collision_kernel(
    const float2* path,
    bool* collision,
    const unsigned char* costmap,
    int num_points,
    float resolution,
    float robot_radius
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        float2 point = path[idx];

        // Convert world coordinates to costmap indices
        int map_x = (int)(point.x / resolution);
        int map_y = (int)(point.y / resolution);

        // Check costmap cell and surrounding area for obstacles
        int robot_cells = (int)(robot_radius / resolution);
        bool local_collision = false;

        for (int dy = -robot_cells; dy <= robot_cells && !local_collision; dy++) {
            for (int dx = -robot_cells; dx <= robot_cells && !local_collision; dx++) {
                int check_x = map_x + dx;
                int check_y = map_y + dy;

                if (check_x >= 0 && check_x < 200 && check_y >= 0 && check_y < 200) {
                    int costmap_idx = check_y * 200 + check_x;
                    if (costmap[costmap_idx] > 200) { // Threshold for obstacle
                        local_collision = true;
                    }
                }
            }
        }

        if (local_collision) {
            *collision = true;
        }
    }
}
```

## Humanoid-Aware Obstacle Avoidance

### Bipedal Navigation Constraints

#### Step-Aware Path Planning
```cpp
class BipedalObstacleAvoider {
public:
    BipedalObstacleAvoider(float max_step_length = 0.3f, float max_step_width = 0.2f)
        : max_step_length_(max_step_length), max_step_width_(max_step_width) {

        // Initialize GPU memory for bipedal-specific planning
        cudaMalloc(&d_footstep_candidates_, max_candidates_ * sizeof(FootstepCandidate));
        cudaMalloc(&d_balance_constraints_, max_constraints_ * sizeof(BalanceConstraint));
    }

    bool plan_path_with_step_constraints(
        const float2& start,
        const float2& goal,
        const std::vector<ObstacleCluster>& obstacles,
        std::vector<Footstep>& footsteps) {

        // Generate footstep candidates that avoid obstacles
        generate_safe_footsteps_gpu(start, goal, obstacles);

        // Evaluate candidates based on balance constraints
        evaluate_footsteps_balance_gpu();

        // Select optimal sequence of footsteps
        bool success = select_optimal_footsteps_gpu(footsteps);

        return success;
    }

    bool check_step_feasibility(const Footstep& step, const std::vector<ObstacleCluster>& obstacles) {
        // Check if a single step is feasible given obstacles
        float2 step_pos = make_float2(step.position.x, step.position.y);

        for (const auto& obstacle : obstacles) {
            float2 diff = make_float2(
                step_pos.x - obstacle.center.x,
                step_pos.y - obstacle.center.y
            );
            float distance = sqrtf(diff.x * diff.x + diff.y * diff.y);

            // Check if step location is too close to obstacle
            if (distance < (obstacle.dimensions.x * 0.5f + robot_foot_size_)) {
                return false;
            }
        }

        return true;
    }

private:
    void generate_safe_footsteps_gpu(
        const float2& start,
        const float2& goal,
        const std::vector<ObstacleCluster>& obstacles) {

        // Copy obstacles to GPU
        cudaMemcpyAsync(d_obstacles_, obstacles.data(),
                       obstacles.size() * sizeof(ObstacleCluster),
                       cudaMemcpyHostToDevice, bipedal_stream_);

        // Generate candidate footsteps that avoid obstacles
        dim3 block_size(256);
        dim3 grid_size((max_candidates_ + block_size.x - 1) / block_size.x);

        generate_safe_footsteps_kernel<<<grid_size, block_size, 0, bipedal_stream_>>>(
            d_footstep_candidates_, start, goal, d_obstacles_, obstacles.size(),
            max_step_length_, max_step_width_, max_candidates_
        );
    }

    void evaluate_footsteps_balance_gpu() {
        // Evaluate each candidate based on balance requirements
        dim3 block_size(256);
        dim3 grid_size((max_candidates_ + block_size.x - 1) / block_size.x);

        evaluate_footsteps_balance_kernel<<<grid_size, block_size, 0, bipedal_stream_>>>(
            d_footstep_candidates_, d_balance_constraints_, max_candidates_
        );
    }

    bool select_optimal_footsteps_gpu(std::vector<Footstep>& footsteps) {
        // Use GPU to find optimal sequence of footsteps
        // This could implement dynamic programming or other optimization algorithms
        return true; // Simplified
    }

    float max_step_length_;
    float max_step_width_;
    float robot_foot_size_ = 0.15f; // 15cm foot size
    int max_candidates_ = 1000;
    int max_constraints_ = 500;

    FootstepCandidate* d_footstep_candidates_;
    BalanceConstraint* d_balance_constraints_;
    ObstacleCluster* d_obstacles_;
    cudaStream_t bipedal_stream_;
};

// CUDA kernel for generating safe footsteps
__global__ void generate_safe_footsteps_kernel(
    FootstepCandidate* candidates,
    float2 start,
    float2 goal,
    const ObstacleCluster* obstacles,
    int num_obstacles,
    float max_step_length,
    float max_step_width,
    int max_candidates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < max_candidates) {
        // Generate candidate in direction of goal with some variation
        float goal_direction = atan2f(goal.y - start.y, goal.x - start.x);
        float angle_variation = (idx % 8) * (2.0f * M_PI / 8.0f);
        float distance = 0.1f + (idx / 8) * (max_step_length / 5.0f);

        float2 candidate_pos;
        candidate_pos.x = start.x + distance * cosf(goal_direction + angle_variation);
        candidate_pos.y = start.y + distance * sinf(goal_direction + angle_variation);

        candidates[idx].position = candidate_pos;
        candidates[idx].step_type = idx % 3; // Different step types

        // Check obstacle collision
        bool is_safe = true;
        for (int i = 0; i < num_obstacles; i++) {
            float2 diff = make_float2(
                candidate_pos.x - obstacles[i].center.x,
                candidate_pos.y - obstacles[i].center.y
            );
            float distance_to_obstacle = sqrtf(diff.x * diff.x + diff.y * diff.y);

            if (distance_to_obstacle < (obstacles[i].dimensions.x * 0.5f + 0.1f)) {
                is_safe = false;
                break;
            }
        }

        candidates[idx].is_safe = is_safe;
        candidates[idx].cost = is_safe ? calculate_step_cost(candidate_pos, goal) : 1e6f;
    }
}

// CUDA kernel for evaluating balance constraints
__global__ void evaluate_footsteps_balance_kernel(
    FootstepCandidate* candidates,
    const BalanceConstraint* constraints,
    int num_candidates
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_candidates && candidates[idx].is_safe) {
        // Evaluate balance feasibility of this step
        // Consider ZMP, CoM trajectory, etc.
        float balance_cost = 0.0f;

        // Add balance-related costs
        // This would involve more complex balance calculations
        candidates[idx].balance_cost = balance_cost;
        candidates[idx].total_cost = candidates[idx].cost + balance_cost;
    }
}

struct FootstepCandidate {
    float2 position;
    int step_type;  // Normal, wide, narrow, etc.
    bool is_safe;
    float cost;          // Path planning cost
    float balance_cost;  // Balance feasibility cost
    float total_cost;    // Combined cost
};

struct BalanceConstraint {
    float2 zmp_limit;     // ZMP stability limits
    float2 com_limit;     // CoM position limits
    float2 com_vel_limit; // CoM velocity limits
    float min_step_time;  // Minimum step duration
    float max_step_time;  // Maximum step duration
};
```

## Social Navigation

### Human-Aware Obstacle Avoidance

#### Social Force Model with GPU Acceleration
```cpp
class SocialNavigationPlanner {
public:
    SocialNavigationPlanner() {
        // Allocate GPU memory for social force calculations
        cudaMalloc(&d_humans_, max_humans_ * sizeof(HumanAgent));
        cudaMalloc(&d_robot_, sizeof(RobotAgent));
        cudaMalloc(&d_social_forces_, max_humans_ * sizeof(float2));
        cudaMalloc(&d_desired_forces_, max_humans_ * sizeof(float2));
    }

    void plan_social_path(const std::vector<HumanAgent>& humans,
                         const RobotAgent& robot,
                         std::vector<float2>& path) {
        // Copy human and robot data to GPU
        cudaMemcpyAsync(d_humans_, humans.data(), humans.size() * sizeof(HumanAgent),
                       cudaMemcpyHostToDevice, social_stream_);
        cudaMemcpyAsync(d_robot_, &robot, sizeof(RobotAgent),
                       cudaMemcpyHostToDevice, social_stream_);

        // Calculate social forces on GPU
        calculate_social_forces_gpu(humans.size());

        // Generate path considering social forces
        generate_social_path_gpu(path);
    }

private:
    void calculate_social_forces_gpu(int num_humans) {
        dim3 block_size(256);
        dim3 grid_size((num_humans + block_size.x - 1) / block_size.x);

        calculate_social_forces_kernel<<<grid_size, block_size, 0, social_stream_>>>(
            d_humans_, d_robot_, d_social_forces_, d_desired_forces_,
            num_humans, social_force_params_
        );
    }

    float social_force_params_[4] = {2.0f, 0.3f, 0.4f, 1.5f}; // Various social force parameters

    HumanAgent* d_humans_;
    RobotAgent* d_robot_;
    float2* d_social_forces_;
    float2* d_desired_forces_;
    cudaStream_t social_stream_;

    int max_humans_ = 100;
};

// CUDA kernel for social force calculation
__global__ void calculate_social_forces_kernel(
    const HumanAgent* humans,
    const RobotAgent* robot,
    float2* social_forces,
    float2* desired_forces,
    int num_humans,
    const float* params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_humans) {
        float2 force = make_float2(0.0f, 0.0f);

        // Calculate force from each human on robot
        for (int i = 0; i < num_humans; i++) {
            if (i != idx) {
                float2 diff = make_float2(
                    humans[idx].position.x - humans[i].position.x,
                    humans[idx].position.y - humans[i].position.y
                );
                float distance = sqrtf(diff.x * diff.x + diff.y * diff.y);

                if (distance > 0.01f) { // Avoid division by zero
                    float force_magnitude = params[0] * expf(-distance / params[1]);
                    float2 repulsion_force = make_float2(
                        force_magnitude * diff.x / distance,
                        force_magnitude * diff.y / distance
                    );
                    force.x += repulsion_force.x;
                    force.y += repulsion_force.y;
                }
            }
        }

        // Add force from walls/obstacles
        // This would consider environment boundaries

        social_forces[idx] = force;
    }
}

struct HumanAgent {
    float2 position;
    float2 velocity;
    float2 desired_velocity;
    float radius;
    float relaxation_time;
};

struct RobotAgent {
    float2 position;
    float2 velocity;
    float2 desired_velocity;
    float radius;
    float mass;
    float relaxation_time;
};
```

Obstacle avoidance for humanoid robots requires sophisticated algorithms that consider both the 3D nature of obstacles and the balance constraints of bipedal locomotion. The GPU acceleration provided by Isaac enables real-time processing of complex sensor data and path planning that would be computationally prohibitive on CPU alone.