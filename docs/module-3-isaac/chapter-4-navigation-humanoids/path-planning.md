---
sidebar_position: 5
title: "Path Planning for Humanoid Navigation"
description: "GPU-accelerated path planning algorithms for humanoid robots"
---

# Path Planning for Humanoid Navigation

Path planning for humanoid robots is significantly more complex than for wheeled robots due to the need to consider bipedal locomotion, balance constraints, and 3D terrain navigation. Isaac provides GPU-accelerated path planning algorithms that can handle these complexities in real-time, enabling humanoid robots to navigate efficiently through complex environments.

## Humanoid-Specific Path Planning Challenges

### 3D Navigation Considerations

Humanoid robots navigate in 3D space with unique constraints:

#### Terrain Traversability
- **Step Height Limits**: Maximum height differences that can be traversed
- **Slope Navigation**: Ability to navigate sloped surfaces
- **Stair Climbing**: Navigating stairs and steps
- **Obstacle Clearance**: Ability to step over or around obstacles

#### Bipedal Motion Constraints
- **Footstep Planning**: Planning where to place each foot
- **Balance Maintenance**: Maintaining stability during navigation
- **Step Timing**: Coordinating step timing with navigation
- **Dynamic Stability**: Maintaining balance during motion

### Multi-Modal Navigation

#### Different Navigation Modes
- **Walking Mode**: Standard bipedal walking
- **Climbing Mode**: Navigating stairs and steps
- **Crouching Mode**: Navigating under low obstacles
- **Sitting/Standing**: Transition between sitting and standing

## Isaac Path Planning Architecture

### GPU-Accelerated Planning Framework

Isaac provides a comprehensive path planning framework with GPU acceleration:

#### Core Planning Components
- **Global Planner**: GPU-accelerated global path planning
- **Local Planner**: Real-time obstacle avoidance
- **Footstep Planner**: Bipedal-specific step planning
- **Trajectory Generator**: Smooth trajectory generation
- **Controller Interface**: Path following and control

#### Planning Pipeline
```
Environment → Perception → Global Planning → Local Planning → Footstep Planning → Control → Humanoid
   ↑           ↑            ↑                 ↑               ↑                ↑        ↑
 GPU       GPU/CPU      GPU/CPU          GPU/CPU        GPU/CPU         GPU/CPU  Hardware
```

### Path Planning Algorithms

#### GPU-Accelerated A* Algorithm
```cpp
#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <map>

class GPUAStarPlanner {
public:
    GPUAStarPlanner(int width, int height, float resolution = 0.05f)
        : width_(width), height_(height), resolution_(resolution) {

        // Allocate GPU memory for A* algorithm
        cudaMalloc(&d_costmap_, width_ * height_ * sizeof(unsigned char));
        cudaMalloc(&d_open_set_, width_ * height_ * sizeof(GridNode));
        cudaMalloc(&d_closed_set_, width_ * height_ * sizeof(bool));
        cudaMalloc(&d_g_scores_, width_ * height_ * sizeof(float));
        cudaMalloc(&d_f_scores_, width_ * height_ * sizeof(float));
        cudaMalloc(&d_parents_, width_ * height_ * sizeof(int2));

        // Initialize CUDA streams
        cudaStreamCreate(&planning_stream_);
    }

    bool plan_path(const int2& start, const int2& goal,
                   std::vector<int2>& path) {
        // Initialize A* algorithm
        initialize_astar_gpu(start, goal);

        // Execute A* iterations on GPU
        bool found_path = execute_astar_gpu(goal);

        if (found_path) {
            // Reconstruct path on GPU
            reconstruct_path_gpu(start, goal, path);
        }

        return found_path;
    }

private:
    void initialize_astar_gpu(const int2& start, const int2& goal) {
        // Initialize g_scores, f_scores, parents, and closed set
        cudaMemsetAsync(d_g_scores_, 0xFF, width_ * height_ * sizeof(float), planning_stream_);
        cudaMemsetAsync(d_f_scores_, 0xFF, width_ * height_ * sizeof(float), planning_stream_);
        cudaMemsetAsync(d_closed_set_, 0, width_ * height_ * sizeof(bool), planning_stream_);

        // Set start node values
        int start_idx = start.y * width_ + start.x;
        float start_g = 0.0f;
        float start_h = heuristic(start, goal);
        float start_f = start_g + start_h;

        cudaMemcpyAsync(&d_g_scores_[start_idx], &start_g, sizeof(float),
                       cudaMemcpyHostToDevice, planning_stream_);
        cudaMemcpyAsync(&d_f_scores_[start_idx], &start_f, sizeof(float),
                       cudaMemcpyHostToDevice, planning_stream_);
    }

    bool execute_astar_gpu(const int2& goal) {
        int goal_idx = goal.y * width_ + goal.x;
        bool goal_reached = false;
        int iterations = 0;

        while (!goal_reached && iterations < max_iterations_) {
            // Find node with minimum f-score in open set
            int current_idx = find_min_f_score_node_gpu();

            if (current_idx == -1) {
                // No path found
                return false;
            }

            // Convert index to coordinates
            int current_y = current_idx / width_;
            int current_x = current_idx % width_;
            int2 current = make_int2(current_x, current_y);

            // Move current node to closed set
            mark_closed_gpu(current_idx);

            // Check if goal is reached
            if (current.x == goal.x && current.y == goal.y) {
                return true;
            }

            // Process neighbors
            process_neighbors_gpu(current, goal);

            iterations++;
        }

        return goal_reached;
    }

    void process_neighbors_gpu(const int2& current, const int2& goal) {
        // Process 8-connected neighbors
        int2 neighbors[8] = {
            make_int2(-1, -1), make_int2(0, -1), make_int2(1, -1),
            make_int2(-1, 0),                     make_int2(1, 0),
            make_int2(-1, 1),  make_int2(0, 1),  make_int2(1, 1)
        };

        for (int i = 0; i < 8; i++) {
            int2 neighbor = make_int2(current.x + neighbors[i].x, current.y + neighbors[i].y);

            // Check bounds
            if (neighbor.x < 0 || neighbor.x >= width_ ||
                neighbor.y < 0 || neighbor.y >= height_) {
                continue;
            }

            // Check if neighbor is in closed set or is obstacle
            int neighbor_idx = neighbor.y * width_ + neighbor.x;
            unsigned char cost = get_cost_from_costmap(neighbor_idx);

            if (cost > 200) { // Obstacle
                continue;
            }

            float tentative_g = get_g_score_gpu(current) +
                               get_distance_gpu(current, neighbor) +
                               get_heuristic_cost_gpu(neighbor, goal);

            if (tentative_g < get_g_score_gpu(neighbor)) {
                // This path to neighbor is better
                update_node_gpu(neighbor, tentative_g, current);
            }
        }
    }

    float heuristic(const int2& a, const int2& b) {
        // Euclidean distance heuristic
        float dx = abs(a.x - b.x) * resolution_;
        float dy = abs(a.y - b.y) * resolution_;
        return sqrtf(dx * dx + dy * dy);
    }

    int find_min_f_score_node_gpu() {
        // Find node with minimum f-score that's not in closed set
        // This would involve parallel reduction or other GPU algorithms
        return 0; // Simplified
    }

    void mark_closed_gpu(int idx) {
        bool closed = true;
        cudaMemcpyAsync(&d_closed_set_[idx], &closed, sizeof(bool),
                       cudaMemcpyHostToDevice, planning_stream_);
    }

    float get_g_score_gpu(const int2& node) {
        int idx = node.y * width_ + node.x;
        float g_score;
        cudaMemcpyAsync(&g_score, &d_g_scores_[idx], sizeof(float),
                       cudaMemcpyDeviceToHost, planning_stream_);
        cudaStreamSynchronize(planning_stream_);
        return g_score;
    }

    void update_node_gpu(const int2& node, float g_score, const int2& parent) {
        int idx = node.y * width_ + node.x;
        float h_score = heuristic(node, make_int2(0, 0)); // Simplified
        float f_score = g_score + h_score;

        cudaMemcpyAsync(&d_g_scores_[idx], &g_score, sizeof(float),
                       cudaMemcpyHostToDevice, planning_stream_);
        cudaMemcpyAsync(&d_f_scores_[idx], &f_score, sizeof(float),
                       cudaMemcpyHostToDevice, planning_stream_);
        cudaMemcpyAsync(&d_parents_[idx], &parent, sizeof(int2),
                       cudaMemcpyHostToDevice, planning_stream_);
    }

    void reconstruct_path_gpu(const int2& start, const int2& goal,
                             std::vector<int2>& path) {
        // Reconstruct path by following parent pointers
        int2 current = goal;
        path.clear();

        while (!(current.x == start.x && current.y == start.y)) {
            path.push_back(current);

            // Get parent
            int idx = current.y * width_ + current.x;
            int2 parent;
            cudaMemcpyAsync(&parent, &d_parents_[idx], sizeof(int2),
                           cudaMemcpyDeviceToHost, planning_stream_);
            cudaStreamSynchronize(planning_stream_);

            current = parent;

            if (path.size() > width_ * height_) {
                // Safety check to prevent infinite loop
                break;
            }
        }

        path.push_back(start);
        std::reverse(path.begin(), path.end());
    }

    unsigned char* d_costmap_;
    GridNode* d_open_set_;
    bool* d_closed_set_;
    float* d_g_scores_;
    float* d_f_scores_;
    int2* d_parents_;
    cudaStream_t planning_stream_;

    int width_, height_;
    float resolution_;
    int max_iterations_ = 100000;
};

struct GridNode {
    int2 position;
    float g_score;
    float f_score;
    int2 parent;
    bool in_open_set;
    bool in_closed_set;
};
```

### GPU-Accelerated RRT Algorithm

#### Rapidly-Exploring Random Trees
```cpp
class GPURRTPlanner {
public:
    GPURRTPlanner(int max_nodes = 10000)
        : max_nodes_(max_nodes) {

        // Allocate GPU memory for RRT
        cudaMalloc(&d_tree_nodes_, max_nodes_ * sizeof(TreeNode));
        cudaMalloc(&d_random_samples_, max_nodes_ * sizeof(float2));
        cudaMalloc(&d_nearest_neighbors_, max_nodes_ * sizeof(int));
        cudaMalloc(&d_distances_, max_nodes_ * sizeof(float));
    }

    bool plan_path_rrt(const float2& start, const float2& goal,
                       std::vector<float2>& path) {
        // Initialize RRT tree
        initialize_rrt_tree_gpu(start);

        bool path_found = false;
        int iterations = 0;

        while (!path_found && iterations < max_iterations_) {
            // Sample random point
            float2 random_point = sample_random_point_gpu();

            // Find nearest node in tree
            int nearest_idx = find_nearest_node_gpu(random_point);

            // Steer towards random point
            float2 new_point = steer_towards_gpu(d_tree_nodes_[nearest_idx].position, random_point);

            // Check collision
            if (!is_collision_free_gpu(d_tree_nodes_[nearest_idx].position, new_point)) {
                continue;
            }

            // Add node to tree
            int new_idx = add_node_to_tree_gpu(nearest_idx, new_point);

            // Check if goal is reached
            if (distance_gpu(new_point, goal) < goal_tolerance_) {
                path_found = true;
            }

            iterations++;
        }

        if (path_found) {
            extract_path_rrt_gpu(start, goal, path);
        }

        return path_found;
    }

private:
    void initialize_rrt_tree_gpu(const float2& start) {
        TreeNode root;
        root.position = start;
        root.parent = -1;
        root.cost = 0.0f;

        cudaMemcpyAsync(d_tree_nodes_, &root, sizeof(TreeNode),
                       cudaMemcpyHostToDevice, planning_stream_);
        num_nodes_ = 1;
    }

    float2 sample_random_point_gpu() {
        // Generate random point with bias towards goal
        float rand_val = static_cast<float>(rand()) / RAND_MAX;
        if (rand_val < goal_bias_) {
            return goal_sample_;
        } else {
            float x = static_cast<float>(rand()) / RAND_MAX * (map_width_ * resolution_);
            float y = static_cast<float>(rand()) / RAND_MAX * (map_height_ * resolution_);
            return make_float2(x, y);
        }
    }

    int find_nearest_node_gpu(const float2& point) {
        // Find nearest node using GPU parallel search
        dim3 block_size(256);
        dim3 grid_size((num_nodes_ + block_size.x - 1) / block_size.x);

        find_nearest_kernel<<<grid_size, block_size, 0, planning_stream_>>>(
            d_tree_nodes_, point, d_distances_, num_nodes_
        );

        // Find minimum distance (simplified - would use parallel reduction in practice)
        return find_min_distance_index_gpu();
    }

    float2 steer_towards_gpu(const float2& from, const float2& to) {
        float2 direction = make_float2(to.x - from.x, to.y - from.y);
        float distance = sqrtf(direction.x * direction.x + direction.y * direction.y);

        if (distance > max_step_size_) {
            // Limit step size
            direction.x = direction.x * max_step_size_ / distance;
            direction.y = direction.y * max_step_size_ / distance;
        }

        return make_float2(from.x + direction.x, from.y + direction.y);
    }

    bool is_collision_free_gpu(const float2& from, const float2& to) {
        // Check collision along the path segment
        int steps = static_cast<int>(distance_gpu(from, to) / resolution_);
        float step_size = 1.0f / steps;

        for (int i = 0; i <= steps; i++) {
            float t = i * step_size;
            float2 point = make_float2(
                from.x + t * (to.x - from.x),
                from.y + t * (to.y - from.y)
            );

            if (is_occupied_gpu(point)) {
                return false;
            }
        }

        return true;
    }

    int add_node_to_tree_gpu(int parent_idx, const float2& position) {
        TreeNode new_node;
        new_node.position = position;
        new_node.parent = parent_idx;
        new_node.cost = d_tree_nodes_[parent_idx].cost +
                       distance_gpu(d_tree_nodes_[parent_idx].position, position);

        int new_idx = num_nodes_++;
        cudaMemcpyAsync(&d_tree_nodes_[new_idx], &new_node, sizeof(TreeNode),
                       cudaMemcpyHostToDevice, planning_stream_);

        return new_idx;
    }

    void extract_path_rrt_gpu(const float2& start, const float2& goal,
                             std::vector<float2>& path) {
        // Find node closest to goal
        int goal_node_idx = find_nearest_node_gpu(goal);

        // Reconstruct path by following parent pointers
        path.clear();
        int current_idx = goal_node_idx;

        while (current_idx != -1) {
            TreeNode node;
            cudaMemcpyAsync(&node, &d_tree_nodes_[current_idx], sizeof(TreeNode),
                           cudaMemcpyDeviceToHost, planning_stream_);
            cudaStreamSynchronize(planning_stream_);

            path.push_back(node.position);
            current_idx = node.parent;
        }

        std::reverse(path.begin(), path.end());
    }

    float distance_gpu(const float2& a, const float2& b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        return sqrtf(dx * dx + dy * dy);
    }

    bool is_occupied_gpu(const float2& point) {
        // Check if point is occupied in costmap
        int x = static_cast<int>(point.x / resolution_);
        int y = static_cast<int>(point.y / resolution_);

        if (x < 0 || x >= map_width_ || y < 0 || y >= map_height_) {
            return true; // Outside map is considered occupied
        }

        unsigned char cost;
        int idx = y * map_width_ + x;
        cudaMemcpyAsync(&cost, &d_costmap_[idx], sizeof(unsigned char),
                       cudaMemcpyDeviceToHost, planning_stream_);
        cudaStreamSynchronize(planning_stream_);

        return cost > 200; // Threshold for obstacle
    }

    TreeNode* d_tree_nodes_;
    float2* d_random_samples_;
    int* d_nearest_neighbors_;
    float* d_distances_;
    cudaStream_t planning_stream_;

    int max_nodes_;
    int num_nodes_ = 0;
    int max_iterations_ = 10000;
    float max_step_size_ = 0.5f; // Maximum distance to extend per step
    float goal_tolerance_ = 0.3f; // Distance to consider goal reached
    float goal_bias_ = 0.1f; // Probability of sampling goal
    float2 goal_sample_;
    int map_width_ = 200;
    int map_height_ = 200;
    float resolution_ = 0.05f;
};

// CUDA kernel for finding nearest node
__global__ void find_nearest_kernel(
    const TreeNode* tree_nodes,
    float2 target_point,
    float* distances,
    int num_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_nodes) {
        float2 node_pos = tree_nodes[idx].position;
        float dx = node_pos.x - target_point.x;
        float dy = node_pos.y - target_point.y;
        distances[idx] = sqrtf(dx * dx + dy * dy);
    }
}

struct TreeNode {
    float2 position;
    int parent;
    float cost;
    int id;
};
```

## Bipedal-Specific Path Planning

### Footstep Planning Integration

#### Step-Aware Path Planning
```cpp
class BipedalPathPlanner {
public:
    BipedalPathPlanner(float max_step_length = 0.3f, float max_step_width = 0.2f)
        : max_step_length_(max_step_length), max_step_width_(max_step_width) {

        // Allocate GPU memory for footstep planning
        cudaMalloc(&d_footsteps_, max_steps_ * sizeof(Footstep));
        cudaMalloc(&d_step_costs_, max_steps_ * sizeof(float));
        cudaMalloc(&d_balance_constraints_, max_constraints_ * sizeof(BalanceConstraint));
    }

    bool plan_bipedal_path(const float2& start, const float2& goal,
                          const std::vector<ObstacleCluster>& obstacles,
                          std::vector<Footstep>& footsteps) {
        // Generate initial path using standard planner
        std::vector<float2> coarse_path;
        bool path_exists = plan_coarse_path_gpu(start, goal, obstacles, coarse_path);

        if (!path_exists) {
            return false;
        }

        // Convert path to footstep sequence
        bool footsteps_valid = generate_footsteps_from_path(coarse_path, footsteps);

        if (footsteps_valid) {
            // Optimize footsteps for balance and efficiency
            optimize_footsteps_gpu(footsteps);
        }

        return footsteps_valid;
    }

private:
    bool plan_coarse_path_gpu(const float2& start, const float2& goal,
                             const std::vector<ObstacleCluster>& obstacles,
                             std::vector<float2>& path) {
        // Plan path considering robot footprint and step constraints
        GPUAStarPlanner astar_planner(200, 200, 0.05f);

        int2 start_grid = world_to_grid(start);
        int2 goal_grid = world_to_grid(goal);

        // Update costmap with obstacles
        update_costmap_with_footprint(obstacles);

        return astar_planner.plan_path(start_grid, goal_grid, path);
    }

    bool generate_footsteps_from_path(const std::vector<float2>& path,
                                     std::vector<Footstep>& footsteps) {
        footsteps.clear();

        if (path.size() < 2) {
            return false;
        }

        // Generate footsteps along the path
        for (size_t i = 0; i < path.size() - 1; i++) {
            float2 start_point = path[i];
            float2 end_point = path[i + 1];

            // Calculate number of steps needed between points
            float distance = distance_2d(start_point, end_point);
            int steps_needed = static_cast<int>(ceil(distance / max_step_length_));

            for (int j = 0; j < steps_needed; j++) {
                float progress = static_cast<float>(j) / steps_needed;
                float2 step_pos = interpolate_2d(start_point, end_point, progress);

                Footstep step;
                step.position = step_pos;
                step.foot_index = (footsteps.size() + 1) % 2; // Alternate feet
                step.orientation = calculate_orientation(start_point, end_point);

                // Check if step is feasible
                if (is_step_feasible(step)) {
                    footsteps.push_back(step);
                } else {
                    // Try alternative step placement
                    if (!find_alternative_step(step, footsteps)) {
                        return false; // Path not feasible
                    }
                }
            }
        }

        return true;
    }

    void optimize_footsteps_gpu(std::vector<Footstep>& footsteps) {
        // Optimize footsteps for balance and smoothness
        if (footsteps.empty()) return;

        // Copy footsteps to GPU
        Footstep* d_footsteps;
        cudaMalloc(&d_footsteps, footsteps.size() * sizeof(Footstep));
        cudaMemcpyAsync(d_footsteps, footsteps.data(),
                       footsteps.size() * sizeof(Footstep),
                       cudaMemcpyHostToDevice, footstep_stream_);

        // Optimize footsteps for balance
        optimize_balance_gpu(d_footsteps, footsteps.size());

        // Optimize for smoothness
        optimize_smoothness_gpu(d_footsteps, footsteps.size());

        // Copy results back
        cudaMemcpyAsync(footsteps.data(), d_footsteps,
                       footsteps.size() * sizeof(Footstep),
                       cudaMemcpyDeviceToHost, footstep_stream_);
        cudaStreamSynchronize(footstep_stream_);

        cudaFree(d_footsteps);
    }

    bool is_step_feasible(const Footstep& step) {
        // Check if step is within step limits and doesn't collide with obstacles
        if (step.position.x < 0 || step.position.y < 0) {
            return false;
        }

        // Check step constraints
        float step_length = distance_2d(last_step_position_, step.position);
        if (step_length > max_step_length_) {
            return false;
        }

        // Check obstacle collision
        return !check_step_collision(step);
    }

    bool find_alternative_step(Footstep& step, const std::vector<Footstep>& existing_steps) {
        // Try different step positions around the desired location
        float search_radius = 0.1f; // 10cm search radius
        int search_resolution = 8; // 8 directions to try

        for (int i = 0; i < search_resolution; i++) {
            float angle = 2.0f * M_PI * i / search_resolution;
            float2 offset = make_float2(
                search_radius * cosf(angle),
                search_radius * sinf(angle)
            );

            step.position = make_float2(
                step.position.x + offset.x,
                step.position.y + offset.y
            );

            if (is_step_feasible(step)) {
                return true;
            }
        }

        return false;
    }

    float2 interpolate_2d(const float2& a, const float2& b, float t) {
        return make_float2(
            a.x + t * (b.x - a.x),
            a.y + t * (b.y - a.y)
        );
    }

    float distance_2d(const float2& a, const float2& b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        return sqrtf(dx * dx + dy * dy);
    }

    float calculate_orientation(const float2& from, const float2& to) {
        return atan2f(to.y - from.y, to.x - from.x);
    }

    int2 world_to_grid(const float2& world_pos) {
        return make_int2(
            static_cast<int>(world_pos.x / resolution_),
            static_cast<int>(world_pos.y / resolution_)
        );
    }

    void update_costmap_with_footprint(const std::vector<ObstacleCluster>& obstacles) {
        // Update costmap considering robot's foot size and safety margins
        // This would expand obstacle areas by robot's footprint
    }

    bool check_step_collision(const Footstep& step) {
        // Check if footstep collides with obstacles
        // Implementation would check against costmap or obstacle list
        return false; // Simplified
    }

    void optimize_balance_gpu(Footstep* d_footsteps, int num_steps) {
        // Optimize footsteps to maintain balance (ZMP constraints)
        dim3 block_size(256);
        dim3 grid_size((num_steps + block_size.x - 1) / block_size.x);

        optimize_balance_kernel<<<grid_size, block_size, 0, footstep_stream_>>>(
            d_footsteps, num_steps, max_step_length_, max_step_width_
        );
    }

    void optimize_smoothness_gpu(Footstep* d_footsteps, int num_steps) {
        // Optimize footsteps for smooth motion
        dim3 block_size(256);
        dim3 grid_size((num_steps + block_size.x - 1) / block_size.x);

        optimize_smoothness_kernel<<<grid_size, block_size, 0, footstep_stream_>>>(
            d_footsteps, num_steps
        );
    }

    Footstep* d_footsteps_;
    float* d_step_costs_;
    BalanceConstraint* d_balance_constraints_;
    cudaStream_t footstep_stream_;

    float max_step_length_;
    float max_step_width_;
    float resolution_ = 0.05f;
    int max_steps_ = 1000;
    int max_constraints_ = 100;
    float2 last_step_position_ = make_float2(0.0f, 0.0f);
};

// CUDA kernel for balance optimization
__global__ void optimize_balance_kernel(
    Footstep* footsteps,
    int num_steps,
    float max_step_length,
    float max_step_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_steps) {
        // Optimize step position for balance
        Footstep current_step = footsteps[idx];

        // Consider previous steps for balance constraints
        if (idx > 0) {
            Footstep prev_step = footsteps[idx - 1];

            // Ensure step is within balance constraints
            float dx = current_step.position.x - prev_step.position.x;
            float dy = current_step.position.y - prev_step.position.y;
            float step_distance = sqrtf(dx * dx + dy * dy);

            if (step_distance > max_step_length) {
                // Adjust step to be within maximum step length
                float scale = max_step_length / step_distance;
                current_step.position.x = prev_step.position.x + dx * scale;
                current_step.position.y = prev_step.position.y + dy * scale;
            }
        }

        footsteps[idx] = current_step;
    }
}

// CUDA kernel for smoothness optimization
__global__ void optimize_smoothness_kernel(
    Footstep* footsteps,
    int num_steps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx > 0 && idx < num_steps - 1) {
        // Apply smoothing to middle steps
        Footstep prev = footsteps[idx - 1];
        Footstep current = footsteps[idx];
        Footstep next = footsteps[idx + 1];

        // Simple smoothing: move towards average of neighbors
        float smoothing_factor = 0.1f;
        float2 smoothed_pos = make_float2(
            current.position.x * (1.0f - smoothing_factor) +
            (prev.position.x + next.position.x) * 0.5f * smoothing_factor,
            current.position.y * (1.0f - smoothing_factor) +
            (prev.position.y + next.position.y) * 0.5f * smoothing_factor
        );

        current.position = smoothed_pos;
        footsteps[idx] = current;
    }
}

struct Footstep {
    float2 position;
    float orientation;
    int foot_index;  // 0 for left, 1 for right
    float step_time;
    float lift_height;
};

struct BalanceConstraint {
    float2 zmp_limits;     // ZMP stability region
    float2 com_limits;     // CoM position limits
    float2 com_velocity_limits; // CoM velocity limits
    float min_step_time;   // Minimum time between steps
    float max_step_time;   // Maximum time between steps
};
```

## Multi-Query Path Planning

### Batch Path Planning for Multiple Goals

#### GPU-Accelerated Multi-Goal Planning
```cpp
class MultiGoalPathPlanner {
public:
    MultiGoalPathPlanner(int max_goals = 100, int max_paths = 1000)
        : max_goals_(max_goals), max_paths_(max_paths) {

        // Allocate GPU memory for multi-goal planning
        cudaMalloc(&d_goals_, max_goals_ * sizeof(float2));
        cudaMalloc(&d_paths_, max_paths_ * sizeof(PathInfo));
        cudaMalloc(&d_path_costs_, max_goals_ * sizeof(float));
        cudaMalloc(&d_path_lengths_, max_goals_ * sizeof(float));
    }

    bool plan_paths_to_multiple_goals(const float2& start,
                                     const std::vector<float2>& goals,
                                     std::vector<std::vector<float2>>& paths) {
        if (goals.empty()) return false;

        // Copy goals to GPU
        cudaMemcpyAsync(d_goals_, goals.data(), goals.size() * sizeof(float2),
                       cudaMemcpyHostToDevice, multi_query_stream_);

        // Plan paths to all goals simultaneously
        bool success = plan_multiple_paths_gpu(start, goals.size());

        if (success) {
            // Extract all paths
            extract_multiple_paths_gpu(goals.size(), paths);
        }

        return success;
    }

    bool find_optimal_goal_sequence(const float2& start,
                                   const std::vector<float2>& goals,
                                   std::vector<int>& goal_order) {
        // Plan paths to all goals
        std::vector<std::vector<float2>> all_paths;
        if (!plan_paths_to_multiple_goals(start, goals, all_paths)) {
            return false;
        }

        // Calculate path costs
        std::vector<float> path_costs;
        for (const auto& path : all_paths) {
            float cost = calculate_path_cost(path);
            path_costs.push_back(cost);
        }

        // Find optimal sequence (simplified - could implement TSP solver)
        goal_order = find_optimal_sequence(start, goals, path_costs);

        return true;
    }

private:
    bool plan_multiple_paths_gpu(const float2& start, int num_goals) {
        // Use GPU parallelization to plan paths to multiple goals
        dim3 block_size(256);
        dim3 grid_size((num_goals + block_size.x - 1) / block_size.x);

        plan_multiple_paths_kernel<<<grid_size, block_size, 0, multi_query_stream_>>>(
            d_goals_, d_paths_, d_path_costs_, start, num_goals
        );

        // Check for success
        bool planning_success = check_planning_success_gpu(num_goals);
        return planning_success;
    }

    float calculate_path_cost(const std::vector<float2>& path) {
        float total_cost = 0.0f;
        for (size_t i = 1; i < path.size(); i++) {
            float2 diff = make_float2(
                path[i].x - path[i-1].x,
                path[i].y - path[i-1].y
            );
            total_cost += sqrtf(diff.x * diff.x + diff.y * diff.y);
        }
        return total_cost;
    }

    std::vector<int> find_optimal_sequence(const float2& start,
                                          const std::vector<float2>& goals,
                                          const std::vector<float>& costs) {
        // Simple greedy approach - visit closest goal first
        std::vector<int> order;
        std::vector<bool> visited(goals.size(), false);

        float2 current_pos = start;
        for (size_t i = 0; i < goals.size(); i++) {
            int best_idx = -1;
            float best_cost = 1e6f;

            for (size_t j = 0; j < goals.size(); j++) {
                if (!visited[j]) {
                    float cost = distance_2d(current_pos, goals[j]);
                    if (cost < best_cost) {
                        best_cost = cost;
                        best_idx = j;
                    }
                }
            }

            if (best_idx != -1) {
                order.push_back(best_idx);
                visited[best_idx] = true;
                current_pos = goals[best_idx];
            }
        }

        return order;
    }

    void extract_multiple_paths_gpu(int num_goals,
                                   std::vector<std::vector<float2>>& paths) {
        // Extract all planned paths from GPU memory
        paths.resize(num_goals);

        for (int i = 0; i < num_goals; i++) {
            // Copy path data for each goal
            // This would involve more complex memory management in practice
        }
    }

    bool check_planning_success_gpu(int num_goals) {
        // Check if all paths were successfully planned
        // This would involve checking success flags from GPU
        return true; // Simplified
    }

    float2* d_goals_;
    PathInfo* d_paths_;
    float* d_path_costs_;
    float* d_path_lengths_;
    cudaStream_t multi_query_stream_;

    int max_goals_;
    int max_paths_;
};

struct PathInfo {
    float2* waypoints;
    int num_waypoints;
    float total_cost;
    bool success;
    int path_id;
};
```

## Dynamic Path Replanning

### Real-time Path Adjustment

#### GPU-Accelerated Dynamic Replanning
```cpp
class DynamicPathReplanner {
public:
    DynamicPathReplanner(float replan_threshold = 0.5f, float replan_rate = 10.0f)
        : replan_threshold_(replan_threshold), replan_rate_(replan_rate) {

        // Allocate GPU memory for dynamic replanning
        cudaMalloc(&d_current_path_, max_path_points_ * sizeof(float2));
        cudaMalloc(&d_obstacle_updates_, max_obstacles_ * sizeof(ObstacleUpdate));
        cudaMalloc(&d_replan_buffer_, max_path_points_ * sizeof(float2));
    }

    bool update_path_for_dynamic_obstacles(
        const std::vector<ObstacleCluster>& new_obstacles,
        const std::vector<float2>& current_path,
        std::vector<float2>& updated_path,
        const float2& robot_position) {

        // Check if replanning is needed
        if (!needs_replanning(current_path, new_obstacles, robot_position)) {
            updated_path = current_path;
            return true;
        }

        // Find safe point in current path
        int safe_point_idx = find_safe_path_point(current_path, new_obstacles, robot_position);

        if (safe_point_idx == -1) {
            // No safe point in current path, replan from robot position
            float2 new_start = robot_position;
            float2 goal = current_path.back();
            return plan_path_gpu(new_start, goal, new_obstacles, updated_path);
        }

        // Replan from safe point to goal
        float2 replan_start = current_path[safe_point_idx];
        float2 goal = current_path.back();

        std::vector<float2> new_path_segment;
        bool success = plan_path_gpu(replan_start, goal, new_obstacles, new_path_segment);

        if (success) {
            // Combine safe portion of old path with new path segment
            updated_path.clear();
            for (int i = 0; i <= safe_point_idx; i++) {
                updated_path.push_back(current_path[i]);
            }
            for (size_t i = 1; i < new_path_segment.size(); i++) { // Skip first point (duplicate)
                updated_path.push_back(new_path_segment[i]);
            }
        }

        return success;
    }

    bool needs_replanning(const std::vector<float2>& current_path,
                         const std::vector<ObstacleCluster>& obstacles,
                         const float2& robot_position) {
        // Check if obstacles intersect with current path
        float safety_distance = 0.5f; // 50cm safety margin

        for (size_t i = 0; i < current_path.size(); i++) {
            float2 path_point = current_path[i];

            // Only check ahead of robot position
            if (distance_2d(path_point, robot_position) <
                distance_2d(current_path[0], robot_position)) {
                continue; // Behind robot, skip
            }

            for (const auto& obstacle : obstacles) {
                float distance = distance_2d(path_point, obstacle.center);
                float min_distance = obstacle.dimensions.x * 0.5f + safety_distance;

                if (distance < min_distance) {
                    return true; // Path intersects with obstacle
                }
            }
        }

        return false; // Path is still clear
    }

private:
    int find_safe_path_point(const std::vector<float2>& path,
                            const std::vector<ObstacleCluster>& obstacles,
                            const float2& robot_position) {
        // Find the first point in path that is safe from all obstacles
        float safety_distance = 0.5f;

        for (int i = 0; i < static_cast<int>(path.size()); i++) {
            float2 path_point = path[i];

            // Only consider points ahead of robot
            if (distance_2d(path_point, robot_position) <
                distance_2d(path[0], robot_position)) {
                continue;
            }

            bool is_safe = true;
            for (const auto& obstacle : obstacles) {
                float distance = distance_2d(path_point, obstacle.center);
                float min_distance = obstacle.dimensions.x * 0.5f + safety_distance;

                if (distance < min_distance) {
                    is_safe = false;
                    break;
                }
            }

            if (is_safe) {
                return i; // Found safe point
            }
        }

        return -1; // No safe point found
    }

    bool plan_path_gpu(const float2& start, const float2& goal,
                      const std::vector<ObstacleCluster>& obstacles,
                      std::vector<float2>& path) {
        // Plan path using GPU-accelerated algorithm
        GPUAStarPlanner planner(200, 200, 0.05f);

        int2 start_grid = world_to_grid(start);
        int2 goal_grid = world_to_grid(goal);

        // Update costmap with new obstacles
        update_costmap_gpu(obstacles);

        std::vector<int2> grid_path;
        bool success = planner.plan_path(start_grid, goal_grid, grid_path);

        if (success) {
            // Convert grid path back to world coordinates
            path.clear();
            for (const auto& grid_pt : grid_path) {
                float2 world_pt = grid_to_world(grid_pt);
                path.push_back(world_pt);
            }
        }

        return success;
    }

    void update_costmap_gpu(const std::vector<ObstacleCluster>& obstacles) {
        // Update costmap with new obstacle information on GPU
        if (obstacles.empty()) return;

        ObstacleCluster* d_obstacles;
        cudaMalloc(&d_obstacles, obstacles.size() * sizeof(ObstacleCluster));
        cudaMemcpyAsync(d_obstacles, obstacles.data(),
                       obstacles.size() * sizeof(ObstacleCluster),
                       cudaMemcpyHostToDevice, replan_stream_);

        dim3 block_size(16, 16);
        dim3 grid_size((200 + block_size.x - 1) / block_size.x,
                      (200 + block_size.y - 1) / block_size.y);

        update_costmap_with_obstacles_kernel<<<grid_size, block_size, 0, replan_stream_>>>(
            d_costmap_, d_obstacles, obstacles.size(), 200, 200, 0.05f
        );

        cudaFree(d_obstacles);
    }

    float2 grid_to_world(const int2& grid_pos) {
        return make_float2(
            grid_pos.x * resolution_,
            grid_pos.y * resolution_
        );
    }

    int2 world_to_grid(const float2& world_pos) {
        return make_int2(
            static_cast<int>(world_pos.x / resolution_),
            static_cast<int>(world_pos.y / resolution_)
        );
    }

    float distance_2d(const float2& a, const float2& b) {
        float dx = a.x - b.x;
        float dy = a.y - b.y;
        return sqrtf(dx * dx + dy * dy);
    }

    float2* d_current_path_;
    ObstacleUpdate* d_obstacle_updates_;
    float2* d_replan_buffer_;
    unsigned char* d_costmap_;
    cudaStream_t replan_stream_;

    float replan_threshold_;
    float replan_rate_;
    float resolution_ = 0.05f;
    int max_path_points_ = 1000;
    int max_obstacles_ = 100;
};

// CUDA kernel for updating costmap with obstacles
__global__ void update_costmap_with_obstacles_kernel(
    unsigned char* costmap,
    const ObstacleCluster* obstacles,
    int num_obstacles,
    int width, int height,
    float resolution
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float2 world_pos = make_float2(x * resolution, y * resolution);
        unsigned char max_cost = 0;

        for (int i = 0; i < num_obstacles; i++) {
            float2 diff = make_float2(
                world_pos.x - obstacles[i].center.x,
                world_pos.y - obstacles[i].center.y
            );
            float distance = sqrtf(diff.x * diff.x + diff.y * diff.y);

            if (distance < obstacles[i].dimensions.x * 0.5f) {
                // Inside obstacle - maximum cost
                max_cost = 254;
                break;
            } else if (distance < obstacles[i].dimensions.x * 0.5f + 0.3f) {
                // Near obstacle - decreasing cost
                float cost = 200 * (1.0f - (distance - obstacles[i].dimensions.x * 0.5f) / 0.3f);
                max_cost = max(max_cost, (unsigned char)cost);
            }
        }

        costmap[y * width + x] = max_cost;
    }
}

struct ObstacleUpdate {
    float2 position;
    float2 dimensions;
    float timestamp;
    ObstacleType type;
};
```

Path planning for humanoid robots requires sophisticated algorithms that can handle the complex constraints of bipedal locomotion while navigating efficiently through 3D environments. The GPU acceleration provided by Isaac enables real-time computation of complex paths that would be computationally prohibitive on CPU alone.