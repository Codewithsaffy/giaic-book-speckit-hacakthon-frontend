---
sidebar_position: 4
title: "Real-time Depth Perception"
description: "GPU-accelerated depth estimation and 3D perception in Isaac ROS"
---

# Real-time Depth Perception

Real-time depth perception is critical for robotics applications that require understanding of 3D spatial relationships. Isaac ROS provides GPU-accelerated depth perception capabilities that enable robots to perceive depth information in real-time, supporting navigation, manipulation, and interaction tasks.

## Understanding Depth Perception

### Depth Perception Fundamentals

Depth perception in robotics involves estimating the distance to objects in the environment:

#### Depth Estimation Methods
- **Stereo Vision**: Using two cameras to calculate depth via triangulation
- **Structured Light**: Projecting known patterns and analyzing distortions
- **Time-of-Flight**: Measuring light travel time for distance calculation
- **Monocular Depth**: Using single camera with learned depth priors

#### Depth Perception Pipeline
1. **Data Acquisition**: Capturing stereo images or depth sensor data
2. **Preprocessing**: Rectifying images and preparing for processing
3. **Depth Computation**: Calculating depth values for each pixel
4. **Post-processing**: Filtering and refining depth estimates
5. **3D Reconstruction**: Building 3D representations from depth data

### Challenges in Real-time Depth Perception

#### Computational Complexity
- **High Resolution**: Processing high-resolution depth maps
- **Real-time Constraints**: Meeting frame rate requirements
- **Memory Bandwidth**: Handling large amounts of depth data
- **Algorithm Complexity**: Running complex depth estimation algorithms

#### Quality Challenges
- **Occlusion Handling**: Dealing with hidden surfaces
- **Textureless Regions**: Estimating depth in uniform areas
- **Reflective Surfaces**: Handling mirrors and shiny objects
- **Dynamic Scenes**: Handling moving objects and lighting changes

## GPU-Accelerated Depth Processing

### CUDA-Based Depth Estimation

Leveraging CUDA for depth computation:

#### Stereo Depth Computation
```cpp
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>

// CUDA kernel for stereo matching
__global__ void stereo_matching_kernel(
    const unsigned char* left_image,
    const unsigned char* right_image,
    float* disparity_map,
    int width,
    int height,
    int max_disparity
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= max_disparity && x < width && y < height) {
        float best_cost = 1e6f;
        int best_disparity = 0;

        // Compute disparity for this pixel
        for (int d = 0; d < max_disparity; d++) {
            if (x - d >= 0) {
                // Compute cost using Sum of Absolute Differences (SAD)
                float cost = 0.0f;
                for (int dy = -2; dy <= 2; dy++) {
                    for (int dx = -2; dx <= 2; dx++) {
                        if (y + dy >= 0 && y + dy < height &&
                            x + dx >= 0 && x + dx < width &&
                            x + dx - d >= 0) {

                            int left_idx = (y + dy) * width + (x + dx);
                            int right_idx = (y + dy) * width + (x + dx - d);

                            cost += abs(left_image[left_idx] - right_image[right_idx]);
                        }
                    }
                }

                if (cost < best_cost) {
                    best_cost = cost;
                    best_disparity = d;
                }
            }
        }

        disparity_map[y * width + x] = best_disparity;
    }
}

class GPUStereoDepthEstimator {
public:
    GPUStereoDepthEstimator(int width, int height, int max_disparity = 64)
        : width_(width), height_(height), max_disparity_(max_disparity) {

        // Allocate GPU memory
        cudaMalloc(&d_left_image_, width_ * height_ * sizeof(unsigned char));
        cudaMalloc(&d_right_image_, width_ * height_ * sizeof(unsigned char));
        cudaMalloc(&d_disparity_map_, width_ * height_ * sizeof(float));

        // Initialize CUDA streams
        cudaStreamCreate(&processing_stream_);
    }

    void compute_depth(const cv::Mat& left_image,
                      const cv::Mat& right_image,
                      cv::Mat& depth_map) {
        // Copy images to GPU
        cudaMemcpyAsync(d_left_image_, left_image.data,
                       width_ * height_ * sizeof(unsigned char),
                       cudaMemcpyHostToDevice, processing_stream_);
        cudaMemcpyAsync(d_right_image_, right_image.data,
                       width_ * height_ * sizeof(unsigned char),
                       cudaMemcpyHostToDevice, processing_stream_);

        // Launch stereo matching kernel
        dim3 block_size(16, 16);
        dim3 grid_size((width_ + block_size.x - 1) / block_size.x,
                      (height_ + block_size.y - 1) / block_size.y);

        stereo_matching_kernel<<<grid_size, block_size, 0, processing_stream_>>>(
            d_left_image_, d_right_image_, d_disparity_map_,
            width_, height_, max_disparity_
        );

        // Convert disparity to depth
        convert_disparity_to_depth_kernel<<<grid_size, block_size, 0, processing_stream_>>>(
            d_disparity_map_, d_depth_map_, width_, height_, baseline_, focal_length_
        );

        // Copy result back to CPU
        cudaMemcpyAsync(depth_map.data, d_depth_map_,
                       width_ * height_ * sizeof(float),
                       cudaMemcpyDeviceToHost, processing_stream_);

        // Synchronize
        cudaStreamSynchronize(processing_stream_);
    }

private:
    void convert_disparity_to_depth_kernel(
        const float* disparity_map,
        float* depth_map,
        int width, int height,
        float baseline, float focal_length
    ) {
        // Depth = (baseline * focal_length) / disparity
        // Launch appropriate CUDA kernel for conversion
    }

    unsigned char* d_left_image_;
    unsigned char* d_right_image_;
    float* d_disparity_map_;
    float* d_depth_map_;
    cudaStream_t processing_stream_;

    int width_, height_, max_disparity_;
    float baseline_ = 0.1f; // Camera baseline in meters
    float focal_length_ = 525.0f; // Focal length in pixels
};
```

### Optimized Depth Algorithms

Advanced GPU-accelerated depth estimation:

#### Semi-Global Block Matching (SGBM)
```cpp
class GPUSGBMDepthEstimator {
public:
    GPUSGBMDepthEstimator(int width, int height)
        : width_(width), height_(height) {

        // Allocate GPU memory for SGBM algorithm
        allocate_sgbm_memory();
    }

    void compute_sgbm_depth(const cv::Mat& left_image,
                           const cv::Mat& right_image,
                           cv::Mat& depth_map) {
        // Upload images to GPU
        cv::cuda::GpuMat gpu_left, gpu_right;
        gpu_left.upload(left_image);
        gpu_right.upload(right_image);

        // Create SGBM matcher
        cv::Ptr<cv::cuda::StereoBM> stereo_bm = cv::cuda::StereoBM::create(64, 21);

        // Compute disparity
        cv::cuda::GpuMat gpu_disparity;
        stereo_bm->compute(gpu_left, gpu_right, gpu_disparity);

        // Convert to depth
        cv::cuda::GpuMat gpu_depth;
        convert_disparity_to_depth_gpu(gpu_disparity, gpu_depth);

        // Download result
        gpu_depth.download(depth_map);
    }

private:
    void allocate_sgbm_memory() {
        // Allocate memory for SGBM path costs and other intermediate data
        size_t path_cost_size = width_ * height_ * max_disparities_ * sizeof(short);

        for (int i = 0; i < num_paths_; i++) {
            short* path_cost;
            cudaMalloc(&path_cost, path_cost_size);
            d_path_costs_.push_back(path_cost);
        }
    }

    void compute_sgbm_paths() {
        // Launch CUDA kernels for SGBM path computation
        // This implements the semi-global optimization
    }

    std::vector<short*> d_path_costs_;
    int width_, height_;
    int max_disparities_ = 64;
    int num_paths_ = 8; // SGBM typically uses 8 paths
};
```

## Isaac ROS Depth Perception Nodes

### Depth Image Processing

Isaac ROS provides GPU-accelerated depth image processing:

#### Depth Image Rectification
```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <image_transport/image_transport.hpp>

class GPUDepthRectifier : public rclcpp::Node {
public:
    GPUDepthRectifier() : Node("gpu_depth_rectifier") {
        // Initialize GPU
        cudaSetDevice(0);
        cudaFree(0);

        // Create subscribers
        depth_sub_ = image_transport::create_subscription(
            this, "input_depth",
            std::bind(&GPUDepthRectifier::depth_callback, this, std::placeholders::_1),
            "raw"
        );

        // Create publishers
        rectified_pub_ = image_transport::create_publisher(this, "output_depth_rectified");

        // Initialize rectification maps
        initialize_rectification();
    }

private:
    void depth_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Perform rectification on GPU
        cv::Mat rectified_depth;
        rectify_depth_on_gpu(cv_ptr->image, rectified_depth);

        // Convert back to ROS message
        cv_bridge::CvImage rectified_cv;
        rectified_cv.header = msg->header;
        rectified_cv.encoding = msg->encoding;
        rectified_cv.image = rectified_depth;

        rectified_pub_.publish(rectified_cv.toImageMsg());
    }

    void rectify_depth_on_gpu(const cv::Mat& input_depth, cv::Mat& output_depth) {
        // Allocate GPU memory
        cv::cuda::GpuMat gpu_input, gpu_output, gpu_map1, gpu_map2;

        // Upload rectification maps to GPU
        gpu_map1.upload(rectification_map1_);
        gpu_map2.upload(rectification_map2_);

        // Upload input depth map
        gpu_input.upload(input_depth);

        // Perform rectification using remap
        cv::cuda::remap(gpu_input, gpu_output, gpu_map1, gpu_map2,
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

        // Download result
        gpu_output.download(output_depth);
    }

    void initialize_rectification() {
        // Load or compute rectification maps
        // These would come from camera calibration
        rectification_map1_ = cv::Mat::zeros(480, 640, CV_32F);
        rectification_map2_ = cv::Mat::zeros(480, 640, CV_32F);

        // In practice, these maps would be computed from stereo calibration
    }

    image_transport::Subscriber depth_sub_;
    image_transport::Publisher rectified_pub_;

    cv::Mat rectification_map1_, rectification_map2_;
};
```

### Point Cloud Generation

GPU-accelerated point cloud generation from depth images:

#### Depth to Point Cloud Conversion
```cpp
#include <sensor_msgs/msg/point_cloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

class GPUPointCloudGenerator {
public:
    GPUPointCloudGenerator(const cv::Mat& camera_matrix, int width, int height)
        : camera_matrix_(camera_matrix), width_(width), height_(height) {

        // Precompute inverse camera matrix
        camera_matrix_inv_ = camera_matrix_.inv();

        // Allocate GPU memory
        cudaMalloc(&d_depth_map_, width_ * height_ * sizeof(float));
        cudaMalloc(&d_points_, width_ * height_ * 3 * sizeof(float));
    }

    void generate_point_cloud_gpu(const cv::Mat& depth_map,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud) {
        // Copy depth map to GPU
        cudaMemcpy(d_depth_map_, depth_map.data,
                  width_ * height_ * sizeof(float), cudaMemcpyHostToDevice);

        // Launch point cloud generation kernel
        dim3 block_size(16, 16);
        dim3 grid_size((width_ + block_size.x - 1) / block_size.x,
                      (height_ + block_size.y - 1) / block_size.y);

        generate_points_kernel<<<grid_size, block_size>>>(
            d_depth_map_, d_points_, width_, height_,
            camera_matrix_inv_.ptr<float>(0)
        );

        // Copy points back to CPU
        std::vector<float> points_cpu(width_ * height_ * 3);
        cudaMemcpy(points_cpu.data(), d_points_,
                  width_ * height_ * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        // Build point cloud
        build_point_cloud_cpu(points_cpu, point_cloud);
    }

private:
    __global__ void generate_points_kernel(
        const float* depth_map,
        float* points,
        int width, int height,
        const float* inv_camera_matrix
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x < width && y < height) {
            int idx = y * width + x;
            float depth = depth_map[idx];

            if (depth > 0 && depth < 100.0f) { // Valid depth range
                // Convert pixel coordinates to camera coordinates
                float x_norm = (x - inv_camera_matrix[2]) / inv_camera_matrix[0];
                float y_norm = (y - inv_camera_matrix[5]) / inv_camera_matrix[4];

                points[idx * 3 + 0] = x_norm * depth; // X
                points[idx * 3 + 1] = y_norm * depth; // Y
                points[idx * 3 + 2] = depth;          // Z
            } else {
                points[idx * 3 + 0] = 0.0f;
                points[idx * 3 + 1] = 0.0f;
                points[idx * 3 + 2] = 0.0f;
            }
        }
    }

    void build_point_cloud_cpu(const std::vector<float>& points_cpu,
                              pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud) {
        point_cloud->clear();
        point_cloud->width = width_;
        point_cloud->height = height_;
        point_cloud->points.resize(width_ * height_);

        for (int i = 0; i < width_ * height_; i++) {
            pcl::PointXYZ& point = point_cloud->points[i];
            point.x = points_cpu[i * 3 + 0];
            point.y = points_cpu[i * 3 + 1];
            point.z = points_cpu[i * 3 + 2];
        }
    }

    cv::Mat camera_matrix_, camera_matrix_inv_;
    float* d_depth_map_;
    float* d_points_;
    int width_, height_;
};
```

## Deep Learning-Based Depth Estimation

### Monocular Depth Estimation

GPU-accelerated monocular depth estimation using deep learning:

#### TensorRT Depth Estimation
```cpp
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class TensorRTDepthEstimator {
public:
    TensorRTDepthEstimator(const std::string& engine_path) {
        // Load TensorRT engine
        load_tensorrt_engine(engine_path);
    }

    void estimate_depth(const cv::Mat& input_image, cv::Mat& depth_map) {
        // Preprocess input
        cv::Mat preprocessed = preprocess_input(input_image);

        // Copy to GPU
        cudaMemcpy(d_input_, preprocessed.data, input_size_, cudaMemcpyHostToDevice);

        // Perform inference
        context_->executeV2(bindings_.data());

        // Copy output back
        cudaMemcpy(h_output_, d_output_, output_size_, cudaMemcpyDeviceToHost);

        // Postprocess output
        depth_map = postprocess_output();
    }

private:
    void load_tensorrt_engine(const std::string& engine_path) {
        // Read engine file
        std::ifstream file(engine_path, std::ios::binary);
        std::vector<char> engine_data((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());

        // Create runtime and deserialize
        runtime_ = nvinfer1::createInferRuntime(g_logger_);
        engine_ = runtime_->deserializeCudaEngine(
            engine_data.data(), engine_data.size()
        );
        context_ = engine_->createExecutionContext();

        // Allocate I/O buffers
        allocate_io_buffers();
    }

    void allocate_io_buffers() {
        // Get binding information
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

            bindings_.push_back(engine_->bindingIsInput(i) ? d_input_ : d_output_);
        }
    }

    cv::Mat preprocess_input(const cv::Mat& image) {
        // Resize and normalize image
        cv::Mat resized, normalized;
        cv::resize(image, resized, cv::Size(640, 480));
        resized.convertTo(normalized, CV_32F, 1.0/255.0);

        // Normalize with ImageNet mean and std
        std::vector<cv::Mat> channels(3);
        cv::split(normalized, channels);

        cv::Mat input_tensor(1, 3 * 480 * 640, CV_32F);
        float* data = (float*)input_tensor.data;

        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 480 * 640; i++) {
                data[c * 480 * 640 + i] =
                    (channels[c].data[i] - imagenet_mean_[c]) / imagenet_std_[c];
            }
        }

        return input_tensor;
    }

    cv::Mat postprocess_output() {
        // Convert network output to depth map
        float* output = (float*)h_output_;

        cv::Mat depth_map(480, 640, CV_32F);
        float* depth_data = (float*)depth_map.data;

        // Apply activation function and scale to depth range
        for (int i = 0; i < 480 * 640; i++) {
            depth_data[i] = sigmoid(output[i]) * max_depth_; // Sigmoid activation
        }

        // Resize back to original size if needed
        cv::resize(depth_map, depth_map, cv::Size(original_width_, original_height_));

        return depth_map;
    }

    float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    std::vector<void*> bindings_;

    void* d_input_;
    void* d_output_;
    void* h_input_;
    void* h_output_;
    size_t input_size_;
    size_t output_size_;

    float imagenet_mean_[3] = {0.485f, 0.456f, 0.406f};
    float imagenet_std_[3] = {0.229f, 0.224f, 0.225f};
    float max_depth_ = 100.0f; // Maximum depth in meters
    int original_width_ = 640;
    int original_height_ = 480;

    Logger g_logger_;
};

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};
```

### Multi-modal Depth Fusion

Combining multiple depth sources:

#### RGB-D Fusion Node
```cpp
#include <sensor_msgs/msg/image.h>
#include <sensor_msgs/msg/point_cloud2.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

class RGBDDepthFusion : public rclcpp::Node {
public:
    RGBDDepthFusion() : Node("rgbd_depth_fusion") {
        // Initialize GPU
        cudaSetDevice(0);
        cudaFree(0);

        // Create synchronized subscribers
        rgb_sub_.subscribe(this, "camera/rgb/image_raw", rmw_qos_profile_sensor_data);
        depth_sub_.subscribe(this, "camera/depth/image_raw", rmw_qos_profile_sensor_data);

        sync_ = std::make_shared<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>>(
            rgb_sub_, depth_sub_, 10
        );
        sync_->registerCallback(std::bind(&RGBDDepthFusion::rgbd_callback, this,
                                         std::placeholders::_1, std::placeholders::_2));

        // Create publisher
        fused_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("fused_pointcloud", 10);
    }

private:
    void rgbd_callback(
        const sensor_msgs::msg::Image::SharedPtr rgb_msg,
        const sensor_msgs::msg::Image::SharedPtr depth_msg
    ) {
        // Convert to OpenCV
        cv_bridge::CvImagePtr rgb_ptr, depth_ptr;
        try {
            rgb_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::RGB8);
            depth_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Fuse RGB and depth data on GPU
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr fused_cloud(
            new pcl::PointCloud<pcl::PointXYZRGB>
        );
        fuse_rgbd_on_gpu(rgb_ptr->image, depth_ptr->image, fused_cloud);

        // Publish fused point cloud
        publish_pointcloud(fused_cloud, rgb_msg->header);
    }

    void fuse_rgbd_on_gpu(const cv::Mat& rgb_image,
                         const cv::Mat& depth_image,
                         pcl::PointCloud<pcl::PointXYZRGB>::Ptr& fused_cloud) {
        // Allocate GPU memory
        cv::cuda::GpuMat gpu_rgb, gpu_depth;
        gpu_rgb.upload(rgb_image);
        gpu_depth.upload(depth_image);

        // Generate colored point cloud
        generate_colored_pointcloud_kernel(
            gpu_depth, gpu_rgb,
            camera_matrix_, fused_cloud
        );
    }

    void generate_colored_pointcloud_kernel(
        const cv::cuda::GpuMat& depth,
        const cv::cuda::GpuMat& rgb,
        const cv::Mat& camera_matrix,
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud
    ) {
        // Launch CUDA kernel to generate colored 3D points
        // This combines depth information with color information
    }

    message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> depth_sub_;
    std::shared_ptr<message_filters::TimeSynchronizer<sensor_msgs::msg::Image, sensor_msgs::msg::Image>> sync_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr fused_pub_;

    cv::Mat camera_matrix_; // Camera intrinsic parameters
};
```

## Performance Optimization

### Memory Management for Depth Data

Efficient GPU memory management for depth processing:

#### Depth Data Memory Pool
```cpp
class DepthMemoryManager {
public:
    static DepthMemoryManager& get_instance() {
        static DepthMemoryManager instance;
        return instance;
    }

    float* allocate_depth_buffer(int width, int height) {
        size_t size = width * height * sizeof(float);

        std::lock_guard<std::mutex> lock(mutex_);

        // Try to reuse existing buffer
        for (auto it = free_depth_buffers_.begin(); it != free_depth_buffers_.end(); ++it) {
            if (it->second.size_bytes >= size) {
                float* buffer = it->second.buffer;
                free_depth_buffers_.erase(it);

                used_depth_buffers_[buffer] = {buffer, size, width, height};
                return buffer;
            }
        }

        // Allocate new buffer
        float* new_buffer;
        cudaMalloc(&new_buffer, size);

        used_depth_buffers_[new_buffer] = {new_buffer, size, width, height};
        return new_buffer;
    }

    void deallocate_depth_buffer(float* buffer) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = used_depth_buffers_.find(buffer);
        if (it != used_depth_buffers_.end()) {
            const auto& buffer_info = it->second;
            used_depth_buffers_.erase(it);

            // Add to free pool if not too large
            if (free_depth_buffers_.size() < max_pool_size_) {
                free_depth_buffers_[buffer_info.size_bytes] = buffer_info;
            } else {
                cudaFree(buffer);
            }
        }
    }

private:
    struct BufferInfo {
        float* buffer;
        size_t size_bytes;
        int width, height;
    };

    std::unordered_map<float*, BufferInfo> used_depth_buffers_;
    std::map<size_t, BufferInfo> free_depth_buffers_; // Ordered by size
    std::mutex mutex_;
    const size_t max_pool_size_ = 10;
};
```

### Multi-resolution Processing

Processing depth at multiple resolutions for efficiency:

#### Hierarchical Depth Processing
```cpp
class HierarchicalDepthProcessor {
public:
    HierarchicalDepthProcessor(int width, int height, int num_levels = 4)
        : width_(width), height_(height), num_levels_(num_levels) {

        // Create pyramid levels
        for (int level = 0; level < num_levels_; level++) {
            int level_width = width_ >> level;
            int level_height = height_ >> level;

            // Allocate GPU memory for this level
            float* level_buffer;
            cudaMalloc(&level_buffer, level_width * level_height * sizeof(float));
            depth_pyramid_.push_back(level_buffer);
        }
    }

    void process_depth_hierarchical(const cv::Mat& input_depth) {
        // Upload full resolution depth
        upload_depth_to_level(0, input_depth);

        // Process at coarsest level first
        for (int level = num_levels_ - 1; level >= 0; level--) {
            process_level_gpu(level);

            // If not at finest level, upsample and refine
            if (level > 0) {
                upsample_level_gpu(level, level - 1);
            }
        }
    }

private:
    void upload_depth_to_level(int level, const cv::Mat& depth) {
        int level_width = width_ >> level;
        int level_height = height_ >> level;

        cv::Mat level_depth;
        if (level == 0) {
            level_depth = depth;
        } else {
            cv::resize(depth, level_depth, cv::Size(level_width, level_height));
        }

        cudaMemcpy(depth_pyramid_[level], level_depth.data,
                  level_width * level_height * sizeof(float),
                  cudaMemcpyHostToDevice);
    }

    void process_level_gpu(int level) {
        int level_width = width_ >> level;
        int level_height = height_ >> level;

        dim3 block_size(16, 16);
        dim3 grid_size((level_width + block_size.x - 1) / block_size.x,
                      (level_height + block_size.y - 1) / block_size.y);

        process_depth_kernel<<<grid_size, block_size>>>(
            depth_pyramid_[level], level_width, level_height
        );
    }

    void upsample_level_gpu(int src_level, int dst_level) {
        int src_width = width_ >> src_level;
        int src_height = height_ >> src_level;
        int dst_width = width_ >> dst_level;
        int dst_height = height_ >> dst_level;

        dim3 block_size(16, 16);
        dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                      (dst_height + block_size.y - 1) / block_size.y);

        upsample_kernel<<<grid_size, block_size>>>(
            depth_pyramid_[src_level], depth_pyramid_[dst_level],
            src_width, src_height, dst_width, dst_height
        );
    }

    std::vector<float*> depth_pyramid_;
    int width_, height_, num_levels_;
};
```

## Quality Enhancement

### Depth Map Filtering

GPU-accelerated depth map filtering and enhancement:

#### Bilateral Filtering for Depth Maps
```cpp
class GPUDepthFilter {
public:
    GPUDepthFilter(float spatial_sigma = 5.0f, float range_sigma = 0.1f)
        : spatial_sigma_(spatial_sigma), range_sigma_(range_sigma) {}

    void bilateral_filter_depth(const cv::Mat& input_depth, cv::Mat& output_depth) {
        // Allocate GPU memory
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input_depth);

        // Launch bilateral filter kernel
        int channels = 1; // Depth is single channel
        cv::cuda::bilateralFilter(gpu_input, gpu_output,
                                 -1, range_sigma_, spatial_sigma_);

        // Download result
        gpu_output.download(output_depth);
    }

    void guided_filter_depth(const cv::Mat& input_depth,
                            const cv::Mat& guidance_image,
                            cv::Mat& output_depth,
                            int radius = 8, float eps = 0.01f) {
        // Upload images to GPU
        cv::cuda::GpuMat gpu_depth, gpu_guidance, gpu_output;
        gpu_depth.upload(input_depth);
        gpu_guidance.upload(guidance_image);

        // Launch guided filter kernel
        guided_filter_kernel(gpu_depth, gpu_guidance, gpu_output, radius, eps);

        gpu_output.download(output_depth);
    }

private:
    void guided_filter_kernel(const cv::cuda::GpuMat& depth,
                             const cv::cuda::GpuMat& guidance,
                             cv::cuda::GpuMat& output,
                             int radius, float eps) {
        // Implementation of guided filter on GPU
        // This preserves edges while smoothing depth
    }

    float spatial_sigma_;
    float range_sigma_;
};
```

Real-time depth perception in Isaac ROS enables robotics applications to understand 3D spatial relationships with GPU-accelerated performance. The combination of traditional computer vision techniques and deep learning approaches, all optimized for GPU execution, provides the foundation for advanced robotics perception capabilities.