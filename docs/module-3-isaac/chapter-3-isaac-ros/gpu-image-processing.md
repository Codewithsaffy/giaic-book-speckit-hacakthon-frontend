---
sidebar_position: 5
title: "GPU Image Processing in Isaac ROS"
description: "Accelerating computer vision operations with GPU computing in Isaac ROS"
---

# GPU Image Processing in Isaac ROS

GPU image processing is fundamental to Isaac ROS, enabling real-time computer vision operations that would be computationally prohibitive on CPU alone. By leveraging NVIDIA's GPU computing capabilities, Isaac ROS can process high-resolution images at frame rates required for robotics applications.

## Fundamentals of GPU Image Processing

### GPU vs CPU Processing

Understanding the differences between GPU and CPU processing for images:

#### GPU Advantages
- **Parallel Processing**: Thousands of cores for parallel pixel processing
- **Memory Bandwidth**: High-bandwidth memory for rapid data access
- **Specialized Units**: Tensor cores for AI operations, RT cores for ray tracing
- **Real-time Performance**: Sustained high frame rates for robotics

#### CPU Advantages
- **Sequential Processing**: Better for complex control flow
- **Memory Management**: More flexible memory allocation
- **General Purpose**: Better for mixed workloads
- **Precision**: Better for high-precision operations

### CUDA Programming Model

GPU programming for image processing:

#### Memory Management
```cpp
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

class GPUImageProcessor {
public:
    GPUImageProcessor(int width, int height)
        : width_(width), height_(height), channels_(3) {

        // Calculate image size
        size_t image_size = width_ * height_ * channels_ * sizeof(unsigned char);

        // Allocate GPU memory
        cudaMalloc(&d_input_, image_size);
        cudaMalloc(&d_output_, image_size);

        // Allocate pinned host memory for faster transfers
        cudaMallocHost(&h_input_, image_size);
        cudaMallocHost(&h_output_, image_size);
    }

    ~GPUImageProcessor() {
        // Free GPU memory
        cudaFree(d_input_);
        cudaFree(d_output_);

        // Free pinned host memory
        cudaFreeHost(h_input_);
        cudaFreeHost(h_output_);
    }

    void process_image(const cv::Mat& input, cv::Mat& output) {
        // Copy image to GPU using pinned memory for faster transfer
        size_t image_size = width_ * height_ * channels_;
        memcpy(h_input_, input.data, image_size);

        // Asynchronous memory transfer
        cudaMemcpyAsync(d_input_, h_input_, image_size,
                       cudaMemcpyHostToDevice, stream_);

        // Launch processing kernel
        dim3 block_size(16, 16);
        dim3 grid_size((width_ + block_size.x - 1) / block_size.x,
                      (height_ + block_size.y - 1) / block_size.y);

        image_processing_kernel<<<grid_size, block_size, 0, stream_>>>(
            d_input_, d_output_, width_, height_, channels_);

        // Copy result back
        cudaMemcpyAsync(h_output_, d_output_, image_size,
                       cudaMemcpyDeviceToHost, stream_);

        // Synchronize stream
        cudaStreamSynchronize(stream_);

        // Copy to output
        output = cv::Mat(height_, width_, CV_8UC3, h_output_);
    }

private:
    unsigned char* d_input_;
    unsigned char* d_output_;
    unsigned char* h_input_;
    unsigned char* h_output_;

    int width_, height_, channels_;
    cudaStream_t stream_;
};
```

#### Basic Image Processing Kernels
```cpp
// Grayscale conversion kernel
__global__ void grayscale_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int rgb_idx = idx * 3;

        // Convert RGB to grayscale using luminance formula
        float gray = 0.299f * input[rgb_idx] +
                    0.587f * input[rgb_idx + 1] +
                    0.114f * input[rgb_idx + 2];

        output[idx] = (unsigned char)gray;
    }
}

// Gaussian blur kernel
__global__ void gaussian_blur_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width, int height,
    float* kernel,
    int kernel_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= kernel_size/2 && x < width - kernel_size/2 &&
        y >= kernel_size/2 && y < height - kernel_size/2) {

        float sum = 0.0f;
        int half_kernel = kernel_size / 2;

        for (int ky = -half_kernel; ky <= half_kernel; ky++) {
            for (int kx = -half_kernel; kx <= half_kernel; kx++) {
                int input_idx = (y + ky) * width + (x + kx);
                int kernel_idx = (ky + half_kernel) * kernel_size + (kx + half_kernel);

                sum += input[input_idx] * kernel[kernel_idx];
            }
        }

        output[y * width + x] = (unsigned char)fminf(255.0f, fmaxf(0.0f, sum));
    }
}
```

## Isaac ROS Image Processing Pipeline

### isaac_ros_image_pipeline Architecture

The Isaac ROS image processing pipeline:

#### Node Structure
```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>

class IsaacROSImageProcessor : public rclcpp::Node {
public:
    IsaacROSImageProcessor() : Node("isaac_ros_image_processor") {
        // Initialize GPU context
        cudaSetDevice(0);
        cudaFree(0); // Initialize context

        // Create image transport
        it_ = std::make_shared<image_transport::ImageTransport>(this);

        // Create subscriber and publisher
        sub_ = it_->subscribe("input_image", 10,
                             &IsaacROSImageProcessor::image_callback, this);
        pub_ = it_->advertise("output_image", 10);

        // Initialize GPU processor
        gpu_processor_ = std::make_unique<GPUImageProcessor>(640, 480);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Process image on GPU
        cv::Mat processed_image;
        gpu_processor_->process_image(cv_ptr->image, processed_image);

        // Convert back to ROS image
        cv_bridge::CvImage out_msg;
        out_msg.header = msg->header;
        out_msg.encoding = sensor_msgs::image_encodings::RGB8;
        out_msg.image = processed_image;

        pub_.publish(out_msg.toImageMsg());
    }

    std::shared_ptr<image_transport::ImageTransport> it_;
    image_transport::Subscriber sub_;
    image_transport::Publisher pub_;
    std::unique_ptr<GPUImageProcessor> gpu_processor_;
};
```

### Advanced Image Processing Operations

#### Feature Detection and Description
```cpp
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>

class GPUFeatureProcessor {
public:
    GPUFeatureProcessor(int max_features = 1000)
        : max_features_(max_features) {

        // Initialize GPU feature detectors
        orb_detector_ = cv::cuda::ORB::create(max_features_);
        surf_detector_ = cv::cuda::SURF_CUDA::create(400);
        harris_detector_ = cv::cuda::FastFeatureDetector::create();
    }

    void detect_and_compute_features(const cv::Mat& image,
                                   std::vector<cv::KeyPoint>& keypoints,
                                   cv::Mat& descriptors) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_image, gpu_gray;
        gpu_image.upload(image);

        // Convert to grayscale on GPU
        cv::cuda::cvtColor(gpu_image, gpu_gray, cv::COLOR_BGR2GRAY);

        // Detect features on GPU
        cv::cuda::GpuMat gpu_keypoints, gpu_descriptors;

        // Use ORB detector (works well on GPU)
        orb_detector_->detectAndCompute(gpu_gray, cv::cuda::GpuMat(),
                                      gpu_keypoints, gpu_descriptors);

        // Download results
        gpu_keypoints.download(keypoints);
        gpu_descriptors.download(descriptors);
    }

    void compute_descriptors_gpu(const cv::Mat& image,
                                const std::vector<cv::KeyPoint>& keypoints,
                                cv::Mat& descriptors) {
        // Upload image and keypoints to GPU
        cv::cuda::GpuMat gpu_image;
        gpu_image.upload(image);

        cv::cuda::GpuMat gpu_keypoints;
        cv::Mat keypoints_mat(keypoints);
        gpu_keypoints.upload(keypoints_mat);

        // Compute descriptors
        cv::cuda::GpuMat gpu_descriptors;
        orb_detector_->computeDescriptors(gpu_image, gpu_keypoints, gpu_descriptors);

        // Download descriptors
        gpu_descriptors.download(descriptors);
    }

private:
    int max_features_;
    cv::Ptr<cv::cuda::ORB> orb_detector_;
    cv::Ptr<cv::cuda::SURF_CUDA> surf_detector_;
    cv::Ptr<cv::cuda::FastFeatureDetector> harris_detector_;
};
```

## Image Enhancement and Filtering

### GPU-Accelerated Image Enhancement

#### Histogram Equalization
```cpp
class GPUHistogramProcessor {
public:
    void histogram_equalization_gpu(const cv::Mat& input, cv::Mat& output) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);

        // Convert to grayscale if needed
        cv::cuda::GpuMat gpu_gray;
        if (input.channels() > 1) {
            cv::cuda::cvtColor(gpu_input, gpu_gray, cv::COLOR_BGR2GRAY);
        } else {
            gpu_gray = gpu_input;
        }

        // Perform histogram equalization
        cv::cuda::equalizeHist(gpu_gray, gpu_output);

        // Download result
        gpu_output.download(output);
    }

    void adaptive_histogram_equalization_gpu(const cv::Mat& input, cv::Mat& output) {
        // Create CLAHE (Contrast Limited Adaptive Histogram Equalization)
        cv::Ptr<cv::cuda::CLAHE> clahe = cv::cuda::createCLAHE();
        clahe->setClipLimit(4.0);
        clahe->setTilesGridSize(cv::Size(8, 8));

        // Upload image to GPU
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);

        // Apply CLAHE
        clahe->apply(gpu_input, gpu_output);

        // Download result
        gpu_output.download(output);
    }
};
```

#### Noise Reduction
```cpp
class GPUNoiseReduction {
public:
    GPUNoiseReduction() {
        // Initialize CUDA streams for asynchronous processing
        cudaStreamCreate(&processing_stream_);
    }

    void denoise_image_gpu(const cv::Mat& input, cv::Mat& output) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);

        // Apply non-local means denoising
        cv::cuda::fastNlMeansDenoisingColored(gpu_input, gpu_output,
                                            3, 3, 7, 21);

        // Download result
        gpu_output.download(output);
    }

    void bilateral_filter_gpu(const cv::Mat& input, cv::Mat& output) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);

        // Apply bilateral filter
        cv::cuda::bilateralFilter(gpu_input, gpu_output,
                                15, 80, 80, cv::BORDER_DEFAULT);

        // Download result
        gpu_output.download(output);
    }

    void total_variation_denoising_gpu(const cv::Mat& input, cv::Mat& output) {
        // For total variation, we'll implement a custom CUDA kernel
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);

        // Launch TV denoising kernel
        launch_tv_denoising_kernel(gpu_input, gpu_output);

        gpu_output.download(output);
    }

private:
    void launch_tv_denoising_kernel(const cv::cuda::GpuMat& input,
                                   cv::cuda::GpuMat& output) {
        // Implementation of total variation denoising using CUDA
        // This is a simplified version - full implementation would be more complex
    }

    cudaStream_t processing_stream_;
};
```

## Color Space Transformations

### GPU-Accelerated Color Processing

#### Color Space Conversions
```cpp
class GPUColorProcessor {
public:
    void convert_color_spaces_gpu(const cv::Mat& input, cv::Mat& output,
                                 int conversion_code) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);

        // Perform color space conversion
        cv::cuda::cvtColor(gpu_input, gpu_output, conversion_code);

        // Download result
        gpu_output.download(output);
    }

    void rgb_to_hsv_parallel(const cv::Mat& input, cv::Mat& output) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);

        // Convert RGB to HSV
        cv::cuda::cvtColor(gpu_input, gpu_output, cv::COLOR_BGR2HSV);

        gpu_output.download(output);
    }

    void custom_color_transformation_gpu(const cv::Mat& input, cv::Mat& output,
                                        const cv::Mat& transformation_matrix) {
        // Upload data to GPU
        cv::cuda::GpuMat gpu_input, gpu_output;
        cv::cuda::GpuMat gpu_transform;

        gpu_input.upload(input);
        gpu_transform.upload(transformation_matrix);

        // Apply custom color transformation
        apply_custom_transform_kernel(gpu_input, gpu_output, gpu_transform);

        gpu_output.download(output);
    }

private:
    void apply_custom_transform_kernel(const cv::cuda::GpuMat& input,
                                      cv::cuda::GpuMat& output,
                                      const cv::cuda::GpuMat& transform) {
        // Launch custom CUDA kernel for color transformation
        dim3 block_size(16, 16);
        dim3 grid_size((input.cols + block_size.x - 1) / block_size.x,
                      (input.rows + block_size.y - 1) / block_size.y);

        // Custom transformation kernel
        custom_color_transform_kernel<<<grid_size, block_size>>>(
            input.ptr(), output.ptr(), transform.ptr(),
            input.cols, input.rows
        );
    }
};

// CUDA kernel for custom color transformation
__global__ void custom_color_transform_kernel(
    const unsigned char* input,
    unsigned char* output,
    const float* transform,
    int width, int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        int rgb_idx = idx * 3;

        // Apply 3x3 transformation matrix
        float r = input[rgb_idx];
        float g = input[rgb_idx + 1];
        float b = input[rgb_idx + 2];

        // Transform colors
        float new_r = transform[0] * r + transform[1] * g + transform[2] * b;
        float new_g = transform[3] * r + transform[4] * g + transform[5] * b;
        float new_b = transform[6] * r + transform[7] * g + transform[8] * b;

        output[rgb_idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, new_r));
        output[rgb_idx + 1] = (unsigned char)fminf(255.0f, fmaxf(0.0f, new_g));
        output[rgb_idx + 2] = (unsigned char)fminf(255.0f, fmaxf(0.0f, new_b));
    }
}
```

## Edge Detection and Feature Extraction

### GPU-Accelerated Edge Detection

#### Canny Edge Detection
```cpp
class GPUEdgeDetector {
public:
    GPUEdgeDetector() {
        // Initialize GPU resources
        initialize_gpu_resources();
    }

    void canny_edge_detection_gpu(const cv::Mat& input, cv::Mat& output,
                                 double low_threshold, double high_threshold) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);

        // Convert to grayscale if needed
        cv::cuda::GpuMat gpu_gray;
        if (input.channels() > 1) {
            cv::cuda::cvtColor(gpu_input, gpu_gray, cv::COLOR_BGR2GRAY);
        } else {
            gpu_gray = gpu_input;
        }

        // Apply Canny edge detection
        cv::cuda::Canny(gpu_gray, gpu_output, low_threshold, high_threshold);

        // Download result
        gpu_output.download(output);
    }

    void sobel_edge_detection_gpu(const cv::Mat& input, cv::Mat& output) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);

        // Convert to grayscale
        cv::cuda::GpuMat gpu_gray;
        if (input.channels() > 1) {
            cv::cuda::cvtColor(gpu_input, gpu_gray, cv::COLOR_BGR2GRAY);
        } else {
            gpu_gray = gpu_input;
        }

        // Apply Sobel operator
        cv::cuda::GpuMat grad_x, grad_y;
        cv::cuda::Sobel(gpu_gray, grad_x, CV_32F, 1, 0, 3);
        cv::cuda::Sobel(gpu_gray, grad_y, CV_32F, 0, 1, 3);

        // Calculate gradient magnitude
        cv::cuda::magnitude(grad_x, grad_y, gpu_output);

        // Convert back to 8-bit
        gpu_output.convertTo(gpu_output, CV_8U);

        gpu_output.download(output);
    }

    void hough_transform_gpu(const cv::Mat& input,
                            std::vector<cv::Vec2f>& lines) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_input;
        gpu_input.upload(input);

        // Perform Hough transform
        cv::cuda::GpuMat gpu_lines;
        cv::Ptr<cv::cuda::HoughLinesDetector> hough_detector =
            cv::cuda::createHoughLinesDetector(1, CV_PI/180, 100);

        hough_detector->detect(gpu_input, gpu_lines);

        // Download results
        cv::Mat cpu_lines;
        gpu_lines.download(cpu_lines);

        // Convert to vector of Vec2f
        for (int i = 0; i < cpu_lines.rows; i++) {
            float rho = cpu_lines.at<float>(i, 0);
            float theta = cpu_lines.at<float>(i, 1);
            lines.push_back(cv::Vec2f(rho, theta));
        }
    }

private:
    void initialize_gpu_resources() {
        // Initialize any required GPU resources
        cudaStreamCreate(&processing_stream_);
    }

    cudaStream_t processing_stream_;
};
```

## Morphological Operations

### GPU-Accelerated Morphological Processing

#### Morphological Transformations
```cpp
class GPUMorphologicalProcessor {
public:
    GPUMorphologicalProcessor() {
        // Create default structuring element
        structuring_element_ = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(5, 5)
        );
    }

    void morphological_operations_gpu(const cv::Mat& input, cv::Mat& output,
                                     int operation, cv::Mat kernel = cv::Mat()) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);

        // Upload or use default kernel
        cv::cuda::GpuMat gpu_kernel;
        if (kernel.empty()) {
            gpu_kernel.upload(structuring_element_);
        } else {
            gpu_kernel.upload(kernel);
        }

        // Apply morphological operation
        cv::cuda::morphologyEx(gpu_input, gpu_output, operation, gpu_kernel);

        // Download result
        gpu_output.download(output);
    }

    void morphological_gradient_gpu(const cv::Mat& input, cv::Mat& output) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);

        // Create structuring element
        cv::cuda::GpuMat gpu_kernel;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        gpu_kernel.upload(kernel);

        // Calculate morphological gradient: dilate - erode
        cv::cuda::GpuMat dilated, eroded;
        cv::cuda::dilate(gpu_input, dilated, gpu_kernel);
        cv::cuda::erode(gpu_input, eroded, gpu_kernel);

        cv::cuda::subtract(dilated, eroded, gpu_output);

        gpu_output.download(output);
    }

    void top_hat_transform_gpu(const cv::Mat& input, cv::Mat& output) {
        // Upload image to GPU
        cv::cuda::GpuMat gpu_input, gpu_output;
        gpu_input.upload(input);

        // Create structuring element
        cv::cuda::GpuMat gpu_kernel;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
        gpu_kernel.upload(kernel);

        // Top-hat = original - open
        cv::cuda::GpuMat opened;
        cv::cuda::morphologyEx(gpu_input, opened, cv::MORPH_OPEN, gpu_kernel);

        cv::cuda::subtract(gpu_input, opened, gpu_output);

        gpu_output.download(output);
    }

private:
    cv::Mat structuring_element_;
};
```

## Performance Optimization

### Memory Optimization Techniques

#### Memory Pool Management
```cpp
class GPUMemoryOptimizer {
public:
    static GPUMemoryOptimizer& get_instance() {
        static GPUMemoryOptimizer instance;
        return instance;
    }

    void* get_memory(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Look for suitable block in free list
        auto it = free_blocks_.lower_bound(size);
        if (it != free_blocks_.end()) {
            void* ptr = it->second;
            size_t actual_size = it->first;
            free_blocks_.erase(it);

            allocated_blocks_[ptr] = actual_size;
            return ptr;
        }

        // Allocate new block
        void* new_ptr;
        cudaMalloc(&new_ptr, size);
        allocated_blocks_[new_ptr] = size;

        return new_ptr;
    }

    void return_memory(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocated_blocks_.find(ptr);
        if (it != allocated_blocks_.end()) {
            size_t size = it->second;
            allocated_blocks_.erase(it);

            // Add to free list if pool isn't too large
            if (free_blocks_.size() < max_pool_size_) {
                free_blocks_[size] = ptr;
            } else {
                cudaFree(ptr);
            }
        }
    }

private:
    GPUMemoryOptimizer() = default;
    ~GPUMemoryOptimizer() {
        // Clean up all allocated memory
        for (auto& pair : allocated_blocks_) {
            cudaFree(pair.first);
        }
        for (auto& pair : free_blocks_) {
            cudaFree(pair.second);
        }
    }

    std::map<size_t, void*> free_blocks_;  // Size -> pointer map
    std::unordered_map<void*, size_t> allocated_blocks_;
    std::mutex mutex_;
    const size_t max_pool_size_ = 50;
};
```

### Stream Management

#### Asynchronous Processing with Streams
```cpp
class GPUStreamManager {
public:
    GPUStreamManager(int num_streams = 4) : num_streams_(num_streams) {
        streams_.resize(num_streams_);
        events_.resize(num_streams_);

        for (int i = 0; i < num_streams_; i++) {
            cudaStreamCreate(&streams_[i]);
            cudaEventCreate(&events_[i]);
        }
    }

    ~GPUStreamManager() {
        for (int i = 0; i < num_streams_; i++) {
            cudaStreamDestroy(streams_[i]);
            cudaEventDestroy(events_[i]);
        }
    }

    cudaStream_t get_stream(int index) {
        return streams_[index % num_streams_];
    }

    void record_event(int index) {
        cudaEventRecord(events_[index % num_streams_],
                       streams_[index % num_streams_]);
    }

    void wait_for_event(int index) {
        cudaStreamWaitEvent(streams_[(index + 1) % num_streams_],
                           events_[index % num_streams_], 0);
    }

private:
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    int num_streams_;
};

// Pipeline processing using streams
class GPUPipelineProcessor {
public:
    GPUPipelineProcessor() : stream_manager_(3), current_stream_(0) {
        // Initialize GPU memory for pipeline stages
        initialize_pipeline_memory();
    }

    void process_frame_pipeline(const cv::Mat& input, cv::Mat& output) {
        int stream_id = current_stream_++;
        cudaStream_t stream = stream_manager_.get_stream(stream_id);

        // Stage 1: Preprocessing
        cv::cuda::GpuMat gpu_input, gpu_preprocessed;
        gpu_input.upload(input, stream);
        preprocess_gpu(gpu_input, gpu_preprocessed, stream);

        // Record event after preprocessing
        stream_manager_.record_event(stream_id);

        // Stage 2: Processing (wait for preprocessing)
        stream_manager_.wait_for_event(stream_id);
        cv::cuda::GpuMat gpu_processed;
        process_gpu(gpu_preprocessed, gpu_processed, stream);

        // Stage 3: Post-processing
        postprocess_gpu(gpu_processed, stream);

        // Download result
        gpu_processed.download(output);
    }

private:
    void preprocess_gpu(const cv::cuda::GpuMat& input,
                       cv::cuda::GpuMat& output,
                       cudaStream_t stream) {
        // Apply preprocessing operations
        cv::cuda::cvtColor(input, output, cv::COLOR_BGR2GRAY, 0, stream);
    }

    void process_gpu(const cv::cuda::GpuMat& input,
                    cv::cuda::GpuMat& output,
                    cudaStream_t stream) {
        // Main processing operations
        cv::cuda::GaussianBlur(input, output, cv::Size(5, 5), 0, 0, cv::BORDER_DEFAULT, stream);
    }

    void postprocess_gpu(const cv::cuda::GpuMat& input,
                        cudaStream_t stream) {
        // Post-processing operations
        // Could include normalization, color conversion, etc.
    }

    void initialize_pipeline_memory() {
        // Initialize any required GPU memory for pipeline
    }

    GPUStreamManager stream_manager_;
    int current_stream_;
};
```

## Isaac ROS Integration

### Isaac ROS Image Pipeline Nodes

#### Complete Image Processing Pipeline
```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

class IsaacROSImagePipeline : public rclcpp::Node {
public:
    IsaacROSImagePipeline() : Node("isaac_ros_image_pipeline") {
        // Initialize GPU context
        cudaSetDevice(0);
        cudaFree(0);

        // Create image transport
        it_ = std::make_shared<image_transport::ImageTransport>(this);

        // Create subscriber and publisher
        sub_ = it_->subscribe("input_image", 10,
                             std::bind(&IsaacROSImagePipeline::image_callback, this,
                                      std::placeholders::_1));
        pub_ = it_->advertise("processed_image", 10);

        // Initialize processing components
        color_processor_ = std::make_unique<GPUColorProcessor>();
        edge_detector_ = std::make_unique<GPUEdgeDetector>();
        morph_processor_ = std::make_unique<GPUMorphologicalProcessor>();
        noise_reducer_ = std::make_unique<GPUNoiseReduction>();

        RCLCPP_INFO(this->get_logger(), "Isaac ROS Image Pipeline initialized");
    }

private:
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat processed_image = cv_ptr->image.clone();

        // Apply image processing pipeline
        try {
            // Noise reduction
            noise_reducer_->denoise_image_gpu(processed_image, processed_image);

            // Color processing
            cv::Mat color_processed;
            color_processor_->convert_color_spaces_gpu(
                processed_image, color_processed, cv::COLOR_BGR2LAB
            );
            processed_image = color_processed;

            // Edge detection (optional, based on parameter)
            if (enable_edge_detection_) {
                cv::Mat edge_map;
                edge_detector_->canny_edge_detection_gpu(
                    processed_image, edge_map, 50, 150
                );
                // Combine with original if needed
            }

            // Morphological operations
            morph_processor_->morphological_operations_gpu(
                processed_image, processed_image, cv::MORPH_CLOSE
            );

        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Processing error: %s", e.what());
            return;
        }

        // Convert back to ROS message
        cv_bridge::CvImage out_msg;
        out_msg.header = msg->header;
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = processed_image;

        pub_.publish(out_msg.toImageMsg());

        // Calculate processing time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time
        ).count();

        RCLCPP_DEBUG(this->get_logger(), "Processing time: %.2f ms", duration / 1000.0);
    }

    std::shared_ptr<image_transport::ImageTransport> it_;
    image_transport::Subscriber sub_;
    image_transport::Publisher pub_;

    std::unique_ptr<GPUColorProcessor> color_processor_;
    std::unique_ptr<GPUEdgeDetector> edge_detector_;
    std::unique_ptr<GPUMorphologicalProcessor> morph_processor_;
    std::unique_ptr<GPUNoiseReduction> noise_reducer_;

    bool enable_edge_detection_ = true;
};
```

GPU image processing in Isaac ROS provides the computational power needed for real-time computer vision in robotics applications. By leveraging GPU acceleration for common image processing operations, these systems can achieve the performance required for demanding robotics tasks while maintaining the flexibility of the ROS ecosystem.