---
sidebar_position: 2
title: "Hardware Accelerated Perception"
description: "GPU-accelerated computer vision and perception in Isaac ROS"
---

# Hardware Accelerated Perception

Hardware accelerated perception is the cornerstone of Isaac ROS, enabling real-time processing of complex sensor data through GPU acceleration. This capability allows robotics applications to perform computationally intensive perception tasks that would be impossible with CPU-only processing.

## GPU-Accelerated Computer Vision

### CUDA Integration

Isaac ROS leverages CUDA for direct GPU acceleration:

#### CUDA Kernel Integration
```cpp
#include <cuda_runtime.h>
#include <npp.h>  // NVIDIA Performance Primitives
#include <opencv2/opencv.hpp>

// Example CUDA kernel for image processing
__global__ void sobel_edge_detection_kernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        // Sobel operator for edge detection
        int idx = y * width + x;

        // Calculate gradients
        int gx = input[idx - width - 1] - input[idx - width + 1] +
                 2 * input[idx - 1] - 2 * input[idx + 1] +
                 input[idx + width - 1] - input[idx + width + 1];

        int gy = input[idx - width - 1] + 2 * input[idx - width] + input[idx - width + 1] -
                 input[idx + width - 1] - 2 * input[idx + width] - input[idx + width + 1];

        // Calculate magnitude
        int magnitude = sqrtf(gx * gx + gy * gy);
        output[idx] = min(255, magnitude);
    }
}

// Isaac ROS node with GPU acceleration
class GPUImageProcessor {
public:
    GPUImageProcessor() {
        // Initialize CUDA context
        cudaSetDevice(0);
        cudaFree(0); // Initialize context

        // Allocate GPU memory
        cudaMalloc(&d_input, width * height * sizeof(unsigned char));
        cudaMalloc(&d_output, width * height * sizeof(unsigned char));
    }

    void process_image(const cv::Mat& input, cv::Mat& output) {
        // Copy image to GPU
        cudaMemcpy(d_input, input.data, input.total() * sizeof(unsigned char), cudaMemcpyHostToDevice);

        // Launch CUDA kernel
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                       (height + block_size.y - 1) / block_size.y);

        sobel_edge_detection_kernel<<<grid_size, block_size>>>(d_input, d_output, width, height);

        // Wait for completion
        cudaDeviceSynchronize();

        // Copy result back to CPU
        cudaMemcpy(output.data, d_output, output.total() * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    }

private:
    unsigned char* d_input;
    unsigned char* d_output;
    int width = 640;
    int height = 480;
};
```

### NVIDIA Performance Primitives (NPP)

Using optimized primitives for common operations:

#### Image Processing with NPP
```cpp
#include <npp.h>
#include <nppi.h>

class NPPImageProcessor {
public:
    void resize_image(const cv::Mat& src, cv::Mat& dst) {
        // Allocate GPU memory
        Npp8u* d_src;
        Npp8u* d_dst;

        cudaMalloc(&d_src, src.rows * src.cols * sizeof(Npp8u));
        cudaMalloc(&d_dst, dst.rows * dst.cols * sizeof(Npp8u));

        // Copy source image to GPU
        cudaMemcpy(d_src, src.data, src.rows * src.cols * sizeof(Npp8u), cudaMemcpyHostToDevice);

        // Perform resize using NPP
        NppiSize src_size = {src.cols, src.rows};
        NppiSize dst_size = {dst.cols, dst.rows};
        NppiRect src_rect = {0, 0, src.cols, src.rows};

        nppiResize_8u_C1R(
            d_src, src.cols * sizeof(Npp8u), src_size,
            src_rect,
            d_dst, dst.cols * sizeof(Npp8u), dst_size,
            0.0, 0.0, NPPI_INTER_LINEAR
        );

        // Copy result back to CPU
        cudaMemcpy(dst.data, d_dst, dst.rows * dst.cols * sizeof(Npp8u), cudaMemcpyDeviceToHost);

        // Free GPU memory
        cudaFree(d_src);
        cudaFree(d_dst);
    }

    void gaussian_filter(const cv::Mat& src, cv::Mat& dst) {
        // Allocate GPU memory
        Npp8u* d_src;
        Npp8u* d_dst;

        cudaMalloc(&d_src, src.rows * src.cols * sizeof(Npp8u));
        cudaMalloc(&d_dst, dst.rows * dst.cols * sizeof(Npp8u));

        cudaMemcpy(d_src, src.data, src.rows * src.cols * sizeof(Npp8u), cudaMemcpyHostToDevice);

        // Apply Gaussian filter
        NppiSize roi_size = {src.cols, src.rows};
        NppiSize mask_size = {5, 5}; // 5x5 Gaussian kernel
        Npp32s anchor = {2, 2}; // Center of kernel

        nppiFilterGauss_8u_C1R(
            d_src, src.cols * sizeof(Npp8u),
            d_dst, dst.cols * sizeof(Npp8u),
            roi_size, NPP_MASK_SIZE_5_X_5
        );

        cudaMemcpy(dst.data, d_dst, dst.rows * dst.cols * sizeof(Npp8u), cudaMemcpyDeviceToHost);

        cudaFree(d_src);
        cudaFree(d_dst);
    }
};
```

## Isaac ROS Perception Nodes

### GPU-Accelerated Image Processing Pipeline

The Isaac ROS image processing pipeline:

#### isaac_ros_image_pipeline
```yaml
# Example launch file for GPU-accelerated image pipeline
image_pipeline:
  ros__parameters:
    # Image rectification parameters
    rectification:
      use_gpu: true
      output_width: 640
      output_height: 480
      processing_rate: 60  # Hz

    # Image enhancement parameters
    enhancement:
      use_gpu: true
      brightness: 0.0
      contrast: 1.0
      saturation: 1.0

    # Feature detection parameters
    feature_detection:
      use_gpu: true
      max_features: 1000
      quality_level: 0.01
      min_distance: 10.0
```

#### Node Configuration
```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cuda_runtime.h>

class GPUImageNode : public rclcpp::Node {
public:
    GPUImageNode() : Node("gpu_image_node") {
        // Initialize CUDA
        cudaSetDevice(0);
        cudaFree(0); // Initialize context

        // Create subscription
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "input_image", 10,
            std::bind(&GPUImageNode::image_callback, this, std::placeholders::_1)
        );

        // Create publisher
        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("output_image", 10);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        // Convert ROS image to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        // Process image on GPU
        cv::Mat processed_image;
        process_on_gpu(cv_ptr->image, processed_image);

        // Convert back to ROS image
        cv_bridge::CvImage out_msg;
        out_msg.header = msg->header;
        out_msg.encoding = sensor_msgs::image_encodings::BGR8;
        out_msg.image = processed_image;

        image_pub_->publish(*out_msg.toImageMsg());
    }

    void process_on_gpu(const cv::Mat& input, cv::Mat& output) {
        output = input.clone(); // Placeholder - actual GPU processing would happen here

        // Example: Convert to grayscale on GPU
        cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);

        // In real implementation, this would involve CUDA kernels
        // or NPP functions for GPU acceleration
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
};
```

### Stereo Processing Acceleration

GPU-accelerated stereo vision:

#### isaac_ros_stereo_image_rectification
```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <stereo_msgs/msg/disparity_image.hpp>

class GPUStereoRectifier {
public:
    GPUStereoRectifier() {
        // Initialize GPU memory for stereo processing
        initialize_gpu_memory();
    }

    void process_stereo_pair(
        const cv::Mat& left_image,
        const cv::Mat& right_image,
        cv::Mat& left_rectified,
        cv::Mat& right_rectified,
        cv::Mat& disparity
    ) {
        // Allocate GPU memory
        allocate_gpu_memory(left_image.size());

        // Copy images to GPU
        copy_images_to_gpu(left_image, right_image);

        // Perform stereo rectification on GPU
        perform_rectification_gpu();

        // Compute disparity on GPU
        compute_disparity_gpu();

        // Copy results back to CPU
        copy_results_to_cpu(left_rectified, right_rectified, disparity);
    }

private:
    void initialize_gpu_memory() {
        // Initialize GPU memory pools and CUDA streams
        cudaStreamCreate(&stream_);

        // Create CUDA events for synchronization
        cudaEventCreate(&start_event_);
        cudaEventCreate(&stop_event_);
    }

    void allocate_gpu_memory(cv::Size image_size) {
        size_t image_bytes = image_size.area() * sizeof(unsigned char);

        if (d_left_image_ && image_size_ != image_size) {
            cudaFree(d_left_image_);
            cudaFree(d_right_image_);
            cudaFree(d_left_rectified_);
            cudaFree(d_right_rectified_);
            cudaFree(d_disparity_);
        }

        if (!d_left_image_) {
            cudaMalloc(&d_left_image_, image_bytes);
            cudaMalloc(&d_right_image_, image_bytes);
            cudaMalloc(&d_left_rectified_, image_bytes);
            cudaMalloc(&d_right_rectified_, image_bytes);
            cudaMalloc(&d_disparity_, image_size.area() * sizeof(float));
            image_size_ = image_size;
        }
    }

    cudaStream_t stream_;
    cudaEvent_t start_event_, stop_event_;

    unsigned char* d_left_image_;
    unsigned char* d_right_image_;
    unsigned char* d_left_rectified_;
    unsigned char* d_right_rectified_;
    float* d_disparity_;

    cv::Size image_size_;
};
```

## Deep Learning Integration

### TensorRT Acceleration

Isaac ROS integrates with TensorRT for optimized neural network inference:

#### Object Detection with TensorRT
```cpp
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

class TensorRTObjectDetector {
public:
    TensorRTObjectDetector(const std::string& engine_path) {
        // Load TensorRT engine
        load_engine(engine_path);

        // Allocate GPU memory for input and output
        allocate_buffers();
    }

    std::vector<Detection> detect_objects(const cv::Mat& input_image) {
        // Preprocess image
        cv::Mat preprocessed = preprocess_image(input_image);

        // Copy image to GPU
        cudaMemcpy(d_input_, preprocessed.data, input_size_, cudaMemcpyHostToDevice);

        // Perform inference
        context_->executeV2(bindings_.data());

        // Copy results back
        cudaMemcpy(h_output_, d_output_, output_size_, cudaMemcpyDeviceToHost);

        // Post-process detections
        return post_process_detections();
    }

private:
    void load_engine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open engine file");
        }

        // Read engine file
        std::vector<char> engine_data((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());

        // Create runtime and deserialize engine
        runtime_ = nvinfer1::createInferRuntime(g_logger_);
        engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
        context_ = engine_->createExecutionContext();
    }

    void allocate_buffers() {
        // Get binding information
        int num_bindings = engine_->getNbBindings();

        for (int i = 0; i < num_bindings; ++i) {
            auto dims = engine_->getBindingDimensions(i);
            size_t size = 1;
            for (int j = 0; j < dims.nbDims; ++j) {
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

    cv::Mat preprocess_image(const cv::Mat& image) {
        // Resize and normalize image
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(640, 640));

        cv::Mat normalized;
        resized.convertTo(normalized, CV_32F, 1.0 / 255.0);

        // Convert BGR to RGB and change data layout
        std::vector<cv::Mat> channels(3);
        cv::split(normalized, channels);

        cv::Mat input_tensor(1, 3 * 640 * 640, CV_32F);
        float* data = (float*)input_tensor.data;

        for (int c = 0; c < 3; ++c) {
            memcpy(data + c * 640 * 640, channels[c].data, 640 * 640 * sizeof(float));
        }

        return input_tensor;
    }

    std::vector<Detection> post_process_detections() {
        // Parse detection results from TensorRT output
        float* output = (float*)h_output_;
        std::vector<Detection> detections;

        // Parse bounding boxes, confidence scores, and class labels
        // Implementation depends on specific model output format

        return detections;
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

### Isaac ROS DNN Inference

Using Isaac ROS for deep learning inference:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from isaac_ros_tensor_rt.tensor_rt_engine import TensorRTEngine

class IsaacROSObjectDetector(Node):
    def __init__(self):
        super().__init__('isaac_ros_object_detector')

        # Create TensorRT engine
        self.tensor_rt_engine = TensorRTEngine(
            engine_path='/path/to/model.plan',
            input_binding_name='input',
            output_binding_name='output'
        )

        # Create subscription and publisher
        self.subscription = self.create_subscription(
            Image,
            'input_image',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(Image, 'output_image', 10)

    def image_callback(self, msg):
        # Convert ROS image to tensor
        input_tensor = self.ros_image_to_tensor(msg)

        # Perform inference with TensorRT
        output_tensor = self.tensor_rt_engine.infer(input_tensor)

        # Process results
        detections = self.process_detections(output_tensor)

        # Publish results
        self.publish_detections(detections)

    def ros_image_to_tensor(self, image_msg):
        """Convert ROS image message to tensor for TensorRT"""
        # Implementation for converting ROS image to tensor format
        pass

    def process_detections(self, output_tensor):
        """Process TensorRT output to detection results"""
        # Implementation for processing detection results
        pass

    def publish_detections(self, detections):
        """Publish detection results"""
        # Implementation for publishing results
        pass
```

## Performance Optimization

### Memory Management

Efficient GPU memory management in Isaac ROS:

#### Memory Pool Management
```cpp
#include <cuda_runtime.h>
#include <unordered_map>
#include <memory>

class GPUMemoryManager {
public:
    static GPUMemoryManager& get_instance() {
        static GPUMemoryManager instance;
        return instance;
    }

    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Check if we have a suitable block in the pool
        auto it = free_blocks_.lower_bound(size);
        if (it != free_blocks_.end()) {
            void* ptr = it->second;
            free_blocks_.erase(it);
            allocated_blocks_[ptr] = size;
            return ptr;
        }

        // Allocate new block
        void* ptr;
        cudaMalloc(&ptr, size);
        allocated_blocks_[ptr] = size;
        return ptr;
    }

    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocated_blocks_.find(ptr);
        if (it != allocated_blocks_.end()) {
            size_t size = it->second;
            allocated_blocks_.erase(it);

            // Add to free pool (with size limit to prevent excessive memory usage)
            if (free_blocks_.size() < max_pool_size_) {
                free_blocks_[size] = ptr;
            } else {
                cudaFree(ptr);
            }
        }
    }

private:
    GPUMemoryManager() = default;
    ~GPUMemoryManager() {
        // Clean up all allocated memory
        for (auto& pair : allocated_blocks_) {
            cudaFree(pair.first);
        }
        for (auto& pair : free_blocks_) {
            cudaFree(pair.second);
        }
    }

    std::unordered_map<void*, size_t> allocated_blocks_;
    std::map<size_t, void*> free_blocks_; // Ordered by size for efficient allocation
    std::mutex mutex_;
    const size_t max_pool_size_ = 100; // Maximum blocks in pool
};
```

### Stream Synchronization

Managing CUDA streams for optimal performance:

#### Asynchronous Processing
```cpp
#include <cuda_runtime.h>

class CUDAStreamManager {
public:
    CUDAStreamManager(int num_streams = 4) : num_streams_(num_streams) {
        streams_.resize(num_streams_);
        events_.resize(num_streams_);

        for (int i = 0; i < num_streams_; ++i) {
            cudaStreamCreate(&streams_[i]);
            cudaEventCreate(&events_[i]);
        }
    }

    ~CUDAStreamManager() {
        for (int i = 0; i < num_streams_; ++i) {
            cudaStreamDestroy(streams_[i]);
            cudaEventDestroy(events_[i]);
        }
    }

    cudaStream_t get_stream(int index) {
        return streams_[index % num_streams_];
    }

    cudaEvent_t get_event(int index) {
        return events_[index % num_streams_];
    }

    void synchronize_stream(int index) {
        cudaStreamSynchronize(streams_[index % num_streams_]);
    }

    void record_event(int index) {
        cudaEventRecord(events_[index % num_streams_], streams_[index % num_streams_]);
    }

private:
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    int num_streams_;
};

// Example usage in Isaac ROS node
class AsyncGPUImageNode : public rclcpp::Node {
public:
    AsyncGPUImageNode() : Node("async_gpu_image_node"),
                         stream_manager_(4),
                         current_stream_(0) {
        // Initialize GPU memory with streams
        initialize_gpu_memory();
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        int stream_id = current_stream_++;

        // Get stream for this processing
        cudaStream_t stream = stream_manager_.get_stream(stream_id);

        // Process image asynchronously on GPU
        process_image_async(msg, stream, stream_id);

        // Record event for synchronization
        stream_manager_.record_event(stream_id);
    }

    void process_image_async(const sensor_msgs::msg::Image::SharedPtr msg,
                            cudaStream_t stream, int stream_id) {
        // Copy image to GPU asynchronously
        cudaMemcpyAsync(d_input_, h_input_, image_size_,
                       cudaMemcpyHostToDevice, stream);

        // Launch kernel asynchronously
        dim3 block_size(16, 16);
        dim3 grid_size((width_ + block_size.x - 1) / block_size.x,
                      (height_ + block_size.y - 1) / block_size.y);

        image_processing_kernel<<<grid_size, block_size, 0, stream>>>(
            d_input_, d_output_, width_, height_);

        // Copy result back asynchronously
        cudaMemcpyAsync(h_output_, d_output_, image_size_,
                       cudaMemcpyDeviceToHost, stream);
    }

    CUDAStreamManager stream_manager_;
    int current_stream_;
    void* d_input_;
    void* d_output_;
    void* h_input_;
    void* h_output_;
    size_t image_size_;
    int width_, height_;
};
```

## Real-time Performance

### Pipeline Optimization

Creating efficient real-time processing pipelines:

#### Multi-stage Pipeline
```cpp
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>

class RealTimePerceptionPipeline {
public:
    RealTimePerceptionPipeline() {
        // Start processing threads
        rectification_thread_ = std::thread(&RealTimePerceptionPipeline::rectification_worker, this);
        detection_thread_ = std::thread(&RealTimePerceptionPipeline::detection_worker, this);
        tracking_thread_ = std::thread(&RealTimePerceptionPipeline::tracking_worker, this);
    }

    void add_input_image(const cv::Mat& image) {
        std::lock_guard<std::mutex> lock(input_mutex_);
        input_queue_.push(image);
        input_cv_.notify_one();
    }

    std::vector<Detection> get_detections() {
        std::unique_lock<std::mutex> lock(output_mutex_);
        while (output_queue_.empty()) {
            output_cv_.wait(lock);
        }

        auto detections = output_queue_.front();
        output_queue_.pop();
        return detections;
    }

private:
    void rectification_worker() {
        while (running_) {
            cv::Mat image;
            {
                std::unique_lock<std::mutex> lock(input_mutex_);
                input_cv_.wait(lock, [this] { return !input_queue_.empty() || !running_; });

                if (!running_ && input_queue_.empty()) break;

                image = input_queue_.front();
                input_queue_.pop();
            }

            // Perform rectification on GPU
            cv::Mat rectified_image = rectify_on_gpu(image);

            // Pass to next stage
            {
                std::lock_guard<std::mutex> lock(rectified_mutex_);
                rectified_queue_.push(rectified_image);
                rectified_cv_.notify_one();
            }
        }
    }

    void detection_worker() {
        while (running_) {
            cv::Mat image;
            {
                std::unique_lock<std::mutex> lock(rectified_mutex_);
                rectified_cv_.wait(lock, [this] { return !rectified_queue_.empty() || !running_; });

                if (!running_ && rectified_queue_.empty()) break;

                image = rectified_queue_.front();
                rectified_queue_.pop();
            }

            // Perform object detection on GPU
            auto detections = detect_objects_on_gpu(image);

            // Pass to next stage
            {
                std::lock_guard<std::mutex> lock(detection_mutex_);
                detection_queue_.push(detections);
                detection_cv_.notify_one();
            }
        }
    }

    void tracking_worker() {
        while (running_) {
            std::vector<Detection> detections;
            {
                std::unique_lock<std::mutex> lock(detection_mutex_);
                detection_cv_.wait(lock, [this] { return !detection_queue_.empty() || !running_; });

                if (!running_ && detection_queue_.empty()) break;

                detections = detection_queue_.front();
                detection_queue_.pop();
            }

            // Perform tracking on GPU
            auto tracked_objects = track_objects_on_gpu(detections);

            // Output results
            {
                std::lock_guard<std::mutex> lock(output_mutex_);
                output_queue_.push(tracked_objects);
                output_cv_.notify_one();
            }
        }
    }

    // GPU processing functions (simplified)
    cv::Mat rectify_on_gpu(const cv::Mat& input) { return input; } // Placeholder
    std::vector<Detection> detect_objects_on_gpu(const cv::Mat& input) { return {}; } // Placeholder
    std::vector<TrackedObject> track_objects_on_gpu(const std::vector<Detection>& input) { return {}; } // Placeholder

    std::queue<cv::Mat> input_queue_;
    std::queue<cv::Mat> rectified_queue_;
    std::queue<std::vector<Detection>> detection_queue_;
    std::queue<std::vector<TrackedObject>> output_queue_;

    std::mutex input_mutex_, rectified_mutex_, detection_mutex_, output_mutex_;
    std::condition_variable input_cv_, rectified_cv_, detection_cv_, output_cv_;

    std::thread rectification_thread_, detection_thread_, tracking_thread_;
    std::atomic<bool> running_{true};

    int width_ = 640, height_ = 480;
};
```

Hardware accelerated perception in Isaac ROS enables robotics applications to perform complex computer vision and deep learning tasks in real-time, providing the computational power needed for advanced robotics capabilities. The combination of GPU acceleration, optimized libraries, and efficient memory management creates a powerful platform for perception-intensive robotics applications.