---
title: "Building Real-time Data Pipelines"
description: "Creating efficient real-time data pipelines for AI-ROS integration with proper buffering and queue management"
sidebar_position: 4
keywords: [real-time, data pipeline, buffering, queue management, performance, ai, ros2]
---

# Building Real-time Data Pipelines

Real-time data pipelines are critical for AI-ROS integration, especially in applications requiring immediate responses to sensor inputs. This section covers efficient image transport, sensor data synchronization, buffering strategies, and performance optimization techniques for high-frequency AI applications.

## Understanding Real-time Requirements

Real-time systems in robotics have strict timing constraints that must be met for proper operation. In AI-ROS integration, this typically involves:

- **Low-latency inference**: Processing sensor data quickly enough to respond to environmental changes
- **High-frequency data handling**: Managing high-bandwidth sensor streams like cameras or LiDAR
- **Deterministic timing**: Ensuring consistent processing times for predictable behavior
- **Resource management**: Efficiently using computational resources without exceeding limits

## Efficient Image Transport

Images are often the largest data source in AI applications. Efficient transport is crucial for real-time performance.

### Image Transport Options

ROS 2 provides several image transport mechanisms:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class EfficientImageTransportNode(Node):
    def __init__(self):
        super().__init__('efficient_image_transport')

        # Initialize bridge for image conversion
        self.cv_bridge = CvBridge()

        # Publishers for different transport types
        self.raw_image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.compressed_pub = self.create_publisher(CompressedImage, 'camera/image_compressed', 10)

        # Subscribers with different QoS profiles
        self.raw_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.raw_image_callback,
            1  # Use small queue size for real-time
        )
        self.compressed_sub = self.create_subscription(
            CompressedImage,
            'camera/image_compressed',
            self.compressed_image_callback,
            1
        )

        # Timer for image processing
        self.timer = self.create_timer(0.1, self.process_images)  # 10 Hz processing

        self.get_logger().info('Efficient Image Transport Node initialized')

    def process_images(self):
        """Simulate image processing for real-time pipeline"""
        # Generate a test image (in real application, this would come from camera)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Convert to ROS Image message
        ros_image = self.cv_bridge.cv2_to_imgmsg(test_image, encoding='bgr8')
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = 'camera_optical_frame'

        # Publish both raw and compressed versions
        self.raw_image_pub.publish(ros_image)

        # Create compressed image
        compressed_msg = CompressedImage()
        compressed_msg.header = ros_image.header
        compressed_msg.format = 'jpeg'

        # Compress using OpenCV
        _, compressed_data = cv2.imencode('.jpg', test_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        compressed_msg.data = compressed_data.tobytes()

        self.compressed_pub.publish(compressed_msg)

    def raw_image_callback(self, msg):
        """Process raw image message"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform AI processing (example: simple edge detection)
            processed_image = cv2.Canny(cv_image, 100, 200)

            # Log processing time
            self.get_logger().info(f'Processed raw image: {msg.width}x{msg.height}')

        except Exception as e:
            self.get_logger().error(f'Raw image processing error: {e}')

    def compressed_image_callback(self, msg):
        """Process compressed image message"""
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Log processing time
            self.get_logger().info(f'Processed compressed image: {cv_image.shape}')

        except Exception as e:
            self.get_logger().error(f'Compressed image processing error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = EfficientImageTransportNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Efficient Image Transport Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Image Transport Configuration

For optimal image transport, configure QoS profiles appropriately:

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class OptimizedImageTransportNode(Node):
    def __init__(self):
        super().__init__('optimized_image_transport')

        # Define QoS for real-time image transport
        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Allow some packet loss for real-time
            history=QoSHistoryPolicy.KEEP_LAST,  # Keep only latest messages
            depth=1,  # Minimal queue depth for low latency
            durability=QoSDurabilityPolicy.VOLATILE  # Don't keep messages after node dies
        )

        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.real_time_image_callback,
            image_qos
        )

    def real_time_image_callback(self, msg):
        """Process image with real-time constraints"""
        # Start timing for performance measurement
        start_time = self.get_clock().now()

        try:
            # Process image for AI inference
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Perform AI processing
            results = self.perform_ai_processing(cv_image)

            # Measure processing time
            end_time = self.get_clock().now()
            processing_time = (end_time.nanoseconds - start_time.nanoseconds) / 1e9

            # Log performance metrics
            self.get_logger().info(f'Processing time: {processing_time:.3f}s')

            # Check if processing meets real-time requirements
            if processing_time > 0.1:  # 100ms threshold
                self.get_logger().warn(f'Processing exceeded real-time threshold: {processing_time:.3f}s')

        except Exception as e:
            self.get_logger().error(f'Real-time processing error: {e}')
```

## Sensor Data Synchronization

Multiple sensors often need to be synchronized for accurate AI processing:

### Time-based Synchronization

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class SensorSynchronizationNode(Node):
    def __init__(self):
        super().__init__('sensor_sync_node')

        # Create QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5  # Slightly larger for synchronization
        )

        # Create subscribers for different sensor types
        self.image_sub = Subscriber(self, Image, 'camera/image_raw', qos_profile=sensor_qos)
        self.imu_sub = Subscriber(self, Imu, 'imu/data', qos_profile=sensor_qos)
        self.laser_sub = Subscriber(self, LaserScan, 'scan', qos_profile=sensor_qos)

        # Synchronize sensors with approximate time sync
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.imu_sub, self.laser_sub],
            queue_size=10,  # Queue size for synchronization
            slop=0.1  # Time tolerance (0.1 seconds)
        )
        self.sync.registerCallback(self.synchronized_callback)

        self.get_logger().info('Sensor Synchronization Node initialized')

    def synchronized_callback(self, image_msg, imu_msg, laser_msg):
        """Process synchronized sensor data"""
        try:
            # All messages have approximately the same timestamp
            sync_time = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec / 1e9

            self.get_logger().info(f'Synchronized data at time: {sync_time:.3f}')

            # Perform multi-sensor AI processing
            ai_results = self.process_multi_sensor_data(image_msg, imu_msg, laser_msg)

            # Publish fused results
            self.publish_fused_results(ai_results, image_msg.header)

        except Exception as e:
            self.get_logger().error(f'Sensor synchronization error: {e}')

    def process_multi_sensor_data(self, image_msg, imu_msg, laser_msg):
        """Process synchronized multi-sensor data"""
        # Convert image for AI processing
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Extract IMU data
        imu_data = {
            'orientation': [imu_msg.orientation.x, imu_msg.orientation.y,
                           imu_msg.orientation.z, imu_msg.orientation.w],
            'angular_velocity': [imu_msg.angular_velocity.x, imu_msg.angular_velocity.y,
                                imu_msg.angular_velocity.z],
            'linear_acceleration': [imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y,
                                   imu_msg.linear_acceleration.z]
        }

        # Extract laser scan data
        laser_ranges = list(laser_msg.ranges)

        # Perform AI processing with multi-sensor fusion
        # (This would be your actual AI algorithm)
        results = {
            'image_features': self.extract_image_features(cv_image),
            'imu_orientation': imu_data['orientation'],
            'obstacle_distances': [r for r in laser_ranges if r < laser_msg.range_max]
        }

        return results

    def extract_image_features(self, image):
        """Extract features from image for AI processing"""
        # Example: simple feature extraction
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        return features if features is not None else []
```

## Buffering and Queue Management

Proper buffering and queue management are essential for real-time performance:

### Adaptive Buffer Management

```python
import collections
import threading
from rclpy.time import Time

class AdaptiveBufferNode(Node):
    def __init__(self):
        super().__init__('adaptive_buffer_node')

        # Initialize buffers with maximum size constraints
        self.max_buffer_size = 10  # Maximum items in buffer
        self.image_buffer = collections.deque(maxlen=self.max_buffer_size)
        self.ai_result_buffer = collections.deque(maxlen=self.max_buffer_size)

        # Buffer statistics
        self.buffer_stats = {
            'processed_count': 0,
            'dropped_count': 0,
            'avg_processing_time': 0.0
        }

        # Lock for thread-safe buffer access
        self.buffer_lock = threading.Lock()

        # Subscribe to image stream
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.buffer_image,
            1  # Minimal queue to prevent backlog
        )

        # Timer for processing buffer
        self.process_timer = self.create_timer(0.033, self.process_buffer)  # ~30 FPS

        self.get_logger().info('Adaptive Buffer Node initialized')

    def buffer_image(self, msg):
        """Add image to processing buffer"""
        with self.buffer_lock:
            # Check if buffer is full
            if len(self.image_buffer) >= self.max_buffer_size:
                # Drop oldest item to make room
                self.buffer_stats['dropped_count'] += 1
                self.image_buffer.popleft()

            # Add new image
            self.image_buffer.append(msg)

            # Log buffer status
            if len(self.image_buffer) == self.max_buffer_size:
                self.get_logger().warn(f'Buffer at maximum capacity: {len(self.image_buffer)}')

    def process_buffer(self):
        """Process images from buffer"""
        with self.buffer_lock:
            if not self.image_buffer:
                return  # No images to process

            # Process oldest image in buffer (FIFO)
            image_msg = self.image_buffer.popleft()

        # Process image with AI model
        start_time = self.get_clock().now()

        try:
            ai_result = self.perform_ai_inference(image_msg)

            # Update processing statistics
            end_time = self.get_clock().now()
            processing_time = (end_time.nanoseconds - start_time.nanoseconds) / 1e9

            self.buffer_stats['processed_count'] += 1
            self.buffer_stats['avg_processing_time'] = (
                (self.buffer_stats['avg_processing_time'] * (self.buffer_stats['processed_count'] - 1) + processing_time) /
                self.buffer_stats['processed_count']
            )

            # Log performance
            self.get_logger().info(
                f'Processed: {self.buffer_stats["processed_count"]}, '
                f'Dropped: {self.buffer_stats["dropped_count"]}, '
                f'Avg time: {self.buffer_stats["avg_processing_time"]:.3f}s'
            )

        except Exception as e:
            self.get_logger().error(f'Buffer processing error: {e}')

    def perform_ai_inference(self, image_msg):
        """Perform AI inference on image"""
        # Convert image for AI processing
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Perform inference (placeholder)
        # In real implementation, this would run your AI model
        results = {'detections': [], 'features': []}
        return results
```

### Priority-based Queue Management

```python
import queue
import threading
import time

class PriorityQueueNode(Node):
    def __init__(self):
        super().__init__('priority_queue_node')

        # Priority queue: (priority, timestamp, data)
        # Lower number = higher priority
        self.priority_queue = queue.PriorityQueue()
        self.queue_thread = threading.Thread(target=self.process_priority_queue)
        self.queue_thread.daemon = True
        self.queue_thread.start()

        # Subscribe to different data streams with different priorities
        self.high_priority_sub = self.create_subscription(
            Image, 'critical_image', self.add_high_priority_data, 1
        )
        self.normal_priority_sub = self.create_subscription(
            Image, 'normal_image', self.add_normal_priority_data, 5
        )

        self.get_logger().info('Priority Queue Node initialized')

    def add_high_priority_data(self, msg):
        """Add high-priority data to queue"""
        priority = 1  # Highest priority
        timestamp = time.time()
        self.priority_queue.put((priority, timestamp, msg))
        self.get_logger().info('Added high-priority data to queue')

    def add_normal_priority_data(self, msg):
        """Add normal-priority data to queue"""
        priority = 5  # Normal priority
        timestamp = time.time()
        self.priority_queue.put((priority, timestamp, msg))
        self.get_logger().info('Added normal-priority data to queue')

    def process_priority_queue(self):
        """Process items from priority queue in background thread"""
        while rclpy.ok():
            try:
                # Get item from queue (blocking with timeout)
                priority, timestamp, data = self.priority_queue.get(timeout=1.0)

                # Process the data
                self.process_priority_data(priority, timestamp, data)

                # Mark task as done
                self.priority_queue.task_done()

            except queue.Empty:
                # Timeout occurred, continue loop
                continue
            except Exception as e:
                self.get_logger().error(f'Priority queue processing error: {e}')

    def process_priority_data(self, priority, timestamp, data):
        """Process priority data"""
        current_time = time.time()
        latency = current_time - timestamp

        self.get_logger().info(f'Processing priority {priority} data, latency: {latency:.3f}s')

        # Perform AI processing based on priority
        if priority <= 2:  # Critical priority
            # Perform immediate processing
            result = self.critical_ai_processing(data)
        else:
            # Perform standard processing
            result = self.standard_ai_processing(data)

        self.get_logger().info(f'Completed processing for priority {priority} data')
```

## Performance Optimization

### Threading and Asynchronous Processing

```python
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class AsyncPerformanceNode(Node):
    def __init__(self):
        super().__init__('async_performance_node')

        # Thread pool for AI processing
        self.ai_executor = ThreadPoolExecutor(max_workers=2)

        # Async event loop for handling concurrent operations
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.run_async_loop, args=(self.loop,), daemon=True).start()

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.async_image_callback, 1
        )
        self.result_pub = self.create_publisher(String, 'ai_results', 10)

        self.get_logger().info('Async Performance Node initialized')

    def run_async_loop(self, loop):
        """Run async event loop in background thread"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def async_image_callback(self, msg):
        """Handle image callback asynchronously"""
        # Submit AI processing to thread pool
        future = self.ai_executor.submit(self.perform_ai_processing, msg)

        # Add callback to handle results when processing completes
        future.add_done_callback(lambda f: self.handle_ai_result(f.result()))

    def perform_ai_processing(self, image_msg):
        """Perform AI processing (runs in separate thread)"""
        try:
            # Convert image
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # Perform AI inference
            # (This would be your actual model inference)
            time.sleep(0.05)  # Simulate processing time
            result = f'Processed image at {image_msg.header.stamp.sec}'

            return result
        except Exception as e:
            self.get_logger().error(f'AI processing error: {e}')
            return f'Error: {e}'

    def handle_ai_result(self, result):
        """Handle AI processing result (called from thread pool)"""
        # Publish result back to main thread context
        self.get_logger().info(f'AI result: {result}')

        result_msg = String()
        result_msg.data = result
        self.result_pub.publish(result_msg)
```

### Memory Management for Real-time Systems

```python
import gc
import psutil
import numpy as np

class MemoryManagedNode(Node):
    def __init__(self):
        super().__init__('memory_managed_node')

        # Pre-allocate buffers to avoid memory fragmentation
        self.image_buffer = np.zeros((480, 640, 3), dtype=np.uint8)
        self.tensor_buffer = np.zeros((1, 3, 224, 224), dtype=np.float32)

        # Memory monitoring
        self.memory_threshold = 0.8  # 80% memory usage threshold

        # Timer for memory management
        self.memory_timer = self.create_timer(1.0, self.check_memory_usage)

        # Image processing
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.memory_efficient_callback, 1
        )

        self.get_logger().info('Memory Managed Node initialized')

    def memory_efficient_callback(self, msg):
        """Process image with memory efficiency in mind"""
        # Check memory usage before processing
        memory_percent = psutil.virtual_memory().percent / 100.0

        if memory_percent > self.memory_threshold:
            self.get_logger().warn(f'High memory usage: {memory_percent:.2%}')
            # Perform garbage collection
            gc.collect()

        try:
            # Use pre-allocated buffer when possible
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Copy to pre-allocated buffer to avoid memory allocation
            if cv_image.shape == self.image_buffer.shape:
                np.copyto(self.image_buffer, cv_image)
                processing_buffer = self.image_buffer
            else:
                processing_buffer = cv_image

            # Perform AI processing using pre-allocated tensor buffer
            ai_result = self.process_with_preallocated_buffer(processing_buffer)

        except Exception as e:
            self.get_logger().error(f'Memory-efficient processing error: {e}')

    def process_with_preallocated_buffer(self, image):
        """Process image using pre-allocated buffers"""
        # Preprocess image into tensor buffer
        if image.dtype == np.uint8:
            # Normalize to [0, 1] and convert to float32
            self.tensor_buffer[0, 0] = image[:, :, 0].astype(np.float32) / 255.0
            self.tensor_buffer[0, 1] = image[:, :, 1].astype(np.float32) / 255.0
            self.tensor_buffer[0, 2] = image[:, :, 2].astype(np.float32) / 255.0

        # Perform AI inference using pre-allocated buffer
        # (This would be your actual model inference)
        result = {'processed': True, 'shape': self.tensor_buffer.shape}
        return result

    def check_memory_usage(self):
        """Monitor and log memory usage"""
        memory = psutil.virtual_memory()
        self.get_logger().info(
            f'Memory usage: {memory.percent:.1f}% '
            f'({memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB)'
        )
```

## Practical Example: Real-time Object Detection Pipeline

Here's a complete example combining all the concepts for a real-time object detection pipeline:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Header
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
import threading
import collections
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class RealTimeObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('real_time_object_detection')

        # Initialize AI model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()

        # Initialize CV bridge
        self.cv_bridge = CvBridge()

        # Real-time processing configuration
        self.processing_rate = 30  # 30 FPS max
        self.last_process_time = 0

        # Buffer for image processing
        self.image_buffer = collections.deque(maxlen=2)  # Only keep latest 2 images
        self.buffer_lock = threading.Lock()

        # QoS profile for real-time performance
        real_time_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            real_time_qos
        )
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            'real_time_detections',
            real_time_qos
        )

        # Timer for processing pipeline
        self.process_timer = self.create_timer(
            1.0 / self.processing_rate,  # Process at target rate
            self.process_pipeline
        )

        self.get_logger().info(f'Real-time Object Detection Node initialized at {self.processing_rate} FPS')

    def image_callback(self, msg):
        """Receive and buffer images"""
        with self.buffer_lock:
            # Only keep the latest image to reduce latency
            while len(self.image_buffer) >= self.image_buffer.maxlen:
                self.image_buffer.popleft()
            self.image_buffer.append(msg)

    def process_pipeline(self):
        """Process images from buffer"""
        with self.buffer_lock:
            if not self.image_buffer:
                return  # No images to process

            # Get latest image
            image_msg = self.image_buffer[-1]
            self.image_buffer.clear()  # Clear buffer to reduce latency

        # Check processing rate to avoid overloading
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_process_time < (1.0 / self.processing_rate * 0.9):
            # Skip processing to maintain target rate
            return

        try:
            # Convert image
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # Perform object detection
            results = self.model(cv_image)

            # Convert results to ROS messages
            detections_msg = self.convert_detections_to_ros(results, image_msg.header)

            # Publish detections
            self.detection_pub.publish(detections_msg)

            # Update timing
            self.last_process_time = current_time

            self.get_logger().info(f'Published {len(detections_msg.detections)} detections')

        except Exception as e:
            self.get_logger().error(f'Real-time processing error: {e}')

    def convert_detections_to_ros(self, results, header):
        """Convert YOLO results to ROS Detection2DArray"""
        detections_msg = Detection2DArray()
        detections_msg.header = header

        # Process detection results
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            if conf > 0.5:  # Confidence threshold
                detection = Detection2D()

                # Set bounding box
                detection.bbox.center.x = float((xyxy[0] + xyxy[2]) / 2)
                detection.bbox.center.y = float((xyxy[1] + xyxy[3]) / 2)
                detection.bbox.size_x = float(xyxy[2] - xyxy[0])
                detection.bbox.size_y = float(xyxy[3] - xyxy[1])

                # Set confidence
                from vision_msgs.msg import ObjectHypothesisWithPose
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(int(cls))
                hypothesis.hypothesis.score = float(conf)
                detection.results.append(hypothesis)

                detections_msg.detections.append(detection)

        return detections_msg

def main(args=None):
    rclpy.init(args=args)
    node = RealTimeObjectDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Real-time Object Detection Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

This section covered building real-time data pipelines with efficient image transport, sensor synchronization, buffering strategies, and performance optimization techniques. In the next section, we'll address latency and synchronization challenges in AI-ROS integration.