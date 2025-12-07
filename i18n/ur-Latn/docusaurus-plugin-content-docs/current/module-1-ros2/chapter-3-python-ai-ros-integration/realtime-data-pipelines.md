---
title: "Real-time Data Pipelines Banana"
description: "AI-ROS integration ke liye efficient real-time data pipelines banana with proper buffering aur queue management"
sidebar_position: 4
keywords: [real-time, data pipeline, buffering, queue management, performance, ai, ros2]
---

# Real-time Data Pipelines Banana

Real-time data pipelines AI-ROS integration ke liye critical hain, khaas kar applications mein jo sensor inputs ke jald response ki requirement karti hain. Yeh section efficient image transport, sensor data synchronization, buffering strategies, aur high-frequency AI applications ke liye performance optimization techniques ko cover karta hai.

## Real-time Requirements ko Samajhna

Robotics mein real-time systems ke strict timing constraints hote hain jo proper operation ke liye meet kiye jana chahiye. AI-ROS integration mein, yeh typically involve karta hai:

- **Low-latency inference**: Environmental changes ko respond karne ke liye sensor data ko jaldi se process karna
- **High-frequency data handling**: Cameras ya LiDAR jaise high-bandwidth sensor streams ko manage karna
- **Deterministic timing**: Predictable behavior ke liye consistent processing times ensure karna
- **Resource management**: Limits exceed kiye bina computational resources ko efficiently use karna

## Efficient Image Transport

Images AI applications mein often largest data source hoti hain. Real-time performance ke liye efficient transport crucial hai.

### Image Transport Options

ROS 2 image transport mechanisms provide karta hai:

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

        # Image conversion ke liye bridge initialize karen
        self.cv_bridge = CvBridge()

        # Different transport types ke liye publishers
        self.raw_image_pub = self.create_publisher(Image, 'camera/image_raw', 10)
        self.compressed_pub = self.create_publisher(CompressedImage, 'camera/image_compressed', 10)

        # Different QoS profiles ke sath subscribers
        self.raw_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.raw_image_callback,
            1  # Real-time ke liye small queue size ka istemal karen
        )
        self.compressed_sub = self.create_subscription(
            CompressedImage,
            'camera/image_compressed',
            self.compressed_image_callback,
            1
        )

        # Image processing ke liye timer
        self.timer = self.create_timer(0.1, self.process_images)  # 10 Hz processing

        self.get_logger().info('Efficient Image Transport Node initialized')

    def process_images(self):
        """Real-time pipeline ke liye image processing simulate karen"""
        # Test image generate karen (real application mein, yeh camera se aayega)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # ROS Image message mein convert karen
        ros_image = self.cv_bridge.cv2_to_imgmsg(test_image, encoding='bgr8')
        ros_image.header.stamp = self.get_clock().now().to_msg()
        ros_image.header.frame_id = 'camera_optical_frame'

        # Dono raw aur compressed versions publish karen
        self.raw_image_pub.publish(ros_image)

        # Compressed image create karen
        compressed_msg = CompressedImage()
        compressed_msg.header = ros_image.header
        compressed_msg.format = 'jpeg'

        # OpenCV ka istemal karke compress karen
        _, compressed_data = cv2.imencode('.jpg', test_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        compressed_msg.data = compressed_data.tobytes()

        self.compressed_pub.publish(compressed_msg)

    def raw_image_callback(self, msg):
        """Raw image message process karen"""
        try:
            # ROS image ko OpenCV mein convert karen
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # AI processing perform karen (example: simple edge detection)
            processed_image = cv2.Canny(cv_image, 100, 200)

            # Processing time log karen
            self.get_logger().info(f'Processed raw image: {msg.width}x{msg.height}')

        except Exception as e:
            self.get_logger().error(f'Raw image processing error: {e}')

    def compressed_image_callback(self, msg):
        """Compressed image message process karen"""
        try:
            # Compressed image decode karen
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Processing time log karen
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

Optimal image transport ke liye, QoS profiles appropriate tareeke se configure karen:

```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

class OptimizedImageTransportNode(Node):
    def __init__(self):
        super().__init__('optimized_image_transport')

        # Real-time image transport ke liye QoS define karen
        image_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # Real-time ke liye some packet loss allow karen
            history=QoSHistoryPolicy.KEEP_LAST,  # Sirf latest messages rakhna
            depth=1,  # Low latency ke liye minimal queue depth
            durability=QoSDurabilityPolicy.VOLATILE  # Node dies ke baad messages rakhna band karen
        )

        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.real_time_image_callback,
            image_qos
        )

    def real_time_image_callback(self, msg):
        """Real-time constraints ke sath image process karen"""
        # Performance measurement ke liye timing start karen
        start_time = self.get_clock().now()

        try:
            # AI inference ke liye image process karen
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # AI processing perform karen
            results = self.perform_ai_processing(cv_image)

            # Processing time measure karen
            end_time = self.get_clock().now()
            processing_time = (end_time.nanoseconds - start_time.nanoseconds) / 1e9

            # Performance metrics log karen
            self.get_logger().info(f'Processing time: {processing_time:.3f}s')

            # Check karen kya processing real-time requirements ko meet karti hai
            if processing_time > 0.1:  # 100ms threshold
                self.get_logger().warn(f'Processing exceeded real-time threshold: {processing_time:.3f}s')

        except Exception as e:
            self.get_logger().error(f'Real-time processing error: {e}')
```

## Sensor Data Synchronization

Multiple sensors ko accurate AI processing ke liye often synchronize kiya jana chahiye:

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

        # Sensor data ke liye QoS profile create karen
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5  # Synchronization ke liye slightly larger
        )

        # Different sensor types ke liye subscribers create karen
        self.image_sub = Subscriber(self, Image, 'camera/image_raw', qos_profile=sensor_qos)
        self.imu_sub = Subscriber(self, Imu, 'imu/data', qos_profile=sensor_qos)
        self.laser_sub = Subscriber(self, LaserScan, 'scan', qos_profile=sensor_qos)

        # Approximate time sync ke sath sensors synchronize karen
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.imu_sub, self.laser_sub],
            queue_size=10,  # Synchronization ke liye queue size
            slop=0.1  # Time tolerance (0.1 seconds)
        )
        self.sync.registerCallback(self.synchronized_callback)

        self.get_logger().info('Sensor Synchronization Node initialized')

    def synchronized_callback(self, image_msg, imu_msg, laser_msg):
        """Synchronized sensor data process karen"""
        try:
            # Saare messages ke approximately same timestamp hain
            sync_time = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec / 1e9

            self.get_logger().info(f'Synchronized data at time: {sync_time:.3f}')

            # Multi-sensor AI processing perform karen
            ai_results = self.process_multi_sensor_data(image_msg, imu_msg, laser_msg)

            # Fused results publish karen
            self.publish_fused_results(ai_results, image_msg.header)

        except Exception as e:
            self.get_logger().error(f'Sensor synchronization error: {e}')

    def process_multi_sensor_data(self, image_msg, imu_msg, laser_msg):
        """Synchronized multi-sensor data process karen"""
        # AI processing ke liye image convert karen
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # IMU data extract karen
        imu_data = {
            'orientation': [imu_msg.orientation.x, imu_msg.orientation.y,
                           imu_msg.orientation.z, imu_msg.orientation.w],
            'angular_velocity': [imu_msg.angular_velocity.x, imu_msg.angular_velocity.y,
                                imu_msg.angular_velocity.z],
            'linear_acceleration': [imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y,
                                   imu_msg.linear_acceleration.z]
        }

        # Laser scan data extract karen
        laser_ranges = list(laser_msg.ranges)

        # Multi-sensor fusion ke sath AI processing perform karen
        # (Yeh aapka actual AI algorithm hoga)
        results = {
            'image_features': self.extract_image_features(cv_image),
            'imu_orientation': imu_data['orientation'],
            'obstacle_distances': [r for r in laser_ranges if r < laser_msg.range_max]
        }

        return results

    def extract_image_features(self, image):
        """AI processing ke liye features extract karen"""
        # Example: simple feature extraction
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        return features if features is not None else []
```

## Buffering aur Queue Management

Real-time performance ke liye proper buffering aur queue management essential hai:

### Adaptive Buffer Management

```python
import collections
import threading
from rclpy.time import Time

class AdaptiveBufferNode(Node):
    def __init__(self):
        super().__init__('adaptive_buffer_node')

        # Maximum size constraints ke sath buffers initialize karen
        self.max_buffer_size = 10  # Buffer mein maximum items
        self.image_buffer = collections.deque(maxlen=self.max_buffer_size)
        self.ai_result_buffer = collections.deque(maxlen=self.max_buffer_size)

        # Buffer statistics
        self.buffer_stats = {
            'processed_count': 0,
            'dropped_count': 0,
            'avg_processing_time': 0.0
        }

        # Thread-safe buffer access ke liye lock
        self.buffer_lock = threading.Lock()

        # Image stream ke liye subscribe karen
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.buffer_image,
            1  # Backlog prevent karne ke liye minimal queue
        )

        # Buffer process karne ke liye timer
        self.process_timer = self.create_timer(0.033, self.process_buffer)  # ~30 FPS

        self.get_logger().info('Adaptive Buffer Node initialized')

    def buffer_image(self, msg):
        """Processing buffer mein image add karen"""
        with self.buffer_lock:
            # Check karen kya buffer full hai
            if len(self.image_buffer) >= self.max_buffer_size:
                # Jagah banane ke liye oldest item drop karen
                self.buffer_stats['dropped_count'] += 1
                self.image_buffer.popleft()

            # New image add karen
            self.image_buffer.append(msg)

            # Buffer status log karen
            if len(self.image_buffer) == self.max_buffer_size:
                self.get_logger().warn(f'Buffer at maximum capacity: {len(self.image_buffer)}')

    def process_buffer(self):
        """Buffer se images process karen"""
        with self.buffer_lock:
            if not self.image_buffer:
                return  # Process karne ke liye koi images nahi hain

            # Buffer mein oldest image process karen (FIFO)
            image_msg = self.image_buffer.popleft()

        # AI model ke sath image process karen
        start_time = self.get_clock().now()

        try:
            ai_result = self.perform_ai_inference(image_msg)

            # Processing statistics update karen
            end_time = self.get_clock().now()
            processing_time = (end_time.nanoseconds - start_time.nanoseconds) / 1e9

            self.buffer_stats['processed_count'] += 1
            self.buffer_stats['avg_processing_time'] = (
                (self.buffer_stats['avg_processing_time'] * (self.buffer_stats['processed_count'] - 1) + processing_time) /
                self.buffer_stats['processed_count']
            )

            # Performance log karen
            self.get_logger().info(
                f'Processed: {self.buffer_stats["processed_count"]}, '
                f'Dropped: {self.buffer_stats["dropped_count"]}, '
                f'Avg time: {self.buffer_stats["avg_processing_time"]:.3f}s'
            )

        except Exception as e:
            self.get_logger().error(f'Buffer processing error: {e}')

    def perform_ai_inference(self, image_msg):
        """Image par AI inference perform karen"""
        # AI processing ke liye image convert karen
        cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        # Inference perform karen (placeholder)
        # Real implementation mein, yeh aapka AI model run karega
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

        # Different priorities ke sath different data streams ko subscribe karen
        self.high_priority_sub = self.create_subscription(
            Image, 'critical_image', self.add_high_priority_data, 1
        )
        self.normal_priority_sub = self.create_subscription(
            Image, 'normal_image', self.add_normal_priority_data, 5
        )

        self.get_logger().info('Priority Queue Node initialized')

    def add_high_priority_data(self, msg):
        """Queue mein high-priority data add karen"""
        priority = 1  # Highest priority
        timestamp = time.time()
        self.priority_queue.put((priority, timestamp, msg))
        self.get_logger().info('Added high-priority data to queue')

    def add_normal_priority_data(self, msg):
        """Queue mein normal-priority data add karen"""
        priority = 5  # Normal priority
        timestamp = time.time()
        self.priority_queue.put((priority, timestamp, msg))
        self.get_logger().info('Added normal-priority data to queue')

    def process_priority_queue(self):
        """Background thread mein priority queue se items process karen"""
        while rclpy.ok():
            try:
                # Queue se item get karen (blocking with timeout)
                priority, timestamp, data = self.priority_queue.get(timeout=1.0)

                # Data process karen
                self.process_priority_data(priority, timestamp, data)

                # Task done mark karen
                self.priority_queue.task_done()

            except queue.Empty:
                # Timeout occurred, continue loop
                continue
            except Exception as e:
                self.get_logger().error(f'Priority queue processing error: {e}')

    def process_priority_data(self, priority, timestamp, data):
        """Priority data process karen"""
        current_time = time.time()
        latency = current_time - timestamp

        self.get_logger().info(f'Processing priority {priority} data, latency: {latency:.3f}s')

        # Priority ke adhar par AI processing perform karen
        if priority <= 2:  # Critical priority
            # Immediate processing perform karen
            result = self.critical_ai_processing(data)
        else:
            # Standard processing perform karen
            result = self.standard_ai_processing(data)

        self.get_logger().info(f'Completed processing for priority {priority} data')
```

## Performance Optimization

### Threading aur Asynchronous Processing

```python
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

class AsyncPerformanceNode(Node):
    def __init__(self):
        super().__init__('async_performance_node')

        # AI processing ke liye thread pool
        self.ai_executor = ThreadPoolExecutor(max_workers=2)

        # Concurrent operations handle karne ke liye async event loop
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.run_async_loop, args=(self.loop,), daemon=True).start()

        # Publishers aur subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.async_image_callback, 1
        )
        self.result_pub = self.create_publisher(String, 'ai_results', 10)

        self.get_logger().info('Async Performance Node initialized')

    def run_async_loop(self, loop):
        """Background thread mein async event loop run karen"""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def async_image_callback(self, msg):
        """Image callback asynchronously handle karen"""
        # Thread pool mein AI processing submit karen
        future = self.ai_executor.submit(self.perform_ai_processing, msg)

        # Results handle karne ke liye callback add karen jab processing complete ho
        future.add_done_callback(lambda f: self.handle_ai_result(f.result()))

    def perform_ai_processing(self, image_msg):
        """AI processing perform karen (separate thread mein chalta hai)"""
        try:
            # Image convert karen
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # AI inference perform karen
            # (Yeh aapka actual model inference hoga)
            time.sleep(0.05)  # Processing time simulate karen
            result = f'Processed image at {image_msg.header.stamp.sec}'

            return result
        except Exception as e:
            self.get_logger().error(f'AI processing error: {e}')
            return f'Error: {e}'

    def handle_ai_result(self, result):
        """AI processing result handle karen (thread pool se call kiya jata hai)"""
        # Main thread context mein result publish karen
        self.get_logger().info(f'AI result: {result}')

        result_msg = String()
        result_msg.data = result
        self.result_pub.publish(result_msg)
```

### Real-time Systems ke liye Memory Management

```python
import gc
import psutil
import numpy as np

class MemoryManagedNode(Node):
    def __init__(self):
        super().__init__('memory_managed_node')

        # Memory fragmentation avoid karne ke liye pre-allocate buffers
        self.image_buffer = np.zeros((480, 640, 3), dtype=np.uint8)
        self.tensor_buffer = np.zeros((1, 3, 224, 224), dtype=np.float32)

        # Memory monitoring
        self.memory_threshold = 0.8  # 80% memory usage threshold

        # Memory management ke liye timer
        self.memory_timer = self.create_timer(1.0, self.check_memory_usage)

        # Image processing
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.memory_efficient_callback, 1
        )

        self.get_logger().info('Memory Managed Node initialized')

    def memory_efficient_callback(self, msg):
        """Memory efficiency ko mind mein rakhkar image process karen"""
        # Processing se pehle memory usage check karen
        memory_percent = psutil.virtual_memory().percent / 100.0

        if memory_percent > self.memory_threshold:
            self.get_logger().warn(f'High memory usage: {memory_percent:.2%}')
            # Garbage collection perform karen
            gc.collect()

        try:
            # Jab possible ho to pre-allocated buffer ka istemal karen
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Memory allocation avoid karne ke liye pre-allocated buffer mein copy karen
            if cv_image.shape == self.image_buffer.shape:
                np.copyto(self.image_buffer, cv_image)
                processing_buffer = self.image_buffer
            else:
                processing_buffer = cv_image

            # Pre-allocated tensor buffer ka istemal karke AI processing perform karen
            ai_result = self.process_with_preallocated_buffer(processing_buffer)

        except Exception as e:
            self.get_logger().error(f'Memory-efficient processing error: {e}')

    def process_with_preallocated_buffer(self, image):
        """Pre-allocated buffers ka istemal karke image process karen"""
        # Tensor buffer mein preprocess image
        if image.dtype == np.uint8:
            # [0, 1] mein normalize karen aur float32 mein convert karen
            self.tensor_buffer[0, 0] = image[:, :, 0].astype(np.float32) / 255.0
            self.tensor_buffer[0, 1] = image[:, :, 1].astype(np.float32) / 255.0
            self.tensor_buffer[0, 2] = image[:, :, 2].astype(np.float32) / 255.0

        # Pre-allocated buffer ka istemal karke AI inference perform karen
        # (Yeh aapka actual model inference hoga)
        result = {'processed': True, 'shape': self.tensor_buffer.shape}
        return result

    def check_memory_usage(self):
        """Monitor aur log memory usage"""
        memory = psutil.virtual_memory()
        self.get_logger().info(
            f'Memory usage: {memory.percent:.1f}% '
            f'({memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB)'
        )
```

## Practical Example: Real-time Object Detection Pipeline

Yeh ek complete example hai saare concepts ko combine karke real-time object detection pipeline ke liye:

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

        # AI model initialize karen
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()

        # CV bridge initialize karen
        self.cv_bridge = CvBridge()

        # Real-time processing configuration
        self.processing_rate = 30  # 30 FPS max
        self.last_process_time = 0

        # Image processing ke liye buffer
        self.image_buffer = collections.deque(maxlen=2)  # Sirf latest 2 images rakhna
        self.buffer_lock = threading.Lock()

        # Real-time performance ke liye QoS profile
        real_time_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers aur subscribers
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

        # Processing pipeline ke liye timer
        self.process_timer = self.create_timer(
            1.0 / self.processing_rate,  # Target rate par process karen
            self.process_pipeline
        )

        self.get_logger().info(f'Real-time Object Detection Node initialized at {self.processing_rate} FPS')

    def image_callback(self, msg):
        """Receive aur buffer images"""
        with self.buffer_lock:
            # Latency reduce karne ke liye sirf latest image rakhna
            while len(self.image_buffer) >= self.image_buffer.maxlen:
                self.image_buffer.popleft()
            self.image_buffer.append(msg)

    def process_pipeline(self):
        """Buffer se images process karen"""
        with self.buffer_lock:
            if not self.image_buffer:
                return  # Process karne ke liye koi images nahi hain

            # Latest image get karen
            image_msg = self.image_buffer[-1]
            self.image_buffer.clear()  # Latency reduce karne ke liye buffer clear karen

        # Overloading avoid karne ke liye processing rate check karen
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.last_process_time < (1.0 / self.processing_rate * 0.9):
            # Target rate maintain karne ke liye processing skip karen
            return

        try:
            # Image convert karen
            cv_image = self.cv_bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

            # Object detection perform karen
            results = self.model(cv_image)

            # Results ko ROS messages mein convert karen
            detections_msg = self.convert_detections_to_ros(results, image_msg.header)

            # Detections publish karen
            self.detection_pub.publish(detections_msg)

            # Timing update karen
            self.last_process_time = current_time

            self.get_logger().info(f'Published {len(detections_msg.detections)} detections')

        except Exception as e:
            self.get_logger().error(f'Real-time processing error: {e}')

    def convert_detections_to_ros(self, results, header):
        """YOLO results ko ROS Detection2DArray mein convert karen"""
        detections_msg = Detection2DArray()
        detections_msg.header = header

        # Detection results process karen
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            if conf > 0.5:  # Confidence threshold
                detection = Detection2D()

                # Bounding box set karen
                detection.bbox.center.x = float((xyxy[0] + xyxy[2]) / 2)
                detection.bbox.center.y = float((xyxy[1] + xyxy[3]) / 2)
                detection.bbox.size_x = float(xyxy[2] - xyxy[0])
                detection.bbox.size_y = float(xyxy[3] - xyxy[1])

                # Confidence set karen
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

Yeh section efficient image transport, sensor synchronization, buffering strategies, aur performance optimization techniques ke sath real-time data pipelines banana ko cover karta hai. Agle section mein, hum AI-ROS integration mein latency aur synchronization challenges ko address karenge.