---
title: "Computer Vision Integration"
description: "Robotic perception ke liye computer vision algorithms ko ROS 2 ke sath integrate karna"
sidebar_position: 3
keywords: [ros2, computer vision, cv, opencv, image processing, perception]
---

# Computer Vision Integration

ROS 2 ke sath computer vision integration robots ko image aur video processing ke zariye apna environment perceive aur understand karne ki allow karti hai. Yeh section OpenCV aur doosre computer vision libraries ko ROS 2 communication patterns ke sath integrate karne ko cover karta hai.

## Image Message Handling

### sensor_msgs/Image ko Samajhna

`sensor_msgs/Image` message type ROS 2 mein image data ke liye standard hai:

```python
from sensor_msgs.msg import Image
import cv2
import numpy as np

class ImageProcessorNode:
    def __init__(self):
        # Image message fields:
        # - header: timestamp aur frame_id
        # - height, width: image dimensions
        # - encoding: pixel format (rgb8, bgr8, mono8, etc.)
        # - is_bigendian: byte order
        # - step: bytes mein full row length
        # - data: actual image data as bytes
        pass
```

### ROS aur OpenCV ke beech Convert Karna

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageConversionNode(Node):
    def __init__(self):
        super().__init__('image_conversion_node')

        # Conversion ke liye CvBridge create karen
        self.bridge = CvBridge()

        # Subscribers aur publishers setup karen
        self.image_subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        self.result_publisher = self.create_publisher(
            Image,
            'processed_image',
            10
        )

        self.status_publisher = self.create_publisher(
            String,
            'cv_status',
            10
        )

    def image_callback(self, msg):
        """ROS Image ko OpenCV mein convert karen aur process karen"
        try:
            # ROS Image ko OpenCV image mein convert karen
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # OpenCV ka istemal karke image process karen
            processed_image = self.process_image(cv_image)

            # Wapas se ROS Image mein convert karen
            result_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
            result_msg.header = msg.header  # Original header preserve karen

            # Processed image publish karen
            self.result_publisher.publish(result_msg)

            # Publish status
            status_msg = String()
            status_msg.data = f'Processed image: {cv_image.shape}'
            self.status_publisher.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Image conversion error: {str(e)}')

    def process_image(self, cv_image):
        """Image par computer vision processing apply karen"
        # Example: Grayscale mein convert karen
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Example: Gaussian blur apply karen
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # Example: Threshold apply karen
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # Output ke liye wapas se 3-channel mein convert karen
        result = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        return result

def main(args=None):
    rclpy.init(args=args)
    node = ImageConversionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Object Detection Integration

### YOLO Integration Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np

class YOLOObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')

        # CvBridge setup karen
        self.bridge = CvBridge()

        # YOLO model load karen
        self.net = self.load_yolo_model()

        # Communication setup karen
        self.image_subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.detection_callback,
            10
        )

        self.detection_publisher = self.create_publisher(
            Detection2DArray,
            'object_detections',
            10
        )

        self.visualized_publisher = self.create_publisher(
            Image,
            'detection_visualization',
            10
        )

    def load_yolo_model(self):
        """Object detection ke liye YOLO model load karen"
        # Example using OpenCV's DNN module
        # Actual model paths se replace karen
        config_path = '/path/to/yolo.cfg'
        weights_path = '/path/to/yolo.weights'

        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

        # Output layer names get karen
        layer_names = net.getLayerNames()
        self.output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return net

    def detection_callback(self, msg):
        """Image process karen aur objects detect karen"
        try:
            # OpenCV format mein convert karen
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Object detection run karen
            detections, visualized_image = self.detect_objects(cv_image)

            # Detections publish karen
            detection_msg = self.create_detection_message(detections, msg.header)
            self.detection_publisher.publish(detection_msg)

            # Visualized image publish karen
            vis_msg = self.bridge.cv2_to_imgmsg(visualized_image, encoding='bgr8')
            vis_msg.header = msg.header
            self.visualized_publisher.publish(vis_msg)

        except Exception as e:
            self.get_logger().error(f'Detection error: {str(e)}')

    def detect_objects(self, image):
        """Image par YOLO object detection run karen"
        height, width, channels = image.shape

        # YOLO ke liye image prepare karen
        blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        # Detections process karen
        class_ids = []
        confidences = []
        boxes = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:  # Confidence threshold
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maximum suppression apply karen
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Bounding boxes ke sath visualized image create karen
        visualized_image = image.copy()
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(class_ids[i])
                confidence = confidences[i]

                # Draw bounding box
                cv2.rectangle(visualized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(visualized_image, f'{label} {confidence:.2f}',
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return list(zip(boxes, confidences, class_ids)), visualized_image

    def create_detection_message(self, detections, header):
        """Detections se Detection2DArray message create karen"
        detection_array = Detection2DArray()
        detection_array.header = header

        for box, confidence, class_id in detections:
            detection = Detection2D()

            # Set bounding box
            detection.bbox.center.x = box[0] + box[2] / 2
            detection.bbox.center.y = box[1] + box[3] / 2
            detection.bbox.size_x = box[2]
            detection.bbox.size_y = box[3]

            # Set classification
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(class_id)
            hypothesis.hypothesis.score = confidence
            detection.results.append(hypothesis)

            detection_array.detections.append(detection)

        return detection_array

def main(args=None):
    rclpy.init(args=args)
    node = YOLOObjectDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Feature Detection aur Tracking

### SIFT Feature Detection

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class FeatureDetectionNode(Node):
    def __init__(self):
        super().__init__('feature_detection_node')

        # CvBridge setup karen
        self.bridge = CvBridge()

        # SIFT detector initialize karen
        self.sift = cv2.SIFT_create()

        # Communication setup karen
        self.image_subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.feature_callback,
            10
        )

        self.feature_publisher = self.create_publisher(
            Image,
            'feature_image',
            10
        )

    def feature_callback(self, msg):
        """Image mein features detect karen"
        try:
            # OpenCV format mein convert karen
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Feature detection ke liye grayscale mein convert karen
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Features detect karen
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)

            # Image par features draw karen
            feature_image = cv2.drawKeypoints(
                cv_image, keypoints, None,
                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )

            # Feature image publish karen
            result_msg = self.bridge.cv2_to_imgmsg(feature_image, encoding='bgr8')
            result_msg.header = msg.header
            self.feature_publisher.publish(result_msg)

            self.get_logger().info(f'Detected {len(keypoints) if keypoints else 0} features')

        except Exception as e:
            self.get_logger().error(f'Feature detection error: {str(e)}')
```

## Image Filtering aur Enhancement

### Real-time Image Processing Pipeline

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageProcessingPipelineNode(Node):
    def __init__(self):
        super().__init__('image_processing_pipeline')

        # CvBridge setup karen
        self.bridge = CvBridge()

        # Processing parameters
        self.declare_parameter('gaussian_blur_kernel', 5)
        self.declare_parameter('canny_low_threshold', 50)
        self.declare_parameter('canny_high_threshold', 150)

        # Communication setup karen
        self.image_subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.processing_callback,
            10
        )

        self.edge_publisher = self.create_publisher(
            Image,
            'edge_image',
            10
        )

        self.processed_publisher = self.create_publisher(
            Image,
            'processed_image',
            10
        )

        self.performance_publisher = self.create_publisher(
            Float32,
            'processing_time',
            10
        )

    def processing_callback(self, msg):
        """Image processing pipeline apply karen"
        start_time = self.get_clock().now()

        try:
            # OpenCV format mein convert karen
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Processing pipeline apply karen
            processed_image = self.apply_pipeline(cv_image)

            # Processing time calculate karen
            end_time = self.get_clock().now()
            processing_time = (end_time.nanoseconds - start_time.nanoseconds) / 1e9

            # Results publish karen
            processed_msg = self.bridge.cv2_to_imgmsg(processed_image, encoding='bgr8')
            processed_msg.header = msg.header
            self.processed_publisher.publish(processed_msg)

            # Publish performance
            perf_msg = Float32()
            perf_msg.data = processing_time
            self.performance_publisher.publish(perf_msg)

        except Exception as e:
            self.get_logger().error(f'Processing pipeline error: {str(e)}')

    def apply_pipeline(self, image):
        """Complete image processing pipeline apply karen"
        # Step 1: Grayscale mein convert karen
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Step 2: Gaussian blur apply karen
        kernel_size = self.get_parameter('gaussian_blur_kernel').value
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Step 3: Canny edge detection apply karen
        low_thresh = self.get_parameter('canny_low_threshold').value
        high_thresh = self.get_parameter('canny_high_threshold').value
        edges = cv2.Canny(blurred, low_thresh, high_thresh)

        # Step 4: Original ko edges ke sath combine karen
        edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(image, 0.7, edge_colored, 0.3, 0)

        return result
```

## Camera Calibration Integration

### Calibration aur Undistortion

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np

class CalibrationNode(Node):
    def __init__(self):
        super().__init__('calibration_node')

        # CvBridge setup karen
        self.bridge = CvBridge()

        # Camera parameters (CameraInfo se fill hoga)
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_received = False

        # Communication setup karen
        self.image_subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.calibration_callback,
            10
        )

        self.info_subscriber = self.create_subscription(
            CameraInfo,
            'camera/camera_info',
            self.camera_info_callback,
            10
        )

        self.undistorted_publisher = self.create_publisher(
            Image,
            'camera/image_undistorted',
            10
        )

    def camera_info_callback(self, msg):
        """Camera calibration parameters receive karen"
        if not self.calibration_received:
            # Camera matrix aur distortion coefficients extract karen
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.calibration_received = True
            self.get_logger().info('Camera calibration received')

    def calibration_callback(self, msg):
        """Camera calibration ko image mein apply karen"
        if not self.calibration_received:
            return

        try:
            # OpenCV format mein convert karen
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Undistortion apply karen
            undistorted_image = cv2.undistort(
                cv_image,
                self.camera_matrix,
                self.dist_coeffs
            )

            # Publish undistorted image
            result_msg = self.bridge.cv2_to_imgmsg(undistorted_image, encoding='bgr8')
            result_msg.header = msg.header
            self.undistorted_publisher.publish(result_msg)

        except Exception as e:
            self.get_logger().error(f'Calibration error: {str(e)}')
```

## CV ke liye Performance Optimization

### Multi-threaded Processing

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
from collections import deque
import time

class MultiThreadedCVNode(Node):
    def __init__(self):
        super().__init__('multithreaded_cv_node')

        # CvBridge setup karen
        self.bridge = CvBridge()

        # Processing queue aur thread
        self.processing_queue = deque(maxlen=2)  # Queue size limit karen
        self.result_queue = deque(maxlen=2)
        self.processing_lock = threading.Lock()
        self.result_lock = threading.Lock()

        # Communication setup karen
        self.image_subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_queue_callback,
            10
        )

        self.result_publisher = self.create_publisher(
            Image,
            'cv_result',
            10
        )

        # Processing thread start karen
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()

    def image_queue_callback(self, msg):
        """Processing queue mein image add karen"
        with self.processing_lock:
            if len(self.processing_queue) < self.processing_queue.maxlen:
                self.processing_queue.append(msg)

    def processing_worker(self):
        """Background thread for image processing"
        while rclpy.ok():
            try:
                with self.processing_lock:
                    if self.processing_queue:
                        msg = self.processing_queue.popleft()
                    else:
                        time.sleep(0.01)  # Busy waiting se bachne ke liye small sleep
                        continue

                # Image process karen
                result = self.process_image_threaded(msg)

                # Output queue mein result add karen
                with self.result_lock:
                    if len(self.result_queue) < self.result_queue.maxlen:
                        self.result_queue.append(result)

            except Exception as e:
                self.get_logger().error(f'Processing thread error: {str(e)}')

    def process_image_threaded(self, msg):
        """Background thread mein image process karen"
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Heavy processing apply karen
            processed = self.heavy_cv_processing(cv_image)

            # Create result message
            result_msg = self.bridge.cv2_to_imgmsg(processed, encoding='bgr8')
            result_msg.header = msg.header

            return result_msg

        except Exception as e:
            self.get_logger().error(f'Threaded processing error: {str(e)}')
            return None

    def heavy_cv_processing(self, image):
        """Heavy computer vision processing simulate karen"
        # Example: Multiple filters apply karen
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        edges = cv2.Canny(blurred, 50, 150)
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return result

    def publish_results(self):
        """Main thread se results publish karen"
        with self.result_lock:
            while self.result_queue:
                result_msg = self.result_queue.popleft()
                if result_msg is not None:
                    self.result_publisher.publish(result_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MultiThreadedCVNode()

    # Main thread se results publish karne ke liye timer
    timer = node.create_timer(0.01, node.publish_results)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Computer Vision Integration

### 1. Memory Management

```python
class MemoryEfficientCVNode(Node):
    def __init__(self):
        super().__init__('memory_efficient_cv_node')

        # Processing ke dauran memory allocation se bachne ke liye pre-allocate buffers
        self.input_buffer = None
        self.output_buffer = None

        # Communication setup karen
        self.image_subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.memory_efficient_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            'cv_result',
            10
        )

    def memory_efficient_callback(self, msg):
        """Efficient memory usage ke sath image process karen"
        try:
            # Agar zarurat ho to pre-allocate karen
            if self.input_buffer is None or self.input_buffer.shape != (msg.height, msg.width, 3):
                self.input_buffer = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)
                self.output_buffer = np.zeros((msg.height, msg.width, 3), dtype=np.uint8)

            # Direct pre-allocated buffer mein convert karen
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Jab possible ho to in-place process karen
            result = self.process_image_inplace(cv_image)

            # Convert aur publish karen
            result_msg = self.bridge.cv2_to_imgmsg(result, encoding='bgr8')
            result_msg.header = msg.header
            self.publisher.publish(result_msg)

        except Exception as e:
            self.get_logger().error(f'Memory efficient processing error: {str(e)}')
```

### 2. Image Streams ke liye Quality of Service

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class QoSCVNode(Node):
    def __init__(self):
        super().__init__('qos_cv_node')

        # Image processing ke liye QoS profile (real-time ke liye best effort)
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,  # Sirf latest image rakhna
            reliability=ReliabilityPolicy.BEST_EFFORT  # Frames drop karne ki allow dena
        )

        # Results ke liye QoS profile (reliable)
        result_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # Appropriate QoS ke sath setup karen
        self.image_subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.process_callback,
            image_qos
        )

        self.result_publisher = self.create_publisher(
            Image,
            'cv_result',
            result_qos
        )
```

ROS 2 ke sath computer vision integration robotic systems ke liye powerful perception capabilities ko enable karta hai, jo sophisticated image processing aur analysis techniques ke zariye robots ko unke visual environment ko understand aur interact karne ki allow karta hai.