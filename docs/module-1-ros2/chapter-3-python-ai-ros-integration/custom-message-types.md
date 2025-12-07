---
title: "Creating Custom ROS Messages for AI Data"
description: "Designing and implementing custom ROS message types for efficient AI data communication"
sidebar_position: 3
keywords: [ros2, custom messages, ai data, message types, tensors, robotics]
---

# Creating Custom ROS Messages for AI Data

Effective communication between AI systems and ROS 2 nodes requires well-designed custom message types. This section covers creating specialized ROS messages for AI data, including tensors, predictions, and detections, along with best practices for efficient data transmission.

## Understanding ROS Message Types

ROS messages are the fundamental data structures used for communication between nodes. For AI applications, standard message types may not be sufficient to efficiently transmit complex data like neural network tensors, prediction results, or model metadata.

### Standard Message Limitations

While ROS provides standard message types like `sensor_msgs/Image` and `std_msgs/Float32MultiArray`, these may not be optimal for AI applications:

- **Performance**: Standard arrays may not efficiently represent multi-dimensional tensors
- **Semantics**: Generic arrays don't convey the meaning of AI-specific data
- **Size**: Large tensor data can overwhelm standard message sizes
- **Metadata**: Missing information about model version, confidence scores, etc.

## Defining Custom Messages for Tensors

For AI applications, we often need to transmit multi-dimensional tensor data. Here's how to create custom message types for tensor communication:

### Tensor.msg Definition

Create a file `msg/Tensor.msg` in your package:

```
# Multi-dimensional tensor message for AI applications
# Represents a tensor with flexible dimensions

# Data type (0=FLOAT32, 1=FLOAT64, 2=INT32, 3=INT64, 4=UINT8, 5=UINT16)
uint8 data_type
# 0 = FLOAT32, 1 = FLOAT64, 2 = INT32, 3 = INT64, 4 = UINT8, 5 = UINT16

# Dimensions of the tensor
uint32[] shape

# Tensor data (flattened)
float32[] float32_data
float64[] float64_data
int32[] int32_data
int64[] int64_data
uint8[] uint8_data
uint16[] uint16_data

# Optional metadata
string name
string description
float64 timestamp
```

### Example Usage in Python

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from your_package.msg import Tensor  # Your custom message
import numpy as np

class TensorPublisherNode(Node):
    def __init__(self):
        super().__init__('tensor_publisher')

        # Publisher for tensor data
        self.tensor_publisher = self.create_publisher(Tensor, 'ai_tensor', 10)

        # Timer to publish tensor data periodically
        self.timer = self.create_timer(1.0, self.publish_tensor_data)

        self.get_logger().info('Tensor Publisher Node initialized')

    def publish_tensor_data(self):
        """Publish example tensor data"""
        # Create a sample tensor (e.g., output from a neural network)
        sample_tensor = np.random.rand(3, 224, 224).astype(np.float32)  # RGB image-like tensor

        # Create ROS message
        tensor_msg = Tensor()
        tensor_msg.header = Header()
        tensor_msg.header.stamp = self.get_clock().now().to_msg()
        tensor_msg.header.frame_id = 'base_link'

        # Set data type
        tensor_msg.data_type = 0  # FLOAT32
        tensor_msg.shape = list(sample_tensor.shape)

        # Flatten and set data
        tensor_msg.float32_data = sample_tensor.flatten().tolist()

        # Set metadata
        tensor_msg.name = 'neural_network_output'
        tensor_msg.description = 'Output tensor from CNN classifier'
        tensor_msg.timestamp = self.get_clock().now().nanoseconds / 1e9

        # Publish
        self.tensor_publisher.publish(tensor_msg)
        self.get_logger().info(f'Published tensor with shape {sample_tensor.shape}')

def main(args=None):
    rclpy.init(args=args)
    node = TensorPublisherNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Tensor Publisher Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Messages for Predictions and Detections

AI systems often produce prediction results or object detections that need specialized message types:

### Prediction.msg Definition

Create `msg/Prediction.msg`:

```
# AI prediction result message
string model_name
string model_version
float64 confidence_threshold

# Multiple possible predictions with confidence scores
string[] labels
float32[] scores

# Top prediction
string top_label
float32 top_score

# Associated image or data reference
string data_reference
float64 timestamp
```

### Detection.msg Definition

Create `msg/Detection.msg`:

```
# Object detection result message
string detector_name
string detector_version

# Bounding box information
float32 x_min
float32 y_min
float32 x_max
float32 y_max
float32 width
float32 height

# Class information
string class_name
int32 class_id
float32 confidence

# Position in 3D space (if available)
float64 x_3d
float64 y_3d
float64 z_3d

# Associated image reference
string image_reference
float64 timestamp
```

### DetectionsArray.msg Definition

Create `msg/DetectionsArray.msg`:

```
# Array of detection results
std_msgs/Header header
Detection[] detections
string frame_id
```

## Best Practices for AI Data Messages

### 1. Efficient Data Representation

```python
# Efficient tensor message with compressed data
# msg/CompressedTensor.msg
uint8 data_type
uint32[] shape
uint8 compression_type  # 0=none, 1=uint8 quantized, 2=compressed
uint8[] compressed_data
float32 scale_factor
float32 zero_point
string name
```

### 2. Metadata Inclusion

Always include relevant metadata for AI data:

```python
# msg/AIPrediction.msg
string model_name
string model_version
string model_hash  # For reproducibility
float64 processing_time  # Time taken for inference
bool is_valid  # Validation flag
string error_message  # If processing failed

# Input information
string input_source
float64 input_timestamp
```

### 3. Size Considerations

For large tensor data, consider chunked messages:

```python
# msg/TensorChunk.msg
uint32 chunk_id
uint32 total_chunks
uint32 tensor_id  # To group chunks belonging to same tensor
float32[] data_chunk
bool is_last_chunk
```

## Building and Using Custom Messages

### Package Configuration

Update your `package.xml` to include message generation dependencies:

```xml
<buildtool_depend>ament_cmake</buildtool_depend>
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

Update your `CMakeLists.txt`:

```cmake
find_package(rosidl_default_generators REQUIRED)

# Find your message files
set(msg_files
  "msg/Tensor.msg"
  "msg/Prediction.msg"
  "msg/Detection.msg"
  "msg/DetectionsArray.msg"
)

# Generate messages
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces std_msgs sensor_msgs geometry_msgs
)

ament_package()
```

### Complete Example: AI Processing Node

Here's a complete example that uses custom message types for AI processing:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from your_package.msg import Tensor, Prediction, DetectionsArray, Detection
from std_msgs.msg import Header
import numpy as np
from PIL import Image as PILImage
import torch
import torchvision.transforms as transforms

class AIProcessingNode(Node):
    def __init__(self):
        super().__init__('ai_processing_node')

        # Initialize AI model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        # Publishers
        self.tensor_pub = self.create_publisher(Tensor, 'feature_tensor', 10)
        self.prediction_pub = self.create_publisher(Prediction, 'prediction_result', 10)
        self.detection_pub = self.create_publisher(DetectionsArray, 'object_detections', 10)

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])

        self.get_logger().info('AI Processing Node initialized')

    def image_callback(self, msg):
        """Process incoming image with AI model"""
        try:
            # Convert ROS image to PIL
            pil_image = self.ros_to_pil_image(msg)

            # Preprocess image
            input_tensor = self.transform(pil_image).unsqueeze(0)  # Add batch dimension

            # Perform inference
            with torch.no_grad():
                results = self.model(input_tensor)

            # Publish tensor data
            self.publish_tensor_data(input_tensor, msg.header.stamp)

            # Publish predictions
            self.publish_predictions(results, msg.header.stamp)

            # Publish detections
            self.publish_detections(results, msg.width, msg.height, msg.header.stamp)

        except Exception as e:
            self.get_logger().error(f'AI processing error: {e}')

    def publish_tensor_data(self, tensor, stamp):
        """Publish tensor data as custom message"""
        tensor_msg = Tensor()
        tensor_msg.header = Header()
        tensor_msg.header.stamp = stamp
        tensor_msg.header.frame_id = 'camera_optical_frame'

        # Set tensor properties
        tensor_msg.data_type = 0  # FLOAT32
        tensor_msg.shape = list(tensor.shape)
        tensor_msg.float32_data = tensor.flatten().tolist()
        tensor_msg.name = 'preprocessed_input'
        tensor_msg.description = 'Input tensor after preprocessing'
        tensor_msg.timestamp = stamp.sec + stamp.nanosec / 1e9

        self.tensor_pub.publish(tensor_msg)

    def publish_predictions(self, results, stamp):
        """Publish prediction results"""
        prediction_msg = Prediction()
        prediction_msg.header = Header()
        prediction_msg.header.stamp = stamp
        prediction_msg.model_name = 'YOLOv5s'
        prediction_msg.model_version = '6.0'
        prediction_msg.confidence_threshold = 0.5

        # Process results to extract predictions
        detections = results.xyxy[0].cpu().numpy()

        if len(detections) > 0:
            # Get top detection
            top_detection = detections[0]
            top_confidence = top_detection[4]
            top_class_id = int(top_detection[5])

            # Set top prediction
            prediction_msg.top_label = str(top_class_id)
            prediction_msg.top_score = float(top_confidence)

            # Set all predictions
            prediction_msg.labels = [str(int(det[5])) for det in detections]
            prediction_msg.scores = [float(det[4]) for det in detections]

        self.prediction_pub.publish(prediction_msg)

    def publish_detections(self, results, img_width, img_height, stamp):
        """Publish object detections"""
        detections_array = DetectionsArray()
        detections_array.header = Header()
        detections_array.header.stamp = stamp
        detections_array.header.frame_id = 'camera_optical_frame'
        detections_array.frame_id = 'camera_optical_frame'

        # Process YOLO results
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            detection = Detection()

            # Convert coordinates to image space
            detection.x_min = float(xyxy[0])
            detection.y_min = float(xyxy[1])
            detection.x_max = float(xyxy[2])
            detection.y_max = float(xyxy[3])
            detection.width = float(xyxy[2] - xyxy[0])
            detection.height = float(xyxy[3] - xyxy[1])

            # Class and confidence
            detection.class_name = str(int(cls))
            detection.class_id = int(cls)
            detection.confidence = float(conf)

            # Timestamp
            detection.timestamp = stamp.sec + stamp.nanosec / 1e9

            detections_array.detections.append(detection)

        self.detection_pub.publish(detections_array)

    def ros_to_pil_image(self, ros_image):
        """Convert ROS Image to PIL Image"""
        height = ros_image.height
        width = ros_image.width
        image_array = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(height, width, -1)
        return PILImage.fromarray(image_array)

def main(args=None):
    rclpy.init(args=args)
    node = AIProcessingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down AI Processing Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization for AI Messages

### 1. Message Size Optimization

For large tensor data, consider compression or downsampling:

```python
def compress_tensor(tensor, method='quantize'):
    """Compress tensor data for efficient transmission"""
    if method == 'quantize':
        # Quantize to 8-bit for transmission
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        scale = (tensor_max - tensor_min) / 255.0
        quantized = ((tensor - tensor_min) / scale).astype(np.uint8)

        return quantized, scale, tensor_min
    elif method == 'downsample':
        # Downsample large tensors
        return tensor[::2, ::2, ::2]  # Downsample by factor of 2
```

### 2. Conditional Publishing

Only publish messages when necessary:

```python
def should_publish_result(self, current_result, previous_result, threshold=0.1):
    """Determine if result should be published based on change threshold"""
    if previous_result is None:
        return True

    # Calculate difference between results
    diff = abs(current_result - previous_result)
    return diff > threshold
```

## Testing Custom Messages

Create test nodes to verify your custom message types:

```python
import rclpy
from rclpy.node import Node
from your_package.msg import Tensor, Prediction

class MessageTester(Node):
    def __init__(self):
        super().__init__('message_tester')

        # Test publishers and subscribers
        self.tensor_sub = self.create_subscription(
            Tensor, 'test_tensor', self.tensor_callback, 10
        )
        self.pred_sub = self.create_subscription(
            Prediction, 'test_prediction', self.pred_callback, 10
        )

        # Test publishers
        self.tensor_pub = self.create_publisher(Tensor, 'test_tensor', 10)
        self.pred_pub = self.create_publisher(Prediction, 'test_prediction', 10)

        # Test timer
        self.timer = self.create_timer(2.0, self.test_publish)

        self.get_logger().info('Message Tester Node initialized')

    def tensor_callback(self, msg):
        self.get_logger().info(f'Received tensor: shape={msg.shape}, name={msg.name}')

    def pred_callback(self, msg):
        self.get_logger().info(f'Received prediction: top_label={msg.top_label}')

    def test_publish(self):
        # Publish test tensor
        tensor_msg = Tensor()
        tensor_msg.shape = [2, 3]
        tensor_msg.float32_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        tensor_msg.name = 'test_tensor'
        self.tensor_pub.publish(tensor_msg)

        # Publish test prediction
        pred_msg = Prediction()
        pred_msg.top_label = 'test_object'
        pred_msg.top_score = 0.95
        self.pred_pub.publish(pred_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MessageTester()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Message Tester')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

This section covered creating custom ROS messages for AI data, including tensor representations, predictions, and detections. In the next section, we'll explore building real-time data pipelines for efficient AI-ROS integration.