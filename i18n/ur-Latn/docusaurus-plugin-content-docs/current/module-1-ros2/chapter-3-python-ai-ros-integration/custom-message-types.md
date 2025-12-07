---
title: "AI Data ke liye Custom ROS Messages banana"
description: "Efficient AI data communication ke liye customized ROS message types design aur implement karna"
sidebar_position: 3
keywords: [ros2, custom messages, ai data, message types, tensors, robotics]
---

# AI Data ke liye Custom ROS Messages banana

AI systems aur ROS 2 nodes ke darmiyan effective communication ke liye well-designed custom message types ki zarurat hoti hai. Yeh section AI data ke liye specialized ROS messages banane ko cover karta hai, jismein tensors, predictions, aur detections included hain, ke sath hi efficient data transmission ke liye best practices bhi included hain.

## ROS Message Types ko Samajhna

ROS messages nodes ke darmiyan communication ke liye istemal kiye jane wale fundamental data structures hain. AI applications ke liye, standard message types complex data jaise neural network tensors, prediction results, ya model metadata efficiently transmit karne ke liye sufficient nahi ho sakti hain.

### Standard Message Limitations

Jabke ROS standard message types provide karta hai jaise `sensor_msgs/Image` aur `std_msgs/Float32MultiArray`, ye AI applications ke liye optimal nahi ho sakti hain:

- **Performance**: Standard arrays multi-dimensional tensors ko efficiently represent nahi kar sakti hain
- **Semantics**: Generic arrays AI-specific data ka meaning convey nahi karti hain
- **Size**: Bade tensor data standard message sizes ko overwhelm kar sakti hain
- **Metadata**: Missing information model version, confidence scores, etc ke baare mein

## Tensors ke liye Custom Messages Define Karna

AI applications ke liye, hum often ko multi-dimensional tensor data transmit karne ki zarurat hoti hai. Yahan dikha gaya hai kaise tensor communication ke liye custom message types create karna hai:

### Tensor.msg Definition

Apne package mein ek file `msg/Tensor.msg` create karen:

```
# AI applications ke liye multi-dimensional tensor message
# Flexible dimensions wala tensor represent karta hai

# Data type (0=FLOAT32, 1=FLOAT64, 2=INT32, 3=INT64, 4=UINT8, 5=UINT16)
uint8 data_type
# 0 = FLOAT32, 1 = FLOAT64, 2 = INT32, 3 = INT64, 4 = UINT8, 5 = UINT16

# Tensor ke dimensions
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

### Python mein Example Usage

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from your_package.msg import Tensor  # Aapka custom message
import numpy as np

class TensorPublisherNode(Node):
    def __init__(self):
        super().__init__('tensor_publisher')

        # Tensor data ke liye publisher
        self.tensor_publisher = self.create_publisher(Tensor, 'ai_tensor', 10)

        # Tensor data periodically publish karne ke liye timer
        self.timer = self.create_timer(1.0, self.publish_tensor_data)

        self.get_logger().info('Tensor Publisher Node initialized')

    def publish_tensor_data(self):
        """Example tensor data publish karen"""
        # Sample tensor create karen (e.g., neural network se output)
        sample_tensor = np.random.rand(3, 224, 224).astype(np.float32)  # RGB image-like tensor

        # ROS message create karen
        tensor_msg = Tensor()
        tensor_msg.header = Header()
        tensor_msg.header.stamp = self.get_clock().now().to_msg()
        tensor_msg.header.frame_id = 'base_link'

        # Data type set karen
        tensor_msg.data_type = 0  # FLOAT32
        tensor_msg.shape = list(sample_tensor.shape)

        # Flatten karen aur data set karen
        tensor_msg.float32_data = sample_tensor.flatten().tolist()

        # Metadata set karen
        tensor_msg.name = 'neural_network_output'
        tensor_msg.description = 'Output tensor from CNN classifier'
        tensor_msg.timestamp = self.get_clock().now().nanoseconds / 1e9

        # Publish karen
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

## Predictions aur Detections ke liye Messages

AI systems often prediction results ya object detections produce karti hain jo specialized message types ki zarurat karti hain:

### Prediction.msg Definition

`msg/Prediction.msg` create karen:

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

# Associated image ya data reference
string data_reference
float64 timestamp
```

### Detection.msg Definition

`msg/Detection.msg` create karen:

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

`msg/DetectionsArray.msg` create karen:

```
# Detection results ka array
std_msgs/Header header
Detection[] detections
string frame_id
```

## AI Data Messages ke liye Best Practices

### 1. Efficient Data Representation

```python
# Compressed data ke sath efficient tensor message
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

AI data ke liye always relevant metadata include karen:

```python
# msg/AIPrediction.msg
string model_name
string model_version
string model_hash  # For reproducibility ke liye
float64 processing_time  # Inference mein laga time
bool is_valid  # Validation flag
string error_message  # Agar processing fail hua ho

# Input information
string input_source
float64 input_timestamp
```

### 3. Size Considerations

Bade tensor data ke liye, chunked messages consider karen:

```python
# msg/TensorChunk.msg
uint32 chunk_id
uint32 total_chunks
uint32 tensor_id  # Same tensor se belong karne wale chunks ko group karne ke liye
float32[] data_chunk
bool is_last_chunk
```

## Custom Messages Build aur Use Karna

### Package Configuration

Apne `package.xml` ko update karen message generation dependencies include karne ke liye:

```xml
<buildtool_depend>ament_cmake</buildtool_depend>
<build_depend>rosidl_default_generators</build_depend>
<exec_depend>rosidl_default_runtime</exec_depend>
<member_of_group>rosidl_interface_packages</member_of_group>
```

Apne `CMakeLists.txt` ko update karen:

```cmake
find_package(rosidl_default_generators REQUIRED)

# Apne message files find karen
set(msg_files
  "msg/Tensor.msg"
  "msg/Prediction.msg"
  "msg/Detection.msg"
  "msg/DetectionsArray.msg"
)

# Messages generate karen
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES builtin_interfaces std_msgs sensor_msgs geometry_msgs
)

ament_package()
```

### Complete Example: AI Processing Node

Yeh ek complete example hai jo AI processing ke liye custom message types ka istemal karta hai:

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

        # AI model initialize karen
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
        """AI model ke sath incoming image process karen"""
        try:
            # ROS image ko PIL mein convert karen
            pil_image = self.ros_to_pil_image(msg)

            # Image preprocess karen
            input_tensor = self.transform(pil_image).unsqueeze(0)  # Batch dimension add karen

            # Inference perform karen
            with torch.no_grad():
                results = self.model(input_tensor)

            # Tensor data publish karen
            self.publish_tensor_data(input_tensor, msg.header.stamp)

            # Predictions publish karen
            self.publish_predictions(results, msg.header.stamp)

            # Detections publish karen
            self.publish_detections(results, msg.width, msg.height, msg.header.stamp)

        except Exception as e:
            self.get_logger().error(f'AI processing error: {e}')

    def publish_tensor_data(self, tensor, stamp):
        """Tensor data ko custom message ke roop mein publish karen"""
        tensor_msg = Tensor()
        tensor_msg.header = Header()
        tensor_msg.header.stamp = stamp
        tensor_msg.header.frame_id = 'camera_optical_frame'

        # Tensor properties set karen
        tensor_msg.data_type = 0  # FLOAT32
        tensor_msg.shape = list(tensor.shape)
        tensor_msg.float32_data = tensor.flatten().tolist()
        tensor_msg.name = 'preprocessed_input'
        tensor_msg.description = 'Input tensor after preprocessing'
        tensor_msg.timestamp = stamp.sec + stamp.nanosec / 1e9

        self.tensor_pub.publish(tensor_msg)

    def publish_predictions(self, results, stamp):
        """Prediction results publish karen"""
        prediction_msg = Prediction()
        prediction_msg.header = Header()
        prediction_msg.header.stamp = stamp
        prediction_msg.model_name = 'YOLOv5s'
        prediction_msg.model_version = '6.0'
        prediction_msg.confidence_threshold = 0.5

        # Results process karen predictions extract karne ke liye
        detections = results.xyxy[0].cpu().numpy()

        if len(detections) > 0:
            # Top detection get karen
            top_detection = detections[0]
            top_confidence = top_detection[4]
            top_class_id = int(top_detection[5])

            # Top prediction set karen
            prediction_msg.top_label = str(top_class_id)
            prediction_msg.top_score = float(top_confidence)

            # Saare predictions set karen
            prediction_msg.labels = [str(int(det[5])) for det in detections]
            prediction_msg.scores = [float(det[4]) for det in detections]

        self.prediction_pub.publish(prediction_msg)

    def publish_detections(self, results, img_width, img_height, stamp):
        """Object detections publish karen"""
        detections_array = DetectionsArray()
        detections_array.header = Header()
        detections_array.header.stamp = stamp
        detections_array.header.frame_id = 'camera_optical_frame'
        detections_array.frame_id = 'camera_optical_frame'

        # YOLO results process karen
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            detection = Detection()

            # Coordinates ko image space mein convert karen
            detection.x_min = float(xyxy[0])
            detection.y_min = float(xyxy[1])
            detection.x_max = float(xyxy[2])
            detection.y_max = float(xyxy[3])
            detection.width = float(xyxy[2] - xyxy[0])
            detection.height = float(xyxy[3] - xyxy[1])

            # Class aur confidence
            detection.class_name = str(int(cls))
            detection.class_id = int(cls)
            detection.confidence = float(conf)

            # Timestamp
            detection.timestamp = stamp.sec + stamp.nanosec / 1e9

            detections_array.detections.append(detection)

        self.detection_pub.publish(detections_array)

    def ros_to_pil_image(self, ros_image):
        """ROS Image ko PIL Image mein convert karen"""
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

## AI Messages ke liye Performance Optimization

### 1. Message Size Optimization

Bade tensor data ke liye, compression ya downsampling consider karen:

```python
def compress_tensor(tensor, method='quantize'):
    """Efficient transmission ke liye tensor data compress karen"""
    if method == 'quantize':
        # Transmission ke liye 8-bit mein quantize karen
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        scale = (tensor_max - tensor_min) / 255.0
        quantized = ((tensor - tensor_min) / scale).astype(np.uint8)

        return quantized, scale, tensor_min
    elif method == 'downsample':
        # Bade tensors downsample karen
        return tensor[::2, ::2, ::2]  # Factor of 2 se downsample karen
```

### 2. Conditional Publishing

Sirf zarurat ke hisab se messages publish karen:

```python
def should_publish_result(self, current_result, previous_result, threshold=0.1):
    """Result ko publish karne ki zarurat change threshold ke adhar par determine karen"""
    if previous_result is None:
        return True

    # Results ke darmiyan difference calculate karen
    diff = abs(current_result - previous_result)
    return diff > threshold
```

## Custom Messages Test Karna

Apne custom message types verify karne ke liye test nodes create karen:

```python
import rclpy
from rclpy.node import Node
from your_package.msg import Tensor, Prediction

class MessageTester(Node):
    def __init__(self):
        super().__init__('message_tester')

        # Test publishers aur subscribers
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
        # Test tensor publish karen
        tensor_msg = Tensor()
        tensor_msg.shape = [2, 3]
        tensor_msg.float32_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        tensor_msg.name = 'test_tensor'
        self.tensor_pub.publish(tensor_msg)

        # Test prediction publish karen
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

Yeh section AI data ke liye custom ROS messages banane ko cover karta hai, jismein tensor representations, predictions, aur detections included hain. Agle section mein, hum efficient AI-ROS integration ke liye real-time data pipelines banana explore karenge.