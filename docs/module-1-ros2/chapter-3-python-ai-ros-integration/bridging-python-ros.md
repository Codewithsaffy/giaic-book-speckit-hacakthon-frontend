---
title: "Connecting Python AI Frameworks with ROS"
description: "Integrating PyTorch and TensorFlow with ROS 2 for intelligent robotics applications"
sidebar_position: 2
keywords: [pytorch, tensorflow, ros2, python, ai, integration, machine learning]
---

# Connecting Python AI Frameworks with ROS

This section explores how to connect popular Python-based AI frameworks with ROS 2, enabling seamless integration of machine learning models into robotic systems. We'll cover PyTorch and TensorFlow integration, model loading and inference in ROS nodes, and best practices for managing dependencies.

## PyTorch Integration with ROS 2

PyTorch is one of the most popular deep learning frameworks, known for its dynamic computation graph and ease of use. Integrating PyTorch models with ROS 2 allows for real-time inference on robotic sensor data.

### Basic PyTorch Node Structure

Here's a complete example of a ROS 2 node that performs image classification using a PyTorch model:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
import numpy as np
import io

class PyTorchImageClassifier(Node):
    def __init__(self):
        super().__init__('pytorch_image_classifier')

        # Initialize PyTorch model
        self.model = self.load_model()
        self.model.eval()

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Create subscription and publisher
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(String, 'classification_result', 10)

        self.get_logger().info('PyTorch Image Classifier node initialized')

    def load_model(self):
        """Load a pre-trained PyTorch model"""
        try:
            # Load a pre-trained ResNet model
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            model.eval()
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return None

    def image_callback(self, msg):
        """Process incoming image and perform classification"""
        if self.model is None:
            return

        try:
            # Convert ROS Image message to PIL Image
            pil_image = self.ros_to_pil_image(msg)

            # Preprocess image
            input_tensor = self.transform(pil_image)
            input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

            # Perform inference
            with torch.no_grad():
                output = self.model(input_batch)

            # Get predicted class
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

            # Publish result
            result_msg = String()
            result_msg.data = f'Class: {predicted_class}, Confidence: {confidence:.2f}'
            self.publisher.publish(result_msg)

            self.get_logger().info(f'Classification result: {result_msg.data}')

        except Exception as e:
            self.get_logger().error(f'Error during inference: {e}')

    def ros_to_pil_image(self, ros_image):
        """Convert ROS Image message to PIL Image"""
        # Convert the image format based on encoding
        if ros_image.encoding == 'rgb8':
            # Convert ROS image data to numpy array
            height = ros_image.height
            width = ros_image.width
            image_array = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(height, width, -1)
            return PILImage.fromarray(image_array)

def main(args=None):
    rclpy.init(args=args)
    node = PyTorchImageClassifier()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down PyTorch Image Classifier node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## TensorFlow Integration with ROS 2

TensorFlow is another leading deep learning framework, particularly popular for production deployments. Here's how to integrate TensorFlow models with ROS 2:

### Basic TensorFlow Node Structure

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import tensorflow as tf
import numpy as np
from PIL import Image as PILImage
import cv2

class TensorFlowImageClassifier(Node):
    def __init__(self):
        super().__init__('tensorflow_image_classifier')

        # Load TensorFlow model
        self.model = self.load_model()

        # Create subscription and publisher
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(String, 'tf_classification_result', 10)

        self.get_logger().info('TensorFlow Image Classifier node initialized')

    def load_model(self):
        """Load a pre-trained TensorFlow model"""
        try:
            # Load MobileNetV2 model pre-trained on ImageNet
            model = tf.keras.applications.MobileNetV2(weights='imagenet')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load TensorFlow model: {e}')
            return None

    def image_callback(self, msg):
        """Process incoming image and perform classification with TensorFlow"""
        if self.model is None:
            return

        try:
            # Convert ROS Image to numpy array
            image_array = self.ros_image_to_numpy(msg)

            # Preprocess image for TensorFlow
            processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
            processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension

            # Perform inference
            predictions = self.model.predict(processed_image)

            # Decode predictions
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
                predictions, top=1
            )[0][0]

            # Extract results
            class_name = decoded_predictions[1]
            confidence = float(decoded_predictions[2])

            # Publish result
            result_msg = String()
            result_msg.data = f'Class: {class_name}, Confidence: {confidence:.2f}'
            self.publisher.publish(result_msg)

            self.get_logger().info(f'TensorFlow result: {result_msg.data}')

        except Exception as e:
            self.get_logger().error(f'Error during TensorFlow inference: {e}')

    def ros_image_to_numpy(self, ros_image):
        """Convert ROS Image message to numpy array for TensorFlow"""
        # Convert ROS image data to numpy array
        height = ros_image.height
        width = ros_image.width

        if ros_image.encoding in ['rgb8', 'bgr8']:
            channels = 3
            image_array = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(height, width, channels)

            # Convert BGR to RGB if needed
            if ros_image.encoding == 'bgr8':
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            # Resize image to model input size (224x224 for MobileNetV2)
            image_array = cv2.resize(image_array, (224, 224))
            return image_array
        else:
            raise ValueError(f"Unsupported image encoding: {ros_image.encoding}")

def main(args=None):
    rclpy.init(args=args)
    node = TensorFlowImageClassifier()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down TensorFlow Image Classifier node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Model Loading and Inference in ROS Nodes

When deploying AI models in ROS nodes, it's crucial to consider memory usage, loading time, and inference performance:

### Best Practices for Model Management

```python
import rclpy
from rclpy.node import Node
import torch
import os

class OptimizedModelNode(Node):
    def __init__(self):
        super().__init__('optimized_model_node')

        # Model configuration
        self.model_path = self.declare_parameter('model_path', '').get_parameter_value().string_value
        self.use_gpu = self.declare_parameter('use_gpu', False).get_parameter_value().bool_value

        # Load model with optimizations
        self.model = self.load_optimized_model()

        # Set model to evaluation mode
        if self.model is not None:
            self.model.eval()

            # Move model to GPU if available
            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                self.get_logger().info('Model moved to GPU')
            else:
                self.get_logger().info('Using CPU for inference')

    def load_optimized_model(self):
        """Load model with various optimizations"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load from file
                model = torch.load(self.model_path)
            else:
                # Load from hub or default model
                model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

            # Apply optimizations
            model = self.apply_optimizations(model)
            return model
        except Exception as e:
            self.get_logger().error(f'Model loading error: {e}')
            return None

    def apply_optimizations(self, model):
        """Apply various optimization techniques"""
        # Convert to TorchScript for faster inference
        try:
            model = torch.jit.script(model)
            self.get_logger().info('Model converted to TorchScript')
        except Exception as e:
            self.get_logger().warning(f'Could not convert to TorchScript: {e}')

        return model

    def perform_inference(self, input_tensor):
        """Perform optimized inference"""
        with torch.no_grad():  # Disable gradient computation for inference
            if self.use_gpu and torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            output = self.model(input_tensor)
            return output.cpu() if self.use_gpu and torch.cuda.is_available() else output
```

## Managing Dependencies

Proper dependency management is crucial for AI-ROS integration:

### Requirements File Example

Create a `requirements.txt` file for your AI dependencies:

```
torch>=1.9.0
torchvision>=0.10.0
tensorflow>=2.6.0
numpy>=1.19.0
Pillow>=8.0.0
opencv-python>=4.5.0
```

### Package.xml Dependencies

In your ROS 2 package.xml, include Python dependencies:

```xml
<depend>python3-pytorch</depend>
<depend>python3-tensorflow</depend>
<depend>python3-numpy</depend>
<depend>python3-opencv</depend>
```

### Installation Script

Create an installation script to ensure all dependencies are properly set up:

```bash
#!/bin/bash
# install_ai_dependencies.sh

# Install Python dependencies
pip3 install torch torchvision tensorflow numpy Pillow opencv-python

# Verify installations
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"

echo "AI dependencies installed successfully"
```

## Performance Considerations

When integrating AI frameworks with ROS 2, consider these performance factors:

1. **Model Size**: Larger models provide better accuracy but require more computational resources
2. **Inference Speed**: Balance between accuracy and real-time performance requirements
3. **Memory Usage**: Monitor GPU/CPU memory consumption, especially on embedded systems
4. **Batch Processing**: Consider batch processing for improved throughput

## Practical Example: Object Detection Pipeline

Here's a complete example combining multiple AI frameworks with ROS 2 for object detection:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
import torch
import torchvision.transforms as transforms
from PIL import Image as PILImage
import numpy as np

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Load object detection model (YOLOv5)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Publishers and subscribers
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        self.detection_publisher = self.create_publisher(
            Detection2DArray,
            'object_detections',
            10
        )

        self.get_logger().info('Object Detection Node initialized')

    def image_callback(self, msg):
        """Process image and detect objects"""
        try:
            # Convert ROS image to PIL
            pil_image = self.ros_to_pil_image(msg)

            # Perform object detection
            results = self.model(pil_image)

            # Convert results to ROS messages
            detections = self.convert_yolo_results_to_ros(results, msg.width, msg.height)

            # Publish detections
            self.detection_publisher.publish(detections)

        except Exception as e:
            self.get_logger().error(f'Object detection error: {e}')

    def convert_yolo_results_to_ros(self, results, img_width, img_height):
        """Convert YOLO results to ROS Detection2DArray message"""
        detections_msg = Detection2DArray()

        # Get detection results
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            detection = Detection2D()

            # Convert bounding box coordinates
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]

            # Set bounding box
            detection.bbox.center.x = x_center
            detection.bbox.center.y = y_center
            detection.bbox.size_x = width
            detection.bbox.size_y = height

            # Set hypothesis
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(int(cls))
            hypothesis.hypothesis.score = float(conf)
            detection.results.append(hypothesis)

            detections_msg.detections.append(detection)

        return detections_msg

    def ros_to_pil_image(self, ros_image):
        """Convert ROS Image to PIL Image"""
        height = ros_image.height
        width = ros_image.width
        image_array = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(height, width, -1)
        return PILImage.fromarray(image_array)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down Object Detection Node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

This section provided a comprehensive overview of connecting Python AI frameworks with ROS 2. In the next section, we'll explore creating custom message types for efficient AI data communication.