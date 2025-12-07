---
title: "Python AI Frameworks ko ROS ke sath connect karna"
description: "PyTorch aur TensorFlow ko intelligent robotics applications ke liye ROS 2 ke sath integrate karna"
sidebar_position: 2
keywords: [pytorch, tensorflow, ros2, python, ai, integration, machine learning]
---

# Python AI Frameworks ko ROS ke sath connect karna

Yeh section popular Python-based AI frameworks ko ROS 2 ke sath connect karne ka exploration karta hai, robotic systems mein machine learning models ka seamless integration enable karta hai. Hum PyTorch aur TensorFlow integration, model loading aur inference in ROS nodes, aur dependencies manage karne ke best practices cover karenge.

## ROS 2 ke sath PyTorch Integration

PyTorch most popular deep learning frameworks mein se ek hai, jo dynamic computation graph aur ease of use ke liye jaana jata hai. PyTorch models ko ROS 2 ke sath integrate karna robotic sensor data par real-time inference ko allow karta hai.

### Basic PyTorch Node Structure

Yeh ek complete example hai ek ROS 2 node ka jo PyTorch model ka istemal karke image classification perform karta hai:

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

        # PyTorch model initialize karen
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

        # Subscription aur publisher create karen
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(String, 'classification_result', 10)

        self.get_logger().info('PyTorch Image Classifier node initialized')

    def load_model(self):
        """Pre-trained PyTorch model load karen"""
        try:
            # Pre-trained ResNet model load karen
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
            model.eval()
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {e}')
            return None

    def image_callback(self, msg):
        """Incoming image ko process karen aur classification perform karen"""
        if self.model is None:
            return

        try:
            # ROS Image message ko PIL Image mein convert karen
            pil_image = self.ros_to_pil_image(msg)

            # Image preprocess karen
            input_tensor = self.transform(pil_image)
            input_batch = input_tensor.unsqueeze(0)  # Batch dimension add karen

            # Inference perform karen
            with torch.no_grad():
                output = self.model(input_batch)

            # Predicted class get karen
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

            # Result publish karen
            result_msg = String()
            result_msg.data = f'Class: {predicted_class}, Confidence: {confidence:.2f}'
            self.publisher.publish(result_msg)

            self.get_logger().info(f'Classification result: {result_msg.data}')

        except Exception as e:
            self.get_logger().error(f'Error during inference: {e}')

    def ros_to_pil_image(self, ros_image):
        """ROS Image message ko PIL Image mein convert karen"""
        # Encoding ke adhar par image format convert karen
        if ros_image.encoding == 'rgb8':
            # ROS image data ko numpy array mein convert karen
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

## ROS 2 ke sath TensorFlow Integration

TensorFlow another leading deep learning framework hai, jo particularly popular hai production deployments ke liye. Yahan dikha gaya hai kaise TensorFlow models ko ROS 2 ke sath integrate karna hai:

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

        # TensorFlow model load karen
        self.model = self.load_model()

        # Subscription aur publisher create karen
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(String, 'tf_classification_result', 10)

        self.get_logger().info('TensorFlow Image Classifier node initialized')

    def load_model(self):
        """Pre-trained TensorFlow model load karen"""
        try:
            # ImageNet par pre-trained MobileNetV2 model load karen
            model = tf.keras.applications.MobileNetV2(weights='imagenet')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load TensorFlow model: {e}')
            return None

    def image_callback(self, msg):
        """Incoming image ko process karen aur TensorFlow ke sath classification perform karen"""
        if self.model is None:
            return

        try:
            # ROS Image ko numpy array mein convert karen
            image_array = self.ros_image_to_numpy(msg)

            # TensorFlow ke liye image preprocess karen
            processed_image = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
            processed_image = np.expand_dims(processed_image, axis=0)  # Batch dimension add karen

            # Inference perform karen
            predictions = self.model.predict(processed_image)

            # Predictions decode karen
            decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(
                predictions, top=1
            )[0][0]

            # Results extract karen
            class_name = decoded_predictions[1]
            confidence = float(decoded_predictions[2])

            # Result publish karen
            result_msg = String()
            result_msg.data = f'Class: {class_name}, Confidence: {confidence:.2f}'
            self.publisher.publish(result_msg)

            self.get_logger().info(f'TensorFlow result: {result_msg.data}')

        except Exception as e:
            self.get_logger().error(f'Error during TensorFlow inference: {e}')

    def ros_image_to_numpy(self, ros_image):
        """ROS Image message ko TensorFlow ke liye numpy array mein convert karen"""
        # ROS image data ko numpy array mein convert karen
        height = ros_image.height
        width = ros_image.width

        if ros_image.encoding in ['rgb8', 'bgr8']:
            channels = 3
            image_array = np.frombuffer(ros_image.data, dtype=np.uint8).reshape(height, width, channels)

            # Agar zarurat ho to BGR ko RGB mein convert karen
            if ros_image.encoding == 'bgr8':
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            # Image ko model input size tak resize karen (MobileNetV2 ke liye 224x224)
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

## ROS Nodes mein Model Loading aur Inference

ROS nodes mein AI models deploy karte samay, memory usage, loading time, aur inference performance ko consider karna crucial hai:

### Model Management ke liye Best Practices

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

        # Model ko optimizations ke sath load karen
        self.model = self.load_optimized_model()

        # Model ko evaluation mode mein set karen
        if self.model is not None:
            self.model.eval()

            # Agar available ho to model ko GPU par move karen
            if self.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                self.get_logger().info('Model moved to GPU')
            else:
                self.get_logger().info('Using CPU for inference')

    def load_optimized_model(self):
        """Model ko various optimizations ke sath load karen"""
        try:
            if self.model_path and os.path.exists(self.model_path):
                # File se load karen
                model = torch.load(self.model_path)
            else:
                # Hub se ya default model load karen
                model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)

            # Optimizations apply karen
            model = self.apply_optimizations(model)
            return model
        except Exception as e:
            self.get_logger().error(f'Model loading error: {e}')
            return None

    def apply_optimizations(self, model):
        """Various optimization techniques apply karen"""
        # Faster inference ke liye TorchScript mein convert karen
        try:
            model = torch.jit.script(model)
            self.get_logger().info('Model converted to TorchScript')
        except Exception as e:
            self.get_logger().warning(f'Could not convert to TorchScript: {e}')

        return model

    def perform_inference(self, input_tensor):
        """Optimized inference perform karen"""
        with torch.no_grad():  # Inference ke liye gradient computation disable karen
            if self.use_gpu and torch.cuda.is_available():
                input_tensor = input_tensor.cuda()

            output = self.model(input_tensor)
            return output.cpu() if self.use_gpu and torch.cuda.is_available() else output
```

## Dependencies Manage Karna

AI-ROS integration ke liye proper dependency management crucial hai:

### Requirements File Example

Apne AI dependencies ke liye ek `requirements.txt` file create karen:

```
torch>=1.9.0
torchvision>=0.10.0
tensorflow>=2.6.0
numpy>=1.19.0
Pillow>=8.0.0
opencv-python>=4.5.0
```

### Package.xml Dependencies

Apne ROS 2 package.xml mein Python dependencies include karen:

```xml
<depend>python3-pytorch</depend>
<depend>python3-tensorflow</depend>
<depend>python3-numpy</depend>
<depend>python3-opencv</depend>
```

### Installation Script

Sare dependencies properly set up hone ka ensure karne ke liye ek installation script create karen:

```bash
#!/bin/bash
# install_ai_dependencies.sh

# Python dependencies install karen
pip3 install torch torchvision tensorflow numpy Pillow opencv-python

# Installations verify karen
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"

echo "AI dependencies installed successfully"
```

## Performance Considerations

ROS 2 ke sath AI frameworks integrate karte samay, in performance factors ko consider karen:

1. **Model Size**: Bade models better accuracy provide karte hain lekin zyada computational resources ki requirement karte hain
2. **Inference Speed**: Accuracy aur real-time performance requirements ke darmiyan balance rakhen
3. **Memory Usage**: GPU/CPU memory consumption ko monitor karen, khaas kar embedded systems par
4. **Batch Processing**: Improved throughput ke liye batch processing consider karen

## Practical Example: Object Detection Pipeline

Yeh ek complete example hai multiple AI frameworks ko ROS 2 ke sath combine karke object detection ke liye:

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

        # Object detection model load karen (YOLOv5)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        # Publishers aur subscribers
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
        """Image ko process karen aur objects detect karen"""
        try:
            # ROS image ko PIL mein convert karen
            pil_image = self.ros_to_pil_image(msg)

            # Object detection perform karen
            results = self.model(pil_image)

            # Results ko ROS messages mein convert karen
            detections = self.convert_yolo_results_to_ros(results, msg.width, msg.height)

            # Detections publish karen
            self.detection_publisher.publish(detections)

        except Exception as e:
            self.get_logger().error(f'Object detection error: {e}')

    def convert_yolo_results_to_ros(self, results, img_width, img_height):
        """YOLO results ko ROS Detection2DArray message mein convert karen"""
        detections_msg = Detection2DArray()

        # Detection results get karen
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            detection = Detection2D()

            # Bounding box coordinates convert karen
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            width = xyxy[2] - xyxy[0]
            height = xyxy[3] - xyxy[1]

            # Bounding box set karen
            detection.bbox.center.x = x_center
            detection.bbox.center.y = y_center
            detection.bbox.size_x = width
            detection.bbox.size_y = height

            # Hypothesis set karen
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(int(cls))
            hypothesis.hypothesis.score = float(conf)
            detection.results.append(hypothesis)

            detections_msg.detections.append(detection)

        return detections_msg

    def ros_to_pil_image(self, ros_image):
        """ROS Image ko PIL Image mein convert karen"""
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

Yeh section Python AI frameworks ko ROS 2 ke sath connect karne ka comprehensive overview provide karta hai. Agle section mein, hum efficient AI data communication ke liye custom message types create karne ka exploration karenge.