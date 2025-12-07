---
title: "ROS 2 ke sath AI Model Integration"
description: "Robotic applications ke liye ROS 2 nodes mein AI models deploy aur integrate karna"
sidebar_position: 2
keywords: [ros2, ai, model, integration, deployment, tensorflow, pytorch]
---

# ROS 2 ke sath AI Model Integration

AI models ko ROS 2 ke sath integrate karna distributed communication capabilities aur artificial intelligence ke computational power ko combine karke intelligent robotic behaviors ko enable karta hai. Yeh section ROS 2 nodes mein AI models deploy aur manage karne ke various approaches ko cover karta hai.

## Model Integration Patterns

### 1. Direct Model Integration

Sabse simple approach hai AI model ko directly ROS 2 node ke andhar load aur run karna:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
import tensorflow as tf  # ya PyTorch ke liye import torch

class AIModelNode(Node):
    def __init__(self):
        super().__init__('ai_model_node')

        # Node initialization ke dauran AI model load karen
        self.model = self.load_model()

        # ROS 2 communication setup karen
        self.image_subscriber = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        self.result_publisher = self.create_publisher(
            String,
            'ai_model_result',
            10
        )

        self.get_logger().info('AI Model Node initialized')

    def load_model(self):
        """AI model load aur initialize karen"""
        try:
            # Example: TensorFlow model load karen
            model = tf.keras.models.load_model('/path/to/model')
            # Ya PyTorch ke liye:
            # model = torch.load('/path/to/model.pth')
            # model.eval()  # Evaluation mode mein set karen
            self.get_logger().info('Model loaded successfully')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            return None

    def image_callback(self, msg):
        """Incoming image process karen aur AI inference run karen"""
        if self.model is None:
            return

        try:
            # ROS Image message ko OpenCV format mein convert karen
            image = self.ros_image_to_cv2(msg)

            # Model input ke liye image preprocess karen
            processed_image = self.preprocess_image(image)

            # Inference run karen
            result = self.model.predict(processed_image)

            # Results process aur publish karen
            output_msg = self.process_results(result)
            self.result_publisher.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'Inference error: {str(e)}')

    def ros_image_to_cv2(self, ros_image):
        """ROS Image message ko OpenCV image mein convert karen"""
        # ROS Image ko OpenCV format mein convert karen
        dtype = np.uint8
        if ros_image.encoding == 'rgb8':
            dtype = np.uint8
        elif ros_image.encoding == '32FC1':
            dtype = np.float32

        # Image data se numpy array create karen
        image = np.frombuffer(ros_image.data, dtype=dtype)
        image = image.reshape(ros_image.height, ros_image.width, -1)

        # Agar zarurat ho to BGR ko RGB mein convert karen
        if ros_image.encoding.startswith('bgr'):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def preprocess_image(self, image):
        """Model input ke liye image preprocess karen"""
        # Resize image ko model input size tak
        input_height, input_width = 224, 224  # Example size
        image = cv2.resize(image, (input_width, input_height))

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0

        # Batch dimension add karen
        image = np.expand_dims(image, axis=0)

        return image

    def process_results(self, results):
        """Model results process karen aur ROS message create karen"""
        # Example: Classification results process karen
        result_msg = String()
        if isinstance(results, np.ndarray):
            # Highest probability wala class find karen
            predicted_class = np.argmax(results[0])
            confidence = np.max(results[0])

            result_msg.data = f'Class: {predicted_class}, Confidence: {confidence:.2f}'

        return result_msg

def main(args=None):
    rclpy.init(args=args)
    node = AIModelNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Service-Based Model Integration

On-demand inference ke liye, AI capabilities provide karne ke liye services ka istemal karen:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from example_interfaces.srv import Trigger  # Generic service type
import tensorflow as tf

class AIModelService(Node):
    def __init__(self):
        super().__init__('ai_model_service')

        # Model load karen
        self.model = self.load_model()

        # Inference requests ke liye service create karen
        self.service = self.create_service(
            Trigger,  # Aap custom service types create kar sakte hain
            'run_inference',
            self.inference_callback
        )

        self.get_logger().info('AI Model Service ready')

    def inference_callback(self, request, response):
        """Inference service requests handle karen"""
        if self.model is None:
            response.success = False
            response.message = 'Model not loaded'
            return response

        try:
            # Inference perform karen (aapke model ke adhar par implement karen)
            # result = self.model.predict(self.get_current_input())

            response.success = True
            response.message = f'Inference completed successfully'
            self.get_logger().info('Inference service called')

        except Exception as e:
            response.success = False
            response.message = f'Inference failed: {str(e)}'
            self.get_logger().error(f'Inference error: {str(e)}')

        return response

def main(args=None):
    rclpy.init(args=args)
    node = AIModelService()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Service interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 3. Action-Based Model Integration

Progress feedback ke sath long-running AI tasks ke liye:

```python
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci  # Example ke roop mein istemal kiya gaya
import tensorflow as tf
import time

class AIModelActionServer(Node):
    def __init__(self):
        super().__init__('ai_model_action_server')

        # Model load karen
        self.model = self.load_model()

        # Action server create karen
        self._action_server = ActionServer(
            self,
            Fibonacci,  # Aapke use case ke liye custom action type define karen
            'ai_inference_action',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

    def goal_callback(self, goal_request):
        """Incoming goals ko accept ya reject karen"""
        self.get_logger().info('Received inference goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Cancel requests ko accept ya reject karen"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """AI inference goal execute karen"""
        self.get_logger().info('Executing inference goal')

        # Feedback message create karen
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = []

        try:
            # Long-running inference simulate karen
            for i in range(goal_handle.request.order):
                if goal_handle.is_canceling():
                    goal_handle.canceled()
                    return Fibonacci.Result()

                # Inference step perform karen
                # result = self.model.predict_step(i)

                # Feedback publish karen
                feedback_msg.partial_sequence.append(i)
                goal_handle.publish_feedback(feedback_msg)

                # Processing time simulate karen
                time.sleep(0.1)

            # Successfully complete karen
            result = Fibonacci.Result()
            result.sequence = feedback_msg.partial_sequence
            goal_handle.succeed()

            return result

        except Exception as e:
            self.get_logger().error(f'Action execution failed: {str(e)}')
            goal_handle.abort()
            return Fibonacci.Result()

def main(args=None):
    rclpy.init(args=args)
    node = AIModelActionServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Action server interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Model Management Strategies

### 1. Model Loading aur Caching

```python
class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_paths = {}

    def load_model(self, model_name, model_path, model_type='tensorflow'):
        """Model load aur cache karen"""
        if model_name in self.models:
            return self.models[model_name]

        try:
            if model_type == 'tensorflow':
                model = tf.keras.models.load_model(model_path)
            elif model_type == 'pytorch':
                import torch
                model = torch.load(model_path)
                model.eval()
            elif model_type == 'onnx':
                import onnxruntime
                model = onnxruntime.InferenceSession(model_path)

            self.models[model_name] = model
            self.model_paths[model_name] = model_path
            return model

        except Exception as e:
            raise Exception(f"Failed to load model {model_name}: {str(e)}")

    def unload_model(self, model_name):
        """Model ko memory se unload aur remove karen"""
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.model_paths:
                del self.model_paths[model_name]

    def get_model(self, model_name):
        """Loaded model get karen"""
        return self.models.get(model_name)
```

### 2. Dynamic Model Switching

```python
class DynamicModelNode(Node):
    def __init__(self):
        super().__init__('dynamic_model_node')

        # Model selection ke liye parameter declare karen
        self.declare_parameter('current_model', 'default_model')

        # Model registry
        self.model_manager = ModelManager()
        self.current_model_name = None

        # Communication setup karen
        self.subscription = self.create_subscription(
            String,
            'input_data',
            self.process_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            'model_output',
            10
        )

        # Parameter callback setup karen
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Initial model load karen
        self.load_initial_model()

    def parameter_callback(self, params):
        """Parameter changes handle karen"""
        for param in params:
            if param.name == 'current_model' and param.type_ == ParameterType.PARAMETER_STRING:
                if self.current_model_name != param.value:
                    self.switch_model(param.value)
        return SetParametersResult(successful=True)

    def load_initial_model(self):
        """Parameter ke adhar par initial model load karen"
        model_name = self.get_parameter('current_model').value
        self.switch_model(model_name)

    def switch_model(self, model_name):
        """Different model par switch karen"
        try:
            # New model load karen
            model = self.model_manager.load_model(model_name, f'/path/to/{model_name}')
            self.current_model = model
            self.current_model_name = model_name
            self.get_logger().info(f'Switched to model: {model_name}')
        except Exception as e:
            self.get_logger().error(f'Failed to switch model: {str(e)}')
```

## Performance Optimization

### 1. Batch Processing

```python
from collections import deque
import threading

class BatchProcessingNode(Node):
    def __init__(self):
        super().__init__('batch_processing_node')

        # Batch processing setup
        self.input_buffer = deque(maxlen=32)  # Batching ke liye buffer
        self.buffer_lock = threading.Lock()

        # Communication setup karen
        self.subscription = self.create_subscription(
            String,
            'input_stream',
            self.buffer_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            'batch_output',
            10
        )

        # Batch processing ke liye timer
        self.batch_timer = self.create_timer(0.1, self.process_batch)

        # Model load karen
        self.model = self.load_model()

    def buffer_callback(self, msg):
        """Processing buffer mein message add karen"
        with self.buffer_lock:
            self.input_buffer.append(msg)

    def process_batch(self):
        """Accumulated inputs ko batch mein process karen"
        with self.buffer_lock:
            if len(self.input_buffer) == 0:
                return

            # Saare inputs extract karen
            inputs = list(self.input_buffer)
            self.input_buffer.clear()

        try:
            # Batch inputs process karen
            results = self.batch_inference(inputs)

            # Results publish karen
            for result in results:
                output_msg = String()
                output_msg.data = result
                self.publisher.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'Batch processing error: {str(e)}')

    def batch_inference(self, inputs):
        """Batch inference perform karen"
        # Inputs ko batch format mein convert karen
        batch_data = [self.preprocess_input(inp) for inp in inputs]
        batch_tensor = np.stack(batch_data)

        # Batch inference run karen
        results = self.model.predict(batch_tensor)

        # Results process karen
        return [self.format_result(res) for res in results]
```

### 2. GPU Acceleration

```python
import tensorflow as tf

class GPUAcceleratedNode(Node):
    def __init__(self):
        super().__init__('gpu_accelerated_node')

        # GPU usage configure karen
        self.configure_gpu()

        # GPU par model load karen
        with tf.device('/GPU:0'):
            self.model = self.load_model()

        # Communication setup karen
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.gpu_inference_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            'gpu_result',
            10
        )

    def configure_gpu(self):
        """GPU settings configure karen"
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Memory growth enable karen taake saara GPU memory allocate na ho
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                self.get_logger().info(f'GPU available: {len(gpus)} device(s)')
            except RuntimeError as e:
                self.get_logger().error(f'GPU configuration error: {str(e)}')
        else:
            self.get_logger().warn('No GPU found, using CPU')

    def gpu_inference_callback(self, msg):
        """GPU par inference run karen"
        try:
            # Image preprocess karen
            image = self.ros_image_to_cv2(msg)
            processed_image = self.preprocess_image(image)

            # GPU par inference run karen
            with tf.device('/GPU:0'):
                result = self.model.predict(processed_image)

            # Result process aur publish karen
            output_msg = self.process_results(result)
            self.publisher.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'GPU inference error: {str(e)}')
```

## Model Deployment Best Practices

### 1. Model Versioning

```python
class VersionedModelNode(Node):
    def __init__(self):
        super().__init__('versioned_model_node')

        # Model version parameter declare karen
        self.declare_parameter('model_version', '1.0.0')
        self.declare_parameter('model_path_pattern', '/models/{version}/model.h5')

        # Versioned model load karen
        self.load_versioned_model()

    def load_versioned_model(self):
        """Version parameter ke adhar par model load karen"
        version = self.get_parameter('model_version').value
        path_pattern = self.get_parameter('model_path_pattern').value

        model_path = path_pattern.format(version=version)

        try:
            self.model = tf.keras.models.load_model(model_path)
            self.get_logger().info(f'Loaded model version {version}')
        except Exception as e:
            self.get_logger().error(f'Failed to load model version {version}: {str(e)}')
```

### 2. Model Monitoring

```python
class MonitoredModelNode(Node):
    def __init__(self):
        super().__init__('monitored_model_node')

        # Performance monitoring
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.error_count = 0

        # Monitoring timer setup karen
        self.monitor_timer = self.create_timer(10.0, self.report_performance)

        # Model load karen
        self.model = self.load_model()

    def monitored_inference(self, input_data):
        """Performance monitoring ke sath inference run karen"
        start_time = time.time()

        try:
            result = self.model.predict(input_data)
            inference_time = time.time() - start_time

            # Performance metrics update karen
            self.inference_count += 1
            self.total_inference_time += inference_time

            return result

        except Exception as e:
            self.error_count += 1
            raise e

    def report_performance(self):
        """Performance metrics report karen"
        if self.inference_count > 0:
            avg_time = self.total_inference_time / self.inference_count
            self.get_logger().info(
                f'Performance: {self.inference_count} inferences, '
                f'avg time: {avg_time:.3f}s, errors: {self.error_count}'
            )
```

ROS 2 ke sath AI model integration powerful intelligent robotic applications ko enable karta hai distributed communication ko advanced computational capabilities ke sath combine karke. Proper implementation ko performance, resource management, aur real-time constraints ke careful consideration ki zarurat hoti hai.