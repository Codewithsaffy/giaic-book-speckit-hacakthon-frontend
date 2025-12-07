---
title: "AI Model Integration with ROS 2"
description: "Deploying and integrating AI models within ROS 2 nodes for robotic applications"
sidebar_position: 2
keywords: [ros2, ai, model, integration, deployment, tensorflow, pytorch]
---

# AI Model Integration with ROS 2

Integrating AI models with ROS 2 enables intelligent robotic behaviors by combining the distributed communication capabilities of ROS 2 with the computational power of artificial intelligence. This section covers various approaches to deploy and manage AI models within ROS 2 nodes.

## Model Integration Patterns

### 1. Direct Model Integration

The simplest approach is to load and run the AI model directly within a ROS 2 node:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
import tensorflow as tf  # or import torch for PyTorch

class AIModelNode(Node):
    def __init__(self):
        super().__init__('ai_model_node')

        # Load AI model during node initialization
        self.model = self.load_model()

        # Setup ROS 2 communication
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
        """Load and initialize the AI model"""
        try:
            # Example: Load a TensorFlow model
            model = tf.keras.models.load_model('/path/to/model')
            # Or for PyTorch:
            # model = torch.load('/path/to/model.pth')
            # model.eval()  # Set to evaluation mode
            self.get_logger().info('Model loaded successfully')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            return None

    def image_callback(self, msg):
        """Process incoming image and run AI inference"""
        if self.model is None:
            return

        try:
            # Convert ROS Image message to OpenCV format
            image = self.ros_image_to_cv2(msg)

            # Preprocess image for model input
            processed_image = self.preprocess_image(image)

            # Run inference
            result = self.model.predict(processed_image)

            # Process and publish results
            output_msg = self.process_results(result)
            self.result_publisher.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'Inference error: {str(e)}')

    def ros_image_to_cv2(self, ros_image):
        """Convert ROS Image message to OpenCV image"""
        # Convert ROS Image to OpenCV format
        dtype = np.uint8
        if ros_image.encoding == 'rgb8':
            dtype = np.uint8
        elif ros_image.encoding == '32FC1':
            dtype = np.float32

        # Create numpy array from image data
        image = np.frombuffer(ros_image.data, dtype=dtype)
        image = image.reshape(ros_image.height, ros_image.width, -1)

        # Convert BGR to RGB if needed
        if ros_image.encoding.startswith('bgr'):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize image to model input size
        input_height, input_width = 224, 224  # Example size
        image = cv2.resize(image, (input_width, input_height))

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def process_results(self, results):
        """Process model results and create ROS message"""
        # Example: Process classification results
        result_msg = String()
        if isinstance(results, np.ndarray):
            # Find class with highest probability
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

For on-demand inference, use services to provide AI capabilities:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from example_interfaces.srv import Trigger  # Generic service type
import tensorflow as tf

class AIModelService(Node):
    def __init__(self):
        super().__init__('ai_model_service')

        # Load model
        self.model = self.load_model()

        # Create service for inference requests
        self.service = self.create_service(
            Trigger,  # You can create custom service types
            'run_inference',
            self.inference_callback
        )

        self.get_logger().info('AI Model Service ready')

    def inference_callback(self, request, response):
        """Handle inference service requests"""
        if self.model is None:
            response.success = False
            response.message = 'Model not loaded'
            return response

        try:
            # Perform inference (implement based on your model)
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

For long-running AI tasks with progress feedback:

```python
import rclpy
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.node import Node
from example_interfaces.action import Fibonacci  # Using as example
import tensorflow as tf
import time

class AIModelActionServer(Node):
    def __init__(self):
        super().__init__('ai_model_action_server')

        # Load model
        self.model = self.load_model()

        # Create action server
        self._action_server = ActionServer(
            self,
            Fibonacci,  # Define custom action type for your use case
            'ai_inference_action',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

    def goal_callback(self, goal_request):
        """Accept or reject incoming goals"""
        self.get_logger().info('Received inference goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel requests"""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the AI inference goal"""
        self.get_logger().info('Executing inference goal')

        # Create feedback message
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = []

        try:
            # Simulate long-running inference
            for i in range(goal_handle.request.order):
                if goal_handle.is_canceling():
                    goal_handle.canceled()
                    return Fibonacci.Result()

                # Perform inference step
                # result = self.model.predict_step(i)

                # Publish feedback
                feedback_msg.partial_sequence.append(i)
                goal_handle.publish_feedback(feedback_msg)

                # Simulate processing time
                time.sleep(0.1)

            # Complete successfully
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

### 1. Model Loading and Caching

```python
class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_paths = {}

    def load_model(self, model_name, model_path, model_type='tensorflow'):
        """Load and cache a model"""
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
        """Unload and remove model from memory"""
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.model_paths:
                del self.model_paths[model_name]

    def get_model(self, model_name):
        """Get a loaded model"""
        return self.models.get(model_name)
```

### 2. Dynamic Model Switching

```python
class DynamicModelNode(Node):
    def __init__(self):
        super().__init__('dynamic_model_node')

        # Declare parameter for model selection
        self.declare_parameter('current_model', 'default_model')

        # Model registry
        self.model_manager = ModelManager()
        self.current_model_name = None

        # Setup communication
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

        # Setup parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

        # Load initial model
        self.load_initial_model()

    def parameter_callback(self, params):
        """Handle parameter changes"""
        for param in params:
            if param.name == 'current_model' and param.type_ == ParameterType.PARAMETER_STRING:
                if self.current_model_name != param.value:
                    self.switch_model(param.value)
        return SetParametersResult(successful=True)

    def load_initial_model(self):
        """Load the initial model based on parameter"""
        model_name = self.get_parameter('current_model').value
        self.switch_model(model_name)

    def switch_model(self, model_name):
        """Switch to a different model"""
        try:
            # Load new model
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
        self.input_buffer = deque(maxlen=32)  # Buffer for batching
        self.buffer_lock = threading.Lock()

        # Setup communication
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

        # Timer for batch processing
        self.batch_timer = self.create_timer(0.1, self.process_batch)

        # Load model
        self.model = self.load_model()

    def buffer_callback(self, msg):
        """Add message to processing buffer"""
        with self.buffer_lock:
            self.input_buffer.append(msg)

    def process_batch(self):
        """Process accumulated inputs in batch"""
        with self.buffer_lock:
            if len(self.input_buffer) == 0:
                return

            # Extract all inputs
            inputs = list(self.input_buffer)
            self.input_buffer.clear()

        try:
            # Batch process inputs
            results = self.batch_inference(inputs)

            # Publish results
            for result in results:
                output_msg = String()
                output_msg.data = result
                self.publisher.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f'Batch processing error: {str(e)}')

    def batch_inference(self, inputs):
        """Perform batch inference"""
        # Convert inputs to batch format
        batch_data = [self.preprocess_input(inp) for inp in inputs]
        batch_tensor = np.stack(batch_data)

        # Run batch inference
        results = self.model.predict(batch_tensor)

        # Process results
        return [self.format_result(res) for res in results]
```

### 2. GPU Acceleration

```python
import tensorflow as tf

class GPUAcceleratedNode(Node):
    def __init__(self):
        super().__init__('gpu_accelerated_node')

        # Configure GPU usage
        self.configure_gpu()

        # Load model on GPU
        with tf.device('/GPU:0'):
            self.model = self.load_model()

        # Setup communication
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
        """Configure GPU settings"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to avoid allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                self.get_logger().info(f'GPU available: {len(gpus)} device(s)')
            except RuntimeError as e:
                self.get_logger().error(f'GPU configuration error: {str(e)}')
        else:
            self.get_logger().warn('No GPU found, using CPU')

    def gpu_inference_callback(self, msg):
        """Run inference on GPU"""
        try:
            # Preprocess image
            image = self.ros_image_to_cv2(msg)
            processed_image = self.preprocess_image(image)

            # Run inference on GPU
            with tf.device('/GPU:0'):
                result = self.model.predict(processed_image)

            # Process and publish result
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

        # Declare model version parameter
        self.declare_parameter('model_version', '1.0.0')
        self.declare_parameter('model_path_pattern', '/models/{version}/model.h5')

        # Load versioned model
        self.load_versioned_model()

    def load_versioned_model(self):
        """Load model based on version parameter"""
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

        # Setup monitoring timer
        self.monitor_timer = self.create_timer(10.0, self.report_performance)

        # Load model
        self.model = self.load_model()

    def monitored_inference(self, input_data):
        """Run inference with performance monitoring"""
        start_time = time.time()

        try:
            result = self.model.predict(input_data)
            inference_time = time.time() - start_time

            # Update performance metrics
            self.inference_count += 1
            self.total_inference_time += inference_time

            return result

        except Exception as e:
            self.error_count += 1
            raise e

    def report_performance(self):
        """Report performance metrics"""
        if self.inference_count > 0:
            avg_time = self.total_inference_time / self.inference_count
            self.get_logger().info(
                f'Performance: {self.inference_count} inferences, '
                f'avg time: {avg_time:.3f}s, errors: {self.error_count}'
            )
```

AI model integration with ROS 2 enables powerful intelligent robotic applications by combining distributed communication with advanced computational capabilities. Proper implementation requires careful consideration of performance, resource management, and real-time constraints.