---
title: "ROS 2 ke sath AI Model Integration"
description: "Robotic applications ke liye ROS 2 nodes mein AI models deploy aur integrate karna"
sidebar_position: 2
keywords: [ros2, ai, model, integration, deployment, tensorflow, pytorch]
---

# ROS 2 ke sath AI Model Integration

AI models ko ROS 2 ke sath integrate karna intelligent robotic behaviors ko enable karta hai jo ROS 2 ke distributed communication capabilities aur artificial intelligence ke computational power ko combine karta hai. Yeh section various approaches ko cover karta hai jo ROS 2 nodes mein AI models deploy aur manage karne ke liye hai.

## Model Integration Patterns

### 1. Direct Model Integration

Sabse simple approach hai AI model ko directly ek ROS 2 node ke andhar load aur run karna:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2
import numpy as np
import tensorflow as tf  # ya torch ko PyTorch ke liye import karen

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
        """AI model load aur initialize karen"
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
        """Incoming image process karen aur AI inference run karen"
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
        """ROS Image message ko OpenCV image mein convert karen"
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
        """Model input ke liye image preprocess karen"
        # Resize image to model input size
        input_height, input_width = 224, 224  # Example size
        image = cv2.resize(image, (input_width, input_height))

        # Normalize pixel values
        image = image.astype(np.float32) / 255.0

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        return image

    def process_results(self, results):
        """Process model results aur ROS message create karen"
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

Demand-based inference ke liye, services ka istemal karke AI capabilities provide karen:

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
        """Handle inference service requests"
        if self.model is None:
            response.success = False
            response.message = 'Model not loaded'
            return response

        try:
            # Perform inference (aapke model ke adhar par implement karen)
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
        """Accept ya reject incoming goals"
        self.get_logger().info('Received inference goal')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept ya reject cancel requests"
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        """Execute the inference goal"
        result = Fibonacci.Result()

        if self.model is None:
            result.sequence = [0]  # Error indicator
            goal_handle.abort()
            return result

        try:
            # Perform long-running inference
            # In real implementation, this would run your actual AI model
            for i in range(10):  # Simulate processing steps
                if goal_handle.is_canceling():
                    result.sequence = [0]
                    goal_handle.canceled()
                    return result

                # Simulate processing
                time.sleep(0.1)

                # Publish feedback
                feedback_msg = Fibonacci.Feedback()
                feedback_msg.partial_sequence = [i]
                goal_handle.publish_feedback(feedback_msg)

            # Complete successfully
            result.sequence = [1]  # Success indicator
            goal_handle.succeed()
            self.get_logger().info('Inference action completed')

        except Exception as e:
            result.sequence = [0]  # Error indicator
            goal_handle.abort()
            self.get_logger().error(f'Action execution failed: {str(e)}')

        return result

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

## Visual vs Collision Geometries

### Key Differences

URDF mein, har link ke alag-alag visual aur collision geometries ho sakti hain:

- **Visual geometry**: Define karta hai kaise robot visualization tools jaise RViz mein look karta hai
- **Collision geometry**: Define karta hai kaise robot physics simulations (Gazebo) mein interact karta hai
- **Inertial properties**: Define karta hai physical mass aur moment of inertia for dynamics

```xml
<link name="example_link">
  <!-- Visual geometry - kaise yeh look karta hai -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://robot_description/meshes/detailed_model.dae"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>

  <!-- Collision geometry - kaise yeh collide karta hai -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.1" length="0.3"/>
    </geometry>
  </collision>

  <!-- Inertial properties - physical properties -->
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>
```

### Why Separate Geometries?

1. **Performance**: Detailed meshes ke sath collision detection computationally expensive hai
2. **Simplicity**: Simple shapes collision detection ke liye zyada reliable hain
3. **Flexibility**: Different purposes ke liye different representations
4. **Quality**: Visual models zyada detailed ho sakti hain collision models se

## Using Primitive Shapes

### Basic Primitive Shapes

URDF kuchh primitive geometric shapes ko support karta hai jo collision detection ke liye efficient hain:

#### Box
```xml
<visual>
  <geometry>
    <box size="0.2 0.1 0.3"/>  <!-- width height depth -->
  </geometry>
</visual>
<collision>
  <geometry>
    <box size="0.2 0.1 0.3"/>
  </geometry>
</collision>
```

#### Cylinder
```xml
<visual>
  <geometry>
    <cylinder radius="0.05" length="0.3"/>
  </geometry>
</visual>
<collision>
  <geometry>
    <cylinder radius="0.05" length="0.3"/>
  </geometry>
</collision>
```

#### Sphere
```xml
<visual>
  <geometry>
    <sphere radius="0.1"/>
  </geometry>
</visual>
<collision>
  <geometry>
    <sphere radius="0.1"/>
  </geometry>
</collision>
```

### Humanoid-Specific Primitive Configurations

#### Torso (Cylinder)
```xml
<link name="torso">
  <visual>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.12" length="0.5"/>
    </geometry>
    <material name="skin_color">
      <color rgba="0.9 0.7 0.5 1.0"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.12" length="0.5"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="15.0"/>
    <inertia ixx="0.3" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.3"/>
  </inertial>
</link>
```

#### Limbs (Cylinders)
```xml
<link name="upper_arm">
  <visual>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.05" length="0.3"/>
    </geometry>
    <material name="skin_color">
      <color rgba="0.9 0.7 0.5 1.0"/>
    </material>
  </visual>
  <collision>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.05" length="0.3"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="1.5"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>
```

#### Feet (Boxes)
```xml
<link name="foot">
  <visual>
    <geometry>
      <box size="0.18 0.08 0.06"/>
    </geometry>
    <material name="black">
      <color rgba="0.1 0.1 0.1 1.0"/>
    </material>
  </visual>
  <collision>
    <geometry>
      <box size="0.18 0.08 0.06"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="0.8"/>
    <inertia ixx="0.002" ixy="0.0" ixz="0.0" iyy="0.003" iyz="0.0" izz="0.004"/>
  </inertial>
</link>
```

## Importing STL/DAE Meshes

### Mesh File Formats

URDF kuchh mesh formats ko support karta hai:
- **STL**: Simple aur widely supported
- **DAE**: Collada format with textures aur materials
- **OBJ**: Wavefront OBJ format

### Basic Mesh Usage

```xml
<link name="detailed_head">
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/head.dae" scale="1 1 1"/>
    </geometry>
    <material name="head_material">
      <color rgba="0.9 0.7 0.5 1.0"/>
    </material>
  </visual>
  <collision>
    <!-- Performance ke liye simpler collision geometry ka istemal karen -->
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <sphere radius="0.1"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="2.0"/>
    <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
  </inertial>
</link>
```

### Mesh Organization Best Practices

#### File Structure
```
robot_description/
├── meshes/
│   ├── base/
│   │   ├── torso.dae
│   │   └── head.dae
│   ├── arms/
│   │   ├── upper_arm.dae
│   │   └── lower_arm.dae
│   └── legs/
│       ├── thigh.dae
│       ├── shin.dae
│       └── foot.dae
├── urdf/
│   └── robot.urdf
└── package.xml
```

#### Mesh Optimization for Collision

Complex visual meshes ke liye, simplified collision versions create karen:

```xml
<link name="detailed_torso">
  <!-- Detailed visual mesh -->
  <visual>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/torso_visual.dae"/>
    </geometry>
  </visual>

  <!-- Simplified collision mesh -->
  <collision>
    <geometry>
      <mesh filename="package://humanoid_description/meshes/torso_collision.stl"/>
    </geometry>
  </collision>

  <!-- Ya multiple simple shapes ka istemal karen -->
  <collision>
    <origin xyz="0 0 0.2" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.1" length="0.3"/>
    </geometry>
  </collision>
  <collision>
    <origin xyz="0 0 0.45" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.08" length="0.1"/>
    </geometry>
  </collision>
</link>
```

## Collision Detection Optimization

### Simple vs Complex Collision Geometries

#### Option 1: Single Complex Shape
```xml
<!-- Kam efficient lekin simpler -->
<collision>
  <geometry>
    <mesh filename="package://robot/meshes/complex_shape.stl"/>
  </geometry>
</collision>
```

#### Option 2: Multiple Simple Shapes (Recommended)
```xml
<!-- Zaida efficient aur reliable -->
<collision>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
  <geometry>
    <cylinder radius="0.05" length="0.2"/>
  </geometry>
</collision>
<collision>
  <origin xyz="0.05 0 0" rpy="0 1.57 0"/>
  <geometry>
    <cylinder radius="0.03" length="0.1"/>
  </geometry>
</collision>
```

Yeh section AI models ko ROS 2 ke sath integrate karne aur robot visualization aur physics simulation ke liye appropriate geometries define karne ke various approaches ko cover karta hai.