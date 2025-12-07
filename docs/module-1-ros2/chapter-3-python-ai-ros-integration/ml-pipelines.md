---
title: "Machine Learning Pipelines"
description: "Building and deploying machine learning pipelines within ROS 2 systems"
sidebar_position: 4
keywords: [ros2, machine learning, ml, pipeline, training, deployment, data collection]
---

# Machine Learning Pipelines

Machine learning pipelines in ROS 2 enable the complete lifecycle of ML model development, from data collection and training to deployment and inference. This section covers how to build robust ML pipelines that integrate seamlessly with ROS 2's distributed architecture.

## Data Collection Pipeline

### Sensor Data Collection Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32
import cv2
import numpy as np
import json
import os
from datetime import datetime
from cv_bridge import CvBridge

class DataCollectionNode(Node):
    def __init__(self):
        super().__init__('data_collection_node')

        # Setup CvBridge for image conversion
        self.bridge = CvBridge()

        # Data collection parameters
        self.declare_parameter('collection_directory', '/tmp/ros2_ml_data')
        self.declare_parameter('collection_rate', 10.0)  # Hz
        self.declare_parameter('enable_image_collection', True)
        self.declare_parameter('enable_laser_collection', True)

        # Initialize data storage
        self.collection_dir = self.get_parameter('collection_directory').value
        self.collection_rate = self.get_parameter('collection_rate').value
        self.enable_image = self.get_parameter('enable_image_collection').value
        self.enable_laser = self.get_parameter('enable_laser_collection').value

        # Create directory for data storage
        os.makedirs(self.collection_dir, exist_ok=True)

        # Data buffers
        self.data_buffer = []
        self.sequence_number = 0

        # Setup subscribers for different sensor types
        if self.enable_image:
            self.image_subscriber = self.create_subscription(
                Image,
                'camera/image_raw',
                self.image_callback,
                10
            )

        if self.enable_laser:
            self.laser_subscriber = self.create_subscription(
                LaserScan,
                'scan',
                self.laser_callback,
                10
            )

        # Control command subscription
        self.cmd_subscriber = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_callback,
            10
        )

        # Status publisher
        self.status_publisher = self.create_publisher(
            String,
            'data_collection_status',
            10
        )

        # Timer for periodic data saving
        self.save_timer = self.create_timer(1.0, self.save_data_buffer)

        self.get_logger().info(f'Data collection initialized in: {self.collection_dir}')

    def image_callback(self, msg):
        """Collect image data with timestamp"""
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Store image data
            image_data = {
                'type': 'image',
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'frame_id': msg.header.frame_id,
                'width': msg.width,
                'height': msg.height,
                'encoding': msg.encoding,
                'data': cv_image.tolist()  # Convert to JSON serializable
            }

            self.add_to_buffer(image_data)

        except Exception as e:
            self.get_logger().error(f'Image collection error: {str(e)}')

    def laser_callback(self, msg):
        """Collect laser scan data"""
        try:
            laser_data = {
                'type': 'laser_scan',
                'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                'frame_id': msg.header.frame_id,
                'angle_min': msg.angle_min,
                'angle_max': msg.angle_max,
                'angle_increment': msg.angle_increment,
                'time_increment': msg.time_increment,
                'scan_time': msg.scan_time,
                'range_min': msg.range_min,
                'range_max': msg.range_max,
                'ranges': list(msg.ranges),
                'intensities': list(msg.intensities)
            }

            self.add_to_buffer(laser_data)

        except Exception as e:
            self.get_logger().error(f'Laser collection error: {str(e)}')

    def cmd_callback(self, msg):
        """Collect control commands"""
        try:
            cmd_data = {
                'type': 'cmd_vel',
                'timestamp': self.get_clock().now().nanoseconds * 1e-9,
                'linear_x': msg.linear.x,
                'linear_y': msg.linear.y,
                'linear_z': msg.linear.z,
                'angular_x': msg.angular.x,
                'angular_y': msg.angular.y,
                'angular_z': msg.angular.z
            }

            self.add_to_buffer(cmd_data)

        except Exception as e:
            self.get_logger().error(f'Command collection error: {str(e)}')

    def add_to_buffer(self, data):
        """Add data to collection buffer"""
        data['sequence'] = self.sequence_number
        self.sequence_number += 1
        self.data_buffer.append(data)

        # Limit buffer size to prevent memory issues
        if len(self.data_buffer) > 1000:  # Keep last 1000 samples
            self.data_buffer = self.data_buffer[-500:]  # Keep last 500 samples

    def save_data_buffer(self):
        """Save collected data to file periodically"""
        if not self.data_buffer:
            return

        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.collection_dir, f'data_{timestamp}.json')

            # Save data
            with open(filename, 'w') as f:
                json.dump(self.data_buffer, f)

            # Clear buffer after saving
            self.data_buffer.clear()

            # Publish status
            status_msg = String()
            status_msg.data = f'Saved {timestamp} - {len(self.data_buffer)} samples'
            self.status_publisher.publish(status_msg)

            self.get_logger().info(f'Saved {timestamp} - {len(self.data_buffer)} samples')

        except Exception as e:
            self.get_logger().error(f'Data save error: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Data collection stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Training Pipeline Integration

### Online Learning Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os

class OnlineLearningNode(Node):
    def __init__(self):
        super().__init__('online_learning_node')

        # Model parameters
        self.declare_parameter('model_path', '/tmp/online_model.pkl')
        self.declare_parameter('feature_size', 10)
        self.declare_parameter('learning_rate', 0.01)

        # Initialize model
        self.model_path = self.get_parameter('model_path').value
        self.feature_size = self.get_parameter('feature_size').value
        self.learning_rate = self.get_parameter('learning_rate').value

        # Load or create model
        self.model = self.load_or_create_model()
        self.scaler = StandardScaler()

        # Training data buffers
        self.feature_buffer = []
        self.target_buffer = []

        # Setup communication
        self.data_subscriber = self.create_subscription(
            String,  # Replace with custom message type for features
            'training_data',
            self.training_callback,
            10
        )

        self.prediction_publisher = self.create_publisher(
            Float32,
            'model_prediction',
            10
        )

        self.performance_publisher = self.create_publisher(
            String,
            'training_status',
            10
        )

        # Timer for periodic model saving
        self.save_timer = self.create_timer(30.0, self.save_model)

        self.get_logger().info('Online learning node initialized')

    def load_or_create_model(self):
        """Load existing model or create new one"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model = pickle.load(f)
                self.get_logger().info('Loaded existing model')
                return model
            except Exception as e:
                self.get_logger().warn(f'Failed to load model: {str(e)}')

        # Create new model
        model = SGDRegressor(learning_rate='constant', eta0=self.learning_rate)
        self.get_logger().info('Created new model')
        return model

    def training_callback(self, msg):
        """Process training data and update model"""
        try:
            # Parse training data (simplified - in practice use custom message types)
            # Format: "feature1,feature2,...,target"
            data_parts = msg.data.split(',')
            if len(data_parts) != self.feature_size + 1:
                return

            features = np.array([float(x) for x in data_parts[:-1]]).reshape(1, -1)
            target = float(data_parts[-1])

            # Add to training buffers
            self.feature_buffer.append(features[0])
            self.target_buffer.append(target)

            # Train when we have enough data
            if len(self.feature_buffer) >= 10:
                # Convert to numpy arrays
                X = np.array(self.feature_buffer)
                y = np.array(self.target_buffer)

                # Fit scaler and transform features
                X_scaled = self.scaler.fit_transform(X)

                # Update model incrementally
                self.model.partial_fit(X_scaled, y)

                # Clear buffers periodically
                if len(self.feature_buffer) > 100:
                    self.feature_buffer = self.feature_buffer[-50:]
                    self.target_buffer = self.target_buffer[-50:]

                # Publish training status
                status_msg = String()
                status_msg.data = f'Model updated - samples: {len(self.feature_buffer)}'
                self.performance_publisher.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Training callback error: {str(e)}')

    def save_model(self):
        """Save the trained model to file"""
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump((self.model, self.scaler), f)

            self.get_logger().info(f'Model saved to {self.model_path}')

        except Exception as e:
            self.get_logger().error(f'Model save error: {str(e)}')
```

## Model Training Service

### Training Service Node

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from example_interfaces.action import Fibonacci  # Using as example, create custom action
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class ModelTrainingService(Node):
    def __init__(self):
        super().__init__('model_training_service')

        # Setup training parameters
        self.declare_parameter('data_path', '/tmp/training_data.csv')
        self.declare_parameter('model_output_path', '/tmp/trained_model.pkl')
        self.declare_parameter('test_size', 0.2)

        # Create action server for training
        self._action_server = ActionServer(
            self,
            Fibonacci,  # Replace with custom training action
            'train_model',
            self.execute_training_callback
        )

        self.get_logger().info('Model training service ready')

    def execute_training_callback(self, goal_handle):
        """Execute model training with progress feedback"""
        self.get_logger().info('Starting model training')

        try:
            # Load training data
            data_path = self.get_parameter('data_path').value
            if not os.path.exists(data_path):
                result = Fibonacci.Result()
                result.sequence = []
                goal_handle.abort()
                return result

            # Load and prepare data
            df = pd.read_csv(data_path)
            X = df.drop('target', axis=1).values
            y = df['target'].values

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.get_parameter('test_size').value
            )

            # Initialize model
            model = RandomForestRegressor(n_estimators=100, random_state=42)

            # Training with progress simulation
            total_steps = 10
            for step in range(total_steps):
                if goal_handle.is_canceling():
                    goal_handle.canceled()
                    return Fibonacci.Result()

                # Simulate training progress
                # In real implementation, this would be actual training steps
                model.fit(X_train, y_train)

                # Publish feedback
                feedback_msg = Fibonacci.Feedback()
                feedback_msg.partial_sequence = [step]
                goal_handle.publish_feedback(feedback_msg)

            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Save trained model
            model_path = self.get_parameter('model_output_path').value
            joblib.dump(model, model_path)

            # Complete training
            result = Fibonacci.Result()
            result.sequence = [int(mse * 1000), int(r2 * 1000)]  # Scale for integer result

            goal_handle.succeed()
            self.get_logger().info(f'Training completed - MSE: {mse:.4f}, R2: {r2:.4f}')

            return result

        except Exception as e:
            self.get_logger().error(f'Training error: {str(e)}')
            goal_handle.abort()
            return Fibonacci.Result()

def main(args=None):
    rclpy.init(args=args)
    node = ModelTrainingService()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Training service interrupted')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Model Evaluation and Validation

### Model Validation Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

class ModelValidationNode(Node):
    def __init__(self):
        super().__init__('model_validation_node')

        # Setup CvBridge
        self.bridge = CvBridge()

        # Validation parameters
        self.declare_parameter('model_path', '/tmp/model.h5')
        self.declare_parameter('validation_dataset_path', '/tmp/validation_data')
        self.declare_parameter('batch_size', 32)

        # Load model
        self.model = self.load_model()

        # Setup communication
        self.validation_subscriber = self.create_subscription(
            String,
            'validation_data',
            self.validation_callback,
            10
        )

        self.metrics_publisher = self.create_publisher(
            String,
            'validation_metrics',
            10
        )

        # Initialize validation metrics
        self.validation_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'loss': []
        }

        self.get_logger().info('Model validation node initialized')

    def load_model(self):
        """Load the model to be validated"""
        try:
            model_path = self.get_parameter('model_path').value
            model = tf.keras.models.load_model(model_path)
            self.get_logger().info('Model loaded for validation')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            return None

    def validation_callback(self, msg):
        """Process validation data"""
        if self.model is None:
            return

        try:
            # Parse validation data (simplified format)
            # In practice, use proper data structures
            data_parts = msg.data.split('|')
            if len(data_parts) != 2:
                return

            # Parse features and true labels
            features_str, labels_str = data_parts
            features = np.array([float(x) for x in features_str.split(',')])
            true_labels = np.array([int(x) for x in labels_str.split(',')])

            # Run inference
            predictions = self.model.predict(features.reshape(1, -1))
            predicted_labels = (predictions > 0.5).astype(int)

            # Calculate metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
            recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

            # Store results
            self.validation_results['accuracy'].append(accuracy)
            self.validation_results['precision'].append(precision)
            self.validation_results['recall'].append(recall)
            self.validation_results['f1_score'].append(f1)

            # Publish metrics
            metrics_msg = String()
            metrics_msg.data = json.dumps({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            self.metrics_publisher.publish(metrics_msg)

        except Exception as e:
            self.get_logger().error(f'Validation error: {str(e)}')

    def get_average_metrics(self):
        """Calculate average validation metrics"""
        avg_metrics = {}
        for metric, values in self.validation_results.items():
            if values:
                avg_metrics[metric] = sum(values) / len(values)
            else:
                avg_metrics[metric] = 0.0
        return avg_metrics
```

## Pipeline Orchestration

### ML Pipeline Orchestrator

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from example_interfaces.action import Fibonacci  # Custom action in practice
import subprocess
import os
from datetime import datetime

class MLPipelineOrchestrator(Node):
    def __init__(self):
        super().__init__('ml_pipeline_orchestrator')

        # Pipeline configuration
        self.declare_parameter('pipeline_config', '/tmp/pipeline_config.json')
        self.declare_parameter('data_collection_enabled', True)
        self.declare_parameter('training_enabled', True)
        self.declare_parameter('evaluation_enabled', True)

        # Pipeline state
        self.pipeline_running = False
        self.current_stage = 'idle'

        # Setup communication
        self.control_subscriber = self.create_subscription(
            String,
            'pipeline_control',
            self.control_callback,
            10
        )

        self.status_publisher = self.create_publisher(
            String,
            'pipeline_status',
            10
        )

        self.completion_publisher = self.create_publisher(
            Bool,
            'pipeline_completion',
            10
        )

        # Action server for pipeline execution
        self._action_server = ActionServer(
            self,
            Fibonacci,  # Replace with custom pipeline action
            'execute_ml_pipeline',
            self.execute_pipeline_callback,
            goal_callback=self.pipeline_goal_callback,
            cancel_callback=self.pipeline_cancel_callback
        )

        self.get_logger().info('ML Pipeline Orchestrator initialized')

    def control_callback(self, msg):
        """Handle pipeline control commands"""
        command = msg.data.lower()

        if command == 'start':
            self.start_pipeline()
        elif command == 'stop':
            self.stop_pipeline()
        elif command == 'status':
            self.publish_status()
        elif command.startswith('stage:'):
            stage = command.split(':')[1]
            self.set_pipeline_stage(stage)

    def start_pipeline(self):
        """Start the complete ML pipeline"""
        if self.pipeline_running:
            self.get_logger().warn('Pipeline already running')
            return

        self.pipeline_running = True
        self.get_logger().info('Starting ML pipeline')

        # Execute pipeline stages
        try:
            if self.get_parameter('data_collection_enabled').value:
                self.current_stage = 'data_collection'
                self.publish_status()
                self.run_data_collection()

            if self.get_parameter('training_enabled').value:
                self.current_stage = 'training'
                self.publish_status()
                self.run_training()

            if self.get_parameter('evaluation_enabled').value:
                self.current_stage = 'evaluation'
                self.publish_status()
                self.run_evaluation()

            self.current_stage = 'completed'
            self.pipeline_running = False
            self.publish_status()

            # Publish completion
            completion_msg = Bool()
            completion_msg.data = True
            self.completion_publisher.publish(completion_msg)

        except Exception as e:
            self.get_logger().error(f'Pipeline execution error: {str(e)}')
            self.pipeline_running = False
            self.current_stage = 'error'

    def run_data_collection(self):
        """Execute data collection stage"""
        self.get_logger().info('Running data collection stage')
        # In practice, this would call data collection nodes
        # or launch data collection processes

    def run_training(self):
        """Execute model training stage"""
        self.get_logger().info('Running training stage')
        # In practice, this would call training nodes
        # or launch training processes

    def run_evaluation(self):
        """Execute model evaluation stage"""
        self.get_logger().info('Running evaluation stage')
        # In practice, this would call evaluation nodes
        # or launch evaluation processes

    def stop_pipeline(self):
        """Stop the running pipeline"""
        self.pipeline_running = False
        self.current_stage = 'stopped'
        self.get_logger().info('Pipeline stopped')
        self.publish_status()

    def publish_status(self):
        """Publish current pipeline status"""
        status_msg = String()
        status_msg.data = f'Pipeline: {self.current_stage}, Running: {self.pipeline_running}'
        self.status_publisher.publish(status_msg)

    def set_pipeline_stage(self, stage):
        """Set pipeline to specific stage"""
        self.current_stage = stage
        self.get_logger().info(f'Set pipeline stage to: {stage}')
        self.publish_status()

    def pipeline_goal_callback(self, goal_request):
        """Handle pipeline execution goal"""
        if self.pipeline_running:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def pipeline_cancel_callback(self, goal_handle):
        """Handle pipeline cancellation"""
        self.stop_pipeline()
        return CancelResponse.ACCEPT

    def execute_pipeline_callback(self, goal_handle):
        """Execute pipeline as action"""
        self.get_logger().info('Executing pipeline via action')

        try:
            self.start_pipeline()

            # Create result
            result = Fibonacci.Result()
            result.sequence = [1]  # Success indicator

            goal_handle.succeed()
            return result

        except Exception as e:
            self.get_logger().error(f'Pipeline action error: {str(e)}')
            goal_handle.abort()
            return Fibonacci.Result()

def main(args=None):
    rclpy.init(args=args)
    node = MLPipelineOrchestrator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Pipeline orchestrator stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Feature Engineering Pipeline

### Feature Processing Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

class FeatureEngineeringNode(Node):
    def __init__(self):
        super().__init__('feature_engineering_node')

        # Feature processing parameters
        self.declare_parameter('feature_scaler', 'standard')  # 'standard', 'minmax', or 'none'
        self.declare_parameter('feature_selection_k', 10)    # Number of features to select
        self.declare_parameter('enable_feature_selection', False)

        # Initialize scalers
        scaler_type = self.get_parameter('feature_scaler').value
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None

        # Feature selection
        if self.get_parameter('enable_feature_selection').value:
            k = self.get_parameter('feature_selection_k').value
            self.feature_selector = SelectKBest(score_func=f_regression, k=k)
            self.feature_selector_fitted = False
        else:
            self.feature_selector = None

        # Setup communication
        self.laser_subscriber = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_feature_callback,
            10
        )

        self.feature_publisher = self.create_publisher(
            Float32MultiArray,
            'engineered_features',
            10
        )

        self.get_logger().info('Feature engineering node initialized')

    def laser_feature_callback(self, msg):
        """Extract features from laser scan data"""
        try:
            # Extract raw features from laser scan
            features = self.extract_laser_features(msg)

            # Apply feature scaling
            if self.scaler is not None:
                if len(features) > 0:
                    features = self.scaler.fit_transform([features])[0]

            # Apply feature selection
            if self.feature_selector is not None and not self.feature_selector_fitted:
                # For feature selection, we need training data with targets
                # This is simplified - in practice you'd have labeled data
                self.feature_selector_fitted = True

            # Publish engineered features
            feature_msg = Float32MultiArray()
            feature_msg.data = features.tolist()
            self.feature_publisher.publish(feature_msg)

        except Exception as e:
            self.get_logger().error(f'Feature engineering error: {str(e)}')

    def extract_laser_features(self, scan_msg):
        """Extract features from laser scan"""
        ranges = np.array(scan_msg.ranges)

        # Remove invalid ranges (inf, nan)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) == 0:
            return np.zeros(10)  # Return default features if no valid ranges

        # Basic statistical features
        features = [
            np.mean(valid_ranges),      # Mean distance
            np.std(valid_ranges),       # Std of distances
            np.min(valid_ranges),       # Min distance
            np.max(valid_ranges),       # Max distance
            len(valid_ranges),          # Number of valid readings
            np.median(valid_ranges),    # Median distance
            np.percentile(valid_ranges, 25),  # 25th percentile
            np.percentile(valid_ranges, 75),  # 75th percentile
        ]

        # Add more sophisticated features
        # Obstacle detection features
        close_obstacles = np.sum(valid_ranges < 1.0)  # Obstacles within 1m
        features.append(close_obstacles)

        return np.array(features)

def main(args=None):
    rclpy.init(args=args)
    node = FeatureEngineeringNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Feature engineering node stopped')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for ML Pipelines

### 1. Model Versioning and Management

```python
class ModelVersionManager:
    def __init__(self, model_directory='/tmp/models'):
        self.model_dir = model_directory
        self.current_version = '0.0.0'
        os.makedirs(model_directory, exist_ok=True)

    def save_model_version(self, model, version, metadata=None):
        """Save model with version and metadata"""
        model_path = os.path.join(self.model_dir, f'model_v{version}.pkl')

        model_data = {
            'model': model,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        return model_path

    def load_model_version(self, version):
        """Load specific model version"""
        model_path = os.path.join(self.model_dir, f'model_v{version}.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['metadata']
```

### 2. Pipeline Monitoring and Logging

```python
import logging
from datetime import datetime

class PipelineMonitor:
    def __init__(self, log_file='/tmp/pipeline.log'):
        self.logger = logging.getLogger('MLPipeline')
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_training_event(self, event_type, details):
        """Log training-related events"""
        self.logger.info(f'TRAINING_{event_type}: {details}')

    def log_inference_event(self, event_type, details):
        """Log inference-related events"""
        self.logger.info(f'INFERENCE_{event_type}: {details}')
```

Machine learning pipelines in ROS 2 provide a comprehensive framework for developing, training, and deploying intelligent robotic systems, enabling continuous learning and adaptation in real-world environments.