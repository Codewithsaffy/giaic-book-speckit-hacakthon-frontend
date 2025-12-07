---
title: "Services and Actions"
description: "Understanding ROS 2 services for synchronous communication and actions for long-running tasks in humanoid robotics"
sidebar_position: 4
keywords: ["ROS 2 services", "actions", "synchronous communication", "humanoid robotics", "robot control"]
---

# Services and Actions

While topics enable asynchronous communication, services and actions provide synchronous and long-running communication patterns essential for humanoid robotics. Services handle request-response interactions, while actions manage complex tasks with feedback and goal management.

## Services: Synchronous Request-Response Communication

Services provide a blocking, synchronous communication pattern where a client sends a request and waits for a response. This is ideal for humanoid robotics applications requiring immediate responses, such as:

- Joint position commands
- Sensor calibration
- System configuration changes
- Diagnostic queries
- Emergency procedures

### Service Server Implementation

```python
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from humanoid_interfaces.srv import SetJointPositions, GetJointLimits

class JointControlService(Node):
    def __init__(self):
        super().__init__('joint_control_service')

        # Create service with custom callback group for parallel handling
        self.joint_pos_service = self.create_service(
            SetJointPositions,
            '/set_joint_positions',
            self.set_joint_positions_callback,
            callback_group=MutuallyExclusiveCallbackGroup()
        )

        self.joint_limits_service = self.create_service(
            GetJointLimits,
            '/get_joint_limits',
            self.get_joint_limits_callback
        )

        # Store current joint positions and limits
        self.current_positions = {}
        self.joint_limits = {
            'left_hip_pitch': (-1.5, 1.5),
            'right_hip_pitch': (-1.5, 1.5),
            'left_knee_pitch': (-0.5, 2.0),
            'right_knee_pitch': (-0.5, 2.0)
        }

    def set_joint_positions_callback(self, request, response):
        """Handle joint position setting requests"""
        try:
            self.get_logger().info(f'Setting joint positions: {request.positions}')

            # Validate joint positions against limits
            for joint_name, target_pos in zip(request.joint_names, request.positions):
                if joint_name in self.joint_limits:
                    min_limit, max_limit = self.joint_limits[joint_name]
                    if not (min_limit <= target_pos <= max_limit):
                        response.success = False
                        response.message = f'Joint {joint_name} position {target_pos} exceeds limits'
                        return response

            # Execute the joint position command (simplified)
            self.execute_joint_command(request.joint_names, request.positions)

            # Update internal state
            for joint_name, pos in zip(request.joint_names, request.positions):
                self.current_positions[joint_name] = pos

            response.success = True
            response.message = 'Joint positions set successfully'
            self.get_logger().info('Joint positions updated successfully')

        except Exception as e:
            response.success = False
            response.message = f'Error setting joint positions: {str(e)}'
            self.get_logger().error(f'Service error: {e}')

        return response

    def get_joint_limits_callback(self, request, response):
        """Return joint limits for requested joints"""
        try:
            if not request.joint_names:  # Return all limits if no specific joints requested
                response.joint_names = list(self.joint_limits.keys())
                response.min_limits = [self.joint_limits[name][0] for name in response.joint_names]
                response.max_limits = [self.joint_limits[name][1] for name in response.joint_names]
            else:
                # Return limits for specific joints
                for joint_name in request.joint_names:
                    if joint_name in self.joint_limits:
                        response.joint_names.append(joint_name)
                        response.min_limits.append(self.joint_limits[joint_name][0])
                        response.max_limits.append(self.joint_limits[joint_name][1])
                    else:
                        response.success = False
                        response.message = f'Joint {joint_name} not found'
                        return response

            response.success = True
            response.message = 'Joint limits retrieved successfully'

        except Exception as e:
            response.success = False
            response.message = f'Error retrieving joint limits: {str(e)}'
            self.get_logger().error(f'Joint limits service error: {e}')

        return response

    def execute_joint_command(self, joint_names, positions):
        """Execute the actual joint command (placeholder)"""
        # In a real implementation, this would interface with the hardware
        # or lower-level control system
        pass
```

### Service Client Implementation

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_services_default
from humanoid_interfaces.srv import SetJointPositions, GetJointLimits

class JointControlClient(Node):
    def __init__(self):
        super().__init__('joint_control_client')

        # Create clients for the services
        self.joint_pos_client = self.create_client(
            SetJointPositions,
            '/set_joint_positions'
        )
        self.joint_limits_client = self.create_client(
            GetJointLimits,
            '/get_joint_limits'
        )

        # Wait for services to be available
        while not self.joint_pos_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Joint position service not available, waiting...')

        while not self.joint_limits_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Joint limits service not available, waiting...')

    def set_joint_positions_async(self, joint_names, positions):
        """Send joint position request asynchronously"""
        request = SetJointPositions.Request()
        request.joint_names = joint_names
        request.positions = positions

        # Send asynchronous request
        future = self.joint_pos_client.call_async(request)
        future.add_done_callback(self.joint_position_response_callback)

        return future

    def joint_position_response_callback(self, future):
        """Handle joint position response"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Success: {response.message}')
            else:
                self.get_logger().error(f'Failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    async def get_joint_limits_async(self, joint_names=None):
        """Get joint limits asynchronously"""
        request = GetJointLimits.Request()
        if joint_names:
            request.joint_names = joint_names

        try:
            response = await self.joint_limits_client.call_async(request)
            if response.success:
                self.get_logger().info(f'Joint limits: {dict(zip(response.joint_names, zip(response.min_limits, response.max_limits)))}')
            else:
                self.get_logger().error(f'Failed to get joint limits: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Get joint limits failed: {e}')
```

## Actions: Long-Running Tasks with Feedback

Actions are designed for long-running operations that require feedback, goal management, and cancellation. Perfect for humanoid robotics tasks like:

- Walking pattern execution
- Complex manipulation sequences
- Navigation to waypoints
- Calibration procedures
- Dance or gesture execution

### Action Server Implementation

```python
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
import threading
import time

from humanoid_interfaces.action import WalkPattern
from geometry_msgs.msg import Pose

class WalkPatternActionServer(Node):
    def __init__(self):
        super().__init__('walk_pattern_action_server')

        # Use reentrant callback group to handle multiple requests
        callback_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self,
            WalkPattern,
            'execute_walk_pattern',
            execute_callback=self.execute_callback,
            callback_group=callback_group,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback
        )

        # Store current robot state
        self.current_pose = Pose()
        self.is_executing = False
        self.cancel_requested = False

    def goal_callback(self, goal_request):
        """Accept or reject incoming goals"""
        self.get_logger().info(f'Received walk pattern goal: {goal_request.pattern_type}')

        # Validate goal parameters
        if goal_request.distance < 0.0:
            self.get_logger().warn('Invalid distance in goal request')
            return GoalResponse.REJECT

        if goal_request.speed <= 0.0:
            self.get_logger().warn('Invalid speed in goal request')
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Accept or reject cancel requests"""
        self.get_logger().info('Received cancel request for walk pattern')
        self.cancel_requested = True
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Execute the walk pattern goal"""
        self.get_logger().info('Executing walk pattern goal')

        feedback_msg = WalkPattern.Feedback()
        result = WalkPattern.Result()

        # Get goal parameters
        pattern_type = goal_handle.request.pattern_type
        distance = goal_handle.request.distance
        speed = goal_handle.request.speed

        # Validate current state
        if self.is_executing:
            result.success = False
            result.message = 'Another walk pattern is already executing'
            return result

        self.is_executing = True
        self.cancel_requested = False

        try:
            # Calculate steps based on distance and step size
            step_size = 0.3  # meters per step
            total_steps = int(distance / step_size)

            for step in range(total_steps):
                # Check for cancellation
                if self.cancel_requested:
                    result.success = False
                    result.message = 'Goal canceled during execution'
                    self.is_executing = False
                    goal_handle.canceled()
                    return result

                # Simulate step execution
                await self.execute_single_step(pattern_type, speed)

                # Update feedback
                feedback_msg.current_step = step + 1
                feedback_msg.total_steps = total_steps
                feedback_msg.progress = (step + 1) / total_steps * 100.0
                feedback_msg.current_pose = self.current_pose

                goal_handle.publish_feedback(feedback_msg)

                # Log progress
                self.get_logger().info(f'Progress: {feedback_msg.progress:.1f}%')

            # Check if we completed the full distance
            if not self.cancel_requested:
                result.success = True
                result.message = f'Walk pattern completed: {total_steps} steps executed'
                goal_handle.succeed()
                self.get_logger().info('Walk pattern completed successfully')
            else:
                result.success = False
                result.message = 'Goal canceled during execution'
                goal_handle.canceled()

        except Exception as e:
            self.get_logger().error(f'Error executing walk pattern: {e}')
            result.success = False
            result.message = f'Execution error: {str(e)}'
            goal_handle.abort()

        finally:
            self.is_executing = False
            self.cancel_requested = False

        return result

    async def execute_single_step(self, pattern_type, speed):
        """Execute a single walking step"""
        # Simulate step execution time based on speed
        step_duration = 1.0 / speed  # seconds per step

        # In a real implementation, this would interface with the walking controller
        time.sleep(step_duration)

        # Update current pose (simplified)
        self.current_pose.position.x += 0.3  # step forward 30cm
```

### Action Client Implementation

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.task import Future

from humanoid_interfaces.action import WalkPattern

class WalkPatternActionClient(Node):
    def __init__(self):
        super().__init__('walk_pattern_action_client')

        self._action_client = ActionClient(
            self,
            WalkPattern,
            'execute_walk_pattern'
        )

    def send_goal(self, pattern_type, distance, speed):
        """Send a walk pattern goal to the action server"""
        goal_msg = WalkPattern.Goal()
        goal_msg.pattern_type = pattern_type
        goal_msg.distance = distance
        goal_msg.speed = speed

        self.get_logger().info(f'Sending walk pattern goal: {pattern_type}, distance: {distance}m, speed: {speed}')

        # Wait for action server to be available
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return None

        # Send goal and get future
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        send_goal_future.add_done_callback(self.goal_response_callback)
        return send_goal_future

    def goal_response_callback(self, future):
        """Handle goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Handle feedback during execution"""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Feedback: Step {feedback.current_step}/{feedback.total_steps}, '
            f'Progress: {feedback.progress:.1f}%'
        )

    def get_result_callback(self, future):
        """Handle the final result"""
        result = future.result().result
        if result.success:
            self.get_logger().info(f'Success: {result.message}')
        else:
            self.get_logger().error(f'Failed: {result.message}')

    def cancel_goal(self, goal_handle_future):
        """Cancel a running goal"""
        goal_handle = goal_handle_future.result()
        cancel_future = goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_response_callback)

    def cancel_response_callback(self, future):
        """Handle cancel response"""
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal successfully canceled')
        else:
            self.get_logger().info('Goal failed to cancel')
```

## Advanced Service and Action Patterns

### Service with Timeout and Retry Logic

```python
import asyncio
from rclpy.qos import qos_profile_services_default

class RobustServiceClient(Node):
    def __init__(self):
        super().__init__('robust_service_client')
        self.service_client = self.create_client(
            SetJointPositions,
            '/set_joint_positions'
        )

    async def call_service_with_retry(self, request, max_retries=3, timeout=2.0):
        """Call service with retry logic and timeout"""
        for attempt in range(max_retries):
            try:
                # Wait for service with timeout
                if not self.service_client.wait_for_service(timeout_sec=timeout):
                    self.get_logger().warn(f'Service not available, attempt {attempt + 1}/{max_retries}')
                    continue

                # Make service call with timeout
                future = self.service_client.call_async(request)

                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(
                        future,
                        timeout=timeout
                    )
                    return response
                except asyncio.TimeoutError:
                    self.get_logger().warn(f'Service call timed out, attempt {attempt + 1}/{max_retries}')
                    continue

            except Exception as e:
                self.get_logger().error(f'Service call failed, attempt {attempt + 1}/{max_retries}: {e}')
                continue

        # All retries failed
        raise Exception(f'Service call failed after {max_retries} attempts')
```

### Action with Preemption

```python
class PreemptableActionServer(Node):
    def __init__(self):
        super().__init__('preemptable_action_server')
        self._action_server = ActionServer(
            self,
            WalkPattern,
            'preemptable_walk_pattern',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            handle_accepted_callback=self.handle_accepted_callback,
            cancel_callback=self.cancel_callback
        )

    def handle_accepted_callback(self, goal_handle):
        """Handle when a goal is accepted"""
        # Start execution in a separate thread
        import threading
        thread = threading.Thread(target=self.execute_goal_thread, args=[goal_handle])
        thread.start()

    def execute_goal_thread(self, goal_handle):
        """Execute goal in separate thread to allow preemption"""
        # Implementation similar to execute_callback but in a thread
        # This allows the main thread to handle new goal requests
        pass
```

## Best Practices for Humanoid Robotics

:::tip Service and Action Best Practices
- Use services for immediate, short-duration operations
- Use actions for long-running tasks that need feedback
- Implement proper error handling and validation
- Use appropriate callback groups for concurrent operations
- Always validate goal parameters before execution
- Provide meaningful feedback during long operations
- Implement cancellation support for actions
:::

:::note Performance Considerations
- Services block the calling thread; use async clients when possible
- Actions can run concurrently with proper callback groups
- Monitor service response times for performance optimization
- Consider using action feedback for progress tracking in humanoid applications
:::

## Key Takeaways

- Services provide synchronous request-response communication
- Actions handle long-running tasks with feedback and cancellation
- Proper error handling and validation are essential for humanoid safety
- Callback groups enable concurrent service and action handling
- Goal validation prevents invalid commands from reaching hardware
- Feedback mechanisms provide transparency for complex humanoid operations
- Timeout and retry logic enhance system robustness

Understanding services and actions is crucial for implementing safe and responsive humanoid robotics systems that can handle both immediate commands and complex, long-duration tasks with appropriate feedback and error handling.