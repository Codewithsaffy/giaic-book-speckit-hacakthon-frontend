---
title: "Services aur Actions"
description: "ROS 2 services ko synchronous communication ke liye aur humanoid robotics mein long-running tasks ke liye actions ko samajhna"
sidebar_position: 4
keywords: ["ROS 2 services", "actions", "synchronous communication", "humanoid robotics", "robot control"]
---

# Services aur Actions

Jabke topics asynchronous communication ko enable karti hain, services aur actions synchronous aur long-running communication patterns provide karti hain jo humanoid robotics ke liye essential hain. Services request-response interactions ko handle karti hain, jabke actions feedback aur goal management ke sath complex tasks ko manage karti hain.

## Services: Synchronous Request-Response Communication

Services blocking, synchronous communication pattern provide karti hain jahan ek client request bhejta hai aur response ka wait karta hai. Yeh humanoid robotics applications ke liye ideal hai jo immediate responses ki requirement karti hain, jaise:

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

        # Parallel handling ke liye custom callback group ke sath service create karen
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

        # Current joint positions aur limits store karen
        self.current_positions = {}
        self.joint_limits = {
            'left_hip_pitch': (-1.5, 1.5),
            'right_hip_pitch': (-1.5, 1.5),
            'left_knee_pitch': (-0.5, 2.0),
            'right_knee_pitch': (-0.5, 2.0)
        }

    def set_joint_positions_callback(self, request, response):
        """Joint position setting requests ko handle karen"""
        try:
            self.get_logger().info(f'Setting joint positions: {request.positions}')

            # Joint positions ko limits ke against validate karen
            for joint_name, target_pos in zip(request.joint_names, request.positions):
                if joint_name in self.joint_limits:
                    min_limit, max_limit = self.joint_limits[joint_name]
                    if not (min_limit <= target_pos <= max_limit):
                        response.success = False
                        response.message = f'Joint {joint_name} position {target_pos} exceeds limits'
                        return response

            # Joint position command execute karen (simplified)
            self.execute_joint_command(request.joint_names, request.positions)

            # Internal state update karen
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
        """Requested joints ke liye joint limits return karen"""
        try:
            if not request.joint_names:  # Koi specific joints requested nahi hain to saare limits return karen
                response.joint_names = list(self.joint_limits.keys())
                response.min_limits = [self.joint_limits[name][0] for name in response.joint_names]
                response.max_limits = [self.joint_limits[name][1] for name in response.joint_names]
            else:
                # Specific joints ke liye limits return karen
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
        """Actual joint command execute karen (placeholder)"""
        # Real implementation mein, yeh hardware ke sath interface karega
        # ya lower-level control system
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

        # Services ke liye clients create karen
        self.joint_pos_client = self.create_client(
            SetJointPositions,
            '/set_joint_positions'
        )
        self.joint_limits_client = self.create_client(
            GetJointLimits,
            '/get_joint_limits'
        )

        # Services available hone tak wait karen
        while not self.joint_pos_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Joint position service not available, waiting...')

        while not self.joint_limits_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Joint limits service not available, waiting...')

    def set_joint_positions_async(self, joint_names, positions):
        """Joint position request asynchronously bhejen"""
        request = SetJointPositions.Request()
        request.joint_names = joint_names
        request.positions = positions

        # Asynchronous request bhejen
        future = self.joint_pos_client.call_async(request)
        future.add_done_callback(self.joint_position_response_callback)

        return future

    def joint_position_response_callback(self, future):
        """Joint position response handle karen"""
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f'Success: {response.message}')
            else:
                self.get_logger().error(f'Failed: {response.message}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    async def get_joint_limits_async(self, joint_names=None):
        """Joint limits asynchronously get karen"""
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

## Actions: Feedback ke sath Long-Running Tasks

Actions long-running operations ke liye designed hain jo feedback, goal management, aur cancellation ki requirement karti hain. Perfect humanoid robotics tasks ke liye jaise:

- Walking pattern execution
- Complex manipulation sequences
- Waypoints tak navigation
- Calibration procedures
- Dance ya gesture execution

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

        # Multiple requests handle karne ke liye reentrant callback group use karen
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

        # Current robot state store karen
        self.current_pose = Pose()
        self.is_executing = False
        self.cancel_requested = False

    def goal_callback(self, goal_request):
        """Incoming goals ko accept ya reject karen"""
        self.get_logger().info(f'Received walk pattern goal: {goal_request.pattern_type}')

        # Goal parameters validate karen
        if goal_request.distance < 0.0:
            self.get_logger().warn('Invalid distance in goal request')
            return GoalResponse.REJECT

        if goal_request.speed <= 0.0:
            self.get_logger().warn('Invalid speed in goal request')
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """Cancel requests ko accept ya reject karen"""
        self.get_logger().info('Received cancel request for walk pattern')
        self.cancel_requested = True
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        """Walk pattern goal execute karen"""
        self.get_logger().info('Executing walk pattern goal')

        feedback_msg = WalkPattern.Feedback()
        result = WalkPattern.Result()

        # Goal parameters get karen
        pattern_type = goal_handle.request.pattern_type
        distance = goal_handle.request.distance
        speed = goal_handle.request.speed

        # Current state validate karen
        if self.is_executing:
            result.success = False
            result.message = 'Another walk pattern is already executing'
            return result

        self.is_executing = True
        self.cancel_requested = False

        try:
            # Step size aur distance ke adhar par steps calculate karen
            step_size = 0.3  # meters per step
            total_steps = int(distance / step_size)

            for step in range(total_steps):
                # Cancellation ke liye check karen
                if self.cancel_requested:
                    result.success = False
                    result.message = 'Goal canceled during execution'
                    self.is_executing = False
                    goal_handle.canceled()
                    return result

                # Step execution simulate karen
                await self.execute_single_step(pattern_type, speed)

                # Feedback update karen
                feedback_msg.current_step = step + 1
                feedback_msg.total_steps = total_steps
                feedback_msg.progress = (step + 1) / total_steps * 100.0
                feedback_msg.current_pose = self.current_pose

                goal_handle.publish_feedback(feedback_msg)

                # Progress log karen
                self.get_logger().info(f'Progress: {feedback_msg.progress:.1f}%')

            # Check karen kya humne poora distance complete kiya
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
        """Single walking step execute karen"""
        # Speed ke adhar par step execution time simulate karen
        step_duration = 1.0 / speed  # seconds per step

        # Real implementation mein, yeh walking controller ke sath interface karega
        time.sleep(step_duration)

        # Current pose update karen (simplified)
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
        """Action server ko walk pattern goal bhejen"""
        goal_msg = WalkPattern.Goal()
        goal_msg.pattern_type = pattern_type
        goal_msg.distance = distance
        goal_msg.speed = speed

        self.get_logger().info(f'Sending walk pattern goal: {pattern_type}, distance: {distance}m, speed: {speed}')

        # Action server available hone tak wait karen
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            return None

        # Goal bhejen aur future get karen
        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )

        send_goal_future.add_done_callback(self.goal_response_callback)
        return send_goal_future

    def goal_response_callback(self, future):
        """Goal response handle karen"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        """Execution ke dauran feedback handle karen"""
        feedback = feedback_msg.feedback
        self.get_logger().info(
            f'Feedback: Step {feedback.current_step}/{feedback.total_steps}, '
            f'Progress: {feedback.progress:.1f}%'
        )

    def get_result_callback(self, future):
        """Final result handle karen"""
        result = future.result().result
        if result.success:
            self.get_logger().info(f'Success: {result.message}')
        else:
            self.get_logger().error(f'Failed: {result.message}')

    def cancel_goal(self, goal_handle_future):
        """Running goal cancel karen"""
        goal_handle = goal_handle_future.result()
        cancel_future = goal_handle.cancel_goal_async()
        cancel_future.add_done_callback(self.cancel_response_callback)

    def cancel_response_callback(self, future):
        """Cancel response handle karen"""
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal successfully canceled')
        else:
            self.get_logger().info('Goal failed to cancel')
```

## Advanced Service aur Action Patterns

### Service with Timeout aur Retry Logic

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
        """Retry logic aur timeout ke sath service call karen"""
        for attempt in range(max_retries):
            try:
                # Timeout ke sath service ke liye wait karen
                if not self.service_client.wait_for_service(timeout_sec=timeout):
                    self.get_logger().warn(f'Service not available, attempt {attempt + 1}/{max_retries}')
                    continue

                # Service call timeout ke sath karen
                future = self.service_client.call_async(request)

                # Response ke liye timeout ke sath wait karen
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

        # Saare retries failed huye
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
        # Separate thread mein execution start karen
        import threading
        thread = threading.Thread(target=self.execute_goal_thread, args=[goal_handle])
        thread.start()

    def execute_goal_thread(self, goal_handle):
        """Execute goal in separate thread to allow preemption"""
        # Implementation execute_callback ke similar hai lekin thread mein
        # Yeh main thread ko allow karta hai new goal requests handle karne ke liye
        pass
```

## Best Practices for Humanoid Robotics

:::tip Service aur Action Best Practices
- Short-duration operations ke liye services ka istemal karen
- Feedback ki zarurat wale long-running tasks ke liye actions ka istemal karen
- Proper error handling aur validation implement karen
- Concurrent operations ke liye appropriate callback groups ka istemal karen
- Execution se pehle goal parameters validate karen
- Long operations mein meaningful feedback provide karen
- Actions ke liye cancellation support implement karen
:::

:::note Performance Considerations
- Services calling thread ko block karti hain; jab bhi possible ho async clients ka istemal karen
- Actions proper callback groups ke sath concurrently run kar sakti hain
- Performance optimization ke liye service response times monitor karen
- Humanoid applications mein progress tracking ke liye action feedback consider karen
:::

## Key Takeaways

- Services synchronous request-response communication provide karti hain
- Actions feedback aur cancellation ke sath long-running tasks handle karti hain
- Proper error handling aur validation humanoid safety ke liye essential hain
- Callback groups concurrent service aur action handling ko enable karti hain
- Goal validation invalid commands ko hardware tak pahunchne se rokta hai
- Feedback mechanisms complex humanoid operations ke liye transparency provide karti hain
- Timeout aur retry logic system robustness ko enhance karti hain

Services aur actions ko samajhna safe aur responsive humanoid robotics systems implement karne ke liye crucial hai jo immediate commands aur complex, long-duration tasks handle kar sakti hain with appropriate feedback aur error handling.