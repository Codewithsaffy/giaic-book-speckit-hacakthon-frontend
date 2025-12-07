---
sidebar_position: 4
title: "LLM Cognitive Planning for Robotics"
description: "Large Language Model integration for high-level task planning and reasoning"
---

# LLM Cognitive Planning for Robotics

Large Language Models (LLMs) have emerged as powerful tools for cognitive planning in robotics, enabling robots to understand complex natural language instructions and decompose them into executable action sequences. This chapter explores how to integrate LLMs for high-level task planning, reasoning, and decision-making in robotic systems.

## LLM Fundamentals for Robotics

### Cognitive Architecture Overview

LLMs serve as cognitive planners by bridging the gap between high-level natural language instructions and low-level robot actions:

#### Cognitive Planning Pipeline
```
Natural Language Instruction → LLM Processing → Task Decomposition → Action Sequences → Robot Execution
         ↓                        ↓                    ↓                  ↓                ↓
    "Bring me coffee" → Understanding → [Go to kitchen, → [Navigate, → [Execute
                       → Intent       →  Find coffee,  →  Locate,      →  Actions]
                       → Extraction   →  Grasp mug,    →  Grasp,
                       →              →  Pour coffee,  →  Pour,
                       →              →  Return]       →  Navigate]
```

### LLM Selection for Robotics

#### Model Characteristics for Robotics Applications

##### Open-Source Models
- **Llama 2/3**: Good balance of capability and accessibility
- **Mistral**: Efficient for instruction following
- **Falcon**: Strong reasoning capabilities
- **MPT**: Good for long-context tasks

##### Commercial APIs
- **GPT-4**: Most capable for complex reasoning
- **Claude**: Strong in following instructions
- **Gemini**: Multimodal capabilities
- **Cohere**: Specialized for enterprise use

#### Model Requirements for Robotics
- **Instruction Following**: Ability to follow step-by-step instructions
- **Reasoning Capabilities**: Logical and spatial reasoning
- **Context Window**: Sufficient length for complex tasks
- **Safety**: Alignment with safety constraints
- **Latency**: Acceptable response times for real-time applications

### LLM Integration Architecture

#### Cognitive Planning Framework
```python
import torch
import json
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class RobotAction:
    """Represents a single robot action"""
    action_type: str
    parameters: Dict[str, Any]
    description: str
    preconditions: List[str]
    effects: List[str]

@dataclass
class TaskPlan:
    """Represents a sequence of actions for task completion"""
    task_description: str
    actions: List[RobotAction]
    success_criteria: List[str]
    estimated_time: float

class LLMInterface(ABC):
    """Abstract interface for LLM integration"""

    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from the LLM"""
        pass

    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        pass

class OpenAILLMInterface(LLMInterface):
    """OpenAI API interface for LLM integration"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model

        # Import only if available
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required for OpenAI LLM interface")

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API"""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content

    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI API"""
        response = await self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

class HuggingFaceLLMInterface(LLMInterface):
    """Hugging Face Transformers interface for LLM integration"""

    def __init__(self, model_name: str, device: str = "cuda"):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        if device == "cuda":
            self.model = self.model.half().cuda()

        self.model.eval()

    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using local Hugging Face model"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")

        if self.device == "cuda":
            inputs = inputs.cuda()

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
                do_sample=kwargs.get("do_sample", True),
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()

    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings using local model"""
        # For local models, we might use a separate embedding model
        # or use the model's hidden states
        inputs = self.tokenizer.encode(text, return_tensors="pt")

        if self.device == "cuda":
            inputs = inputs.cuda()

        with torch.no_grad():
            outputs = self.model(inputs, output_hidden_states=True)
            # Use last hidden state as embedding
            embedding = outputs.hidden_states[-1][0, -1, :].cpu().numpy().tolist()

        return embedding
```

## Task Decomposition and Planning

### Natural Language to Action Mapping

#### Task Decomposition Pipeline
```python
class TaskDecomposer:
    """Decompose natural language tasks into executable actions"""

    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface

        # Define action vocabulary for the robot
        self.action_vocabulary = {
            "navigation": [
                "go_to", "navigate_to", "move_to", "approach",
                "follow_path", "avoid_obstacle", "stop"
            ],
            "manipulation": [
                "grasp", "release", "pick_up", "put_down",
                "open", "close", "push", "pull", "turn"
            ],
            "perception": [
                "detect", "identify", "locate", "count",
                "measure", "recognize", "classify"
            ],
            "communication": [
                "speak", "listen", "ask", "answer",
                "signal", "notify", "report"
            ]
        }

    async def decompose_task(self, instruction: str, context: Dict[str, Any] = None) -> TaskPlan:
        """Decompose natural language instruction into task plan"""
        prompt = self._create_decomposition_prompt(instruction, context)

        response = await self.llm_interface.generate_text(
            prompt,
            max_tokens=1000,
            temperature=0.3
        )

        return self._parse_task_plan(response, instruction)

    def _create_decomposition_prompt(self, instruction: str, context: Dict[str, Any] = None) -> str:
        """Create prompt for task decomposition"""
        context_str = json.dumps(context) if context else "No context available"

        prompt = f"""
You are a robot task planner. Given a natural language instruction and context,
decompose the task into a sequence of executable actions. Each action should be
from the robot's action vocabulary and include parameters, preconditions, and effects.

Context: {context_str}

Instruction: "{instruction}"

Please respond with a JSON object containing:
- task_description: Brief description of the task
- actions: Array of action objects with structure:
  - action_type: Type of action (navigation, manipulation, perception, communication)
  - parameters: Dictionary of parameters needed for the action
  - description: Human-readable description of the action
  - preconditions: List of preconditions that must be satisfied
  - effects: List of effects that result from the action
- success_criteria: List of conditions that indicate task completion
- estimated_time: Estimated time to complete the task in seconds

Example response format:
{{
    "task_description": "Go to kitchen and bring water bottle",
    "actions": [
        {{
            "action_type": "navigation",
            "parameters": {{"destination": "kitchen"}},
            "description": "Navigate to the kitchen",
            "preconditions": ["robot_is_idle"],
            "effects": ["robot_is_at_kitchen"]
        }},
        {{
            "action_type": "perception",
            "parameters": {{"object": "water_bottle"}},
            "description": "Locate water bottle",
            "preconditions": ["robot_is_at_kitchen"],
            "effects": ["water_bottle_location_known"]
        }}
    ],
    "success_criteria": ["robot_has_water_bottle", "robot_is_at_user_location"],
    "estimated_time": 120.0
}}
"""
        return prompt

    def _parse_task_plan(self, response: str, original_instruction: str) -> TaskPlan:
        """Parse LLM response into TaskPlan object"""
        try:
            # Clean response to extract JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]

            data = json.loads(json_str)

            # Convert action dictionaries to RobotAction objects
            actions = []
            for action_data in data["actions"]:
                action = RobotAction(
                    action_type=action_data["action_type"],
                    parameters=action_data["parameters"],
                    description=action_data["description"],
                    preconditions=action_data["preconditions"],
                    effects=action_data["effects"]
                )
                actions.append(action)

            return TaskPlan(
                task_description=data["task_description"],
                actions=actions,
                success_criteria=data["success_criteria"],
                estimated_time=data["estimated_time"]
            )
        except json.JSONDecodeError:
            # Fallback: create simple plan with single action
            return TaskPlan(
                task_description=original_instruction,
                actions=[RobotAction(
                    action_type="unknown",
                    parameters={"instruction": original_instruction},
                    description=f"Execute: {original_instruction}",
                    preconditions=["robot_ready"],
                    effects=["task_attempted"]
                )],
                success_criteria=["task_completed"],
                estimated_time=60.0
            )
```

### Hierarchical Task Planning

#### Multi-Level Planning Architecture
```python
class HierarchicalTaskPlanner:
    """Manage hierarchical task planning with multiple abstraction levels"""

    def __init__(self, task_decomposer: TaskDecomposer):
        self.task_decomposer = task_decomposer
        self.high_level_tasks = []
        self.mid_level_tasks = []
        self.low_level_tasks = []

    async def plan_task_hierarchically(self, instruction: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Plan task at multiple levels of abstraction"""
        # High-level: Overall task structure
        high_level_plan = await self._plan_high_level(instruction, context)

        # Mid-level: Detailed action sequences
        mid_level_plan = await self._plan_mid_level(high_level_plan, context)

        # Low-level: Primitive robot actions
        low_level_plan = await self._plan_low_level(mid_level_plan, context)

        return {
            "high_level": high_level_plan,
            "mid_level": mid_level_plan,
            "low_level": low_level_plan,
            "execution_graph": self._create_execution_graph(low_level_plan)
        }

    async def _plan_high_level(self, instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan at high level of abstraction"""
        prompt = f"""
Given the following task: "{instruction}"

Provide a high-level decomposition into major phases. Each phase should represent
a significant milestone in task completion. Consider the context: {json.dumps(context)}

Respond in JSON format:
{{
    "task": "{instruction}",
    "phases": [
        {{
            "name": "Phase name",
            "description": "What needs to be accomplished",
            "subtasks": ["list of high-level subtasks"]
        }}
    ]
}}
"""
        response = await self.task_decomposer.llm_interface.generate_text(prompt)
        return self._parse_json_response(response)

    async def _plan_mid_level(self, high_level_plan: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan at mid level of abstraction"""
        phases = high_level_plan.get("phases", [])

        mid_level_plan = {
            "task": high_level_plan["task"],
            "phases": []
        }

        for phase in phases:
            prompt = f"""
Given the phase: "{phase['name']}" with description: "{phase['description']}"

Decompose this phase into specific, executable subtasks. Each subtask should be
clearly defined and achievable by the robot. Consider the context: {json.dumps(context)}

Respond in JSON format:
{{
    "phase_name": "{phase['name']}",
    "subtasks": [
        {{
            "name": "Subtask name",
            "description": "Detailed description",
            "action_sequence": ["sequence of robot actions"],
            "success_criteria": ["list of success criteria"]
        }}
    ]
}}
"""
            response = await self.task_decomposer.llm_interface.generate_text(prompt)
            phase_plan = self._parse_json_response(response)
            mid_level_plan["phases"].append(phase_plan)

        return mid_level_plan

    async def _plan_low_level(self, mid_level_plan: Dict[str, Any], context: Dict[str, Any]) -> List[RobotAction]:
        """Plan at low level with primitive robot actions"""
        all_actions = []

        for phase in mid_level_plan["phases"]:
            for subtask in phase["subtasks"]:
                # Use the task decomposer for each subtask
                task_plan = await self.task_decomposer.decompose_task(
                    subtask["description"],
                    context
                )
                all_actions.extend(task_plan.actions)

        return all_actions

    def _create_execution_graph(self, low_level_actions: List[RobotAction]) -> Dict[str, Any]:
        """Create execution graph showing dependencies between actions"""
        graph = {
            "nodes": [],
            "edges": [],
            "start_nodes": [],
            "end_nodes": []
        }

        # Create nodes for each action
        for i, action in enumerate(low_level_actions):
            node = {
                "id": i,
                "action": action.action_type,
                "description": action.description,
                "parameters": action.parameters,
                "preconditions": action.preconditions,
                "effects": action.effects
            }
            graph["nodes"].append(node)

        # Create edges based on preconditions and effects
        for i in range(len(low_level_actions) - 1):
            # Simple linear dependency for now
            edge = {
                "from": i,
                "to": i + 1,
                "type": "sequential"
            }
            graph["edges"].append(edge)

        if graph["nodes"]:
            graph["start_nodes"] = [0]
            graph["end_nodes"] = [len(graph["nodes"]) - 1]

        return graph

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            json_str = response[start_idx:end_idx]
            return json.loads(json_str)
        except:
            # Return empty dict if parsing fails
            return {}

class ExecutionManager:
    """Manage execution of planned tasks"""

    def __init__(self):
        self.current_task = None
        self.execution_history = []
        self.failure_recovery = True

    async def execute_task_plan(self, task_plan: TaskPlan, robot_interface) -> Dict[str, Any]:
        """Execute a task plan with monitoring and recovery"""
        results = {
            "task": task_plan.task_description,
            "executed_actions": [],
            "failed_actions": [],
            "success": False,
            "reason": ""
        }

        for i, action in enumerate(task_plan.actions):
            try:
                # Check preconditions
                if not await self._check_preconditions(action, robot_interface):
                    results["failed_actions"].append({
                        "action": action,
                        "reason": "Preconditions not met"
                    })

                    if self.failure_recovery:
                        recovery_result = await self._attempt_recovery(action, robot_interface)
                        if not recovery_result:
                            results["reason"] = f"Failed to recover from precondition failure for action {action.description}"
                            return results
                    else:
                        results["reason"] = f"Preconditions not met for action {action.description}"
                        return results

                # Execute action
                execution_result = await self._execute_action(action, robot_interface)

                if execution_result["success"]:
                    results["executed_actions"].append({
                        "action": action,
                        "result": execution_result,
                        "timestamp": execution_result.get("timestamp")
                    })

                    # Update robot state based on effects
                    await self._apply_effects(action, robot_interface)
                else:
                    results["failed_actions"].append({
                        "action": action,
                        "reason": execution_result.get("error", "Unknown error")
                    })

                    if self.failure_recovery:
                        recovery_result = await self._attempt_recovery(action, robot_interface)
                        if not recovery_result:
                            results["reason"] = f"Action failed and recovery unsuccessful: {action.description}"
                            return results
                    else:
                        results["reason"] = f"Action failed: {action.description}"
                        return results

            except Exception as e:
                results["failed_actions"].append({
                    "action": action,
                    "reason": f"Exception during execution: {str(e)}"
                })
                results["reason"] = f"Exception during execution: {str(e)}"
                return results

        # Check success criteria
        success = await self._check_success_criteria(task_plan.success_criteria, robot_interface)
        results["success"] = success
        results["reason"] = "All actions completed successfully" if success else "Success criteria not met"

        return results

    async def _check_preconditions(self, action: RobotAction, robot_interface) -> bool:
        """Check if action preconditions are met"""
        # In a real implementation, this would check robot state
        # For now, assume preconditions are met
        return True

    async def _execute_action(self, action: RobotAction, robot_interface) -> Dict[str, Any]:
        """Execute a single robot action"""
        try:
            # This would interface with the actual robot
            # For now, simulate execution
            import time
            start_time = time.time()

            # Simulate different action types
            if action.action_type == "navigation":
                result = await robot_interface.navigate_to(action.parameters.get("destination", "unknown"))
            elif action.action_type == "manipulation":
                result = await robot_interface.manipulate_object(
                    action.parameters.get("object", "unknown"),
                    action.parameters.get("action", "grasp")
                )
            elif action.action_type == "perception":
                result = await robot_interface.perceive_environment(
                    action.parameters.get("target", "unknown")
                )
            else:
                result = {"status": "executed", "details": f"Executed {action.action_type}"}

            execution_time = time.time() - start_time

            return {
                "success": result.get("status") == "success",
                "result": result,
                "execution_time": execution_time,
                "timestamp": time.time()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }

    async def _apply_effects(self, action: RobotAction, robot_interface):
        """Apply effects of action to robot state"""
        # Update robot state based on action effects
        # This would modify the robot's internal state representation
        pass

    async def _attempt_recovery(self, failed_action: RobotAction, robot_interface) -> bool:
        """Attempt to recover from action failure"""
        # Implement recovery strategies
        # This could involve retrying, alternative approaches, or asking for help
        return False  # For now, no recovery implemented

    async def _check_success_criteria(self, criteria: List[str], robot_interface) -> bool:
        """Check if task success criteria are met"""
        # Check if all success criteria are satisfied
        # In a real implementation, this would query robot state
        return True  # For now, assume success
```

## Prompt Engineering for Robotics

### Effective Prompting Strategies

#### Role-Based Prompting
```python
class PromptEngineer:
    """Engineer effective prompts for robotics applications"""

    @staticmethod
    def create_role_based_prompt(role: str, instruction: str, context: Dict[str, Any]) -> str:
        """Create role-based prompt for LLM"""
        roles = {
            "robot_planner": {
                "description": "You are an expert robot task planner with deep knowledge of robotics, navigation, manipulation, and human-robot interaction.",
                "expertise": [
                    "Robot kinematics and dynamics",
                    "Navigation and path planning",
                    "Object manipulation and grasping",
                    "Human-robot interaction",
                    "Task decomposition and sequencing",
                    "Safety considerations"
                ]
            },
            "spatial_reasoner": {
                "description": "You are an expert in spatial reasoning and environmental understanding for robotics applications.",
                "expertise": [
                    "3D spatial relationships",
                    "Object affordances",
                    "Navigation space analysis",
                    "Collision detection and avoidance",
                    "Environmental mapping"
                ]
            },
            "safety_analyst": {
                "description": "You are a safety analyst specializing in robotics applications with focus on risk assessment and mitigation.",
                "expertise": [
                    "Risk assessment",
                    "Safety constraints",
                    "Emergency procedures",
                    "Human safety considerations",
                    "Fail-safe mechanisms"
                ]
            }
        }

        if role not in roles:
            role = "robot_planner"  # Default role

        role_info = roles[role]

        prompt = f"""
{role_info['description']}

Your expertise includes:
"""
        for expertise in role_info['expertise']:
            prompt += f"- {expertise}\n"

        prompt += f"""

Context information:
{json.dumps(context, indent=2)}

Task: {instruction}

Please provide your analysis, planning, or response considering your role and expertise.
"""
        return prompt

    @staticmethod
    def create_chain_of_thought_prompt(instruction: str, context: Dict[str, Any]) -> str:
        """Create chain-of-thought prompt for complex reasoning"""
        prompt = f"""
Task: {instruction}

Context: {json.dumps(context, indent=2)}

Let's approach this step by step:

Step 1: Understand the task and requirements
- What is the main objective?
- What are the constraints?
- What resources are available?

Step 2: Analyze the environment and situation
- What is the current state?
- What obstacles or challenges exist?
- What opportunities are available?

Step 3: Plan the approach
- What high-level strategy should be used?
- What sequence of actions is needed?
- What alternatives should be considered?

Step 4: Consider safety and error handling
- What could go wrong?
- How should errors be handled?
- What safety measures are needed?

Step 5: Execute and monitor
- How will progress be monitored?
- When is the task complete?
- What constitutes success?

Please provide your detailed analysis and plan based on this reasoning process.
"""
        return prompt

    @staticmethod
    def create_few_shot_prompt(instruction: str, examples: List[Dict[str, str]]) -> str:
        """Create few-shot prompt with examples"""
        prompt = "Here are examples of task planning:\n\n"

        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Instruction: {example['instruction']}\n"
            prompt += f"Plan: {example['plan']}\n\n"

        prompt += f"Now, for the new instruction:\n"
        prompt += f"Instruction: {instruction}\n"
        prompt += f"Plan:"

        return prompt

# Example usage of prompt engineering
def example_prompt_engineering():
    """Example of using prompt engineering techniques"""

    # Example task
    instruction = "Go to the kitchen, get a glass of water, and bring it to the living room"

    # Context information
    context = {
        "robot_capabilities": ["navigation", "grasping", "speech"],
        "environment_map": {
            "kitchen": {"x": 10.0, "y": 5.0},
            "living_room": {"x": 2.0, "y": 3.0}
        },
        "object_locations": {
            "glass": "kitchen_counter",
            "water": "kitchen_sink"
        },
        "safety_constraints": ["avoid_people", "no_speed_above_1m_s"]
    }

    # Create different types of prompts
    role_prompt = PromptEngineer.create_role_based_prompt("robot_planner", instruction, context)
    cot_prompt = PromptEngineer.create_chain_of_thought_prompt(instruction, context)

    # Few-shot examples
    examples = [
        {
            "instruction": "Pick up the red ball from the table",
            "plan": "1. Navigate to table location\n2. Identify red ball\n3. Plan grasp trajectory\n4. Execute grasp\n5. Verify grasp success"
        },
        {
            "instruction": "Go to the bedroom and turn off the lights",
            "plan": "1. Navigate to bedroom\n2. Locate light switch\n3. Approach switch\n4. Execute turn-off action\n5. Verify lights are off"
        }
    ]

    few_shot_prompt = PromptEngineer.create_few_shot_prompt(instruction, examples)

    return {
        "role_based": role_prompt,
        "chain_of_thought": cot_prompt,
        "few_shot": few_shot_prompt
    }
```

## Safety and Validation

### Safe LLM Integration

#### Safety Constraints and Validation
```python
class SafetyValidator:
    """Validate LLM outputs for safety in robotics applications"""

    def __init__(self):
        # Define safety constraint categories
        self.safety_categories = [
            "physical_harm",
            "property_damage",
            "privacy_violation",
            "security_breach",
            "social_norm_violation",
            "task_impossibility"
        ]

        # Define prohibited actions
        self.prohibited_actions = [
            "shoot", "hurt", "damage", "break", "destroy", "steal",
            "spy", "monitor", "record", "follow", "chase", "attack"
        ]

    def validate_task_plan(self, task_plan: TaskPlan, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate task plan for safety"""
        issues = []

        # Check for prohibited actions
        for action in task_plan.actions:
            if any(prohibited in action.action_type.lower() for prohibited in self.prohibited_actions):
                issues.append({
                    "type": "prohibited_action",
                    "action": action.action_type,
                    "severity": "high"
                })

            # Check action parameters
            for param_value in action.parameters.values():
                if isinstance(param_value, str):
                    if any(prohibited in param_value.lower() for prohibited in self.prohibited_actions):
                        issues.append({
                            "type": "prohibited_parameter",
                            "parameter": param_value,
                            "severity": "medium"
                        })

        # Check success criteria
        for criterion in task_plan.success_criteria:
            if any(prohibited in criterion.lower() for prohibited in self.prohibited_actions):
                issues.append({
                    "type": "unsafe_success_criterion",
                    "criterion": criterion,
                    "severity": "high"
                })

        # Validate against context constraints
        context_issues = self._validate_against_context(task_plan, context)
        issues.extend(context_issues)

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "risk_level": self._assess_risk_level(issues)
        }

    def _validate_against_context(self, task_plan: TaskPlan, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate task plan against context constraints"""
        issues = []

        # Check safety constraints from context
        safety_constraints = context.get("safety_constraints", [])

        for constraint in safety_constraints:
            if constraint == "avoid_people":
                for action in task_plan.actions:
                    if action.action_type == "navigation":
                        # Check if navigation path avoids people
                        # This would require spatial reasoning
                        pass

            elif constraint == "no_speed_above_1m_s":
                for action in task_plan.actions:
                    if action.action_type == "navigation":
                        # Check if parameters respect speed limits
                        speed = action.parameters.get("speed", 1.0)
                        if speed > 1.0:
                            issues.append({
                                "type": "speed_limit_violation",
                                "action": action.action_type,
                                "expected": "<= 1.0 m/s",
                                "actual": f"{speed} m/s",
                                "severity": "high"
                            })

        return issues

    def _assess_risk_level(self, issues: List[Dict[str, Any]]) -> str:
        """Assess overall risk level based on issues"""
        if not issues:
            return "low"

        high_severity = [issue for issue in issues if issue["severity"] == "high"]
        medium_severity = [issue for issue in issues if issue["severity"] == "medium"]

        if high_severity:
            return "high"
        elif medium_severity:
            return "medium"
        else:
            return "low"

class CognitivePlanner:
    """Main cognitive planning system with safety integration"""

    def __init__(self, llm_interface: LLMInterface):
        self.llm_interface = llm_interface
        self.task_decomposer = TaskDecomposer(llm_interface)
        self.hierarchical_planner = HierarchicalTaskPlanner(self.task_decomposer)
        self.safety_validator = SafetyValidator()
        self.execution_manager = ExecutionManager()

    async def plan_and_execute(self, instruction: str, context: Dict[str, Any], robot_interface) -> Dict[str, Any]:
        """Plan and execute task with full safety validation"""

        # Step 1: Decompose task
        task_plan = await self.task_decomposer.decompose_task(instruction, context)

        # Step 2: Validate safety
        safety_check = self.safety_validator.validate_task_plan(task_plan, context)

        if not safety_check["valid"]:
            return {
                "success": False,
                "error": "Safety validation failed",
                "safety_issues": safety_check["issues"],
                "risk_level": safety_check["risk_level"]
            }

        # Step 3: Execute plan
        execution_result = await self.execution_manager.execute_task_plan(task_plan, robot_interface)

        return {
            "success": execution_result["success"],
            "task_description": task_plan.task_description,
            "execution_result": execution_result,
            "safety_validation": safety_check
        }

    async def plan_with_feedback(self, instruction: str, context: Dict[str, Any], robot_interface) -> Dict[str, Any]:
        """Plan with iterative feedback and refinement"""

        # Initial plan
        initial_plan = await self.task_decomposer.decompose_task(instruction, context)

        # Validate and get feedback
        safety_check = self.safety_validator.validate_task_plan(initial_plan, context)

        if not safety_check["valid"]:
            # Request plan revision based on safety issues
            revised_plan = await self._revise_plan(initial_plan, safety_check["issues"], context)

            # Validate revised plan
            revised_safety_check = self.safety_validator.validate_task_plan(revised_plan, context)

            if not revised_safety_check["valid"]:
                return {
                    "success": False,
                    "error": "Could not create safe plan after revision",
                    "initial_plan": initial_plan,
                    "revised_plan": revised_plan,
                    "safety_issues": revised_safety_check["issues"]
                }

            # Execute revised plan
            execution_result = await self.execution_manager.execute_task_plan(revised_plan, robot_interface)
        else:
            # Execute initial plan
            execution_result = await self.execution_manager.execute_task_plan(initial_plan, robot_interface)

        return {
            "success": execution_result["success"],
            "task_description": initial_plan.task_description,
            "execution_result": execution_result,
            "safety_validation": safety_check
        }

    async def _revise_plan(self, original_plan: TaskPlan, issues: List[Dict[str, Any]], context: Dict[str, Any]) -> TaskPlan:
        """Revise plan based on safety issues"""

        issue_descriptions = []
        for issue in issues:
            issue_descriptions.append(f"- {issue['type']}: {issue.get('action', issue.get('parameter', 'Unknown'))}")

        prompt = f"""
Original task: {original_plan.task_description}

Original plan:
{json.dumps([{
    'action_type': action.action_type,
    'parameters': action.parameters,
    'description': action.description
} for action in original_plan.actions], indent=2)}

Safety issues identified:
{chr(10).join(issue_descriptions)}

Context: {json.dumps(context, indent=2)}

Please revise the plan to address the safety issues while still accomplishing the task.
Provide the revised plan in the same format as the original.
"""

        response = await self.llm_interface.generate_text(prompt)

        # Parse the revised plan (simplified)
        try:
            # Extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                revised_actions_data = json.loads(json_str)

                # Convert back to RobotAction objects
                revised_actions = []
                for action_data in revised_actions_data:
                    action = RobotAction(
                        action_type=action_data["action_type"],
                        parameters=action_data["parameters"],
                        description=action_data["description"],
                        preconditions=action_data.get("preconditions", []),
                        effects=action_data.get("effects", [])
                    )
                    revised_actions.append(action)

                return TaskPlan(
                    task_description=original_plan.task_description,
                    actions=revised_actions,
                    success_criteria=original_plan.success_criteria,
                    estimated_time=original_plan.estimated_time
                )
        except:
            pass

        # If parsing fails, return original plan (unsafe, but better than nothing)
        return original_plan
```

## Integration with Real Systems

### Complete Integration Example

#### ROS 2 Integration
```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from ament_index_python.packages import get_package_share_directory

class LLMCognitivePlannerNode(Node):
    """ROS 2 node for LLM-based cognitive planning"""

    def __init__(self):
        super().__init__('llm_cognitive_planner')

        # Initialize cognitive planner
        # For this example, we'll use a mock LLM interface
        # In practice, you'd initialize with real API credentials
        self.llm_interface = self._create_mock_llm_interface()
        self.cognitive_planner = CognitivePlanner(self.llm_interface)

        # ROS interfaces
        self.instruction_sub = self.create_subscription(
            String, 'natural_language_instruction', self.instruction_callback, 10)
        self.status_pub = self.create_publisher(String, 'planning_status', 10)
        self.action_pub = self.create_publisher(String, 'robot_action', 10)

        # Robot state and environment
        self.robot_state = {}
        self.environment_map = {}

        self.get_logger().info('LLM Cognitive Planner node initialized')

    def _create_mock_llm_interface(self):
        """Create a mock LLM interface for demonstration"""
        class MockLLMInterface(LLMInterface):
            async def generate_text(self, prompt: str, **kwargs) -> str:
                # Mock response - in practice, this would call real LLM
                if "decompose" in prompt.lower() or "plan" in prompt.lower():
                    return '''{
                        "task_description": "Mock task",
                        "actions": [
                            {
                                "action_type": "navigation",
                                "parameters": {"destination": "kitchen"},
                                "description": "Navigate to kitchen",
                                "preconditions": ["robot_idle"],
                                "effects": ["robot_moving"]
                            }
                        ],
                        "success_criteria": ["robot_at_destination"],
                        "estimated_time": 60.0
                    }'''
                else:
                    return "Mock response"

            async def embed_text(self, text: str) -> List[float]:
                return [0.0] * 128  # Mock embedding

        return MockLLMInterface()

    async def instruction_callback(self, msg: String):
        """Handle incoming natural language instruction"""
        instruction = msg.data
        self.get_logger().info(f'Received instruction: {instruction}')

        # Create context for planning
        context = {
            "robot_state": self.robot_state,
            "environment_map": self.environment_map,
            "current_time": self.get_clock().now().to_msg(),
            "safety_constraints": ["avoid_collisions", "respect_human_space"]
        }

        # Plan and execute
        try:
            result = await self.cognitive_planner.plan_and_execute(
                instruction, context, self.robot_interface)

            # Publish status
            status_msg = String()
            status_msg.data = f"Planning result: {result['success']}"
            self.status_pub.publish(status_msg)

            self.get_logger().info(f'Planning completed: {result["success"]}')

        except Exception as e:
            self.get_logger().error(f'Error in cognitive planning: {e}')

    def robot_interface(self):
        """Mock robot interface for demonstration"""
        class MockRobotInterface:
            async def navigate_to(self, destination):
                print(f"Navigating to {destination}")
                return {"status": "success"}

            async def manipulate_object(self, obj, action):
                print(f"Manipulating {obj} with {action}")
                return {"status": "success"}

            async def perceive_environment(self, target):
                print(f"Perceiving {target}")
                return {"status": "success", "result": "object_found"}

        return MockRobotInterface()

def main(args=None):
    rclpy.init(args=args)
    node = LLMCognitivePlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down LLM cognitive planner')
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

The LLM cognitive planning system provides high-level reasoning and task decomposition capabilities that enable robots to understand and execute complex natural language instructions. By combining LLMs with safety validation and hierarchical planning, robots can perform sophisticated tasks while maintaining safety and reliability.