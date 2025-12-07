---
sidebar_position: 3
title: "Physics Simulation in Gazebo"
description: "Understanding and configuring physics simulation for realistic robot behavior"
---

# Physics Simulation in Gazebo

Physics simulation is the cornerstone of realistic robot simulation in Gazebo. Understanding how to configure and optimize physics parameters is crucial for creating accurate digital twins that faithfully represent real-world robot behavior.

## Physics Engine Fundamentals

### Core Physics Concepts

Gazebo's physics simulation is based on fundamental physical principles:

#### Newtonian Mechanics
- **Newton's Laws**: Motion, force, and acceleration relationships
- **Linear Dynamics**: Translation motion of rigid bodies
- **Angular Dynamics**: Rotational motion and moments of inertia
- **Conservation Laws**: Conservation of momentum and energy

#### Rigid Body Dynamics
- **Mass Properties**: Mass, center of mass, and inertia tensors
- **Kinematics**: Position, velocity, and acceleration relationships
- **Dynamics**: Force, torque, and motion relationships
- **Constraints**: Joint limitations and contact constraints

### Physics Engine Options

Gazebo supports multiple physics engines, each with different strengths:

#### ODE (Open Dynamics Engine)
- **Default Engine**: Used in most Gazebo installations
- **Strengths**: Stable, well-tested, good for most applications
- **Limitations**: Can be less accurate for complex contacts
- **Use Cases**: General robotics simulation, basic contact scenarios

#### Bullet Physics
- **Modern Engine**: Advanced physics simulation capabilities
- **Strengths**: Better contact handling, more accurate collisions
- **Limitations**: Potentially more computationally expensive
- **Use Cases**: Complex contact scenarios, high-fidelity simulation

#### DART (Dynamic Animation and Robotics Toolkit)
- **Advanced Engine**: Specialized for robotics and animation
- **Strengths**: Excellent kinematic and dynamic simulation
- **Limitations**: More complex to configure
- **Use Cases**: Humanoid robots, complex kinematic chains

## Physics Configuration

### World Physics Parameters

Physics parameters are configured in world files:

```xml
<physics type="ode" name="default_physics">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Key Parameters Explained

#### Time Step (`max_step_size`)
- **Purpose**: Defines the simulation time increment
- **Typical Values**: 0.001 to 0.01 seconds
- **Trade-offs**: Smaller steps = more accuracy but slower performance
- **Guidelines**: Use 0.001s for high-precision applications

#### Real Time Factor (`real_time_factor`)
- **Purpose**: Controls simulation speed relative to real time
- **Value 1.0**: Simulation runs at real-time speed
- **Value > 1.0**: Simulation runs faster than real time
- **Value < 1.0**: Simulation runs slower than real time

#### Update Rate (`real_time_update_rate`)
- **Purpose**: Defines how often physics updates occur per second
- **Relationship**: Inverse of max_step_size (1000 Hz for 0.001s step)
- **Performance**: Higher rates require more computational resources

## Contact Simulation

### Contact Parameters

Accurate contact simulation is crucial for realistic robot behavior:

#### Error Reduction Parameter (ERP)
- **Purpose**: Controls how quickly constraint errors are corrected
- **Range**: 0.0 to 1.0
- **Default**: 0.2
- **Higher Values**: More aggressive error correction, potentially unstable
- **Lower Values**: Softer constraints, more stable but less accurate

#### Constraint Force Mixing (CFM)
- **Purpose**: Adds regularization to constraint equations
- **Range**: 0.0 to 1.0 (typically very small)
- **Default**: 0.0
- **Higher Values**: Softer constraints, more stable
- **Lower Values**: Harder constraints, more accurate

### Contact Properties

#### Surface Properties
```xml
<surface>
  <friction>
    <ode>
      <mu>1.0</mu>
      <mu2>1.0</mu2>
      <fdir1>0 0 0</fdir1>
      <slip1>0.0</slip1>
      <slip2>0.0</slip2>
    </ode>
  </friction>
  <bounce>
    <restitution_coefficient>0.0</restitution_coefficient>
    <threshold>100000.0</threshold>
  </bounce>
  <contact>
    <ode>
      <soft_cfm>0.0</soft_cfm>
      <soft_erp>0.2</soft_erp>
      <kp>1000000000000.0</kp>
      <kd>1.0</kd>
      <max_vel>100.0</max_vel>
      <min_depth>0.001</min_depth>
    </ode>
  </contact>
</surface>
```

#### Friction Modeling
- **Static Friction (mu)**: Resistance to initial motion
- **Dynamic Friction (mu2)**: Resistance during sliding motion
- **Anisotropic Friction**: Different friction in different directions
- **Slip Parameters**: Allow controlled sliding for stability

## Robot-Specific Physics Configuration

### Mass and Inertia Properties

Accurate mass properties are essential for realistic simulation:

#### Mass Distribution
```xml
<inertial>
  <mass>1.0</mass>
  <inertia>
    <ixx>0.01</ixx>
    <ixy>0.0</ixy>
    <ixz>0.0</ixz>
    <iyy>0.01</iyy>
    <iyz>0.0</iyz>
    <izz>0.01</izz>
  </inertia>
</inertial>
```

#### Center of Mass
- **Location**: Should match the physical robot's center of mass
- **Impact**: Affects balance, stability, and motion
- **Measurement**: Use CAD tools or physical measurement
- **Verification**: Compare simulated and real robot behavior

### Joint Dynamics

#### Joint Friction and Damping
```xml
<joint name="joint1" type="revolute">
  <physics>
    <ode>
      <damping>0.1</damping>
      <friction>0.05</friction>
      <spring_reference>0</spring_reference>
      <spring_stiffness>0</spring_stiffness>
    </ode>
  </physics>
</joint>
```

#### Joint Limits
- **Effort Limits**: Maximum torque/force a joint can apply
- **Velocity Limits**: Maximum speed of joint motion
- **Position Limits**: Physical range of motion constraints
- **Acceleration Limits**: Rate of change constraints

## Humanoid-Specific Physics Considerations

### Balance and Stability

Humanoid robots require special attention to physics parameters:

#### Center of Mass Management
- **Low Center of Mass**: Improve stability during walking
- **Dynamic Adjustment**: Consider load changes affecting balance
- **Real-time Monitoring**: Track CoM position during simulation
- **Stability Metrics**: Calculate stability margins (ZMP, etc.)

#### Ground Contact Modeling
- **Foot Contact**: Accurate modeling of foot-ground interaction
- **Pressure Distribution**: Simulate pressure sensors in feet
- **Slip Prevention**: Configure friction for stable stance
- **Impact Absorption**: Model compliant behavior during footsteps

### Walking Dynamics

#### Zero Moment Point (ZMP)
- **Stability Criterion**: Ensure ZMP remains within support polygon
- **Trajectory Planning**: Generate stable walking patterns
- **Simulation Validation**: Verify ZMP behavior in simulation
- **Real-World Transfer**: Compare ZMP between sim and reality

#### Dynamic Balance
- **Inertia Properties**: Accurate inertia for stable motion
- **Control Integration**: Physics-aware control algorithms
- **Disturbance Handling**: Test robustness to external forces
- **Recovery Strategies**: Simulate balance recovery behaviors

## Performance Optimization

### Physics Performance Tuning

#### Solver Configuration
- **Iteration Count**: Balance accuracy vs. performance (typically 10-50)
- **Solver Type**: Quick vs. PGS vs. other options
- **Convergence Criteria**: Tolerance settings for solver
- **Adaptive Stepping**: Variable time steps where possible

#### Model Simplification
- **Collision Geometry**: Use simplified shapes for collision detection
- **Visual vs. Collision**: Separate detailed visual from simple collision models
- **Level of Detail**: Reduce complexity at a distance
- **Proxy Objects**: Use simplified models for distant objects

### Real-Time Simulation

#### Deterministic Simulation
- **Fixed Time Steps**: Ensure consistent behavior
- **Synchronization**: Coordinate with control systems
- **Timing Analysis**: Monitor simulation timing
- **Performance Profiling**: Identify bottlenecks

#### Multi-Threading
- **Physics Threading**: Separate physics from rendering
- **Parallel Processing**: Utilize multiple CPU cores
- **Load Balancing**: Distribute computation efficiently
- **Thread Safety**: Ensure safe multi-threaded operation

## Validation and Tuning

### Physics Model Validation

#### Parameter Identification
- **System Identification**: Use real robot data to tune parameters
- **Response Matching**: Match simulated and real robot responses
- **Sensitivity Analysis**: Identify critical parameters
- **Uncertainty Quantification**: Understand parameter uncertainty

#### Experimental Validation
- **Simple Tests**: Start with basic motion validation
- **Complex Behaviors**: Progress to complex robot behaviors
- **Edge Cases**: Test boundary conditions and failure modes
- **Statistical Validation**: Use multiple trials for robust validation

### Tuning Methodology

#### Iterative Approach
1. **Initial Estimation**: Use CAD or physical measurements
2. **Coarse Tuning**: Adjust parameters for general behavior
3. **Fine Tuning**: Optimize for specific behaviors
4. **Validation**: Test against real robot performance

#### Automated Tuning
- **Optimization Algorithms**: Use optimization for parameter tuning
- **Machine Learning**: Learn parameters from data
- **Bayesian Methods**: Probabilistic parameter estimation
- **Genetic Algorithms**: Evolutionary parameter optimization

## Advanced Physics Features

### Multi-Physics Simulation

#### Fluid Dynamics
- **Air Resistance**: Model aerodynamic effects
- **Water Simulation**: For aquatic robots
- **Wind Effects**: Environmental disturbance modeling
- **Buoyancy**: Floating robot simulation

#### Flexible Body Dynamics
- **Soft Body Simulation**: Deformable objects
- **Finite Element Methods**: Detailed deformation modeling
- **Reduced Models**: Simplified flexible body models
- **Contact with Deformable Objects**: Complex interaction modeling

### Contact Modeling Enhancements

#### Advanced Contact Models
- **Hertzian Contact**: Accurate sphere-plane contact
- **Coulomb Friction**: Classical friction modeling
- **Stribeck Effects**: Mixed friction regimes
- **Adhesion**: Surface adhesion modeling

Physics simulation in Gazebo provides the foundation for creating realistic digital twins of humanoid robots. Proper configuration and tuning of physics parameters ensure that simulated robots behave similarly to their real-world counterparts, enabling safe and effective development of complex robotic behaviors.