---
title: "Visual and Collision Geometries"
description: "Understanding visual vs collision meshes, primitive shapes, STL/DAE import, collision optimization, materials"
sidebar_position: 5
keywords: [visual geometry, collision geometry, meshes, stl,dae, materials, optimization]
---

# Visual and Collision Geometries

Visual and collision geometries define how your robot appears in visualization tools and how it interacts in physics simulations. This section covers the differences between visual and collision models, how to use primitive shapes and mesh files, and techniques for optimizing collision detection.

## Visual vs Collision Geometries

### Key Differences

In URDF, each link can have separate visual and collision geometries:

- **Visual geometry**: Defines how the robot looks in visualization tools like RViz
- **Collision geometry**: Defines how the robot interacts in physics simulations (Gazebo)
- **Inertial properties**: Defines the physical mass and moment of inertia for dynamics

```xml
<link name="example_link">
  <!-- Visual geometry - how it looks -->
  <visual>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <mesh filename="package://robot_description/meshes/detailed_model.stl"/>
    </geometry>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
  </visual>

  <!-- Collision geometry - how it collides -->
  <collision>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <geometry>
      <cylinder radius="0.1" length="0.3"/>
    </geometry>
  </collision>

  <!-- Inertial properties - physical properties -->
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial>
</link>
```

### Why Separate Geometries?

1. **Performance**: Collision detection with detailed meshes is computationally expensive
2. **Simplicity**: Simple shapes are more reliable for collision detection
3. **Flexibility**: Different representations for different purposes
4. **Quality**: Visual models can be more detailed than collision models

## Using Primitive Shapes

### Basic Primitive Shapes

URDF supports several primitive geometric shapes that are efficient for collision detection:

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

URDF supports several mesh formats:
- **STL**: Simple and widely supported
- **DAE**: Collada format with textures and materials
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
    <!-- Use a simpler collision geometry for performance -->
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

For complex visual meshes, create simplified collision versions:

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

  <!-- Or use multiple simple shapes -->
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

## Optimizing Collision Detection

### Simple vs Complex Collision Geometries

#### Option 1: Single Complex Shape
```xml
<!-- Less efficient but simpler -->
<collision>
  <geometry>
    <mesh filename="package://robot/meshes/complex_shape.stl"/>
  </geometry>
</collision>
```

#### Option 2: Multiple Simple Shapes (Recommended)
```xml
<!-- More efficient and reliable -->
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

### Convex Hulls for Complex Shapes

For complex collision detection, you can use convex hulls:

```xml
<collision>
  <!-- If your mesh is not convex, consider creating a convex hull -->
  <geometry>
    <mesh filename="package://robot/meshes/convex_hull.dae"/>
  </geometry>
</collision>
```

### Collision Filtering

Use collision filtering to prevent certain parts from colliding:

```xml
<!-- In URDF, you can't directly filter collisions, but you can structure your model -->
<!-- to minimize unnecessary collision checks -->

<!-- Group related links that shouldn't collide with each other -->
<link name="arm_assembly">
  <!-- This would be a compound link in a more complex setup -->
</link>
```

## Material and Color Properties

### Defining Materials

Materials define the visual appearance of your robot:

```xml
<!-- Define materials at the top of your URDF or in a separate file -->
<material name="red">
  <color rgba="1.0 0.0 0.0 1.0"/>
</material>

<material name="blue">
  <color rgba="0.0 0.0 1.0 1.0"/>
</material>

<material name="skin_color">
  <color rgba="0.9 0.7 0.5 1.0"/>
</material>

<material name="metal_gray">
  <color rgba="0.5 0.5 0.5 1.0"/>
</material>

<material name="black">
  <color rgba="0.1 0.1 0.1 1.0"/>
</material>
```

### Using Textures (DAE format)

For more complex visual effects, use DAE files with textures:

```xml
<visual>
  <geometry>
    <mesh filename="package://robot/meshes/textured_part.dae"/>
  </geometry>
  <!-- Materials are typically defined within the DAE file -->
</visual>
```

### Complete Material Example

```xml
<?xml version="1.0"?>
<robot name="colored_humanoid">
  <!-- Material definitions -->
  <material name="head_material">
    <color rgba="0.9 0.7 0.5 1.0"/>  <!-- Skin tone -->
  </material>

  <material name="torso_material">
    <color rgba="0.2 0.4 0.8 1.0"/>  <!-- Blue shirt -->
  </material>

  <material name="limb_material">
    <color rgba="0.3 0.3 0.3 1.0"/>  <!-- Dark gray for joints -->
  </material>

  <material name="foot_material">
    <color rgba="0.1 0.1 0.1 1.0"/>  <!-- Black shoes -->
  </material>

  <!-- Links using materials -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="head_material"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <link name="torso">
    <visual>
      <geometry>
        <cylinder radius="0.12" length="0.5"/>
      </geometry>
      <material name="torso_material"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.12" length="0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="15.0"/>
      <inertia ixx="0.3" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.3"/>
    </inertial>
  </link>
</robot>
```

## Practical Example: Complete Humanoid with Optimized Geometries

Here's a complete example showing optimized visual and collision geometries for a humanoid:

```xml
<?xml version="1.0"?>
<robot name="optimized_humanoid">
  <!-- Materials -->
  <material name="skin">
    <color rgba="0.9 0.7 0.5 1.0"/>
  </material>
  <material name="shirt">
    <color rgba="0.2 0.4 0.8 1.0"/>
  </material>
  <material name="pants">
    <color rgba="0.1 0.1 0.6 1.0"/>
  </material>
  <material name="shoes">
    <color rgba="0.1 0.1 0.1 1.0"/>
  </material>

  <!-- Base/Pelvis -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.25 0.1"/>
      </geometry>
      <material name="pants"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.25 0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="8.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Torso -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0.0 0.0 0.05"/>
  </joint>

  <link name="torso">
    <visual>
      <origin xyz="0 0 0.25" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.12" length="0.5"/>
      </geometry>
      <material name="shirt"/>
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

  <!-- Head -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0.0 0.0 0.5"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="5.0" velocity="1.0"/>
  </joint>

  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Left Upper Arm -->
  <joint name="left_shoulder_pitch" type="revolute">
    <parent link="torso"/>
    <child link="left_upper_arm"/>
    <origin xyz="-0.05 -0.15 0.2" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="20.0" velocity="1.5"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
      <material name="skin"/>
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

  <!-- Left Lower Arm -->
  <joint name="left_elbow" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="2.5" effort="15.0" velocity="2.0"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <origin xyz="0 0 -0.125" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.125" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.04" length="0.25"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <!-- Left Hand -->
  <joint name="left_wrist" type="revolute">
    <parent link="left_lower_arm"/>
    <child link="left_hand"/>
    <origin xyz="0 0 -0.25" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="5.0" velocity="3.0"/>
  </joint>

  <link name="left_hand">
    <visual>
      <geometry>
        <box size="0.1 0.06 0.04"/>
      </geometry>
      <material name="skin"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.1 0.06 0.04"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0005" ixy="0.0" ixz="0.0" iyy="0.0005" iyz="0.0" izz="0.0005"/>
    </inertial>
  </link>

  <!-- Left Thigh -->
  <joint name="left_hip_pitch" type="revolute">
    <parent link="base_link"/>
    <child link="left_thigh"/>
    <origin xyz="0.0 -0.125 -0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="0.7" effort="50.0" velocity="1.0"/>
  </joint>

  <link name="left_thigh">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.07" length="0.4"/>
      </geometry>
      <material name="pants"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.07" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
  </link>

  <!-- Left Shin -->
  <joint name="left_knee" type="revolute">
    <parent link="left_thigh"/>
    <child link="left_shin"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="2.5" effort="40.0" velocity="1.0"/>
  </joint>

  <link name="left_shin">
    <visual>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
      <material name="pants"/>
    </visual>
    <collision>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.06" length="0.4"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.04"/>
    </inertial>
  </link>

  <!-- Left Foot -->
  <joint name="left_ankle" type="revolute">
    <parent link="left_shin"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="20.0" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.18 0.08 0.06"/>
      </geometry>
      <material name="shoes"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.18 0.08 0.06"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.005" ixy="0.0" ixz="0.0" iyy="0.005" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>
</robot>
```

## Performance Optimization Tips

### 1. Use Simple Collision Geometries
- Replace complex meshes with simple shapes when possible
- Use multiple simple shapes instead of one complex mesh
- Consider convex hulls for moderately complex shapes

### 2. Optimize Visual Meshes
- Use appropriate level of detail for visual meshes
- Consider different LOD (Level of Detail) models
- Compress mesh files to reduce load times

### 3. Organize Mesh Files
- Keep mesh files in a logical directory structure
- Use descriptive names for mesh files
- Maintain both visual and collision versions when needed

### 4. Balance Quality and Performance
- More complex visual models look better but use more resources
- Simpler collision models are faster but may be less accurate
- Test performance in your target simulation environment

In the next section, we'll explore practical applications and complete examples of humanoid robots that bring together all the concepts covered in this chapter.