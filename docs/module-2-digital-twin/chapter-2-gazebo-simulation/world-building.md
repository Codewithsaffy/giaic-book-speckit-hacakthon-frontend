---
sidebar_position: 4
title: "Building Gazebo Worlds and Environments"
description: "Creating and configuring simulation environments for robotics testing"
---

# Building Gazebo Worlds and Environments

Creating realistic and appropriate simulation environments is crucial for effective digital twin development. Gazebo provides powerful tools for building complex worlds that accurately represent real-world scenarios where robots will operate.

## World Structure and Components

### Basic World File Structure

A Gazebo world file is an SDF (Simulation Description Format) file that defines the entire simulation environment:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- Physics configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Environment models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom models and objects -->
    <model name="my_robot">
      <!-- Robot definition -->
    </model>

    <!-- Lighting and environment -->
    <light name="sun" type="directional">
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
  </world>
</sdf>
```

### Essential World Components

#### Physics Configuration
- **Gravity Settings**: Define gravitational acceleration
- **Time Parameters**: Simulation time step and real-time factor
- **Solver Settings**: Physics engine configuration
- **Damping**: Global damping parameters

#### Environment Elements
- **Ground Plane**: Basic surface for robot operation
- **Sky**: Background environment and lighting
- **Lighting**: Sun, ambient, and artificial lights
- **Atmosphere**: Optional atmospheric effects

## Creating Custom Environments

### Simple Indoor Environment

Creating a basic indoor environment for humanoid robot testing:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="indoor_hallway">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Walls -->
    <model name="wall_1">
      <pose>0 5 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>20 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>20 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <model name="wall_2">
      <pose>0 -5 1 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>20 0.2 3</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>20 0.2 3</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

    <!-- Obstacles -->
    <model name="obstacle_1">
      <pose>2 0 0.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.4 0.2 1</ambient>
            <diffuse>0.8 0.4 0.2 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
  </world>
</sdf>
```

### Complex Indoor Environment

For more complex indoor scenarios:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="complex_indoor">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Lighting -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Rooms with furniture -->
    <model name="room_wall_1">
      <pose>0 8 1.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>16 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>16 0.2 3</size></box>
          </geometry>
          <material><ambient>0.8 0.8 0.8 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Doorway -->
    <model name="door_frame">
      <pose>0 4 1.5 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>4 0.2 3</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>4 0.2 3</size></box>
          </geometry>
          <material><ambient>0.5 0.3 0.1 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Tables and chairs -->
    <model name="table_1">
      <pose>-3 2 0.4 0 0 0</pose>
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry>
            <box><size>1.5 0.8 0.8</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>1.5 0.8 0.8</size></box>
          </geometry>
          <material><ambient>0.6 0.4 0.2 1</ambient></material>
        </visual>
      </link>
    </model>

    <!-- Artificial lighting -->
    <light name="room_light_1" type="point">
      <pose>-3 2 2 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.5 0.5 0.5 1</specular>
      <attenuation>
        <range>10</range>
        <constant>0.2</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
    </light>
  </world>
</sdf>
```

## Model Integration

### Including Pre-built Models

Gazebo provides a library of pre-built models:

```xml
<!-- Include a model from the model database -->
<include>
  <uri>model://cylinder</uri>
  <pose>1 1 1 0 0 0</pose>
</include>

<!-- Include a more complex model -->
<include>
  <uri>model://table</uri>
  <pose>0 0 0 0 0 0</pose>
</include>

<!-- Include a robot model -->
<include>
  <uri>model://pr2</uri>
  <pose>0 0 0 0 0 0</pose>
</include>
```

### Custom Model Creation

Creating custom models for specific use cases:

```xml
<model name="custom_obstacle">
  <pose>5 0 1 0 0 0</pose>
  <static>true</static>
  <link name="link">
    <!-- Collision geometry (for physics) -->
    <collision name="collision">
      <geometry>
        <mesh>
          <uri>model://custom_obstacle/meshes/obstacle.dae</uri>
        </mesh>
      </geometry>
      <surface>
        <friction>
          <ode>
            <mu>0.8</mu>
            <mu2>0.8</mu2>
          </ode>
        </friction>
      </surface>
    </collision>

    <!-- Visual geometry (for rendering) -->
    <visual name="visual">
      <geometry>
        <mesh>
          <uri>model://custom_obstacle/meshes/obstacle.dae</uri>
        </mesh>
      </geometry>
      <material>
        <ambient>0.2 0.6 0.8 1</ambient>
        <diffuse>0.2 0.6 0.8 1</diffuse>
        <specular>0.1 0.1 0.1 1</specular>
      </material>
    </visual>
  </link>
</model>
```

## Advanced Environment Features

### Dynamic Environments

Creating environments that change during simulation:

```xml
<!-- Moving obstacles -->
<model name="moving_obstacle">
  <pose>0 0 0.5 0 0 0</pose>
  <link name="link">
    <collision name="collision">
      <geometry>
        <sphere><radius>0.3</radius></sphere>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <sphere><radius>0.3</radius></sphere>
      </geometry>
      <material>
        <ambient>1 0 0 1</ambient>
      </material>
    </visual>
    <!-- Plugin to move the obstacle -->
    <plugin name="model_move_plugin" filename="libModelMovePlugin.so">
      <velocity>0.5 0 0</velocity>
      <amplitude>2.0</amplitude>
      <frequency>0.5</frequency>
    </plugin>
  </link>
</model>
```

### Sensor Testing Environments

Environments specifically designed for sensor testing:

```xml
<world name="sensor_test_world">
  <!-- Physics -->
  <physics type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
  </physics>

  <!-- Ground plane with texture -->
  <model name="ground_plane">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <plane><normal>0 0 1</normal></plane>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <plane><normal>0 0 1</normal><size>20 20</size></plane>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/WhitePVC</name>
          </script>
        </material>
      </visual>
    </link>
  </model>

  <!-- Different texture regions -->
  <model name="checkerboard_region">
    <pose>5 0 0.01 0 0 0</pose>
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <box><size>4 4 0.02</size></box>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/CheckerBlue</name>
          </script>
        </material>
      </visual>
    </link>
  </model>

  <!-- Various objects for sensor testing -->
  <model name="high_contrast_target">
    <pose>0 3 0.5 0 0 0</pose>
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box><size>0.5 0.5 1</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>0.5 0.5 1</size></box>
        </geometry>
        <material>
          <ambient>0 0 0 1</ambient>
          <diffuse>0 0 0 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</world>
```

## Environment Customization

### Terrain Generation

Creating custom terrains for outdoor scenarios:

```xml
<model name="custom_terrain">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <geometry>
        <heightmap>
          <uri>model://terrain/heightmap.png</uri>
          <size>40 40 5</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="visual">
      <geometry>
        <heightmap>
          <uri>model://terrain/heightmap.png</uri>
          <size>40 40 5</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
      <material>
        <script>
          <uri>file://media/materials/scripts/gazebo.material</uri>
          <name>Gazebo/Dirt</name>
        </script>
      </material>
    </visual>
  </link>
</model>
```

### Weather and Atmospheric Effects

Adding environmental conditions:

```xml
<!-- Atmosphere configuration -->
<atmosphere type="adiabatic">
  <temperature>288.15</temperature>
  <pressure>101325</pressure>
</atmosphere>

<!-- Wind effects -->
<model name="wind_generator">
  <static>true</static>
  <link name="link">
    <plugin name="wind_plugin" filename="libWindPlugin.so">
      <force>0.1 0 0</force>
      <direction>0.1 0.05 0</direction>
      <perturbation>0.05</perturbation>
    </plugin>
  </link>
</model>
```

## World Building Best Practices

### Performance Optimization

#### Level of Detail (LOD)
- **Simplify Geometry**: Use simpler collision meshes than visual meshes
- **Reduce Polygons**: Optimize mesh complexity for performance
- **Distance-Based Detail**: Reduce detail for distant objects
- **Static Objects**: Mark non-moving objects as static

#### Resource Management
- **Texture Optimization**: Use compressed textures where possible
- **Model Caching**: Cache frequently used models
- **Memory Management**: Monitor and optimize memory usage
- **Streaming**: Load/unload parts of large environments as needed

### Realism Considerations

#### Physical Accuracy
- **Material Properties**: Accurate friction, restitution, and surface properties
- **Lighting Conditions**: Match real-world lighting scenarios
- **Environmental Factors**: Include relevant environmental conditions
- **Scale Accuracy**: Maintain proper scale relationships

#### Visual Fidelity
- **Texture Quality**: High-resolution textures for important objects
- **Lighting Effects**: Realistic shadows, reflections, and lighting
- **Environmental Details**: Add small details that enhance realism
- **Consistency**: Maintain visual consistency across the environment

## Humanoid-Specific Environments

### Indoor Navigation Environments

For humanoid robot navigation testing:

```xml
<world name="humanoid_navigation_test">
  <!-- Standard physics -->
  <physics type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
  </physics>

  <!-- Ground with appropriate friction for walking -->
  <model name="floor">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <plane><normal>0 0 1</normal></plane>
        </geometry>
        <surface>
          <friction>
            <ode><mu>0.7</mu><mu2>0.7</mu2></ode>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <plane><normal>0 0 1</normal><size>20 20</size></plane>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/WhitePVC</name>
          </script>
        </material>
      </visual>
    </link>
  </model>

  <!-- Navigation obstacles -->
  <model name="narrow_passage">
    <pose>0 0 0.5 0 0 0</pose>
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box><size>4 0.1 1</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>4 0.1 1</size></box>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
        </material>
      </visual>
    </link>
  </model>

  <!-- Stairs for testing locomotion -->
  <model name="stairs">
    <pose>5 0 0.15 0 0 0</pose>
    <static>true</static>
    <link name="link1">
      <collision name="collision">
        <geometry>
          <box><size>2 1 0.3</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>2 1 0.3</size></box>
        </geometry>
        <material>
          <ambient>0.4 0.4 0.4 1</ambient>
        </material>
      </visual>
    </link>
    <link name="link2">
      <pose>0 0 0.3 0 0 0</pose>
      <collision name="collision">
        <geometry>
          <box><size>2 1 0.3</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>2 1 0.3</size></box>
        </geometry>
        <material>
          <ambient>0.4 0.4 0.4 1</ambient>
        </material>
      </visual>
    </link>
  </model>
</world>
```

### Manipulation Environments

For humanoid manipulation tasks:

```xml
<world name="manipulation_test">
  <!-- Physics setup -->
  <physics type="ode">
    <max_step_size>0.001</max_step_size>
    <real_time_factor>1</real_time_factor>
  </physics>

  <!-- Work surface -->
  <model name="work_table">
    <pose>0 0 0.4 0 0 0</pose>
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry>
          <box><size>1.5 0.8 0.8</size></box>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <box><size>1.5 0.8 0.8</size></box>
        </geometry>
        <material>
          <ambient>0.6 0.4 0.2 1</ambient>
        </material>
      </visual>
    </link>
  </model>

  <!-- Objects for manipulation -->
  <model name="object_1">
    <pose>0.3 0.2 0.85 0 0 0</pose>
    <link name="link">
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.15</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.15</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>1 0 0 1</ambient>
        </material>
      </visual>
      <inertial>
        <mass>0.1</mass>
        <inertia>
          <ixx>0.0001</ixx>
          <iyy>0.0001</iyy>
          <izz>0.000025</izz>
        </inertia>
      </inertial>
    </link>
  </model>
</world>
```

## Environment Validation

### Testing and Validation

#### Visual Validation
- **Render Testing**: Verify all objects render correctly
- **Lighting Validation**: Check lighting and shadows
- **Texture Mapping**: Ensure textures apply correctly
- **Camera Views**: Test different camera perspectives

#### Physics Validation
- **Collision Detection**: Verify objects interact correctly
- **Stability Testing**: Check for physics instabilities
- **Performance Testing**: Monitor simulation performance
- **Accuracy Validation**: Compare with real-world scenarios

Building effective Gazebo worlds requires careful consideration of both visual and physical aspects to create realistic digital twins that accurately represent real-world environments for humanoid robot testing and development.