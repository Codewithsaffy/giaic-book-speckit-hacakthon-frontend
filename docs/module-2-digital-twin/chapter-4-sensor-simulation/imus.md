---
sidebar_position: 4
title: "IMU Simulation in Robotics"
description: "Simulating Inertial Measurement Units for robotics applications"
---

# IMU Simulation in Robotics

Inertial Measurement Units (IMUs) are critical sensors for robotics applications, providing essential information about a robot's orientation, acceleration, and angular velocity. IMUs enable robots to maintain balance, navigate accurately, and perform precise movements. Simulating IMUs accurately is crucial for creating realistic digital twins that can effectively test and validate robotic control systems.

## Understanding IMU Technology

### IMU Components

An IMU typically combines multiple sensor types:

#### Accelerometer
- **Function**: Measures linear acceleration along three axes
- **Range**: Typically ±2g to ±16g
- **Sensitivity**: Measured in mg/digit or mg/LSB
- **Applications**: Gravity detection, tilt measurement, motion detection

#### Gyroscope
- **Function**: Measures angular velocity around three axes
- **Range**: Typically ±250°/s to ±2000°/s
- **Sensitivity**: Measured in °/s/digit or °/s/LSB
- **Applications**: Rotation detection, stabilization, heading

#### Magnetometer (Optional)
- **Function**: Measures magnetic field strength and direction
- **Range**: Typically ±1300μT to ±81900μT
- **Sensitivity**: Measured in μT/LSB
- **Applications**: Compass heading, magnetic field mapping

### IMU Characteristics

#### Key Parameters
- **Sample Rate**: How frequently measurements are taken
- **Resolution**: Precision of measurements
- **Noise Density**: Noise level per square root of bandwidth
- **Bias Stability**: Long-term stability of zero-point offset
- **Scale Factor Error**: Deviation from ideal sensitivity

#### Performance Metrics
- **Accuracy**: Closeness to true values
- **Precision**: Repeatability of measurements
- **Stability**: Consistency over time and temperature
- **Linearity**: Proportionality of output to input
- **Cross-Axis Sensitivity**: Interference between axes

## IMU Simulation in Gazebo

### IMU Sensor Configuration

Configuring IMU sensors in Gazebo:

```xml
<sdf version="1.7">
  <model name="imu_model">
    <link name="imu_link">
      <sensor name="imu_sensor" type="imu">
        <pose>0 0 0 0 0 0</pose>
        <topic>imu/data</topic>
        <update_rate>100</update_rate>
        <imu>
          <angular_velocity>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.001</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0001</bias_stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.001</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0001</bias_stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.001</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0001</bias_stddev>
              </noise>
            </z>
          </angular_velocity>
          <linear_acceleration>
            <x>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0017</bias_stddev>
              </noise>
            </x>
            <y>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0017</bias_stddev>
              </noise>
            </y>
            <z>
              <noise type="gaussian">
                <mean>0.0</mean>
                <stddev>0.017</stddev>
                <bias_mean>0.0</bias_mean>
                <bias_stddev>0.0017</bias_stddev>
              </noise>
            </z>
          </linear_acceleration>
        </imu>
        <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
          <alwaysOn>true</alwaysOn>
          <bodyName>imu_link</bodyName>
          <topicName>imu/data</topicData>
          <serviceName>imu/service</serviceName>
          <gaussianNoise>0.017</gaussianNoise>
          <updateRate>100.0</updateRate>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>
```

### Custom IMU Plugin Development

Creating advanced IMU simulation with custom processing:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/common/common.hh>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Vector3.h>
#include <tf2/LinearMath/Quaternion.h>

namespace gazebo
{
  class CustomIMUPlugin : public SensorPlugin
  {
    public: void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
    {
      // Cast to IMU sensor
      this->parentSensor = std::dynamic_pointer_cast<sensors::ImuSensor>(_sensor);
      if (!this->parentSensor)
      {
        gzerr << "CustomIMUPlugin requires an ImuSensor\n";
        return;
      }

      // Initialize ROS if needed
      if (!ros::isInitialized())
      {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "gazebo_client", ros::init_options::NoSigintHandler);
      }

      this->rosNode.reset(new ros::NodeHandle("gazebo_client"));

      // Create publisher
      this->pub = this->rosNode->advertise<sensor_msgs::Imu>("/imu/data", 1);

      // Get sensor parameters from SDF
      if (_sdf->HasElement("accel_noise_std"))
        this->accelNoiseStd = _sdf->Get<double>("accel_noise_std");
      if (_sdf->HasElement("gyro_noise_std"))
        this->gyroNoiseStd = _sdf->Get<double>("gyro_noise_std");

      // Connect to sensor update event
      this->updateConnection = this->parentSensor->ImuUpdated.Connect(
          boost::bind(&CustomIMUPlugin::OnUpdate, this));
    }

    public: void OnUpdate()
    {
      // Get raw IMU data from Gazebo
      ignition::math::Vector3d linearAcc = this->parentSensor->LinearAcceleration();
      ignition::math::Vector3d angularVel = this->parentSensor->AngularVelocity();
      ignition::math::Quaterniond orientation = this->parentSensor->Orientation();

      // Apply noise and bias models
      ApplyNoiseAndBias(linearAcc, angularVel);

      // Create IMU message
      sensor_msgs::Imu imu_msg;
      imu_msg.header.stamp = ros::Time::now();
      imu_msg.header.frame_id = "imu_link";

      // Set linear acceleration
      imu_msg.linear_acceleration.x = linearAcc.X();
      imu_msg.linear_acceleration.y = linearAcc.Y();
      imu_msg.linear_acceleration.z = linearAcc.Z();

      // Set angular velocity
      imu_msg.angular_velocity.x = angularVel.X();
      imu_msg.angular_velocity.y = angularVel.Y();
      imu_msg.angular_velocity.z = angularVel.Z();

      // Set orientation (if available from ground truth)
      imu_msg.orientation.w = orientation.W();
      imu_msg.orientation.x = orientation.X();
      imu_msg.orientation.y = orientation.Y();
      imu_msg.orientation.z = orientation.Z();

      // Set covariance matrices
      SetCovarianceMatrices(imu_msg);

      // Publish the message
      this->pub.publish(imu_msg);
    }

    private: void ApplyNoiseAndBias(ignition::math::Vector3d &_linearAcc,
                                   ignition::math::Vector3d &_angularVel)
    {
      // Apply Gaussian noise to linear acceleration
      _linearAcc.X() += this->GenerateGaussianNoise(this->accelNoiseStd);
      _linearAcc.Y() += this->GenerateGaussianNoise(this->accelNoiseStd);
      _linearAcc.Z() += this->GenerateGaussianNoise(this->accelNoiseStd);

      // Apply Gaussian noise to angular velocity
      _angularVel.X() += this->GenerateGaussianNoise(this->gyroNoiseStd);
      _angularVel.Y() += this->GenerateGaussianNoise(this->gyroNoiseStd);
      _angularVel.Z() += this->GenerateGaussianNoise(this->gyroNoiseStd);

      // Apply bias drift over time
      ApplyBiasDrift();
    }

    private: double GenerateGaussianNoise(double stdDev)
    {
      // Simple Gaussian noise generator
      static double n2 = 0.0;
      static int n2_cached = 0;
      if (!n2_cached)
      {
        double x, y, r;
        do
        {
          x = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1;
          y = 2.0 * static_cast<double>(rand()) / RAND_MAX - 1;
          r = x * x + y * y;
        }
        while (r >= 1.0 || r == 0.0);
        double d = sqrt(-2.0 * log(r) / r);
        double n1 = x * d;
        n2 = y * d;
        n2_cached = 1;
        return stdDev * n1;
      }
      else
      {
        n2_cached = 0;
        return stdDev * n2;
      }
    }

    private: void ApplyBiasDrift()
    {
      // Simulate bias drift over time
      this->timeSinceStart += 0.01; // Assuming 100Hz update rate

      // Apply random walk for bias
      this->accelBias.X() += this->GenerateGaussianNoise(0.0001);
      this->accelBias.Y() += this->GenerateGaussianNoise(0.0001);
      this->accelBias.Z() += this->GenerateGaussianNoise(0.0001);

      this->gyroBias.X() += this->GenerateGaussianNoise(0.00001);
      this->gyroBias.Y() += this->GenerateGaussianNoise(0.00001);
      this->gyroBias.Z() += this->GenerateGaussianNoise(0.00001);
    }

    private: void SetCovarianceMatrices(sensor_msgs::Imu &_msg)
    {
      // Set covariance matrices (information about uncertainty)
      // Acceleration covariance
      _msg.linear_acceleration_covariance[0] = this->accelNoiseStd * this->accelNoiseStd;
      _msg.linear_acceleration_covariance[4] = this->accelNoiseStd * this->accelNoiseStd;
      _msg.linear_acceleration_covariance[8] = this->accelNoiseStd * this->accelNoiseStd;

      // Angular velocity covariance
      _msg.angular_velocity_covariance[0] = this->gyroNoiseStd * this->gyroNoiseStd;
      _msg.angular_velocity_covariance[4] = this->gyroNoiseStd * this->gyroNoiseStd;
      _msg.angular_velocity_covariance[8] = this->gyroNoiseStd * this->gyroNoiseStd;

      // Orientation covariance (if orientation is estimated)
      _msg.orientation_covariance[0] = -1; // Set to -1 if not available
    }

    private: sensors::ImuSensorPtr parentSensor;
    private: std::unique_ptr<ros::NodeHandle> rosNode;
    private: ros::Publisher pub;
    private: event::ConnectionPtr updateConnection;

    // Noise parameters
    private: double accelNoiseStd = 0.017;
    private: double gyroNoiseStd = 0.001;
    private: double timeSinceStart = 0.0;

    // Bias parameters
    private: ignition::math::Vector3d accelBias = ignition::math::Vector3d::Zero;
    private: ignition::math::Vector3d gyroBias = ignition::math::Vector3d::Zero;
  };

  GZ_REGISTER_SENSOR_PLUGIN(CustomIMUPlugin)
}
```

## IMU Simulation in Unity

### Unity IMU Implementation

Creating IMU simulation in Unity:

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class UnityIMUSimulation : MonoBehaviour
{
    [Header("IMU Configuration")]
    public float accelerometerNoiseStd = 0.017f;
    public float gyroscopeNoiseStd = 0.001f;
    public float magnetometerNoiseStd = 0.01f;
    public float updateRate = 100f; // Hz

    [Header("ROS Integration")]
    public string imuTopic = "/imu/data";
    public bool publishToROS = true;

    private ROSConnection ros;
    private Rigidbody rb;
    private float lastUpdateTime;

    // Bias parameters
    private Vector3 accelerometerBias = Vector3.zero;
    private Vector3 gyroscopeBias = Vector3.zero;
    private Vector3 magnetometerBias = Vector3.zero;

    // Bias drift parameters
    private float biasDriftRate = 0.00001f;

    void Start()
    {
        if (publishToROS)
        {
            ros = ROSConnection.GetOrCreateInstance();
            ros.RegisterPublisher<ImuMsg>(imuTopic);
        }

        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
            rb.isKinematic = true; // Don't let physics affect this rigidbody
        }
    }

    void Update()
    {
        if (Time.time - lastUpdateTime >= 1f / updateRate)
        {
            PublishIMUData();
            lastUpdateTime = Time.time;
        }

        UpdateBiasDrift();
    }

    void PublishIMUData()
    {
        ImuMsg imuMsg = new ImuMsg();
        imuMsg.header.stamp = new TimeStamp(Time.time);
        imuMsg.header.frame_id = transform.name;

        // Get true values from Unity's physics system
        Vector3 trueAngularVelocity = GetTrueAngularVelocity();
        Vector3 trueLinearAcceleration = GetTrueLinearAcceleration();
        Vector3 trueMagneticField = GetTrueMagneticField();

        // Apply noise and bias
        Vector3 noisyAngularVelocity = AddNoiseAndBias(trueAngularVelocity, gyroscopeNoiseStd, ref gyroscopeBias);
        Vector3 noisyLinearAcceleration = AddNoiseAndBias(trueLinearAcceleration, accelerometerNoiseStd, ref accelerometerBias);
        Vector3 noisyMagneticField = AddNoiseAndBias(trueMagneticField, magnetometerNoiseStd, ref magnetometerBias);

        // Set angular velocity
        imuMsg.angular_velocity = new Vector3Msg(
            noisyAngularVelocity.x,
            noisyAngularVelocity.y,
            noisyAngularVelocity.z
        );

        // Set linear acceleration
        imuMsg.linear_acceleration = new Vector3Msg(
            noisyLinearAcceleration.x,
            noisyLinearAcceleration.y,
            noisyLinearAcceleration.z
        );

        // Set orientation (if using magnetometer for heading)
        imuMsg.orientation = CalculateOrientationFromMagneticField(noisyMagneticField);

        // Set covariance matrices
        SetCovarianceMatrices(imuMsg);

        if (publishToROS)
        {
            ros.Publish(imuTopic, imuMsg);
        }
    }

    Vector3 GetTrueAngularVelocity()
    {
        // Calculate true angular velocity from Unity's rotation
        // This is a simplified approach - in practice, you'd track this more precisely
        if (Time.deltaTime > 0)
        {
            Quaternion deltaRotation = transform.rotation * Quaternion.Inverse(transform.rotation);
            Vector3 angularVelocity = (2.0f * Vector3.Scale(deltaRotation.eulerAngles, new Vector3(
                Mathf.Sign(deltaRotation.w), Mathf.Sign(deltaRotation.w), Mathf.Sign(deltaRotation.w)
            ))) / Time.deltaTime;
            return angularVelocity * Mathf.Deg2Rad; // Convert to rad/s
        }
        return Vector3.zero;
    }

    Vector3 GetTrueLinearAcceleration()
    {
        // Calculate true linear acceleration
        // This would typically come from physics simulation
        Vector3 currentVelocity = rb.velocity;
        Vector3 acceleration = (currentVelocity - rb.velocity) / Time.deltaTime;

        // Add gravity compensation
        acceleration -= Physics.gravity;

        // Transform to IMU frame
        return transform.InverseTransformDirection(acceleration);
    }

    Vector3 GetTrueMagneticField()
    {
        // Simulate magnetic field (Earth's magnetic field + local disturbances)
        Vector3 magneticField = new Vector3(0.2f, 0.0f, 0.4f); // Earth's magnetic field approximation

        // Add local magnetic disturbances
        magneticField += new Vector3(
            Mathf.Sin(Time.time * 0.1f) * 0.01f,
            Mathf.Cos(Time.time * 0.15f) * 0.01f,
            Mathf.Sin(Time.time * 0.2f) * 0.01f
        );

        return transform.InverseTransformDirection(magneticField);
    }

    Vector3 AddNoiseAndBias(Vector3 trueValue, float noiseStd, ref Vector3 bias)
    {
        Vector3 noise = new Vector3(
            RandomGaussian() * noiseStd,
            RandomGaussian() * noiseStd,
            RandomGaussian() * noiseStd
        );

        return trueValue + bias + noise;
    }

    QuaternionMsg CalculateOrientationFromMagneticField(Vector3 magneticField)
    {
        // Simplified orientation calculation using magnetic field
        // In practice, you'd use sensor fusion algorithms like Madgwick or Mahony
        Vector3 gravity = Physics.gravity.normalized;
        Vector3 magneticNorth = magneticField.normalized;

        // Create coordinate system
        Vector3 up = -gravity; // Z-axis
        Vector3 east = Vector3.Cross(up, magneticNorth).normalized; // X-axis
        Vector3 north = Vector3.Cross(east, up).normalized; // Y-axis

        // Create rotation matrix and convert to quaternion
        Matrix4x4 rotationMatrix = Matrix4x4.identity;
        rotationMatrix.SetColumn(0, new Vector4(east.x, east.y, east.z, 0));
        rotationMatrix.SetColumn(1, new Vector4(north.x, north.y, north.z, 0));
        rotationMatrix.SetColumn(2, new Vector4(up.x, up.y, up.z, 0));

        Quaternion rotation = Quaternion.LookRotation(north, up);

        return new QuaternionMsg(
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w
        );
    }

    void SetCovarianceMatrices(ImuMsg msg)
    {
        // Set acceleration covariance
        msg.linear_acceleration_covariance = new double[9];
        msg.linear_acceleration_covariance[0] = accelerometerNoiseStd * accelerometerNoiseStd; // XX
        msg.linear_acceleration_covariance[4] = accelerometerNoiseStd * accelerometerNoiseStd; // YY
        msg.linear_acceleration_covariance[8] = accelerometerNoiseStd * accelerometerNoiseStd; // ZZ

        // Set angular velocity covariance
        msg.angular_velocity_covariance = new double[9];
        msg.angular_velocity_covariance[0] = gyroscopeNoiseStd * gyroscopeNoiseStd; // XX
        msg.angular_velocity_covariance[4] = gyroscopeNoiseStd * gyroscopeNoiseStd; // YY
        msg.angular_velocity_covariance[8] = gyroscopeNoiseStd * gyroscopeNoiseStd; // ZZ

        // Set orientation covariance (set to -1 if not available)
        msg.orientation_covariance = new double[9];
        msg.orientation_covariance[0] = -1; // Indicates covariance is not available
    }

    float RandomGaussian()
    {
        // Box-Muller transform for Gaussian random numbers
        float u1 = Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }

    void UpdateBiasDrift()
    {
        // Simulate bias drift over time (random walk)
        accelerometerBias += new Vector3(
            RandomGaussian() * biasDriftRate,
            RandomGaussian() * biasDriftRate,
            RandomGaussian() * biasDriftRate
        ) * Time.deltaTime;

        gyroscopeBias += new Vector3(
            RandomGaussian() * biasDriftRate * 0.1f, // Gyro bias drift is typically slower
            RandomGaussian() * biasDriftRate * 0.1f,
            RandomGaussian() * biasDriftRate * 0.1f
        ) * Time.deltaTime;
    }
}
```

### Advanced IMU Features

#### Temperature Compensation

Simulating temperature effects on IMU performance:

```csharp
using UnityEngine;

public class IMUTemperatureCompensation : MonoBehaviour
{
    [Header("Temperature Configuration")]
    public float baseTemperature = 25f; // Celsius
    public float currentTemperature = 25f;
    public float temperatureDriftRate = 0.001f; // Change per degree per second
    public float tempCoefficientAccel = 0.0001f; // Scale factor per degree
    public float tempCoefficientGyro = 0.00005f; // Scale factor per degree

    private float temperatureChangeRate = 0.1f; // Degrees per second

    void Update()
    {
        UpdateTemperature();
    }

    void UpdateTemperature()
    {
        // Simulate temperature changes based on environment or robot operation
        float targetTemp = baseTemperature + Mathf.Sin(Time.time * 0.1f) * 10f; // Oscillating temperature
        currentTemperature = Mathf.Lerp(currentTemperature, targetTemp, Time.deltaTime * temperatureChangeRate);
    }

    public Vector3 ApplyTemperatureEffects(Vector3 rawValue, SensorType sensorType)
    {
        float tempDiff = currentTemperature - baseTemperature;

        switch (sensorType)
        {
            case SensorType.ACCELEROMETER:
                // Apply temperature-induced bias and scale factor changes
                Vector3 tempBias = new Vector3(
                    tempDiff * tempCoefficientAccel,
                    tempDiff * tempCoefficientAccel,
                    tempDiff * tempCoefficientAccel
                );

                float tempScale = 1.0f + tempDiff * 0.001f; // 0.1% per degree

                return (rawValue + tempBias) * tempScale;

            case SensorType.GYROSCOPE:
                Vector3 gyroTempBias = new Vector3(
                    tempDiff * tempCoefficientGyro,
                    tempDiff * tempCoefficientGyro,
                    tempDiff * tempCoefficientGyro
                );

                float gyroTempScale = 1.0f + tempDiff * 0.0005f; // 0.05% per degree

                return (rawValue + gyroTempBias) * gyroTempScale;

            default:
                return rawValue;
        }
    }
}

public enum SensorType
{
    ACCELEROMETER,
    GYROSCOPE,
    MAGNETOMETER
}
```

#### Cross-Axis Sensitivity

Simulating cross-axis interference:

```csharp
using UnityEngine;

public class IMUCrossAxisSensitivity : MonoBehaviour
{
    [Header("Cross-Axis Configuration")]
    public float crossAxisAccel = 0.001f; // Cross-axis sensitivity for accelerometer
    public float crossAxisGyro = 0.0005f; // Cross-axis sensitivity for gyroscope

    public Vector3 ApplyCrossAxisEffects(Vector3 rawValue, SensorType sensorType)
    {
        Matrix4x4 crossAxisMatrix = Matrix4x4.identity;

        switch (sensorType)
        {
            case SensorType.ACCELEROMETER:
                // Set cross-axis sensitivity coefficients
                float ca = crossAxisAccel;
                crossAxisMatrix[0, 1] = ca; // Y affects X
                crossAxisMatrix[0, 2] = ca; // Z affects X
                crossAxisMatrix[1, 0] = ca; // X affects Y
                crossAxisMatrix[1, 2] = ca; // Z affects Y
                crossAxisMatrix[2, 0] = ca; // X affects Z
                crossAxisMatrix[2, 1] = ca; // Y affects Z
                break;

            case SensorType.GYROSCOPE:
                float cg = crossAxisGyro;
                crossAxisMatrix[0, 1] = cg;
                crossAxisMatrix[0, 2] = cg;
                crossAxisMatrix[1, 0] = cg;
                crossAxisMatrix[1, 2] = cg;
                crossAxisMatrix[2, 0] = cg;
                crossAxisMatrix[2, 1] = cg;
                break;
        }

        Vector4 inputVector = new Vector4(rawValue.x, rawValue.y, rawValue.z, 1);
        Vector4 outputVector = crossAxisMatrix * inputVector;

        return new Vector3(outputVector.x, outputVector.y, outputVector.z);
    }
}
```

#### Scale Factor Non-Linearity

Simulating non-linear sensor response:

```csharp
using UnityEngine;

public class IMUNonLinearity : MonoBehaviour
{
    [Header("Non-Linearity Configuration")]
    public AnimationCurve accelNonLinearity = AnimationCurve.Linear(0, 0, 1, 1);
    public AnimationCurve gyroNonLinearity = AnimationCurve.Linear(0, 0, 1, 1);

    public Vector3 ApplyNonLinearity(Vector3 rawValue, SensorType sensorType)
    {
        switch (sensorType)
        {
            case SensorType.ACCELEROMETER:
                return new Vector3(
                    ApplyCurve(rawValue.x, accelNonLinearity),
                    ApplyCurve(rawValue.y, accelNonLinearity),
                    ApplyCurve(rawValue.z, accelNonLinearity)
                );

            case SensorType.GYROSCOPE:
                return new Vector3(
                    ApplyCurve(rawValue.x, gyroNonLinearity),
                    ApplyCurve(rawValue.y, gyroNonLinearity),
                    ApplyCurve(rawValue.z, gyroNonLinearity)
                );

            default:
                return rawValue;
        }
    }

    float ApplyCurve(float value, AnimationCurve curve)
    {
        // Normalize the input based on sensor range and apply curve
        float normalized = Mathf.Abs(value) / 100f; // Assuming 100 as max range
        float curveValue = curve.Evaluate(normalized);
        return Mathf.Sign(value) * curveValue * 100f; // Scale back to original range
    }
}
```

## Sensor Fusion Integration

### Complementary Filter

Implementing a simple complementary filter for orientation estimation:

```csharp
using UnityEngine;

public class IMUSensorFusion : MonoBehaviour
{
    [Header("Filter Configuration")]
    public float accelerometerWeight = 0.05f; // Weight for accelerometer in fusion
    public float gyroscopeWeight = 0.95f;     // Weight for gyroscope in fusion
    public float filterCutoffFreq = 1f;       // Cutoff frequency for accelerometer filter

    private Quaternion estimatedOrientation = Quaternion.identity;
    private Vector3 lastAccelReading = Vector3.zero;
    private Vector3 filteredAccel = Vector3.zero;
    private float timeConstant;

    void Start()
    {
        timeConstant = 1f / (2f * Mathf.PI * filterCutoffFreq);
    }

    public Quaternion UpdateOrientation(Vector3 accelerometer, Vector3 gyroscope, float deltaTime)
    {
        // Apply low-pass filter to accelerometer data
        filteredAccel = LowPassFilter(accelerometer, lastAccelReading, deltaTime);
        lastAccelReading = accelerometer;

        // Get orientation from accelerometer (pitch and roll)
        Quaternion accelOrientation = Quaternion.FromToRotation(Vector3.up, filteredAccel) * Quaternion.Euler(0, 0, 0);

        // Update orientation using gyroscope integration
        Vector3 gyroRotation = gyroscope * deltaTime;
        Quaternion gyroUpdate = Quaternion.Euler(gyroRotation * Mathf.Rad2Deg);
        Quaternion predictedOrientation = estimatedOrientation * gyroUpdate;

        // Fuse accelerometer and gyroscope data using complementary filter
        float angleDiff = Quaternion.Angle(estimatedOrientation, accelOrientation);
        if (angleDiff > 1f) // Only use accelerometer when there's significant difference
        {
            estimatedOrientation = Quaternion.Slerp(
                predictedOrientation,
                accelOrientation,
                accelerometerWeight
            );
        }
        else
        {
            estimatedOrientation = predictedOrientation;
        }

        return estimatedOrientation;
    }

    Vector3 LowPassFilter(Vector3 current, Vector3 previous, float deltaTime)
    {
        float alpha = deltaTime / (timeConstant + deltaTime);
        return Vector3.Lerp(current, previous, alpha);
    }

    // Alternative: Madgwick filter implementation
    public class MadgwickFilter
    {
        private float beta = 0.1f; // Algorithm gain
        private Quaternion quaternion = Quaternion.identity;

        public Quaternion Update(Vector3 gyroscope, Vector3 accelerometer, Vector3 magnetometer, float deltaTime)
        {
            if (deltaTime <= 0) return quaternion;

            // Convert gyroscope from deg/s to rad/s
            Vector3 gyro = gyroscope * Mathf.Deg2Rad;

            // Normalize accelerometer measurement
            accelerometer = accelerometer.normalized;

            // Normalize magnetometer measurement
            if (magnetometer.magnitude > 0)
                magnetometer = magnetometer.normalized;

            // Rate of change of quaternion from gyroscope
            Quaternion qDot = new Quaternion(
                -0.5f * (quaternion.x * gyro.x + quaternion.y * gyro.y + quaternion.z * gyro.z),
                -0.5f * (-quaternion.w * gyro.x - quaternion.z * gyro.y + quaternion.y * gyro.z),
                -0.5f * (-quaternion.z * gyro.x + quaternion.w * gyro.y - quaternion.x * gyro.z),
                -0.5f * (-quaternion.y * gyro.x + quaternion.x * gyro.y + quaternion.w * gyro.z)
            );

            // Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)
            if (accelerometer.magnitude > 0.1f)
            {
                // Normalise accelerometer measurement
                Vector3 v = new Vector3(
                    2.0f * (quaternion.x * quaternion.z - quaternion.w * quaternion.y),
                    2.0f * (quaternion.w * quaternion.x + quaternion.y * quaternion.z),
                    quaternion.w * quaternion.w - quaternion.x * quaternion.x - quaternion.y * quaternion.y + quaternion.z * quaternion.z
                ).normalized;

                // Error is sum of cross product between estimated direction and measured direction of gravity
                Vector3 error = Vector3.Cross(v, accelerometer);

                // Compute and apply integral feedback if enabled
                // Apply feedback step
                qDot = new Quaternion(
                    qDot.w + beta * error.x,
                    qDot.x + beta * error.y,
                    qDot.y + beta * error.z,
                    qDot.z
                );
            }

            // Integrate rate of change of quaternion to yield quaternion
            quaternion.w += qDot.w * deltaTime;
            quaternion.x += qDot.x * deltaTime;
            quaternion.y += qDot.y * deltaTime;
            quaternion.z += qDot.z * deltaTime;

            // Normalise quaternion
            float norm = Mathf.Sqrt(quaternion.w * quaternion.w +
                                  quaternion.x * quaternion.x +
                                  quaternion.y * quaternion.y +
                                  quaternion.z * quaternion.z);
            if (norm > 0)
            {
                quaternion.w /= norm;
                quaternion.x /= norm;
                quaternion.y /= norm;
                quaternion.z /= norm;
            }

            return quaternion;
        }
    }
}
```

## Performance Optimization

### Efficient IMU Processing

Optimizing IMU simulation performance:

```csharp
using UnityEngine;

public class OptimizedIMUSimulation : MonoBehaviour
{
    [Header("Performance Configuration")]
    public int updateRate = 100; // Hz
    public bool enableFiltering = true;
    public bool useFixedTimestep = true;

    private float lastUpdateTime;
    private UnityIMUSimulation.IMUSample[] sampleBuffer;
    private int bufferIndex = 0;
    private const int BUFFER_SIZE = 10;

    [System.Serializable]
    public class IMUSample
    {
        public Vector3 acceleration;
        public Vector3 angularVelocity;
        public Vector3 magneticField;
        public float timestamp;
    }

    void Start()
    {
        sampleBuffer = new UnityIMUSimulation.IMUSample[BUFFER_SIZE];
        for (int i = 0; i < BUFFER_SIZE; i++)
        {
            sampleBuffer[i] = new UnityIMUSimulation.IMUSample();
        }
    }

    void Update()
    {
        if (useFixedTimestep || Time.time - lastUpdateTime >= 1f / updateRate)
        {
            ProcessIMUSample();
            lastUpdateTime = Time.time;
        }
    }

    void ProcessIMUSample()
    {
        // Get raw sensor data
        Vector3 rawAccel = GetRawAcceleration();
        Vector3 rawGyro = GetRawAngularVelocity();
        Vector3 rawMag = GetRawMagneticField();

        // Apply all sensor effects efficiently
        Vector3 processedAccel = ApplyAllEffects(rawAccel, SensorType.ACCELEROMETER);
        Vector3 processedGyro = ApplyAllEffects(rawGyro, SensorType.GYROSCOPE);
        Vector3 processedMag = ApplyAllEffects(rawMag, SensorType.MAGNETOMETER);

        // Store in buffer for filtering
        StoreSample(processedAccel, processedGyro, processedMag);

        if (enableFiltering)
        {
            ApplyFiltering();
        }

        // Publish data
        PublishIMUData(processedAccel, processedGyro, processedMag);
    }

    Vector3 ApplyAllEffects(Vector3 rawValue, SensorType sensorType)
    {
        // Apply all effects in a single pass for efficiency
        Vector3 value = rawValue;

        // Apply temperature effects
        value = ApplyTemperatureEffects(value, sensorType);

        // Apply cross-axis effects
        value = ApplyCrossAxisEffects(value, sensorType);

        // Apply non-linearity
        value = ApplyNonLinearity(value, sensorType);

        // Add noise and bias
        value = AddNoiseAndBias(value, GetNoiseStd(sensorType));

        return value;
    }

    float GetNoiseStd(SensorType sensorType)
    {
        switch (sensorType)
        {
            case SensorType.ACCELEROMETER: return 0.017f;
            case SensorType.GYROSCOPE: return 0.001f;
            case SensorType.MAGNETOMETER: return 0.01f;
            default: return 0.0f;
        }
    }

    void StoreSample(Vector3 accel, Vector3 gyro, Vector3 mag)
    {
        sampleBuffer[bufferIndex].acceleration = accel;
        sampleBuffer[bufferIndex].angularVelocity = gyro;
        sampleBuffer[bufferIndex].magneticField = mag;
        sampleBuffer[bufferIndex].timestamp = Time.time;

        bufferIndex = (bufferIndex + 1) % BUFFER_SIZE;
    }

    void ApplyFiltering()
    {
        // Apply simple filtering to reduce noise
        // This could be extended to more sophisticated filtering
    }

    Vector3 GetRawAcceleration()
    {
        // Get raw acceleration from physics simulation
        return Physics.gravity + Vector3.zero; // Placeholder
    }

    Vector3 GetRawAngularVelocity()
    {
        // Get raw angular velocity from physics simulation
        return Vector3.zero; // Placeholder
    }

    Vector3 GetRawMagneticField()
    {
        // Get raw magnetic field
        return new Vector3(0.2f, 0.0f, 0.4f); // Earth's magnetic field
    }

    Vector3 ApplyTemperatureEffects(Vector3 rawValue, SensorType sensorType)
    {
        // Temperature compensation logic
        return rawValue;
    }

    Vector3 ApplyCrossAxisEffects(Vector3 rawValue, SensorType sensorType)
    {
        // Cross-axis sensitivity logic
        return rawValue;
    }

    Vector3 ApplyNonLinearity(Vector3 rawValue, SensorType sensorType)
    {
        // Non-linearity compensation logic
        return rawValue;
    }

    Vector3 AddNoiseAndBias(Vector3 rawValue, float noiseStd)
    {
        // Add noise and bias
        return rawValue + new Vector3(
            RandomGaussian() * noiseStd,
            RandomGaussian() * noiseStd,
            RandomGaussian() * noiseStd
        );
    }

    float RandomGaussian()
    {
        float u1 = Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }

    void PublishIMUData(Vector3 accel, Vector3 gyro, Vector3 mag)
    {
        // Publish IMU data to ROS or other systems
    }
}
```

## Calibration and Validation

### IMU Calibration

Implementing IMU calibration procedures:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class IMUCalibration : MonoBehaviour
{
    [Header("Calibration Configuration")]
    public int calibrationSamples = 1000;
    public float calibrationDuration = 10f; // seconds
    public bool isCalibrating = false;

    private List<Vector3> accelSamples = new List<Vector3>();
    private List<Vector3> gyroSamples = new List<Vector3>();
    private float calibrationStartTime;

    // Calibration results
    public Vector3 accelerometerBias = Vector3.zero;
    public Vector3 gyroscopeBias = Vector3.zero;
    public Matrix4x4 accelerometerScale = Matrix4x4.identity;
    public Matrix4x4 gyroscopeScale = Matrix4x4.identity;

    public void StartCalibration()
    {
        isCalibrating = true;
        calibrationStartTime = Time.time;
        accelSamples.Clear();
        gyroSamples.Clear();

        Debug.Log("Starting IMU calibration. Keep device stationary.");
    }

    void Update()
    {
        if (isCalibrating)
        {
            CollectCalibrationSamples();

            if (Time.time - calibrationStartTime >= calibrationDuration)
            {
                CompleteCalibration();
                isCalibrating = false;
            }
        }
    }

    void CollectCalibrationSamples()
    {
        // In a real implementation, you'd get these from your IMU simulation
        Vector3 rawAccel = GetRawAcceleration(); // Placeholder
        Vector3 rawGyro = GetRawAngularVelocity(); // Placeholder

        accelSamples.Add(rawAccel);
        gyroSamples.Add(rawGyro);
    }

    void CompleteCalibration()
    {
        // Calculate accelerometer bias (should be at gravity level when stationary)
        Vector3 avgAccel = Vector3.zero;
        foreach (Vector3 sample in accelSamples)
        {
            avgAccel += sample;
        }
        avgAccel /= accelSamples.Count;

        // For stationary IMU, acceleration should be [0, 0, g]
        accelerometerBias = new Vector3(avgAccel.x, avgAccel.y, avgAccel.z - Physics.gravity.magnitude);

        // Calculate gyroscope bias (should be zero when stationary)
        Vector3 avgGyro = Vector3.zero;
        foreach (Vector3 sample in gyroSamples)
        {
            avgGyro += sample;
        }
        avgGyro /= gyroSamples.Count;
        gyroscopeBias = avgGyro;

        Debug.Log($"Calibration complete. Accel bias: {accelerometerBias}, Gyro bias: {gyroscopeBias}");
    }

    public Vector3 ApplyCalibration(Vector3 rawValue, SensorType sensorType)
    {
        switch (sensorType)
        {
            case SensorType.ACCELEROMETER:
                // Apply bias correction
                Vector3 corrected = rawValue - accelerometerBias;
                // Apply scale factor correction (simplified)
                return new Vector3(
                    corrected.x * accelerometerScale.m00,
                    corrected.y * accelerometerScale.m11,
                    corrected.z * accelerometerScale.m22
                );

            case SensorType.GYROSCOPE:
                // Apply bias correction
                Vector3 gyroCorrected = rawValue - gyroscopeBias;
                // Apply scale factor correction
                return new Vector3(
                    gyroCorrected.x * gyroscopeScale.m00,
                    gyroCorrected.y * gyroscopeScale.m11,
                    gyroCorrected.z * gyroscopeScale.m22
                );

            default:
                return rawValue;
        }
    }

    Vector3 GetRawAcceleration()
    {
        // Placeholder - in real implementation, get from physics
        return Vector3.zero;
    }

    Vector3 GetRawAngularVelocity()
    {
        // Placeholder - in real implementation, get from physics
        return Vector3.zero;
    }
}
```

IMU simulation is essential for creating realistic digital twins that can accurately represent a robot's motion and orientation. Proper simulation of IMUs enables effective testing of navigation, stabilization, and control algorithms in virtual environments before deployment on real robots.