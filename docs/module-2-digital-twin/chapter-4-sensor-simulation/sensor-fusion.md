---
sidebar_position: 5
title: "Sensor Fusion in Robotics"
description: "Combining multiple sensor data for enhanced robotics perception"
---

# Sensor Fusion in Robotics

Sensor fusion is the process of combining data from multiple sensors to achieve more accurate, reliable, and robust perception than would be possible with any single sensor alone. In robotics, sensor fusion is essential for creating accurate digital twins that can effectively represent real-world conditions and support complex robotic behaviors.

## Fundamentals of Sensor Fusion

### Why Sensor Fusion is Necessary

Robotics systems face several challenges that make sensor fusion essential:

#### Individual Sensor Limitations
- **Limited Field of View**: Single sensors only perceive part of the environment
- **Range Limitations**: Different sensors have different effective ranges
- **Environmental Sensitivity**: Performance degrades under certain conditions
- **Noise and Uncertainty**: All sensors have inherent measurement errors
- **Temporal Constraints**: Different sensors may have different update rates

#### Benefits of Fusion
- **Redundancy**: Backup if one sensor fails
- **Complementary Information**: Different sensors provide different types of data
- **Enhanced Accuracy**: Combined measurements are more accurate than individual ones
- **Robustness**: System continues functioning despite individual sensor failures
- **Increased Reliability**: Confidence in measurements through cross-validation

### Types of Sensor Fusion

#### Data-Level Fusion
- **Early Fusion**: Raw sensor data is combined before processing
- **Advantages**: Maximum information preservation
- **Challenges**: High computational requirements, synchronization needed
- **Applications**: Multi-camera systems, LiDAR-camera integration

#### Feature-Level Fusion
- **Mid-Level Fusion**: Features extracted from sensors are combined
- **Advantages**: Reduced data volume, focused on relevant information
- **Challenges**: Feature extraction must be robust and consistent
- **Applications**: Object detection, tracking systems

#### Decision-Level Fusion
- **Late Fusion**: Individual sensor decisions are combined
- **Advantages**: Modular design, easy to implement
- **Challenges**: Less information available for combination
- **Applications**: Classification systems, behavior selection

## Mathematical Foundations

### Probability Theory

Sensor fusion is fundamentally based on probability theory:

#### Bayes' Theorem
```
P(A|B) = P(B|A) * P(A) / P(B)
```
- **P(A|B)**: Posterior probability (belief after evidence)
- **P(B|A)**: Likelihood (probability of evidence given belief)
- **P(A)**: Prior probability (belief before evidence)
- **P(B)**: Evidence (normalization factor)

#### Kalman Filter Basics

The Kalman filter is a fundamental algorithm for sensor fusion:

```csharp
public class KalmanFilter
{
    private Matrix4x4 state;           // State vector
    private Matrix4x4 covariance;      // Error covariance matrix
    private Matrix4x4 processNoise;    // Process noise covariance
    private Matrix4x4 measurementNoise; // Measurement noise covariance
    private Matrix4x4 stateTransition;  // State transition model
    private Matrix4x4 observation;      // Observation model

    public void Predict()
    {
        // Predict next state
        state = stateTransition * state;

        // Predict next covariance
        covariance = stateTransition * covariance * stateTransition.transpose() + processNoise;
    }

    public void Update(Vector3 measurement)
    {
        // Calculate Kalman gain
        Matrix4x4 innovationCovariance = observation * covariance * observation.transpose() + measurementNoise;
        Matrix4x4 kalmanGain = covariance * observation.transpose() * innovationCovariance.inverse();

        // Update state
        Vector3 innovation = measurement - observation * state; // measurement - predicted measurement
        state = state + kalmanGain * innovation;

        // Update covariance
        Matrix4x4 identity = Matrix4x4.identity;
        covariance = (identity - kalmanGain * observation) * covariance;
    }
}
```

### Extended Kalman Filter (EKF)

For non-linear systems:

```csharp
public class ExtendedKalmanFilter
{
    public Vector3 State { get; private set; }
    public Matrix4x4 Covariance { get; private set; }

    public delegate Vector3 StateTransitionFunction(Vector3 state, float deltaTime);
    public delegate Vector3 ObservationFunction(Vector3 state);
    public delegate Matrix4x4 JacobianFunction(Vector3 state);

    private JacobianFunction stateJacobianFunc;
    private JacobianFunction observationJacobianFunc;
    private Matrix4x4 processNoise;
    private Matrix4x4 measurementNoise;

    public void Predict(float deltaTime)
    {
        // Linearize the non-linear state transition function
        Matrix4x4 F = stateJacobianFunc(State); // Jacobian of state transition

        // Predict state (non-linear function)
        State = StateTransitionFunction(State, deltaTime);

        // Predict covariance (linearized)
        Covariance = F * Covariance * F.transpose() + processNoise;
    }

    public void Update(Vector3 measurement)
    {
        // Linearize the non-linear observation function
        Matrix4x4 H = observationJacobianFunc(State); // Jacobian of observation

        // Calculate innovation
        Vector3 predictedMeasurement = ObservationFunction(State);
        Vector3 innovation = measurement - predictedMeasurement;

        // Calculate innovation covariance
        Matrix4x4 innovationCov = H * Covariance * H.transpose() + measurementNoise;

        // Calculate Kalman gain
        Matrix4x4 kalmanGain = Covariance * H.transpose() * innovationCov.inverse();

        // Update state and covariance
        State = State + kalmanGain * innovation;
        Covariance = (Matrix4x4.identity - kalmanGain * H) * Covariance;
    }
}
```

### Particle Filter

For highly non-linear or non-Gaussian systems:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class ParticleFilter
{
    public struct Particle
    {
        public Vector3 state;
        public float weight;

        public Particle(Vector3 state, float weight)
        {
            this.state = state;
            this.weight = weight;
        }
    }

    private List<Particle> particles;
    private int particleCount;

    public ParticleFilter(int particleCount)
    {
        this.particleCount = particleCount;
        particles = new List<Particle>(particleCount);

        // Initialize particles with uniform distribution
        for (int i = 0; i < particleCount; i++)
        {
            Vector3 randomState = GenerateRandomState();
            particles.Add(new Particle(randomState, 1.0f / particleCount));
        }
    }

    public Vector3 GetEstimatedState()
    {
        Vector3 estimatedState = Vector3.zero;
        float totalWeight = 0f;

        foreach (Particle particle in particles)
        {
            estimatedState += particle.state * particle.weight;
            totalWeight += particle.weight;
        }

        return estimatedState / totalWeight;
    }

    public void Predict(float deltaTime)
    {
        foreach (int i in particles.Count)
        {
            // Sample from motion model
            Vector3 noise = GenerateMotionNoise();
            Vector3 newState = MotionModel(particles[i].state, deltaTime) + noise;

            particles[i] = new Particle(newState, particles[i].weight);
        }
    }

    public void Update(Vector3 measurement)
    {
        float totalWeight = 0f;

        // Update particle weights based on measurement likelihood
        for (int i = 0; i < particles.Count; i++)
        {
            float likelihood = MeasurementLikelihood(particles[i].state, measurement);
            float newWeight = particles[i].weight * likelihood;

            particles[i] = new Particle(particles[i].state, newWeight);
            totalWeight += newWeight;
        }

        // Normalize weights
        if (totalWeight > 0)
        {
            for (int i = 0; i < particles.Count; i++)
            {
                float normalizedWeight = particles[i].weight / totalWeight;
                particles[i] = new Particle(particles[i].state, normalizedWeight);
            }
        }

        // Resample particles if needed (to avoid degeneracy)
        if (EffectiveSampleSize() < particleCount / 2)
        {
            Resample();
        }
    }

    private float EffectiveSampleSize()
    {
        float sumOfSquaredWeights = 0f;
        foreach (Particle p in particles)
        {
            sumOfSquaredWeights += p.weight * p.weight;
        }
        return 1.0f / sumOfSquaredWeights;
    }

    private void Resample()
    {
        List<Particle> newParticles = new List<Particle>(particleCount);
        float weightSum = 0f;

        foreach (Particle p in particles)
        {
            weightSum += p.weight;
        }

        float step = weightSum / particleCount;
        float start = Random.value * step;
        int index = 0;
        float cumulativeWeight = particles[0].weight;

        for (int i = 0; i < particleCount; i++)
        {
            float position = start + i * step;

            while (position > cumulativeWeight)
            {
                index++;
                cumulativeWeight += particles[index].weight;
            }

            newParticles.Add(new Particle(particles[index].state, 1.0f / particleCount));
        }

        particles = newParticles;
    }

    private Vector3 GenerateRandomState()
    {
        // Generate random state based on initial uncertainty
        return new Vector3(
            Random.Range(-1f, 1f),
            Random.Range(-1f, 1f),
            Random.Range(-Mathf.PI, Mathf.PI)
        );
    }

    private Vector3 GenerateMotionNoise()
    {
        // Generate noise for motion model
        return new Vector3(
            RandomGaussian() * 0.1f,
            RandomGaussian() * 0.1f,
            RandomGaussian() * 0.05f
        );
    }

    private Vector3 MotionModel(Vector3 state, float deltaTime)
    {
        // Implement motion model (e.g., constant velocity)
        return state; // Placeholder
    }

    private float MeasurementLikelihood(Vector3 predictedState, Vector3 measurement)
    {
        // Calculate likelihood of measurement given predicted state
        Vector3 diff = measurement - predictedState;
        float distance = diff.magnitude;

        // Use Gaussian likelihood
        float stdDev = 0.5f; // Measurement uncertainty
        return Mathf.Exp(-(distance * distance) / (2 * stdDev * stdDev));
    }

    private float RandomGaussian()
    {
        float u1 = Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }
}
```

## Common Sensor Fusion Applications

### Localization Fusion

Combining multiple sensors for robot localization:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class LocalizationFusion : MonoBehaviour
{
    [Header("Sensor Weights")]
    public float odometryWeight = 0.6f;
    public float imuWeight = 0.2f;
    public float gpsWeight = 0.15f;
    public float landmarkWeight = 0.05f;

    [Header("Fusion Configuration")]
    public float fusionRate = 50f; // Hz

    private ExtendedKalmanFilter ekf;
    private float lastFusionTime;

    // Sensor data buffers
    private Queue<OdometryData> odometryBuffer;
    private Queue<IMUData> imuBuffer;
    private Queue<GPSData> gpsBuffer;
    private Queue<LandmarkData> landmarkBuffer;

    void Start()
    {
        ekf = new ExtendedKalmanFilter();
        odometryBuffer = new Queue<OdometryData>();
        imuBuffer = new Queue<IMUData>();
        gpsBuffer = new Queue<GPSData>();
        landmarkBuffer = new Queue<LandmarkData>();
    }

    void Update()
    {
        if (Time.time - lastFusionTime >= 1f / fusionRate)
        {
            PerformFusion();
            lastFusionTime = Time.time;
        }
    }

    void PerformFusion()
    {
        // Get the latest data from each sensor
        OdometryData? latestOdometry = GetLatestData(odometryBuffer);
        IMUData? latestIMU = GetLatestData(imuBuffer);
        GPSData? latestGPS = GetLatestData(gpsBuffer);
        LandmarkData? latestLandmark = GetLatestData(landmarkBuffer);

        if (latestOdometry.HasValue)
        {
            // Predict state using odometry
            PredictWithOdometry(latestOdometry.Value);
        }

        // Update with different sensor measurements based on availability and reliability
        if (latestIMU.HasValue)
        {
            UpdateWithIMU(latestIMU.Value);
        }

        if (latestGPS.HasValue)
        {
            UpdateWithGPS(latestGPS.Value);
        }

        if (latestLandmark.HasValue)
        {
            UpdateWithLandmarks(latestLandmark.Value);
        }
    }

    void PredictWithOdometry(OdometryData odometry)
    {
        // Use odometry data to predict next state
        // This involves motion models based on wheel encoders or visual odometry
        Vector3 controlInput = new Vector3(odometry.deltaX, odometry.deltaY, odometry.deltaTheta);

        // Apply motion model in EKF
        ekf.Predict(odometry.timeDelta);
    }

    void UpdateWithIMU(IMUData imu)
    {
        // Use IMU data to correct position and orientation estimates
        // IMU provides acceleration and angular velocity
        Vector3 measurement = new Vector3(imu.acceleration.x, imu.acceleration.y, imu.angularVelocity.z);

        ekf.Update(measurement);
    }

    void UpdateWithGPS(GPSData gps)
    {
        // Use GPS data for global position correction
        // GPS provides absolute position with low frequency but high accuracy
        Vector3 measurement = new Vector3(gps.latitude, gps.longitude, gps.altitude);

        // Apply measurement with appropriate weight and noise model
        ekf.Update(measurement);
    }

    void UpdateWithLandmarks(LandmarkData landmark)
    {
        // Use landmark observations to correct position
        // Landmarks provide known reference points in the environment
        Vector3 measurement = new Vector3(landmark.range, landmark.bearing, landmark.landmarkId);

        ekf.Update(measurement);
    }

    T? GetLatestData<T>(Queue<T> buffer) where T : struct
    {
        if (buffer.Count > 0)
        {
            return buffer.Dequeue();
        }
        return null;
    }

    public Vector3 GetEstimatedPosition()
    {
        return ekf.State; // Return the fused position estimate
    }

    public Matrix4x4 GetUncertainty()
    {
        return ekf.Covariance; // Return uncertainty estimate
    }
}

[System.Serializable]
public struct OdometryData
{
    public float deltaX, deltaY, deltaTheta;
    public float timeDelta;
    public float confidence;
}

[System.Serializable]
public struct IMUData
{
    public Vector3 acceleration;
    public Vector3 angularVelocity;
    public float timestamp;
}

[System.Serializable]
public struct GPSData
{
    public double latitude, longitude, altitude;
    public float hdop, vdop; // Horizontal and vertical dilution of precision
    public float timestamp;
}

[System.Serializable]
public struct LandmarkData
{
    public float range, bearing;
    public int landmarkId;
    public float confidence;
}
```

### SLAM (Simultaneous Localization and Mapping)

```csharp
using UnityEngine;
using System.Collections.Generic;

public class SLAMFusion : MonoBehaviour
{
    [Header("SLAM Configuration")]
    public float mappingUpdateRate = 1f; // Hz
    public int maxMapFeatures = 1000;

    // Core SLAM components
    private ExtendedKalmanFilter slamEKF;
    private List<MapFeature> mapFeatures;
    private List<SensorObservation> observationBuffer;

    void Start()
    {
        slamEKF = new ExtendedKalmanFilter();
        mapFeatures = new List<MapFeature>();
        observationBuffer = new List<SensorObservation>();
    }

    public void ProcessSensorData(Vector3[] sensorReadings, Vector3 robotPose)
    {
        // Extract features from sensor data
        List<Feature> detectedFeatures = ExtractFeatures(sensorReadings);

        // Associate observations with existing map features
        List<FeatureAssociation> associations = DataAssociate(detectedFeatures);

        // Update robot pose using EKF
        UpdateRobotPose(associations);

        // Add new features to map if they are consistently observed
        AddNewFeaturesToMap(detectedFeatures, associations);

        // Update map features using observations
        UpdateMapFeatures(associations);
    }

    List<Feature> ExtractFeatures(Vector3[] sensorData)
    {
        List<Feature> features = new List<Feature>();

        // Extract features from sensor data (e.g., corners, edges, planes)
        for (int i = 0; i < sensorData.Length; i++)
        {
            // Simple example: detect planar surfaces in point cloud
            if (IsPlanarSurface(sensorData, i))
            {
                Feature planarFeature = new Feature
                {
                    type = FeatureType.PLANE,
                    position = CalculatePlaneCenter(sensorData, i),
                    normal = CalculatePlaneNormal(sensorData, i)
                };
                features.Add(planarFeature);
            }
        }

        return features;
    }

    List<FeatureAssociation> DataAssociate(List<Feature> detectedFeatures)
    {
        List<FeatureAssociation> associations = new List<FeatureAssociation>();

        foreach (Feature detected in detectedFeatures)
        {
            // Find closest map feature or create new one
            MapFeature closestFeature = FindClosestFeature(detected);

            if (closestFeature != null &&
                Vector3.Distance(detected.position, closestFeature.position) < 0.5f) // Association threshold
            {
                // Existing feature association
                associations.Add(new FeatureAssociation
                {
                    detectedFeature = detected,
                    mapFeature = closestFeature,
                    isNew = false
                });
            }
            else
            {
                // New feature to add to map
                associations.Add(new FeatureAssociation
                {
                    detectedFeature = detected,
                    mapFeature = null,
                    isNew = true
                });
            }
        }

        return associations;
    }

    void UpdateRobotPose(List<FeatureAssociation> associations)
    {
        // Use EKF to update robot pose based on feature observations
        // This involves the Jacobian matrices for robot pose and feature locations
    }

    void AddNewFeaturesToMap(List<Feature> detectedFeatures, List<FeatureAssociation> associations)
    {
        foreach (var association in associations)
        {
            if (association.isNew && mapFeatures.Count < maxMapFeatures)
            {
                // Add new feature to map with initial uncertainty
                MapFeature newFeature = new MapFeature
                {
                    position = association.detectedFeature.position,
                    uncertainty = CalculateInitialUncertainty()
                };

                mapFeatures.Add(newFeature);
            }
        }
    }

    void UpdateMapFeatures(List<FeatureAssociation> associations)
    {
        // Update map features using observations
        // This involves updating the state covariance matrix for the joint robot-feature system
    }

    bool IsPlanarSurface(Vector3[] points, int centerIndex, int neighborhoodSize = 10)
    {
        // Simple planarity test using least squares fitting
        List<Vector3> neighborhood = GetNeighborhood(points, centerIndex, neighborhoodSize);

        if (neighborhood.Count < 3) return false;

        // Calculate plane parameters using SVD
        // ... plane fitting algorithm ...

        // Return true if planarity measure is above threshold
        return true; // Placeholder
    }

    Vector3 CalculatePlaneCenter(Vector3[] points, int centerIndex)
    {
        // Calculate center of planar surface
        return points[centerIndex]; // Placeholder
    }

    Vector3 CalculatePlaneNormal(Vector3[] points, int centerIndex)
    {
        // Calculate normal of planar surface
        return Vector3.up; // Placeholder
    }

    List<Vector3> GetNeighborhood(Vector3[] points, int centerIndex, int size)
    {
        // Get neighboring points for surface analysis
        List<Vector3> neighborhood = new List<Vector3>();

        int start = Mathf.Max(0, centerIndex - size / 2);
        int end = Mathf.Min(points.Length, centerIndex + size / 2);

        for (int i = start; i < end; i++)
        {
            neighborhood.Add(points[i]);
        }

        return neighborhood;
    }

    MapFeature FindClosestFeature(Feature detected)
    {
        MapFeature closest = null;
        float minDistance = float.MaxValue;

        foreach (MapFeature feature in mapFeatures)
        {
            float distance = Vector3.Distance(detected.position, feature.position);
            if (distance < minDistance)
            {
                minDistance = distance;
                closest = feature;
            }
        }

        return closest;
    }

    Matrix4x4 CalculateInitialUncertainty()
    {
        // Calculate initial uncertainty for new map feature
        Matrix4x4 uncertainty = Matrix4x4.identity;
        uncertainty.m00 = 1.0f; // Position uncertainty
        uncertainty.m11 = 1.0f;
        uncertainty.m22 = 1.0f;
        return uncertainty;
    }
}

public enum FeatureType
{
    CORNER,
    EDGE,
    PLANE,
    POINT
}

public struct Feature
{
    public FeatureType type;
    public Vector3 position;
    public Vector3 normal;
    public float descriptor; // For feature matching
}

public struct MapFeature
{
    public Vector3 position;
    public Matrix4x4 uncertainty;
    public int observationCount;
}

public struct FeatureAssociation
{
    public Feature detectedFeature;
    public MapFeature mapFeature;
    public bool isNew;
}
```

### Multi-Sensor Integration

Combining LiDAR, cameras, and IMUs:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class MultiSensorFusion : MonoBehaviour
{
    [Header("Sensor Configuration")]
    public UnityLidarSimulation lidar;
    public UnityDepthCamera depthCamera;
    public UnityIMUSimulation imu;

    [Header("Fusion Parameters")]
    public float lidarWeight = 0.4f;
    public float cameraWeight = 0.3f;
    public float imuWeight = 0.3f;

    private ParticleFilter particleFilter;
    private List<Observation> observationBuffer;

    void Start()
    {
        particleFilter = new ParticleFilter(1000);
        observationBuffer = new List<Observation>();
    }

    void Update()
    {
        // Collect observations from all sensors
        CollectLidarObservations();
        CollectCameraObservations();
        CollectIMUObservations();

        // Perform fusion if we have enough observations
        if (observationBuffer.Count > 0)
        {
            PerformMultiSensorFusion();
            observationBuffer.Clear();
        }
    }

    void CollectLidarObservations()
    {
        // Get latest LiDAR point cloud
        List<Vector3> lidarPoints = lidar.GetPointCloud();

        foreach (Vector3 point in lidarPoints)
        {
            observationBuffer.Add(new Observation
            {
                sensorType = SensorType.LIDAR,
                position = point,
                weight = lidarWeight,
                timestamp = Time.time
            });
        }
    }

    void CollectCameraObservations()
    {
        // Process camera data for feature extraction
        // This would involve computer vision algorithms
    }

    void CollectIMUObservations()
    {
        // Get latest IMU data
        Vector3 accel = new Vector3(0, 0, 0); // Placeholder
        Vector3 gyro = new Vector3(0, 0, 0);  // Placeholder

        observationBuffer.Add(new Observation
        {
            sensorType = SensorType.IMU,
            position = accel,
            orientation = gyro,
            weight = imuWeight,
            timestamp = Time.time
        });
    }

    void PerformMultiSensorFusion()
    {
        // Predict step using motion model
        particleFilter.Predict(Time.deltaTime);

        // Update step using all observations
        foreach (Observation obs in observationBuffer)
        {
            particleFilter.Update(obs);
        }

        // Get fused estimate
        Vector3 fusedPosition = particleFilter.GetEstimatedState();
        UpdateRobotVisualization(fusedPosition);
    }

    void UpdateRobotVisualization(Vector3 position)
    {
        // Update robot position in simulation based on fused estimate
        transform.position = position;
    }
}

public enum SensorType
{
    LIDAR,
    CAMERA,
    IMU,
    GPS,
    ODOMETRY
}

public struct Observation
{
    public SensorType sensorType;
    public Vector3 position;
    public Vector3 orientation;
    public float weight;
    public float timestamp;
    public float confidence;
}
```

## Implementation Considerations

### Synchronization

Proper synchronization is crucial for sensor fusion:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class SensorSynchronizer : MonoBehaviour
{
    [Header("Synchronization Configuration")]
    public float timeWindow = 0.01f; // 10ms window for synchronization
    public float maxLatency = 0.1f;  // Maximum acceptable latency

    private Dictionary<SensorType, Queue<SensorData>> sensorBuffers;
    private List<SynchronizedData> synchronizedData;

    void Start()
    {
        sensorBuffers = new Dictionary<SensorType, Queue<SensorData>>();
        synchronizedData = new List<SynchronizedData>();

        // Initialize buffers for each sensor type
        foreach (SensorType sensor in System.Enum.GetValues(typeof(SensorType)))
        {
            sensorBuffers[sensor] = new Queue<SensorData>();
        }
    }

    public void AddSensorData(SensorData data)
    {
        // Add data to appropriate buffer
        if (sensorBuffers.ContainsKey(data.type))
        {
            sensorBuffers[data.type].Enqueue(data);
        }

        // Try to synchronize data
        TrySynchronizeData();
    }

    void TrySynchronizeData()
    {
        // Find the latest timestamp across all sensors
        float latestTime = FindLatestTimestamp();

        // Look for data within the synchronization window
        while (AllSensorsHaveData(latestTime - timeWindow))
        {
            // Extract synchronized data bundle
            SynchronizedData bundle = ExtractSynchronizedBundle(latestTime - timeWindow);
            synchronizedData.Add(bundle);

            // Remove processed data from buffers
            RemoveProcessedData();
        }

        // Clean old data that's beyond max latency
        CleanOldData();
    }

    float FindLatestTimestamp()
    {
        float latest = float.MinValue;
        foreach (var buffer in sensorBuffers.Values)
        {
            if (buffer.Count > 0)
            {
                float newestTime = buffer.Peek().timestamp;
                latest = Mathf.Max(latest, newestTime);
            }
        }
        return latest;
    }

    bool AllSensorsHaveData(float minTime)
    {
        foreach (var buffer in sensorBuffers.Values)
        {
            if (buffer.Count == 0) return false;
            if (buffer.Peek().timestamp < minTime) return false;
        }
        return true;
    }

    SynchronizedData ExtractSynchronizedBundle(float minTime)
    {
        SynchronizedData bundle = new SynchronizedData();
        bundle.timestamp = minTime;

        foreach (var pair in sensorBuffers)
        {
            // Find closest data point to target time
            SensorData closest = FindClosestData(pair.Value, minTime);
            bundle.sensorData.Add(pair.Key, closest);
        }

        return bundle;
    }

    SensorData FindClosestData(Queue<SensorData> buffer, float targetTime)
    {
        SensorData closest = buffer.Peek();
        float minDiff = Mathf.Abs(closest.timestamp - targetTime);

        // This is a simplified approach - in practice, you might want to
        // interpolate between two closest points
        foreach (var data in buffer)
        {
            float diff = Mathf.Abs(data.timestamp - targetTime);
            if (diff < minDiff)
            {
                minDiff = diff;
                closest = data;
            }
        }

        return closest;
    }

    void RemoveProcessedData()
    {
        foreach (var buffer in sensorBuffers.Values)
        {
            if (buffer.Count > 0)
            {
                buffer.Dequeue();
            }
        }
    }

    void CleanOldData()
    {
        float cutoffTime = Time.time - maxLatency;

        foreach (var buffer in sensorBuffers.Values)
        {
            while (buffer.Count > 0 && buffer.Peek().timestamp < cutoffTime)
            {
                buffer.Dequeue();
            }
        }
    }
}

[System.Serializable]
public struct SensorData
{
    public SensorType type;
    public Vector3 data;
    public float timestamp;
    public float confidence;
}

public struct SynchronizedData
{
    public float timestamp;
    public Dictionary<SensorType, SensorData> sensorData;

    public SynchronizedData()
    {
        timestamp = 0f;
        sensorData = new Dictionary<SensorType, SensorData>();
    }
}
```

### Computational Efficiency

Optimizing sensor fusion for real-time performance:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class EfficientSensorFusion : MonoBehaviour
{
    [Header("Efficiency Configuration")]
    public int maxParticles = 500;
    public int fusionSubsampleRate = 1; // Process every Nth sensor reading
    public bool useMultithreading = true;

    // Pre-allocated buffers to avoid garbage collection
    private Vector3[] tempVectorBuffer;
    private float[] tempFloatBuffer;
    private Matrix4x4 tempMatrix;

    // Caching to avoid repeated calculations
    private Dictionary<string, object> calculationCache;

    void Start()
    {
        // Pre-allocate buffers
        tempVectorBuffer = new Vector3[1000];
        tempFloatBuffer = new float[1000];
        tempMatrix = Matrix4x4.identity;

        calculationCache = new Dictionary<string, object>();
    }

    public Vector3 EfficientFusionUpdate(List<SensorReading> readings)
    {
        // Subsample if needed
        if (Time.frameCount % fusionSubsampleRate != 0)
        {
            return GetCachedEstimate();
        }

        // Use pre-allocated buffers
        int validReadingsCount = FilterValidReadings(readings, tempVectorBuffer);

        // Perform efficient fusion calculation
        return PerformFusion(tempVectorBuffer, validReadingsCount);
    }

    int FilterValidReadings(List<SensorReading> readings, Vector3[] buffer)
    {
        int count = 0;
        for (int i = 0; i < readings.Count && i < buffer.Length; i++)
        {
            if (readings[i].IsValid())
            {
                buffer[count] = readings[i].GetVector();
                count++;
            }
        }
        return count;
    }

    Vector3 PerformFusion(Vector3[] readings, int count)
    {
        // Efficient weighted average fusion
        Vector3 weightedSum = Vector3.zero;
        float totalWeight = 0f;

        for (int i = 0; i < count; i++)
        {
            float weight = CalculateReadingWeight(readings[i]);
            weightedSum += readings[i] * weight;
            totalWeight += weight;
        }

        if (totalWeight > 0)
        {
            return weightedSum / totalWeight;
        }

        return Vector3.zero;
    }

    float CalculateReadingWeight(Vector3 reading)
    {
        // Cache key based on reading characteristics
        string cacheKey = $"weight_{reading.magnitude:F3}";

        if (calculationCache.ContainsKey(cacheKey))
        {
            return (float)calculationCache[cacheKey];
        }

        // Calculate weight based on uncertainty model
        float weight = 1.0f / (0.01f + reading.magnitude * 0.001f); // Example model

        // Cache the result
        calculationCache[cacheKey] = weight;

        // Limit cache size
        if (calculationCache.Count > 1000)
        {
            // Remove oldest entries (simplified)
            var keys = new List<string>(calculationCache.Keys);
            calculationCache.Remove(keys[0]);
        }

        return weight;
    }

    Vector3 GetCachedEstimate()
    {
        // Return most recent estimate if skipping fusion
        return transform.position; // Placeholder
    }
}

[System.Serializable]
public struct SensorReading
{
    public SensorType sensorType;
    public Vector3 data;
    public float confidence;
    public float timestamp;
    public Matrix4x4 covariance;

    public bool IsValid()
    {
        return !float.IsNaN(data.x) && !float.IsInfinity(data.x);
    }

    public Vector3 GetVector()
    {
        return data;
    }
}
```

## Quality Assessment

### Performance Metrics

Evaluating sensor fusion performance:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class FusionPerformanceAnalyzer : MonoBehaviour
{
    [Header("Performance Metrics")]
    public bool enableAnalysis = true;
    public float analysisInterval = 1.0f;

    private List<FusionErrorSample> errorSamples;
    private float lastAnalysisTime;

    [System.Serializable]
    public struct FusionErrorSample
    {
        public float timestamp;
        public Vector3 estimationError;
        public float uncertainty;
        public float consistencyMeasure;
    }

    void Start()
    {
        errorSamples = new List<FusionErrorSample>();
    }

    void Update()
    {
        if (enableAnalysis && Time.time - lastAnalysisTime >= analysisInterval)
        {
            AnalyzeFusionPerformance();
            lastAnalysisTime = Time.time;
        }
    }

    void AnalyzeFusionPerformance()
    {
        if (errorSamples.Count == 0) return;

        // Calculate various performance metrics
        float rmse = CalculateRMSE();
        float avgUncertainty = CalculateAverageUncertainty();
        float consistency = CalculateConsistency();
        float efficiency = CalculateComputationalEfficiency();

        // Log performance metrics
        Debug.Log($"Fusion Performance - RMSE: {rmse:F3}, Avg Uncertainty: {avgUncertainty:F3}, " +
                 $"Consistency: {consistency:P2}, Efficiency: {efficiency:P2}");
    }

    float CalculateRMSE()
    {
        Vector3 sumSquares = Vector3.zero;
        foreach (var sample in errorSamples)
        {
            sumSquares += new Vector3(
                sample.estimationError.x * sample.estimationError.x,
                sample.estimationError.y * sample.estimationError.y,
                sample.estimationError.z * sample.estimationError.z
            );
        }

        Vector3 meanSquares = sumSquares / errorSamples.Count;
        return Mathf.Sqrt((meanSquares.x + meanSquares.y + meanSquares.z) / 3);
    }

    float CalculateAverageUncertainty()
    {
        float sum = 0f;
        foreach (var sample in errorSamples)
        {
            sum += sample.uncertainty;
        }
        return sum / errorSamples.Count;
    }

    float CalculateConsistency()
    {
        // Check if errors are within expected uncertainty bounds
        int consistentCount = 0;
        foreach (var sample in errorSamples)
        {
            // Check if error magnitude is within 3-sigma bounds
            if (sample.estimationError.magnitude <= 3 * sample.uncertainty)
            {
                consistentCount++;
            }
        }
        return (float)consistentCount / errorSamples.Count;
    }

    float CalculateComputationalEfficiency()
    {
        // Calculate ratio of useful work to computational cost
        // This is a simplified example
        return 0.8f; // Placeholder
    }

    public void LogFusionSample(Vector3 trueValue, Vector3 estimatedValue, float uncertainty)
    {
        FusionErrorSample sample = new FusionErrorSample();
        sample.timestamp = Time.time;
        sample.estimationError = estimatedValue - trueValue;
        sample.uncertainty = uncertainty;
        sample.consistencyMeasure = Mathf.Abs(sample.estimationError.magnitude - uncertainty);

        errorSamples.Add(sample);

        // Keep only recent samples
        if (errorSamples.Count > 1000)
        {
            errorSamples.RemoveAt(0);
        }
    }
}
```

Sensor fusion is a critical component of modern robotics systems, enabling robots to create more accurate and reliable representations of their environment and state. Proper implementation of sensor fusion techniques allows digital twins to provide realistic and useful simulations for testing and validation of robotic systems.