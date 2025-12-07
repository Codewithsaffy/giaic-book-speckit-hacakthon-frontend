---
sidebar_position: 3
title: "Depth Camera Simulation"
description: "Simulating RGB-D cameras and depth sensors for robotics applications"
---

# Depth Camera Simulation

Depth cameras, also known as RGB-D cameras, provide both color and depth information simultaneously, making them invaluable for robotics applications that require both visual recognition and 3D spatial understanding. Simulating depth cameras accurately is essential for creating realistic digital twins that can support computer vision and 3D perception tasks.

## Understanding Depth Cameras

### Types of Depth Cameras

#### Time-of-Flight (ToF) Cameras
- **Principle**: Measure time for light to travel to object and back
- **Range**: Typically 0.3m to 5m
- **Accuracy**: Good accuracy within range
- **Lighting**: Works in various lighting conditions
- **Resolution**: Moderate resolution depth maps

#### Structured Light Cameras
- **Principle**: Project known light pattern and analyze distortions
- **Accuracy**: High accuracy for close range
- **Range**: Short to medium range (0.3m to 2m)
- **Lighting**: Requires controlled lighting
- **Resolution**: High-resolution depth maps

#### Stereo Vision Systems
- **Principle**: Use two cameras to calculate depth via triangulation
- **Range**: Long range capabilities
- **Resolution**: Variable based on baseline and resolution
- **Computation**: Requires significant processing power
- **Calibration**: Complex calibration requirements

### Depth Camera Characteristics

#### Key Parameters
- **Depth Range**: Minimum and maximum measurable distances
- **Depth Accuracy**: Precision of distance measurements
- **Depth Resolution**: Number of depth measurements per frame
- **Frame Rate**: How often depth data is updated
- **Field of View**: Angular coverage of the sensor

#### Performance Metrics
- **Accuracy**: Closeness to true distance values
- **Precision**: Repeatability of measurements
- **Resolution**: Spatial resolution of depth map
- **Update Rate**: Frequency of depth measurements
- **Noise Level**: Amount of random variation in measurements

## Depth Camera Simulation in Gazebo

### Depth Camera Sensor Configuration

Configuring depth camera sensors in Gazebo:

```xml
<sdf version="1.7">
  <model name="depth_camera_model">
    <link name="camera_link">
      <sensor name="depth_camera" type="depth">
        <pose>0 0 0.1 0 0 0</pose>
        <camera>
          <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>10</far>
          </clip>
          <noise>
            <type>gaussian</type>
            <mean>0.0</mean>
            <stddev>0.01</stddev>
          </noise>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_openni_kinect.so">
          <baseline>0.2</baseline>
          <alwaysOn>true</alwaysOn>
          <updateRate>30.0</updateRate>
          <cameraName>depth_camera</cameraName>
          <imageTopicName>/rgb/image_raw</imageTopicName>
          <depthImageTopicName>/depth/image_raw</depthImageTopicName>
          <pointCloudTopicName>/depth/points</pointCloudTopicName>
          <cameraInfoTopicName>/rgb/camera_info</cameraInfoTopicName>
          <depthImageCameraInfoTopicName>/depth/camera_info</depthImageCameraInfoTopicName>
          <frameName>camera_link</frameName>
          <pointCloudCutoff>0.1</pointCloudCutoff>
          <pointCloudCutoffMax>5.0</pointCloudCutoffMax>
          <distortion_k1>0.0</distortion_k1>
          <distortion_k2>0.0</distortion_k2>
          <distortion_k3>0.0</distortion_k3>
          <distortion_t1>0.0</distortion_t1>
          <distortion_t2>0.0</distortion_t2>
          <CxPrime>0.0</CxPrime>
          <Cx>320.5</Cx>
          <Cy>240.5</Cy>
          <focalLength>525.0</focalLength>
        </plugin>
      </sensor>
    </link>
  </model>
</sdf>
```

### Depth Camera Plugin Development

Creating custom depth camera processing:

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/common/common.hh>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

namespace gazebo
{
  class CustomDepthCameraPlugin : public SensorPlugin
  {
    public: void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
    {
      // Cast to depth camera sensor
      this->parentSensor = std::dynamic_pointer_cast<sensors::DepthCameraSensor>(_sensor);
      if (!this->parentSensor)
      {
        gzerr << "CustomDepthCameraPlugin requires a DepthCameraSensor\n";
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

      // Create publishers
      this->imagePub = this->rosNode->advertise<sensor_msgs::Image>("/camera/rgb/image_raw", 1);
      this->depthPub = this->rosNode->advertise<sensor_msgs::Image>("/camera/depth/image_raw", 1);
      this->pointsPub = this->rosNode->advertise<sensor_msgs::PointCloud2>("/camera/depth/points", 1);

      // Connect to sensor update event
      this->newImageConnection = this->parentSensor->DepthCamera()->ConnectNewDepthFrame(
          boost::bind(&CustomDepthCameraPlugin::OnNewFrame, this, _1, _2, _3, _4, _5));
    }

    public: void OnNewFrame(const float *_image,
                           unsigned int _width, unsigned int _height,
                           unsigned int _depth, const std::string &_format)
    {
      // Process depth image
      ProcessDepthImage(_image, _width, _height);

      // Create and publish RGB image
      CreateAndPublishRGBImage();

      // Create and publish depth image
      CreateAndPublishDepthImage();

      // Create and publish point cloud
      CreateAndPublishPointCloud();
    }

    private: void ProcessDepthImage(const float *_image,
                                   unsigned int _width, unsigned int _height)
    {
      // Process the depth data here
      // Apply noise, filtering, or other processing as needed
    }

    private: void CreateAndPublishRGBImage()
    {
      // Create and publish RGB image message
      sensor_msgs::Image rgb_msg;
      // Fill in RGB image data
      this->imagePub.publish(rgb_msg);
    }

    private: void CreateAndPublishDepthImage()
    {
      // Create and publish depth image message
      sensor_msgs::Image depth_msg;
      // Fill in depth image data
      this->depthPub.publish(depth_msg);
    }

    private: void CreateAndPublishPointCloud()
    {
      // Create and publish point cloud message
      sensor_msgs::PointCloud2 points_msg;
      // Fill in point cloud data
      this->pointsPub.publish(points_msg);
    }

    private: sensors::DepthCameraSensorPtr parentSensor;
    private: std::unique_ptr<ros::NodeHandle> rosNode;
    private: ros::Publisher imagePub;
    private: ros::Publisher depthPub;
    private: ros::Publisher pointsPub;
    private: event::ConnectionPtr newImageConnection;
  };

  GZ_REGISTER_SENSOR_PLUGIN(CustomDepthCameraPlugin)
}
```

## Depth Camera Simulation in Unity

### Unity Depth Camera Implementation

Creating depth camera simulation in Unity:

```csharp
using UnityEngine;
using System.Collections.Generic;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;

public class UnityDepthCamera : MonoBehaviour
{
    [Header("Camera Configuration")]
    public int imageWidth = 640;
    public int imageHeight = 480;
    public float fieldOfView = 60f;
    public float nearClip = 0.1f;
    public float farClip = 10f;

    [Header("Depth Configuration")]
    public float depthNear = 0.1f;
    public float depthFar = 10f;
    public bool enablePointCloud = true;

    [Header("ROS Integration")]
    public string rgbTopic = "/camera/rgb/image_raw";
    public string depthTopic = "/camera/depth/image_raw";
    public string pointsTopic = "/camera/depth/points";
    public bool publishToROS = true;

    private Camera cam;
    private RenderTexture depthTexture;
    private Texture2D rgbTexture;
    private Texture2D depthTexture2D;
    private ROSConnection ros;
    private float[] depthData;

    void Start()
    {
        InitializeCamera();
        InitializeTextures();

        if (publishToROS)
        {
            ros = ROSConnection.GetOrCreateInstance();
            ros.RegisterPublisher<ImageMsg>(rgbTopic);
            ros.RegisterPublisher<ImageMsg>(depthTopic);
            ros.RegisterPublisher<PointCloud2Msg>(pointsTopic);
        }
    }

    void InitializeCamera()
    {
        cam = GetComponent<Camera>();
        if (cam == null)
        {
            cam = gameObject.AddComponent<Camera>();
        }

        cam.fieldOfView = fieldOfView;
        cam.nearClipPlane = nearClip;
        cam.farClipPlane = farClip;
        cam.depthTextureMode = DepthTextureMode.Depth;
    }

    void InitializeTextures()
    {
        // Create depth render texture
        depthTexture = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.Depth);
        cam.targetTexture = depthTexture;

        // Create textures for reading
        rgbTexture = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);
        depthTexture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RFloat, false);

        // Initialize depth data array
        depthData = new float[imageWidth * imageHeight];
    }

    void Update()
    {
        CaptureAndPublishImages();
    }

    void CaptureAndPublishImages()
    {
        // Capture RGB image
        RenderTexture.active = null;
        cam.targetTexture = null;
        cam.Render();

        // Capture RGB texture
        RenderTexture.active = cam.targetTexture;
        rgbTexture.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        rgbTexture.Apply();

        // Capture depth texture
        depthTexture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
        depthTexture2D.Apply();

        // Extract depth data
        Color[] depthColors = depthTexture2D.GetPixels();
        for (int i = 0; i < depthColors.Length; i++)
        {
            depthData[i] = depthColors[i].r; // Depth is stored in red channel
        }

        if (publishToROS)
        {
            PublishRGBImage();
            PublishDepthImage();
            if (enablePointCloud)
            {
                PublishPointCloud();
            }
        }

        // Reset render texture
        RenderTexture.active = null;
        cam.targetTexture = depthTexture;
    }

    void PublishRGBImage()
    {
        ImageMsg rgbMsg = new ImageMsg();
        rgbMsg.header.stamp = new TimeStamp(Time.time);
        rgbMsg.header.frame_id = transform.name;
        rgbMsg.height = (uint)imageHeight;
        rgbMsg.width = (uint)imageWidth;
        rgbMsg.encoding = "rgb8";
        rgbMsg.is_bigendian = 0;
        rgbMsg.step = (uint)(imageWidth * 3); // 3 bytes per pixel for RGB

        // Convert texture to byte array
        Color32[] colors = rgbTexture.GetPixels32();
        byte[] imageData = new byte[colors.Length * 3];
        for (int i = 0; i < colors.Length; i++)
        {
            imageData[i * 3] = colors[i].r;
            imageData[i * 3 + 1] = colors[i].g;
            imageData[i * 3 + 2] = colors[i].b;
        }
        rgbMsg.data = imageData;

        ros.Publish(rgbTopic, rgbMsg);
    }

    void PublishDepthImage()
    {
        ImageMsg depthMsg = new ImageMsg();
        depthMsg.header.stamp = new TimeStamp(Time.time);
        depthMsg.header.frame_id = transform.name;
        depthMsg.height = (uint)imageHeight;
        depthMsg.width = (uint)imageWidth;
        depthMsg.encoding = "32FC1"; // 32-bit float, 1 channel
        depthMsg.is_bigendian = 0;
        depthMsg.step = (uint)(imageWidth * 4); // 4 bytes per float

        // Convert depth data to byte array
        byte[] depthBytes = new byte[depthData.Length * 4];
        for (int i = 0; i < depthData.Length; i++)
        {
            byte[] floatBytes = System.BitConverter.GetBytes(depthData[i]);
            System.Buffer.BlockCopy(floatBytes, 0, depthBytes, i * 4, 4);
        }
        depthMsg.data = depthBytes;

        ros.Publish(depthTopic, depthMsg);
    }

    void PublishPointCloud()
    {
        PointCloud2Msg pointsMsg = new PointCloud2Msg();
        pointsMsg.header.stamp = new TimeStamp(Time.time);
        pointsMsg.header.frame_id = transform.name;

        // Define point cloud structure
        pointsMsg.fields = new PointFieldMsg[4];
        pointsMsg.fields[0] = new PointFieldMsg("x", 0, 7, 1);   // FLOAT32
        pointsMsg.fields[1] = new PointFieldMsg("y", 4, 7, 1);   // FLOAT32
        pointsMsg.fields[2] = new PointFieldMsg("z", 8, 7, 1);   // FLOAT32
        pointsMsg.fields[3] = new PointFieldMsg("rgb", 12, 7, 1); // FLOAT32

        pointsMsg.point_step = 16; // 4 fields * 4 bytes each
        pointsMsg.height = 1;
        pointsMsg.width = (uint)(imageWidth * imageHeight);

        // Generate point cloud data
        List<byte> pointCloudData = new List<byte>();
        for (int y = 0; y < imageHeight; y++)
        {
            for (int x = 0; x < imageWidth; x++)
            {
                int index = y * imageWidth + x;
                float depth = depthData[index];

                if (depth > 0 && depth < float.PositiveInfinity)
                {
                    // Convert pixel coordinates to 3D world coordinates
                    Vector3 point3D = PixelTo3D(x, y, depth);

                    // Add point data to byte array
                    pointCloudData.AddRange(System.BitConverter.GetBytes(point3D.x));
                    pointCloudData.AddRange(System.BitConverter.GetBytes(point3D.y));
                    pointCloudData.AddRange(System.BitConverter.GetBytes(point3D.z));
                    pointCloudData.AddRange(System.BitConverter.GetBytes(0f)); // RGB placeholder
                }
            }
        }

        pointsMsg.data = pointCloudData.ToArray();
        pointsMsg.row_step = (uint)pointCloudData.Count;

        ros.Publish(pointsTopic, pointsMsg);
    }

    Vector3 PixelTo3D(int x, int y, float depth)
    {
        // Convert pixel coordinates to normalized device coordinates
        float normX = (x - imageWidth / 2.0f) / (imageWidth / 2.0f);
        float normY = (y - imageHeight / 2.0f) / (imageHeight / 2.0f);

        // Convert to view space
        float fovRad = fieldOfView * Mathf.Deg2Rad;
        float aspect = (float)imageWidth / imageHeight;

        float viewX = normX * Mathf.Tan(fovRad / 2f) * aspect * depth;
        float viewY = normY * Mathf.Tan(fovRad / 2f) * depth;

        // Transform to world space
        Vector3 viewSpacePoint = new Vector3(viewX, viewY, depth);
        Vector3 worldPoint = transform.TransformPoint(viewSpacePoint);

        return worldPoint;
    }
}
```

### Advanced Depth Camera Features

#### Noise and Distortion Simulation

Adding realistic noise and distortion to depth cameras:

```csharp
using UnityEngine;

public class DepthCameraNoise : MonoBehaviour
{
    [Header("Noise Configuration")]
    public float depthNoiseStd = 0.01f;      // Depth noise standard deviation
    public float spatialNoiseStd = 0.005f;   // Spatial noise standard deviation
    public float biasError = 0.001f;         // Systematic bias error

    [Header("Distortion Configuration")]
    public float radialDistortionK1 = -0.1f;
    public float radialDistortionK2 = 0.02f;
    public float tangentialDistortionP1 = 0.0f;
    public float tangentialDistortionP2 = 0.0f;

    public float[] ApplyNoiseAndDistortion(float[] depthData, int width, int height)
    {
        float[] noisyData = new float[depthData.Length];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = y * width + x;
                float originalDepth = depthData[index];

                if (originalDepth > 0 && originalDepth < float.PositiveInfinity)
                {
                    // Apply systematic bias
                    float biasedDepth = originalDepth + biasError;

                    // Apply random noise
                    float noise = RandomGaussian() * depthNoiseStd;
                    float noisyDepth = biasedDepth + noise;

                    // Apply distance-dependent noise (closer objects have more noise)
                    float distanceFactor = 1.0f + (noisyDepth - 1.0f) * 0.1f; // 10% increase per meter
                    noisyDepth += RandomGaussian() * depthNoiseStd * distanceFactor;

                    noisyData[index] = Mathf.Max(0f, noisyDepth);
                }
                else
                {
                    noisyData[index] = originalDepth;
                }
            }
        }

        return ApplyDistortion(noisyData, width, height);
    }

    float[] ApplyDistortion(float[] depthData, int width, int height)
    {
        // For depth cameras, distortion primarily affects the mapping from pixel to 3D coordinates
        // This is typically handled in the point cloud generation stage
        return depthData; // In this simplified version, we return the same data
    }

    float RandomGaussian()
    {
        // Box-Muller transform for Gaussian random numbers
        float u1 = Random.value;
        float u2 = Random.value;
        return Mathf.Sqrt(-2f * Mathf.Log(u1)) * Mathf.Cos(2f * Mathf.PI * u2);
    }

    public void ApplyNoiseToTexture(Texture2D depthTexture)
    {
        Color[] pixels = depthTexture.GetPixels();
        for (int i = 0; i < pixels.Length; i++)
        {
            float depth = pixels[i].r;
            if (depth > 0 && depth < 1.0f) // Assuming normalized depth
            {
                float noise = RandomGaussian() * depthNoiseStd;
                float noisyDepth = Mathf.Clamp(depth + noise, 0f, 1f);
                pixels[i] = new Color(noisyDepth, noisyDepth, noisyDepth, 1f);
            }
        }
        depthTexture.SetPixels(pixels);
        depthTexture.Apply();
    }
}
```

#### Multi-Camera Depth Systems

Simulating stereo vision and multi-camera depth systems:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class StereoDepthCamera : MonoBehaviour
{
    [Header("Stereo Configuration")]
    public Camera leftCamera;
    public Camera rightCamera;
    public float baseline = 0.1f;  // Distance between cameras in meters
    public float focalLength = 525f;  // Focal length in pixels
    public int disparityRange = 64;   // Maximum disparity to search

    [Header("Output Configuration")]
    public string disparityTopic = "/stereo/disparity";
    public string pointCloudTopic = "/stereo/points";

    private Texture2D leftImage;
    private Texture2D rightImage;
    private float[,] disparityMap;

    void Start()
    {
        InitializeStereoCameras();
    }

    void InitializeStereoCameras()
    {
        // Set up left camera
        if (leftCamera == null)
        {
            leftCamera = transform.Find("LeftCamera")?.GetComponent<Camera>();
            if (leftCamera == null)
            {
                GameObject leftCamObj = new GameObject("LeftCamera");
                leftCamObj.transform.SetParent(transform);
                leftCamObj.transform.localPosition = new Vector3(-baseline / 2, 0, 0);
                leftCamera = leftCamObj.AddComponent<Camera>();
            }
        }

        // Set up right camera
        if (rightCamera == null)
        {
            rightCamera = transform.Find("RightCamera")?.GetComponent<Camera>();
            if (rightCamera == null)
            {
                GameObject rightCamObj = new GameObject("RightCamera");
                rightCamObj.transform.SetParent(transform);
                rightCamObj.transform.localPosition = new Vector3(baseline / 2, 0, 0);
                rightCamera = rightCamObj.AddComponent<Camera>();
            }
        }

        // Initialize image textures
        int width = leftCamera.pixelWidth;
        int height = leftCamera.pixelHeight;
        leftImage = new Texture2D(width, height);
        rightImage = new Texture2D(width, height);
        disparityMap = new float[height, width];
    }

    public float[,] ComputeDisparityMap(Texture2D leftImg, Texture2D rightImg)
    {
        int width = leftImg.width;
        int height = leftImg.height;
        float[,] dispMap = new float[height, width];

        // Simple block matching algorithm for disparity computation
        int blockSize = 15; // Size of matching window
        int halfBlock = blockSize / 2;

        for (int y = halfBlock; y < height - halfBlock; y++)
        {
            for (int x = halfBlock; x < width - halfBlock; x++)
            {
                float bestMatch = float.MaxValue;
                int bestDisp = 0;

                // Search along the epipolar line (same y-coordinate)
                for (int d = 0; d < Mathf.Min(disparityRange, x); d++)
                {
                    if (x - d >= halfBlock)
                    {
                        float ssd = ComputeBlockSsd(leftImg, rightImg, x, y, d, blockSize);
                        if (ssd < bestMatch)
                        {
                            bestMatch = ssd;
                            bestDisp = d;
                        }
                    }
                }

                dispMap[y, x] = bestDisp;
            }
        }

        return dispMap;
    }

    float ComputeBlockSsd(Texture2D leftImg, Texture2D rightImg, int x, int y, int disparity, int blockSize)
    {
        int halfBlock = blockSize / 2;
        float ssd = 0f;

        for (int dy = -halfBlock; dy <= halfBlock; dy++)
        {
            for (int dx = -halfBlock; dx <= halfBlock; dx++)
            {
                Color leftColor = leftImg.GetPixel(x + dx, y + dy);
                Color rightColor = rightImg.GetPixel(x + dx - disparity, y + dy);

                float diffR = leftColor.r - rightColor.r;
                float diffG = leftColor.g - rightColor.g;
                float diffB = leftColor.b - rightColor.b;

                ssd += diffR * diffR + diffG * diffG + diffB * diffB;
            }
        }

        return ssd;
    }

    public float[,] ConvertDisparityToDepth(float[,] disparityMap)
    {
        int height = disparityMap.GetLength(0);
        int width = disparityMap.GetLength(1);
        float[,] depthMap = new float[height, width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float disparity = disparityMap[y, x];
                if (disparity > 0)
                {
                    // Depth = (Baseline * Focal_Length) / Disparity
                    depthMap[y, x] = (baseline * focalLength) / disparity;
                }
                else
                {
                    depthMap[y, x] = float.PositiveInfinity; // Invalid depth
                }
            }
        }

        return depthMap;
    }
}
```

## Depth Camera Applications

### 3D Reconstruction

Using depth camera data for 3D reconstruction:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class DepthTo3DReconstruction : MonoBehaviour
{
    [Header("Reconstruction Configuration")]
    public float resolution = 0.01f;  // Size of each voxel
    public float maxDistance = 5f;    // Maximum reconstruction distance
    public Material pointMaterial;

    private Dictionary<Vector3Int, float> voxelGrid = new Dictionary<Vector3Int, float>();
    private List<Vector3> pointCloud = new List<Vector3>();

    public void ProcessDepthFrame(float[,] depthMap, int width, int height, Matrix4x4 cameraMatrix)
    {
        voxelGrid.Clear();
        pointCloud.Clear();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float depth = depthMap[y, x];

                if (depth > 0 && depth < maxDistance)
                {
                    // Convert pixel coordinates to 3D world coordinates
                    Vector3 worldPoint = PixelTo3D(x, y, depth, cameraMatrix);

                    // Add to point cloud
                    pointCloud.Add(worldPoint);

                    // Add to voxel grid
                    Vector3Int voxelCoord = WorldToVoxel(worldPoint);
                    if (voxelGrid.ContainsKey(voxelCoord))
                    {
                        voxelGrid[voxelCoord] = (voxelGrid[voxelCoord] + depth) / 2f; // Average
                    }
                    else
                    {
                        voxelGrid[voxelCoord] = depth;
                    }
                }
            }
        }
    }

    Vector3 PixelTo3D(int x, int y, float depth, Matrix4x4 cameraMatrix)
    {
        // Convert pixel coordinates to normalized coordinates
        Vector3 normalized = new Vector3(
            (x - cameraMatrix.m02) / cameraMatrix.m00,  // (x - cx) / fx
            (y - cameraMatrix.m12) / cameraMatrix.m11,  // (y - cy) / fy
            1f
        );

        // Scale by depth to get 3D coordinates in camera space
        Vector3 cameraSpace = normalized * depth;

        // Transform to world space
        Vector3 worldPoint = transform.TransformPoint(cameraSpace);

        return worldPoint;
    }

    Vector3Int WorldToVoxel(Vector3 worldPos)
    {
        return new Vector3Int(
            Mathf.RoundToInt(worldPos.x / resolution),
            Mathf.RoundToInt(worldPos.y / resolution),
            Mathf.RoundToInt(worldPos.z / resolution)
        );
    }

    public GameObject GeneratePointVisualization()
    {
        GameObject pointCloudObj = new GameObject("PointCloud");

        foreach (Vector3 point in pointCloud)
        {
            GameObject pointObj = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            pointObj.transform.SetParent(pointCloudObj.transform);
            pointObj.transform.position = point;
            pointObj.transform.localScale = Vector3.one * resolution * 0.5f;
            pointObj.GetComponent<Renderer>().material = pointMaterial;
            DestroyImmediate(pointObj.GetComponent<Collider>()); // Remove collider for performance
        }

        return pointCloudObj;
    }

    public GameObject GenerateMeshFromVoxels()
    {
        // Create mesh from voxel grid using marching cubes or similar algorithm
        // This is a simplified approach - full implementation would be more complex
        GameObject meshObj = new GameObject("ReconstructedMesh");

        // For now, just create a simple representation
        foreach (var kvp in voxelGrid)
        {
            Vector3 worldPos = new Vector3(
                kvp.Key.x * resolution,
                kvp.Key.y * resolution,
                kvp.Key.z * resolution
            );

            GameObject voxel = GameObject.CreatePrimitive(PrimitiveType.Cube);
            voxel.transform.SetParent(meshObj.transform);
            voxel.transform.position = worldPos;
            voxel.transform.localScale = Vector3.one * resolution * 0.9f;
            voxel.GetComponent<Renderer>().material = pointMaterial;
        }

        return meshObj;
    }
}
```

### Object Detection and Recognition

Using depth data for object detection:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class DepthObjectDetector : MonoBehaviour
{
    [Header("Detection Configuration")]
    public float minObjectSize = 0.1f;    // Minimum size to consider as object
    public float maxObjectSize = 2.0f;    // Maximum size to consider as object
    public float distanceThreshold = 0.02f; // Threshold for grouping points

    public class DetectedObject
    {
        public List<Vector3> points;
        public Vector3 center;
        public float size;
        public string type;

        public DetectedObject()
        {
            points = new List<Vector3>();
        }
    }

    public List<DetectedObject> DetectObjects(float[,] depthMap, int width, int height, Matrix4x4 cameraMatrix)
    {
        List<DetectedObject> objects = new List<DetectedObject>();
        List<Vector3> allPoints = new List<Vector3>();

        // Extract all valid 3D points from depth map
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                float depth = depthMap[y, x];
                if (depth > 0 && depth < float.PositiveInfinity)
                {
                    Vector3 point = PixelTo3D(x, y, depth, cameraMatrix);
                    allPoints.Add(point);
                }
            }
        }

        // Cluster points into objects using a simple clustering algorithm
        List<List<Vector3>> clusters = ClusterPoints(allPoints);

        // Filter clusters based on size and create detected objects
        foreach (var cluster in clusters)
        {
            if (cluster.Count > 10) // Minimum number of points
            {
                float size = CalculateClusterSize(cluster);
                if (size >= minObjectSize && size <= maxObjectSize)
                {
                    DetectedObject obj = new DetectedObject();
                    obj.points = cluster;
                    obj.center = CalculateCenter(cluster);
                    obj.size = size;
                    obj.type = ClassifyObject(cluster, size);

                    objects.Add(obj);
                }
            }
        }

        return objects;
    }

    List<List<Vector3>> ClusterPoints(List<Vector3> points)
    {
        List<List<Vector3>> clusters = new List<List<Vector3>>();
        List<bool> assigned = new List<bool>(new bool[points.Count]);

        for (int i = 0; i < points.Count; i++)
        {
            if (!assigned[i])
            {
                List<Vector3> cluster = new List<Vector3>();
                Queue<int> queue = new Queue<int>();
                queue.Enqueue(i);

                while (queue.Count > 0)
                {
                    int currentIndex = queue.Dequeue();
                    if (assigned[currentIndex]) continue;

                    assigned[currentIndex] = true;
                    cluster.Add(points[currentIndex]);

                    // Find nearby points
                    for (int j = 0; j < points.Count; j++)
                    {
                        if (!assigned[j] && Vector3.Distance(points[currentIndex], points[j]) < distanceThreshold)
                        {
                            queue.Enqueue(j);
                        }
                    }
                }

                if (cluster.Count > 0)
                {
                    clusters.Add(cluster);
                }
            }
        }

        return clusters;
    }

    float CalculateClusterSize(List<Vector3> cluster)
    {
        if (cluster.Count == 0) return 0f;

        Vector3 min = cluster[0], max = cluster[0];
        foreach (Vector3 point in cluster)
        {
            min = Vector3.Min(min, point);
            max = Vector3.Max(max, point);
        }

        Vector3 size = max - min;
        return size.magnitude;
    }

    Vector3 CalculateCenter(List<Vector3> cluster)
    {
        Vector3 sum = Vector3.zero;
        foreach (Vector3 point in cluster)
        {
            sum += point;
        }
        return sum / cluster.Count;
    }

    string ClassifyObject(List<Vector3> cluster, float size)
    {
        // Simple classification based on size and shape
        if (size < 0.2f)
        {
            return "small_object";
        }
        else if (size < 0.5f)
        {
            return "medium_object";
        }
        else
        {
            return "large_object";
        }
    }

    Vector3 PixelTo3D(int x, int y, float depth, Matrix4x4 cameraMatrix)
    {
        Vector3 normalized = new Vector3(
            (x - cameraMatrix.m02) / cameraMatrix.m00,
            (y - cameraMatrix.m12) / cameraMatrix.m11,
            1f
        );

        Vector3 cameraSpace = normalized * depth;
        return transform.TransformPoint(cameraSpace);
    }
}
```

## Performance Optimization

### Efficient Depth Processing

Optimizing depth camera simulation performance:

```csharp
using UnityEngine;
using System.Threading.Tasks;

public class OptimizedDepthCamera : MonoBehaviour
{
    [Header("Performance Configuration")]
    public int resolutionReduction = 2;  // Process every Nth pixel
    public bool useComputeShader = true;  // Use compute shader for processing
    public int updateRate = 30;          // Processing frame rate

    private float lastProcessTime;
    private ComputeShader depthComputeShader;
    private int kernelHandle;

    void Start()
    {
        if (useComputeShader)
        {
            InitializeComputeShader();
        }
    }

    void Update()
    {
        if (Time.time - lastProcessTime >= 1f / updateRate)
        {
            ProcessDepthData();
            lastProcessTime = Time.time;
        }
    }

    void InitializeComputeShader()
    {
        // Load compute shader for depth processing
        // This would typically be a custom compute shader
        depthComputeShader = Resources.Load<ComputeShader>("DepthProcessing");
        if (depthComputeShader != null)
        {
            kernelHandle = depthComputeShader.FindKernel("CSMain");
        }
    }

    void ProcessDepthData()
    {
        if (useComputeShader && depthComputeShader != null)
        {
            // Use compute shader for parallel processing
            ProcessWithComputeShader();
        }
        else
        {
            // Use CPU processing
            ProcessWithCPU();
        }
    }

    void ProcessWithComputeShader()
    {
        // Set up compute shader parameters
        depthComputeShader.SetTexture(kernelHandle, "DepthTexture", GetDepthTexture());
        depthComputeShader.SetFloat("Time", Time.time);
        depthComputeShader.SetFloat("NoiseStd", 0.01f);

        // Dispatch compute shader
        int threadGroupsX = Mathf.CeilToInt(Screen.width / 8.0f);
        int threadGroupsY = Mathf.CeilToInt(Screen.height / 8.0f);
        depthComputeShader.Dispatch(kernelHandle, threadGroupsX, threadGroupsY, 1);
    }

    void ProcessWithCPU()
    {
        // Process depth data on CPU with optimizations
        // Use parallel processing where possible
        ProcessDepthInParallel();
    }

    void ProcessDepthInParallel()
    {
        // Use Unity's job system or Task for parallel processing
        int width = 640;
        int height = 480;
        int step = resolutionReduction;

        // Process image in chunks
        int chunkSize = height / 4; // Process in 4 chunks
        for (int chunk = 0; chunk < 4; chunk++)
        {
            int startY = chunk * chunkSize;
            int endY = (chunk == 3) ? height : (chunk + 1) * chunkSize;

            // Process this chunk
            ProcessDepthChunk(startY, endY, step);
        }
    }

    void ProcessDepthChunk(int startY, int endY, int step)
    {
        // Process a chunk of the depth image
        for (int y = startY; y < endY; y += step)
        {
            for (int x = 0; x < 640; x += step)
            {
                // Process depth pixel at (x, y)
                ProcessDepthPixel(x, y);
            }
        }
    }

    void ProcessDepthPixel(int x, int y)
    {
        // Process individual depth pixel
        // This could include noise addition, filtering, etc.
    }

    Texture2D GetDepthTexture()
    {
        // Return the current depth texture
        return null; // Placeholder
    }
}
```

Depth camera simulation is crucial for robotics applications that require both visual and spatial understanding. Proper simulation of depth cameras enables realistic testing of 3D perception algorithms, object detection systems, and spatial reasoning capabilities in digital twin environments.