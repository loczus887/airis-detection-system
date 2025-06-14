# AIRIS - Advanced Intelligent Recognition & Interception System

[![Organization](https://img.shields.io/badge/Organization-MyAI%20Research%20Group-blue?logo=github)](https://github.com/SKNUM-MyAI)

A sophisticated security monitoring solution utilizing artificial intelligence for real-time threat detection and facial recognition, featuring a custom-built security glasses prototype.

## Overview

AIRIS is an intelligent security monitoring system that combines computer vision, machine learning, and wearable technology to provide real-time threat detection and surveillance capabilities. The project includes a **custom-built security glasses prototype** integrating ESP32-CAM modules with wearable technology for personal security monitoring. The system can operate with various camera sources, including our prototype ESP32-CAM glasses, as well as standard USB cameras and IP cameras.

## Features

### Core Functionality
- **Real-time Face Detection and Tracking**: Advanced face detection using dlib and OpenCV with temporal tracking capabilities
- **Dangerous Object Recognition**: YOLO-based object detection optimized for identifying potentially dangerous items
- **Multi-camera Support**: Compatible with ESP32-CAM, USB cameras, and IP cameras
- **Alert System**: Hierarchical threat classification with temporal event aggregation
- **Cross-platform Interface**: Built with Kivy for multi-platform compatibility

### Advanced Capabilities
- **Temporal Face Analysis**: Monitors exposure time and generates alerts for prolonged observation
- **Adaptive Confidence Thresholds**: Dynamic confidence adjustment for different object classes
- **Background Processing**: Asynchronous detection processing for real-time performance
- **Network Camera Discovery**: Automatic ESP32-CAM device scanning on local network
- **Debug Mode**: Comprehensive debugging features for development and testing

## System Architecture

The system follows a three-layer architecture:

1. **Data Acquisition Layer**: Camera modules (ESP32-CAM, USB, IP cameras)
2. **Processing Layer**: AI-powered detection algorithms (face recognition, object detection)
3. **Presentation Layer**: Cross-platform user interface with real-time monitoring

## Installation

### Prerequisites

#### Python Dependencies
```bash
pip install kivy opencv-python numpy scipy requests
```

#### Optional Dependencies (for enhanced functionality)
```bash
pip install dlib  # For advanced face recognition
```

#### Computer Vision Models
The system requires YOLO model files for object detection. Download and place in the project root:

- `yolov4-tiny.weights` and `yolov4-tiny.cfg` (recommended for performance)
- `yolov3-tiny.weights` and `yolov3-tiny.cfg`
- `yolov4.weights` and `yolov4.cfg`
- `yolov3.weights` and `yolov3.cfg`
- `coco.names` (class labels file)

For enhanced face recognition (optional):
- `shape_predictor_68_face_landmarks.dat`
- `dlib_face_recognition_resnet_model_v1.dat`

### Hardware Setup

#### ESP32-CAM Integration & Security Glasses Prototype
The system supports ESP32-CAM modules for wearable applications, including our **custom-built security glasses prototype**. The prototype integrates miniature ESP32-CAM modules mounted on wearable glasses frames, enabling hands-free personal security monitoring. While the ESP32-CAM sketch is not included in this repository, the system can connect to any ESP32-CAM device configured to serve video streams over HTTP.

**Supported ESP32-CAM endpoints:**
- Stream endpoints: `/stream`, `/video`, `/mjpeg`, `/`
- Capture endpoints: `/capture`, `/still`, `/jpg`, `/photo`

#### Alternative Camera Sources
- USB cameras (index 0, 1, 2, etc.)
- IP cameras with HTTP streams
- Standard webcams

## Usage

### Starting the Application
```bash
python main.py
```

### Basic Operations

1. **Camera Management**
   - Click "Start Camera" to begin monitoring
   - Use "Toggle Camera" to switch between available cameras
   - Click "Connect ESP32-CAM" to connect to network cameras

2. **Network Discovery**
   - Use "Scan Network" to automatically discover ESP32-CAM devices
   - Configure IP addresses in Settings

3. **Configuration**
   - Access Settings to configure detection parameters
   - Adjust confidence thresholds and time limits
   - Set ESP32-CAM connection parameters

4. **Monitoring**
   - View real-time camera feed with detection overlays
   - Monitor alerts in the dedicated alerts panel
   - Toggle debug mode for detailed detection information

### Configuration Options

#### Face Detection Settings
- **Time Limit**: Duration threshold for face exposure alerts (default: 10 seconds)
- **Face Threshold**: Confidence threshold for face recognition (default: 0.6)

#### Object Detection Settings
- **Object Confidence**: General confidence threshold for object detection (default: 0.5)
- **Enable/Disable**: Toggle face detection and object detection independently

#### Network Settings
- **ESP32-CAM IP**: IP address of ESP32-CAM device (default: 192.168.0.249)
- **ESP32-CAM Port**: HTTP port for camera communication (default: 80)

## Technical Details

### Face Detection Algorithm
- **Two-stage approach**: Haar classifiers for initial detection, dlib models for detailed analysis
- **128-dimensional descriptors**: Advanced face identification and tracking
- **Temporal persistence**: Compensation for brief detection failures
- **Exposure time monitoring**: Alert generation for prolonged observation

### Object Detection System
- **Modified YOLO architecture**: Optimized for dangerous object identification
- **Dynamic confidence thresholds**: Adaptive sensitivity per object class
- **Spatial tracking**: Position-based object persistence between frames
- **Temporal aggregation**: False positive reduction through frame-to-frame analysis

### Performance Optimization
- **Asynchronous processing**: Separate threads for detection and UI
- **Adaptive frame rate**: Dynamic processing frequency adjustment
- **Resource management**: Graceful degradation under limited resources
- **Memory optimization**: Efficient frame buffering and processing

## Supported Object Classes

The system can detect various potentially dangerous objects including:
- Knives and sharp objects
- Firearms and weapons
- Tools (hammers, axes)
- Bottles and containers
- Electronic devices

Confidence thresholds are individually tuned for each object class to optimize detection accuracy.

## Development

### Project Structure
```
airis/
├── main.py                    # Main application entry point
├── detection/
│   ├── face_detector.py       # Face detection and tracking
│   ├── object_detector.py     # Object detection system
│   └── __init__.py
├── utils/
│   ├── camera_manager.py      # Camera handling and management
│   └── __init__.py
└── README.md
```

### Extending the System

#### Adding New Camera Sources
Extend the `CameraManager` class to support additional camera types or protocols.

#### Custom Object Detection
Modify the `dangerous_objects` dictionary in `object_detector.py` to add new object classes or adjust confidence thresholds.

#### UI Customization
The Kivy-based interface can be extended with additional screens and functionality.

## Performance Metrics

Based on experimental testing:
- **Object Detection Accuracy**: 87% at 0.5 confidence threshold
- **Processing Frequency**: 5-10 frames per second
- **Alert Generation Accuracy**: 91% for temporal exposure detection
- **WiFi Transmission Latency**: 120-180ms on standard 802.11n networks

## Hardware Requirements

### Minimum Requirements
- **CPU**: Dual-core processor (Intel i3 or equivalent)
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional CUDA-compatible GPU for enhanced performance
- **Camera**: USB camera, webcam, or network camera
- **Network**: WiFi or Ethernet for ESP32-CAM connectivity

### Recommended Setup
- **CPU**: Quad-core processor with good single-thread performance
- **RAM**: 8GB or more
- **GPU**: NVIDIA GPU with CUDA support for accelerated inference
- **Storage**: SSD for faster model loading


## License

This project is developed as part of academic research at the [Machine Learning Student Research Group MyAI](https://github.com/SKNUM-MyAI), Faculty of Computer Science and Telecommunications.

## Research Team

**Authors**: [@Alicja-Kowalska](https://github.com/loczus887), [@Wiktoria-Kosek](https://github.com/wiktoriakk), 
**Supervisor**: Dr. Ilona Urbaniak
**Organization**: [Machine Learning Student Research Group MyAI](https://github.com/SKNUM-MyAI)  
**Institution**: Faculty of Computer Science and Telecommunications


## Future Development

Planned enhancements include:
- Augmented Reality (AR) system integration
- Incremental learning techniques for environment-specific adaptation
- Enhanced mobile platform optimization
- Integration with cloud-based analytics services
- Advanced behavioral analysis algorithms