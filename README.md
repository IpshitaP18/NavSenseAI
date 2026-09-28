# NavSense AI

### Context-Aware Navigation Assistance Using Artificial Intelligence

NavSense AI is an ongoing AI-based navigation assistance project designed to help visually impaired users understand their surroundings through computer vision, depth estimation, context-aware reasoning, and spoken guidance.

The project is being developed incrementally, starting with real-time object detection and directional awareness and gradually extending toward distance estimation, contextual guidance, voice output, mobile deployment, and wearable integration.

> **Project Status:** 🚧 Under Active Development
> **Current Stage:** Stage 1 — Object Detection & Direction Awareness

---

## 🎯 Project Goal

The goal of NavSense AI is to develop a context-aware navigation assistance system that can analyze a user's surroundings through a camera and provide simple, actionable guidance.

The planned system combines:

* Real-time object detection
* Direction awareness
* Monocular depth estimation
* Context fusion
* Natural-language guidance
* Text-to-speech
* Mobile deployment
* Future wearable/glasses integration

The system is being developed as a research/prototype project and is **not intended to be a certified safety device**.

---

# 🧠 Current Implementation

## Stage 1 — Object Detection & Direction Awareness ✅

The current prototype uses a webcam as the camera input and processes the live video using OpenCV and YOLOv8.

### Current Pipeline

```text
Webcam
   ↓
OpenCV
   ↓
YOLOv8 Object Detection
   ↓
Bounding Boxes
   ↓
Object Direction
   ↓
LEFT / CENTER / RIGHT
```

The current implementation can:

* Capture live webcam frames
* Detect objects using YOLOv8
* Draw bounding boxes around detected objects
* Display object class names
* Display detection confidence
* Determine whether an object is on the LEFT, CENTER, or RIGHT side of the frame

The direction is currently determined by dividing the camera frame into three horizontal regions.

---

# 📂 Project Structure

```text
NavSenseAI/
│
├── app.py
├── detection.py
├── direction.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── .streamlit/
    └── config.toml
```

### `app.py`

Main entry point of the current prototype.

It:

* Opens the webcam
* Captures frames using OpenCV
* Sends frames to the object detector
* Calculates object direction
* Displays the detection results

The current prototype uses `cv2.VideoCapture()` for webcam input and displays the processed frames using OpenCV.

### `detection.py`

Contains the YOLOv8 object detection module.

The `ObjectDetector` class loads the YOLO model and returns detected objects with:

* Object class
* Confidence score
* Bounding-box coordinates

### `direction.py`

Contains the direction-detection logic.

Objects are currently classified into:

```text
LEFT
CENTER
RIGHT
```

based on the horizontal position of their bounding-box center.

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/NavSenseAI.git
```

```bash
cd NavSenseAI
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

## 3. Activate the virtual environment

On Windows:

```powershell
venv\Scripts\activate
```

## 4. Install dependencies

```bash
python -m pip install --upgrade pip
```

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Current Prototype

Run:

```bash
python app.py
```

The webcam will open and detected objects will be displayed with their bounding boxes, class names, confidence values, and directions.

Press:

```text
Q
```

to exit the application.

On the first run, Ultralytics may download the YOLOv8 model weights automatically.

---

# 🖥️ Current Demo Interface

The project may also contain a Streamlit-based demonstration interface for presenting the current prototype.

If retained, it can be launched using:

```bash
python -m streamlit run <streamlit_file>.py
```

The Streamlit interface is currently treated as a prototype/demo layer. The long-term project direction is to move toward a React-based frontend while keeping the AI pipeline in Python.

---

# 🚀 Development Roadmap

NavSense AI is being developed in multiple stages.

### Stage 1 — Object Detection & Direction Awareness ✅

```text
Webcam
   ↓
OpenCV
   ↓
YOLOv8
   ↓
Bounding Boxes
   ↓
LEFT / CENTER / RIGHT
```

**Status:** Completed

---

### Stage 2 — Depth & Distance Awareness 🔜

The next stage will introduce monocular depth estimation using MiDaS.

Planned functionality:

```text
Object Detection
        +
Depth Estimation
        ↓
Distance Awareness
```

This will allow the system to move beyond simply knowing **where** an object is and begin estimating **how close** it is.

---

### Stage 3 — Context Fusion & Voice Guidance 🔜

Object detection and depth information will be combined to generate context-aware navigation instructions.

Planned pipeline:

```text
Object Detection
        +
Depth Estimation
        ↓
Context Fusion
        ↓
Navigation Instruction
        ↓
Text-to-Speech
```

Example guidance could eventually take the form of short actionable instructions such as:

```text
"Obstacle ahead."

"Object on your left."

"Move slightly to the right."
```

The exact guidance logic will be developed and refined during implementation.

---

### Stage 4 — React Interface 🔜

The project will gradually move toward a dedicated frontend rather than relying on Streamlit as the final interface.

Planned architecture:

```text
React Frontend
      ↓
Python AI Backend
      ↓
Computer Vision Pipeline
      ↓
Navigation Guidance
```

The React interface will eventually provide a cleaner interface for monitoring the navigation system and its outputs.

---

### Stage 5 — Mobile Deployment 🔜

The AI pipeline will eventually be adapted for smartphone-based deployment.

Possible architecture:

```text
Phone Camera
     ↓
AI Processing
     ↓
Navigation Guidance
     ↓
Audio Output
```

The smartphone can act as the main processing and communication device.

---

### Stage 6 — Wearable / Smart Glasses Integration 🔮

A future version may integrate a wearable camera or smart glasses.

Possible architecture:

```text
Wearable Camera
      ↓
Phone / Edge Device
      ↓
NavSense AI
      ↓
Audio Guidance
```

This stage will be explored after the core AI pipeline and mobile implementation are stable.

---

# 🏗️ Planned Overall Architecture

The long-term NavSense AI pipeline is planned as:

```text
Camera Input
     ↓
Object Detection
     +
Depth Estimation
     ↓
Context Fusion
     ↓
Navigation Reasoning
     ↓
Natural-Language Instruction
     ↓
Text-to-Speech
     ↓
User
```

The architecture is being implemented incrementally so that each component can be tested independently before being integrated into the complete system.

---

# 🔬 Technology Stack

### Current

* Python
* OpenCV
* YOLOv8
* Ultralytics
* Streamlit (prototype interface)

### Planned

* MiDaS
* Depth Estimation
* Context Fusion
* Natural-Language Generation
* Text-to-Speech
* React
* Python Backend
* Mobile Deployment
* Wearable Camera Integration

---

# 📌 Current Limitations

The current Stage 1 prototype does **not** yet provide:

* Distance estimation
* Depth-aware obstacle analysis
* Context-aware navigation decisions
* Voice guidance
* Mobile deployment
* Wearable integration

These capabilities are part of the planned future stages.

---

# 📈 Project Progress

| Component            | Status      |
| -------------------- | ----------- |
| Webcam Input         | ✅ Completed |
| OpenCV Processing    | ✅ Completed |
| YOLOv8 Detection     | ✅ Completed |
| Bounding Boxes       | ✅ Completed |
| Confidence Display   | ✅ Completed |
| Direction Detection  | ✅ Completed |
| Depth Estimation     | 🔜 Planned  |
| Distance Awareness   | 🔜 Planned  |
| Context Fusion       | 🔜 Planned  |
| Voice Guidance       | 🔜 Planned  |
| React Interface      | 🔜 Planned  |
| Mobile Deployment    | 🔜 Planned  |
| Wearable Integration | 🔮 Future   |

---

# 🤝 Development

NavSense AI is an ongoing project. New modules will be added progressively as development continues.

The current repository represents the **first working stage of the system**, rather than the final version.

Future commits will progressively introduce depth estimation, context-aware reasoning, voice guidance, frontend development, and deployment capabilities.

---

# ⚠️ Disclaimer

NavSense AI is currently a research/prototype project intended for experimentation and development.

It should not be relied upon as a certified assistive or safety-critical navigation device.
