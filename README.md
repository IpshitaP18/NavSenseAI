# 🚀 NavSense AI
### AI-Powered Navigation Assistance System

NavSense AI is an AI-powered navigation assistance prototype that uses **Computer Vision and Artificial Intelligence** to detect obstacles, estimate their relative distance, and provide real-time navigation instructions through voice feedback.

The system combines YOLOv8 object detection, MiDaS depth estimation, and intelligent navigation logic to help users understand their surroundings.

---

## ✨ Key Features

- **Real-Time Object Detection:** Detects surrounding objects using YOLOv8.
- **Depth Estimation:** Estimates relative object depth using MiDaS.
- **Distance Classification:** Classifies obstacles into Near, Medium, and Far categories.
- **Obstacle Prioritization:** Identifies obstacles based on their proximity and position.
- **Smart Navigation Guidance:** Generates instructions such as Stop, Slow Down, Keep Left, and Keep Right.
- **Voice Assistance:** Provides audio instructions using text-to-speech.
- **Interactive Dashboard:** Displays live detection, depth maps, and navigation guidance through Streamlit.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| OpenCV | Image processing and webcam access |
| YOLOv8 | Object detection |
| MiDaS | Depth estimation |
| PyTorch | Deep learning framework |
| NumPy | Numerical computation |
| Streamlit | Interactive user interface |
| pyttsx3 | Voice assistance |

---

## 📂 Project Structure

```text
NavSenseAI/
│
├── app.py                 # Standalone webcam detection
├── streamlit_app.py       # Main Streamlit application
│
├── detection.py           # YOLOv8 object detection
├── direction.py           # Object direction detection
├── depth.py               # MiDaS depth estimation
├── distance.py            # Distance classification
├── priority.py            # Obstacle risk prioritization
├── navigation.py          # Navigation decision logic
├── voice.py               # Voice guidance system
│
├── requirements.txt       # Project dependencies
├── yolov8n.pt             # YOLOv8 model weights
└── README.md
```

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/IpshitaP18/NavSenseAI.git
```

### 2. Navigate to the Project Directory

```bash
cd NavSenseAI
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
streamlit run streamlit_app.py
```

Open the local URL displayed in your terminal to access the application.

---

## 🔄 System Workflow

1. **Camera Input:** Captures real-time video using OpenCV.
2. **Object Detection:** YOLOv8 detects objects in the video frames.
3. **Depth Estimation:** MiDaS generates a relative depth map.
4. **Distance Classification:** Estimates obstacle proximity using depth information.
5. **Obstacle Prioritization:** Ranks detected obstacles according to their potential risk.
6. **Navigation Decision:** Determines suitable navigation instructions.
7. **Voice Feedback:** Converts navigation instructions into speech.
8. **Dashboard Display:** Displays camera output, depth maps, and navigation information.

---

## 🧭 Navigation Instructions

| Detected Situation | Guidance |
|---|---|
| Immediate obstacle detected | Stop |
| Obstacle ahead | Slow Down |
| Obstacle on the left | Keep Right |
| Obstacle on the right | Keep Left |
| No immediate obstacle detected | Path Clear |

*Navigation decisions depend on the detected objects, their relative depth, and their positions in the camera frame.*

---

## 🔮 Future Enhancements

- Develop a modern React-based web interface.
- Integrate a FastAPI backend.
- Implement WebSocket communication for real-time updates.
- Improve object tracking and distance estimation.
- Optimize model inference for better performance.
- Explore future Android and iOS application support.

---

## ⚠️ Limitations

- MiDaS provides relative depth estimation, not exact physical distance measurements.
- Detection accuracy depends on lighting, camera quality, and environmental conditions.
- Navigation guidance is limited by the camera's field of view.
- This project is a prototype and is not intended to replace established mobility aids such as white canes or guide dogs.

---

## 👨‍💻 Project Information

**Project Name:** NavSense AI  
**Domain:** Artificial Intelligence and Computer Vision  
**Primary Language:** Python  
**Interface:** Streamlit  

**GitHub Repository:**  
[https://github.com/IpshitaP18/NavSenseAI](https://github.com/IpshitaP18/NavSenseAI)

---

### 💡 Making Navigation Smarter Through AI and Computer Vision.