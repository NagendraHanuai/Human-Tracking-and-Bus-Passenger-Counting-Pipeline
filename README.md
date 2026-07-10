# Human Tracking and Bus Passenger Counting Pipeline

A real-time **Human Tracking and Bus Passenger Counting** system built using **YOLO**, **BoT-SORT**, and **OpenCV**. This project detects people, assigns persistent IDs, counts passengers entering and exiting the bus, estimates occupancy, and saves both an annotated video and detailed JSON results.

---

## Features

* Real-time person detection using YOLO
* Persistent person tracking using BoT-SORT
* Passenger Entry (IN) counting
* Passenger Exit (OUT) counting
* Real-time Occupancy estimation
* Annotated output video
* Frame-wise JSON output
* GPU acceleration (CUDA supported)
* Performance benchmarking (FPS, latency, RAM)

---

# Repository Structure

```text
Human-Tracking-and-Bus-Passenger-Counting-Pipeline/
│
├── pipeline_1.py
├── pipeline_2.py
├── requirements.txt
├── README.md
```

---

# Pipeline Versions

## Pipeline 1

Basic passenger counting pipeline.

### Features

* YOLO person detection
* BoT-SORT tracking
* Persistent IDs
* Entry and Exit counting
* Occupancy calculation
* Annotated output video
* JSON output

---

## Pipeline 2 (Improved)

Pipeline 2 includes all Pipeline 1 features plus several improvements for more accurate passenger counting.

### Additional Features

* Outside bus window person filtering
* Appearance-based person re-identification
* Improved persistent ID recovery
* Reduced duplicate counting
* Better occupancy estimation
* Ignored detection logging
* Enhanced JSON output

---

# Pipeline Comparison

| Feature                   | Pipeline 1 | Pipeline 2 |
| ------------------------- | ---------- | ---------- |
| Person Detection          | ✅          | ✅          |
| BoT-SORT Tracking         | ✅          | ✅          |
| Persistent Person IDs     | ✅          | ✅          |
| Entry Counting            | ✅          | ✅          |
| Exit Counting             | ✅          | ✅          |
| Occupancy Estimation      | ✅          | ✅          |
| Annotated Video           | ✅          | ✅          |
| JSON Output               | ✅          | ✅          |
| Outside Window Filtering  | ❌          | ✅          |
| Appearance Matching       | ❌          | ✅          |
| Improved ID Recovery      | Basic      | Advanced   |
| Duplicate Count Reduction | Basic      | Improved   |
| Ignored Detection Logging | ❌          | ✅          |

---

# Requirements

* Python 3.10+
* CUDA (Recommended)
* OpenCV
* Ultralytics YOLO
* NumPy
* psutil

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/NagendraHanuai/Human-Tracking-and-Bus-Passenger-Counting-Pipeline.git

cd Human-Tracking-and-Bus-Passenger-Counting-Pipeline
```

Create a virtual environment (Optional)

```bash
python3 -m venv venv
```

Activate the environment

Linux

```bash
source venv/bin/activate
```

Windows

```cmd
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Model

Download your trained YOLO model and update the following variable inside the script.

```python
MODEL_PATH = "yolo26x.pt"
```

---

# Input Video

Update these variables inside the script.

```python
VIDEO_PATH = "input_video.mp4"
OUTPUT_PATH = "output_video.mp4"
JSON_OUTPUT_PATH = "output.json"
```

---

# Running Pipeline 1

```bash
python3 pipeline_1.py
```

---

# Running Pipeline 2

```bash
python3 pipeline_2.py
```

---

# Outputs

After processing, the pipeline generates:

* Annotated output video (.mp4)
* Frame-wise JSON file (.json)

Example

```text
output_video.mp4

output.json
```

---

# JSON Output

Each frame contains

* Frame ID
* Timestamp
* Person IDs
* Bounding Boxes
* Detection Confidence
* Entry Events
* Exit Events
* Occupancy
* Total Unique Persons

Pipeline 2 additionally stores

* Ignored detections
* Outside bus filtering information
* Appearance-based ID matching information

---

# Dashboard

The annotated output video displays

* IN Count
* OUT Count
* Current Occupancy
* Total Passenger Count
* Virtual Entry/Exit Lines
* Person Bounding Boxes
* Human Labels

---

# Performance Report

After processing, the script prints

* Total Frames
* Video Length
* Processing Time
* Average FPS
* Median FPS
* Latency
* RAM Usage
* Unique Person Count
* IN Count
* OUT Count
* Final Occupancy

---

# Applications

* Automatic Passenger Counting
* Smart Bus Analytics
* Public Transportation Monitoring
* Intelligent Transportation Systems (ITS)
* CCTV Video Analytics
* Edge AI Applications

---

# Future Improvements

* Multi-camera support
* RTSP live stream support
* ByteTrack integration
* DeepSORT integration
* TensorRT optimization
* Web dashboard
* REST API
* Passenger heatmap visualization

---

# Author

**Nagendra**

GitHub: https://github.com/NagendraHanuai

---

