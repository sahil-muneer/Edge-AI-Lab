# 🚀 Edge AI Lab: Full-Stack Machine Learning Simulator

An enterprise-grade, full-stack web application designed to simulate, process, and map real-time hardware telemetry into machine learning models. Built to handle edge-device data streams, visualize complex mathematical boundaries, and ensure pipeline stability through active data sanitation.

## 🧠 Core Architecture
This project bridges the gap between raw hardware sensor data and actionable data science:
* **Frontend UI (React/Vite):** A high-performance interactive canvas for real-time data plotting and batch processing visualization.
* **Math/ML Engine (Python/FastAPI):** A dedicated backend server crunching matrix mathematics and predictive algorithms using `NumPy` and `Scikit-Learn`.
* **Hardware Telemetry Simulation:** Engineered to mimic data streams from microcontrollers like the ESP32-CAM, preparing models for real-world physical deployment.

## 🔥 Key Engineering Features

### 1. Active Outlier Detection Guardrails
Physical sensors can malfunction and send voltage spikes that corrupt ML models. This pipeline features a hardcoded safety net that actively intercepts coordinates outside safe physical bounds (e.g., catching impossible X: 5000, Y: 6666 values) and drops the bad data packet before it touches the math engine.

### 2. Dynamic Math Fallback Engine
A self-healing architecture. If the primary Python machine learning server goes offline, the React frontend automatically engages a local JavaScript physics fallback, ensuring the user interface and basic calculations continue running without crashing.

### 3. Real-Time Algorithm Processing
* **K-Nearest Neighbors (KNN):** Real-time classification boundary mapping across the UI canvas.
* **Linear & Polynomial Regression:** Dynamic curve fitting and trend-line calculation based on live coordinate injections.

## 🛠️ Tech Stack
* **Frontend:** React, JavaScript, Tailwind CSS
* **Backend:** Python, FastAPI, Uvicorn
* **Data Science:** Scikit-Learn, NumPy

## 💻 Local Setup & Execution

### 1. Start the Frontend
```bash
npm install
npm run dev

### 2. Start the Python ML Server
Open a second terminal and initialize the backend engine:

```bash
cd backend
pip install fastapi uvicorn numpy scikit-learn
python -m uvicorn main:app --reload