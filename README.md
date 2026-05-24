# Real-Time Indian Sign Language Translation

## 📌 Overview

Real-Time Indian Sign Language Translation is an AI-powered web application developed to bridge the communication gap between Deaf/Hard-of-Hearing (DHH) individuals and hearing people.

This project provides a **bidirectional communication system**:

* **Sign-to-Text Translation** → Converts Indian Sign Language (ISL) gestures into readable text in real time.
* **Text/Voice-to-Sign Translation** → Converts spoken or written English into ISL animations.

Unlike traditional systems that require sensor gloves or expensive hardware, this solution uses only a standard webcam and AI-based computer vision techniques.

---

## 🎯 Objectives

* Enable seamless communication between deaf and hearing individuals
* Recognize ISL gestures accurately in real time
* Eliminate the need for sensor-based gloves
* Provide two-way communication support
* Develop a cost-effective and accessible solution
* Improve inclusivity using AI and assistive technology

---

## 🚀 Features

✅ Real-time ISL gesture recognition
✅ Bidirectional translation system
✅ Webcam-based hand tracking
✅ Speech-to-sign conversion
✅ Text-to-speech support
✅ Hybrid avatar engine
✅ Translation history storage
✅ Dark mode & accessibility controls
✅ Browser-based user interface
✅ Works without specialized hardware

---

## 🧠 Technologies Used

### Programming & Frameworks

* Python 3.10
* Flask
* HTML5
* CSS3
* JavaScript

### AI & Computer Vision

* TensorFlow / Keras
* OpenCV
* Google MediaPipe
* NumPy

### Database

* SQLite

### Tools

* Visual Studio Code
* Google Chrome

---

## 🏗️ System Architecture

The system follows an MVC-based Client–Server Architecture.

### Modules

### 1️⃣ Sign-to-Text Module

* Captures webcam video
* Detects 21 hand landmarks using MediaPipe
* Normalizes coordinates
* Predicts ISL gestures using trained ML model
* Displays translated text output

### 2️⃣ Text-to-Sign Module

* Accepts text or voice input
* Converts text into ISL animations
* Uses word-level sign videos
* Falls back to finger-spelling if word is unavailable

### 3️⃣ Data Management Module

* Stores translation history using SQLite
* Maintains timestamps and logs

### 4️⃣ Web Interface & Controller

* Flask backend handling APIs and routing
* Real-time frontend updates using JavaScript Fetch API

---

## ⚙️ Methodology

1. Capture real-time webcam frames
2. Extract hand landmarks using MediaPipe
3. Normalize landmark coordinates
4. Predict gesture using trained Keras model
5. Convert predictions into text
6. Translate text/voice into ISL animations
7. Display output in real time

---

## 📊 Model Details

* Gesture Classes: **37 ISL Gestures**
* Hand Tracking: **21 Landmarks**
* Framework: **TensorFlow/Keras**
* Real-Time Inference with CPU Support
* Approximate Accuracy: **96.5%**

---

## 💻 Installation

### Clone Repository

```bash
git clone https://github.com/ChethanGS18/RealTimeSignLanguageTranslation-_-ISL-.git
cd RealTimeSignLanguageTranslation-_-ISL-
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python app.py
```

or

```bash
flask run
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## 📂 Project Structure

```text
RealTimeSignLanguageTranslation/
│
├── static/
│   ├── videos/
│   ├── images/
│   ├── css/
│   └── js/
│
├── templates/
│
├── model/
│
├── database/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📈 Results

* Real-time gesture prediction
* Smooth ISL avatar generation
* Fast and accurate translation
* User-friendly interface
* Reliable two-way communication

---

## 🔮 Future Enhancements

* Mobile application deployment
* 3D animated avatar support
* Regional language support
* Continuous learning for new signs
* NLP-based grammar correction
* Cloud integration

---

## 📚 Research References

1. Real-Time Sign Language Translation Using Deep CNN–RNN (IEEE, 2024)
2. Sign2Speech: End-to-End Sign Language to Speech Translation Using Transformers (ACM, 2024)
3. Multimodal Fusion for Robust Sign Language Interpretation (Springer, 2024)
4. Graph Neural Networks for Sign Gesture Dynamics (IEEE, 2025)
5. Incremental Learning for Adaptive Sign Language Translation (ACM, 2024)

---

## 👨‍💻 Author
* Chethan G S

---

## 📜 License

This project is developed for educational and research purposes only.

---

## 🤝 Contribution

Contributions, suggestions, and improvements are welcome.

Feel free to fork this repository and submit pull requests.

---


