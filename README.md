# 🎓 Student Emotion Recognition System

AI-powered system for recognizing students’ emotional states and engagement levels in online and offline learning environments using facial emotion recognition, deep learning, and computer vision.

---

## 🚀 Overview

This project was developed as part of an academic research initiative focused on affective computing and adaptive learning technologies.

The system analyzes students’ emotions from:
- uploaded video files,
- webcam recordings,
- facial expressions in real time.

Based on detected emotions, the system calculates an **Engagement Score** and generates recommendations for improving the educational process.

---

## ✨ Features

✅ Real-time facial emotion recognition  
✅ Webcam emotion analysis  
✅ Offline video analysis  
✅ Emotion distribution visualization  
✅ Student engagement estimation  
✅ Automatic pedagogical recommendations  
✅ CSV export for experiment tracking  
✅ TXT report generation  
✅ Interactive command-line interface  
✅ Google Colab support  

---

## 🧠 Recognized Emotions

The system detects the following emotional states:

| Emotion | Description |
|---|---|
| Happiness | Positive engagement |
| Neutral | Stable state |
| Surprise | Active reaction |
| Sadness | Low motivation |
| Fear | Confusion or stress |
| Anger | Frustration |
| Disgust | Strong negative reaction |

---

## 📊 Engagement Score

The engagement score is calculated using weighted emotional statistics.

| Score Range | Engagement Level |
|---|---|
| 70–100 | High engagement |
| 40–69 | Medium engagement |
| 0–39 | Low engagement |

---

## 🏗️ System Architecture

The project contains two main analytical modules:

### 1️⃣ Text Emotion Analysis
Neural network models:
- CNN
- BiLSTM with Attention
- CNN-LSTM

Used for classifying student feedback into emotional categories.

### 2️⃣ Video Emotion Analysis
Based on:
- DeepFace
- OpenCV
- Facial Emotion Recognition (FER)

Supports:
- online webcam mode;
- offline video processing.

---

## 📈 Experimental Results

| Model | Accuracy |
|---|---:|
| CNN | 91.6% |
| BiLSTM + Attention | **92.2%** |
| CNN-LSTM | 89.8% |

Best model:
- **F1-score:** 0.922
- **AUC-ROC:** 0.944

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- DeepFace
- OpenCV
- NumPy
- Matplotlib
- Pillow
- Google Colab

---

## 🔒 Ethics & Privacy

This project is intended for educational and research purposes.

The system processes facial emotion data and therefore should be used responsibly and in compliance with privacy regulations and ethical AI practices.

Recommendations:
- obtain user consent before analysis;
- avoid storing personal biometric data;
- use anonymized educational datasets whenever possible;
- ensure transparency of AI-generated decisions.

---

## 📂 Project Structure

```text
student-emotion-recognition-system/
│
├── notebook/
│   └── student_emotion_recognition.ipynb
│
├── assets/
│
├── reports/
│
├── results/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 👤 Author

**Khadzhimurad Khutraev**

Data science student with a focus on machine learning and neural networks.
