# 🏭 SteelSight AI – Steel Surface Defect Detection using YOLO26

## 📌 Project Overview

Steel surface defects can affect product quality, manufacturing efficiency, and production costs. Detecting these defects manually can be time-consuming and may lead to human errors.

**SteelSight AI** is an AI-powered steel surface defect detection system developed using the YOLO26 object detection model. It automatically detects and identifies defects in steel surface images and videos using computer vision.

The application provides an interactive web interface built with Streamlit, allowing users to upload images or videos and perform automated defect inspection.

---

## 🎯 Objectives

- Develop an automated steel surface defect detection system using YOLO26.
- Detect and classify multiple types of steel surface defects.
- Perform object detection on both images and videos.
- Display bounding boxes and confidence scores for detected defects.
- Analyze defect distribution and generate detection statistics.
- Build an interactive Streamlit application for real-time inspection.
- Demonstrate the application of deep learning in industrial quality control.

---

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Deep Learning:** YOLO26
- **Computer Vision:** OpenCV
- **Libraries:** Ultralytics, NumPy, Pandas, Pillow
- **Visualization:** Matplotlib / Streamlit Charts
- **Web Application:** Streamlit
- **Model Format:** PyTorch (`.pt`)
- **Version Control:** Git and GitHub

---

## 📊 Dataset

- **Dataset:** NEU-DET (Northeastern University Surface Defect Database)
- Contains six categories of steel surface defects.
- Includes images and corresponding annotations for model training and evaluation.

### 🔍 Defect Categories

1. **Crazing** – Fine cracks appearing on the steel surface.
2. **Inclusion** – Foreign particles or impurities embedded in the steel.
3. **Patches** – Irregular patch-like surface defects.
4. **Pitted Surface** – Small holes or pits on the steel surface.
5. **Rolled-in Scale** – Oxide layers pressed into the steel during rolling.
6. **Scratches** – Linear marks or scratches on the steel surface.

---

## 🧠 Model Development

The project uses YOLO26, a modern object detection model, to identify and locate defects in steel surface images.

### Model Workflow

1. Dataset collection and preparation.
2. Image annotation and preprocessing.
3. Dataset configuration.
4. YOLO26 model training.
5. Model validation and performance evaluation.
6. Saving the trained model weights (`best.pt`).
7. Integrating the trained model into a Streamlit application.

---

## 🚀 Application Features

### 1. Image Inspection
- Upload steel surface images.
- Detect multiple defects in a single image.
- Display bounding boxes around detected defects.
- Show defect labels and confidence scores.
- Generate defect count and distribution statistics.

### 2. Video Inspection
- Upload steel surface inspection videos.
- Process video frames using the trained YOLO26 model.
- Detect defects frame by frame.
- Generate annotated output videos.
- View detection summaries and download processed videos.

### 3. Analytics Dashboard
- Display total detected defects.
- Show class-wise defect counts.
- Visualize defect distribution.
- Provide an easy-to-understand inspection summary.

---

## 💻 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/k-v-s-vinay/SteelSight-AI.git
cd SteelSight-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv env
```

Activate the environment on Windows:

```bash
env\Scripts\activate
```

On Linux/Mac:

```bash
source env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the Trained Model

Place your trained model weights file inside the project directory:

```text
best.pt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📁 Project Structure

```text
SteelSight-AI/
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
│
├── create_test_video.py
│
└── sample_images/
```

---

## 🌐 Deployment

🔗 **Live Demo:** [Click Here](https://steelsight-ai-1.streamlit.app/)

🔗 **GitHub Repository:** [Link](https://github.com/k-v-s-vinay/SteelSight-AI/)

---

## 📈 Expected Outcomes

- Automated identification of six categories of steel surface defects.
- Faster and more consistent inspection compared with manual visual checking.
- Visual localization of defects using bounding boxes.
- Confidence scores to understand model predictions.
- Class-wise defect statistics for inspection analysis.
- An interactive application demonstrating the practical use of deep learning in manufacturing.

---

## 🏫 Acknowledgements

This project was developed as part of my Data Science with Generative AI learning journey at Innomatics Research Labs.

I would like to thank my trainers and mentors for their guidance and support throughout the project development process.

---

## 📌 Conclusion

SteelSight AI demonstrates how deep learning and computer vision can be applied to industrial quality inspection.

By integrating YOLO26 with Streamlit, this project provides an interactive solution for detecting and analyzing steel surface defects in images and videos.

The system highlights the potential of AI-powered inspection to support manufacturing quality control, reduce manual inspection effort, and improve defect identification.

---

## 📎 Contact

- **Author:** VINAY
- **LinkedIn:** https://www.linkedin.com/in/vinaykamsala7
- **GitHub:** https://github.com/k-v-s-vinay
- **Email:** vinaykamsala03@gmail.com

**Developed with ❤️ by Vinay**
