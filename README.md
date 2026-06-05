# 🔫 Weapon Detection System Using YOLOv10 & OpenCV

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat&logo=python)
![YOLOv10](https://img.shields.io/badge/YOLOv10-Object_Detection-cyan?style=flat)
![OpenCV](https://img.shields.io/badge/OpenCV-Real--Time-purple?style=flat&logo=opencv)
![mAP](https://img.shields.io/badge/mAP@0.50-96%25-brightgreen?style=flat)
![FPS](https://img.shields.io/badge/Speed-25_FPS-orange?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

> A real-time firearm detection system using **YOLOv10 deep learning** and **OpenCV**, capable of identifying weapons in live video streams, webcams, CCTV feeds, and recorded videos at **25 FPS** with **96% mAP@0.50**.

---

## 🎯 Key Results

| Metric | Score |
|--------|-------|
| mAP@0.50 | **96.0%** |
| Localization Precision | **95.2%** |
| Processing Speed | **25 FPS** |
| Improvement over baseline | **+3.1% mAP** |
| Training Dataset | **9,065 images** |

---

## ✨ Features

- 🎥 Supports **live webcam**, **video files** (.mp4, .avi, .mov), and **images** (.jpg, .png)
- 📦 YOLOv10 fine-tuned on a custom 9,065-image gun detection dataset
- 🟦 Real-time bounding box overlay with class name and confidence score
- ⚡ Zero-frame-drop pipeline at 25 FPS on mid-level hardware
- 🔌 Compatible with USB cameras, laptop webcams, and CCTV RTSP streams

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Detection Model | YOLOv10 (Ultralytics) |
| Video Processing | OpenCV |
| Visualization | cvzone |
| Training Environment | Google Colab (GPU) |
| Language | Python 3.9+ |

---

## 🚀 Getting Started

### Prerequisites
```
Python 3.9+
GPU recommended for training (CPU works for inference)
```

### Installation
```bash
git clone https://github.com/NikhilDunnala/weapon-detection.git
cd weapon-detection
pip install -r requirements.txt
```

### Download the trained model
Download `model.pt` from Google Drive and place it in the project folder:

👉 [Download model.pt](https://drive.google.com/file/d/1ck8w9N7nYc1acy-BjFIkbVg2YD400eW0/view?usp=drive_link)

### Run detection
```bash
python detect.py
```

> Edit `input_source` in `detect.py` to point to your image/video file or set to `0` for webcam.

---

## 📂 Project Structure

```
weapon-detection/
├── train.py              # YOLOv10 training script (Google Colab)
├── detect.py             # Real-time detection script
├── data.yaml             # Dataset config (paths, class names)
├── model.pt              # Trained YOLOv10 weights (best.pt)
└── README.md
```

---

## 🔬 How It Works

### Training Pipeline
1. YOLOv10s pre-trained weights downloaded from official source
2. Custom dataset (9,065 images, 1 class: `gun`) loaded via `data.yaml`
3. Fine-tuned for **50 epochs** at 640×640 resolution on Google Colab GPU
4. Best weights saved from validation sweeps (eliminates bounding-box drift)

### Inference Pipeline
1. Video/webcam feed is opened via `cv2.VideoCapture`
2. Each frame passed to YOLOv10 for inference
3. Detected bounding boxes drawn with class label + confidence score
4. Frame displayed in real-time window; press **ESC** to exit

---

## 📸 Sample Output

*(Add a screenshot of weapon detection output here)*

---

## 🔮 Future Work

- [ ] Night-vision / low-light detection support
- [ ] Multi-weapon type detection (rifles, knives)
- [ ] Multi-camera centralised dashboard
- [ ] Edge deployment on NVIDIA Jetson Nano / Raspberry Pi
- [ ] Automated alert system (SMS / email notifications)

---

## 📜 License

MIT License

---

## 👨‍💻 Author

**Nikhil Dunnala** — [GitHub](https://github.com/NikhilDunnala) · [LinkedIn](https://linkedin.com/in/nikhildunnala)

*B.Tech CSE @ VIT-AP University, 2026*# weapon-detection
