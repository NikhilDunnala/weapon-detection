# ============================================================
# Weapon Detection System — Model Training Script
# Stack: YOLOv10, Ultralytics, Google Colab (GPU)
# Author: Nikhil Dunnala | github.com/NikhilDunnala
# ============================================================
# Run this script on Google Colab with GPU runtime enabled.
# Runtime > Change runtime type > T4 GPU
# ============================================================

# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Install Ultralytics (YOLOv10)
import subprocess
subprocess.run(["pip", "install", "ultralytics"], check=True)

import os

# Step 3: Disable W&B logging (not needed for training)
os.environ['WANDB_DISABLED'] = 'true'
print("W&B logging disabled.")

# Step 4: Fix a compatibility issue in Ultralytics for Colab
subprocess.run([
    "sed", "-i",
    's/torch.load(file, map_location="cpu")/torch.load(file, map_location="cpu", weights_only=False)/',
    "/usr/local/lib/python3.12/dist-packages/ultralytics/nn/tasks.py"
])

# Step 5: Download YOLOv10s pre-trained weights
weights_dir = '/content/drive/MyDrive/yolov10_weights'
os.makedirs(weights_dir, exist_ok=True)
subprocess.run([
    "wget", "-P", weights_dir,
    "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt"
])

# Step 6: Unzip custom weapon dataset
subprocess.run([
    "unzip", "-q",
    "/content/drive/MyDrive/archive.zip",
    "-d", "/content/yolov10_data"
])

# Step 7: Create data.yaml config file
data_yaml_content = """path: /content/yolov10_data
train: ./train/images
val: ./valid/images
test: ./test/images
nc: 1
names: ['gun']
"""

with open("/content/yolov10_data/data.yaml", "w") as f:
    f.write(data_yaml_content)

print("data.yaml created successfully.")

# Step 8: Train YOLOv10 on custom dataset
subprocess.run([
    "yolo", "detect", "train",
    "data=/content/yolov10_data/data.yaml",
    f"model={weights_dir}/yolov10s.pt",
    "epochs=50",
    "imgsz=640",
    "project=/content/drive/MyDrive/yolov10_training_results",
    "name=yolov10s_results",
    "save_json=False"
])

# Step 9: Test on a sample image
subprocess.run([
    "yolo", "detect", "predict",
    "model=/content/drive/MyDrive/yolov10_training_results/yolov10s_results/weights/best.pt",
    "source=/content/gun_test.jpg",
    "save=True"
])

print("\nTraining complete!")
print("Best model weights saved at:")
print("/content/drive/MyDrive/yolov10_training_results/yolov10s_results/weights/best.pt")
