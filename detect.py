# ============================================================
# Weapon Detection System — Real-Time Detection Script
# Stack: YOLOv10 (Ultralytics), OpenCV, cvzone
# Author: Nikhil Dunnala | github.com/NikhilDunnala
# ============================================================
# Supports: images (.jpg, .png), videos (.mp4, .avi, .mov),
#           live webcam (set input_source = 0)
# Press ESC to stop video/webcam detection
# ============================================================

import cv2
import cvzone
from ultralytics import YOLO
import os

# ============================================================
# CONFIGURATION — Edit these paths before running
# ============================================================

# Path to your trained YOLOv10 model weights
# Download model.pt from Google Drive:
# https://drive.google.com/file/d/1ck8w9N7nYc1acy-BjFIkbVg2YD400eW0/view?usp=drive_link
MODEL_PATH = "model.pt"

# Input source:
#   - Image:  "path/to/image.jpg"
#   - Video:  "path/to/video.mp4"
#   - Webcam: 0  (integer, not string)
INPUT_SOURCE = 0  # default: webcam

# Confidence threshold (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.4

# ============================================================
# Load YOLO model
# ============================================================
print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# ============================================================
# Detect input type
# ============================================================
is_webcam = str(INPUT_SOURCE).isdigit() or INPUT_SOURCE == 0
input_str = str(INPUT_SOURCE)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

ext = os.path.splitext(input_str)[-1].lower()
is_image = ext in IMAGE_EXTENSIONS
is_video = ext in VIDEO_EXTENSIONS


# ============================================================
# Core detection function — runs on a single frame
# ============================================================
def process_frame(frame):
    results = model(frame, conf=CONFIDENCE_THRESHOLD)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 255), 3)

            # Draw label with confidence score
            cvzone.putTextRect(
                frame,
                f"{class_name} | {confidence:.2f}",
                (x1 + 8, y1 - 12),
                scale=2,
                thickness=2,
                colorR=(50, 50, 255)
            )

    return frame


# ============================================================
# IMAGE mode
# ============================================================
if is_image:
    print(f"Running detection on image: {INPUT_SOURCE}")
    frame = cv2.imread(INPUT_SOURCE)

    if frame is None:
        print(f"Error: Unable to read image at '{INPUT_SOURCE}'")
        exit(1)

    output = process_frame(frame)
    cv2.imshow("Weapon Detection — Image", output)
    print("Detection complete. Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output
    out_path = "detection_output.jpg"
    cv2.imwrite(out_path, output)
    print(f"Output saved to: {out_path}")


# ============================================================
# VIDEO / WEBCAM mode
# ============================================================
elif is_video or is_webcam:
    source = 0 if is_webcam else INPUT_SOURCE
    label = "Webcam" if is_webcam else f"Video: {INPUT_SOURCE}"
    print(f"Starting real-time detection — {label}")
    print("Press ESC to stop.")

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Unable to open source '{source}'")
        exit(1)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or frame not available.")
            break

        output = process_frame(frame)
        frame_count += 1

        # Show FPS on frame
        cv2.putText(
            output,
            f"Frame: {frame_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        window_title = "Weapon Detection — Live" if is_webcam else "Weapon Detection — Video"
        cv2.imshow(window_title, output)

        # ESC key to exit
        if cv2.waitKey(10) & 0xFF == 27:
            print("ESC pressed — stopping detection.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Detection stopped. Total frames processed: {frame_count}")


# ============================================================
# INVALID input
# ============================================================
else:
    print(f"Invalid input source: '{INPUT_SOURCE}'")
    print("Set INPUT_SOURCE to an image path, video path, or 0 for webcam.")
