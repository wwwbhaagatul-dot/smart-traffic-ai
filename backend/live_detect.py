import requests
import cv2
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open webcam or video
cap = cv2.VideoCapture(0)  # use "traffic.mp4" for a video file
if not cap.isOpened():
    print("Error: Cannot open video file or webcam.")
    exit()

# Create output folder
os.makedirs("output_videos", exist_ok=True)
output_path = "output_videos/live_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

print("🚦 Smart Traffic Detection started... Press Ctrl+C to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    annotated_frame = results[0].plot()

    # Initialize video writer once
    if out is None:
        height, width, _ = frame.shape
        out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    out.write(annotated_frame)

    # Count vehicles
    names = model.names
    counts = {}
    for box in results[0].boxes.cls:
        cls = names[int(box)]
        if cls in ["car", "truck", "bus", "motorbike"]:
            counts[cls] = counts.get(cls, 0) + 1

    print("Vehicle Counts:", counts)

    # Send counts to backend
    try:
        requests.post("http://127.0.0.1:5000/update", json=counts, timeout=1)
    except:
        pass

# Release resources
cap.release()
out.release()
print(f"✅ Video saved at: {output_path}")
