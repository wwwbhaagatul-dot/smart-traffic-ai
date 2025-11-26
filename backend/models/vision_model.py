# vision_model.py
# Handles vehicle detection using YOLOv8

from ultralytics import YOLO
import os

# Load the YOLO model once globally
model = YOLO('yolov8n.pt')

def detect_vehicles(image_path, save_dir="backend/output"):
    """
    Detect vehicles in an image and return:
    - results object
    - vehicle counts
    - annotated image path
    - bounding box coordinates
    """

    os.makedirs(save_dir, exist_ok=True)

    # Run YOLO detection and save annotated output
    results = model.predict(source=image_path, save=True, project=save_dir, name="")

    # Initialize counters
    vehicle_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

    # Prepare coordinate list
    coordinates = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            if cls_name in vehicle_counts:
                vehicle_counts[cls_name] += 1

            # Save bounding box coordinates
            coords = box.xyxy[0].tolist()
            coordinates.append({"class": cls_name, "coords": coords})

    # Get saved annotated image path
    annotated_path = os.path.join(save_dir, "image0.jpg")

    print("Vehicle counts:", vehicle_counts)
    print("Annotated image saved at:", annotated_path)

    return results, vehicle_counts, annotated_path, coordinates
