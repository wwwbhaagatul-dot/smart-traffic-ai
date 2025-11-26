latest_counts = {"car": 0, "bus": 0, "truck": 0, "motorcycle": 0}
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
from models.vision_model import detect_vehicles

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Create folder to store uploaded images
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- ROUTES ---

@app.route('/')
def home():
    return jsonify({"message": "Smart Traffic Backend Running!"})


@app.route('/traffic', methods=['GET'])
def get_traffic():
    sample_data = {
        "intersection_1": {"cars": 10, "buses": 2, "2W": 5},
        "intersection_2": {"cars": 7, "buses": 1, "2W": 3}
    }
    return jsonify(sample_data)


# ---- VEHICLE DETECTION API ----
@app.route('/detect', methods=['POST'])
def detect_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    results, counts, annotated_path, coordinates = detect_vehicles(image_path)

    return jsonify({
        "vehicle_counts": counts,
        "annotated_image_path": annotated_path,
        "coordinates": coordinates
    })


# ---- ROUTE TO SERVE UPLOADED/ANNOTATED IMAGES ----
@app.route('/update_traffic', methods=['POST'])
def update_traffic():
    data = request.get_json()
    vehicle_counts = data.get('vehicle_counts', {})
    print("🚗 Live Traffic Update Received:", vehicle_counts)
    return jsonify({"status": "success", "received": vehicle_counts})

@app.route('/traffic_data', methods=['GET'])
def traffic_data():
    return jsonify(latest_counts)

@app.route('/update', methods=['POST'])
def update_simple():
    data = request.get_json()
    latest_counts.update(data)
    print("Updated via /update →", latest_counts)
    return {"status": "ok"}

if __name__ == '__main__':
    app.run(debug=True)
