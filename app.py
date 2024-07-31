from flask import Flask, render_template, Response, jsonify
import cv2
import logging
import time
from detector import load_known_faces, detect_objects
from ultralytics import YOLO
from database import initialize_db, clear_status_log, fetch_status_logs  # Import database functions

# Set up logging
logging.basicConfig(filename='camera_debug.log', level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)

# Function to find the first available camera index
def get_camera_index():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cap.release()
            logging.debug(f"Camera found at index: {index}")
            return index
        cap.release()
        index += 1
    logging.error("No camera found.")
    return -1

# Get the camera index
camera_index = get_camera_index()
if camera_index == -1:
    raise RuntimeError("No camera available.")

# Initialize the camera
cap = cv2.VideoCapture(camera_index)
cap.set(3, 1280)
cap.set(4, 720)

# Load YOLO model and known faces
model = YOLO("ppe.pt")
known_faces_dir = "C:\\Users\\sambita\\webapp\\known_faces"
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

# Initialize the database and clear previous session logs
initialize_db()
clear_status_log()

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            logging.error("Failed to read frame from camera.")
            break
        else:
            frame = detect_objects(frame, model, known_face_encodings, known_face_names)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to fetch and display status logs as JSON
@app.route('/status_logs')
def status_logs():
    logs = fetch_status_logs()
    return jsonify(logs)

if __name__ == "__main__":
    # Ensure some delay for camera initialization
    time.sleep(2)
    app.run(host="0.0.0.0", port=80, debug=False)

