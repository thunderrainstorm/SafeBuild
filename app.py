from flask import Flask, render_template, Response
import cv2
from detector import load_known_faces, detect_objects
from ultralytics import YOLO

app = Flask(__name__)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("ppe.pt")
known_faces_dir = "C:\\Users\\sambita\\webapp\\known_faces"
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
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

if __name__ == "__main__":
    from flask import Flask, render_template, Response
import cv2
from detector import load_known_faces, detect_objects
from ultralytics import YOLO

app = Flask(__name__)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("ppe.pt")
known_faces_dir = "C:\\Users\\sambita\\webapp\\known_faces"
known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=False)
