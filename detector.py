import cv2
import math
import face_recognition
import os
from ultralytics import YOLO
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
from database import log_status  # Import the log_status function

def calculate_intersection_area(boxA, boxB):
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxB[3])

    # Compute the area of the intersection rectangle
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    return intersection_area

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    # Load all images from the known_faces directory
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            known_image = face_recognition.load_image_file(image_path)
            known_face_encoding = face_recognition.face_encodings(known_image)[0]
            known_face_encodings.append(known_face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])  # Use the filename without extension as the name

    return known_face_encodings, known_face_names

def detect_objects(img, model, known_face_encodings, known_face_names):
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest',
                  'machinery', 'vehicle']
    myColor = (0, 0, 255)

    results = model(img, stream=True)
    person_boxes = []
    hardhat_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Ignore Mask and Safety Vest detections
            if currentClass in ['Mask', 'NO-Mask', 'Safety Vest', 'NO-Safety Vest']:
                continue

            # Append the bounding box to the respective list without drawing them
            if conf > 0.5 and (currentClass == 'Hardhat' or currentClass == 'NO-Hardhat' or currentClass == 'Person'):
                if currentClass == 'NO-Hardhat':
                    myColor = (0, 0, 255)
                elif currentClass == 'Hardhat':
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                # Append the bounding box to the respective list
                if currentClass == 'Hardhat' or currentClass == 'NO-Hardhat':
                    hardhat_boxes.append((x1, y1, x2, y2, currentClass))
                if currentClass == 'Person':
                    person_boxes.append((x1, y1, x2, y2))

    # Face recognition and drawing rectangles and text for recognized faces
    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

    # QR code detection and processing with error handling
    barcode_data = None
    try:
        for barcode in decode(img, symbols=[ZBarSymbol.QRCODE]):
            barcode_data = barcode.data.decode('utf-8')
            pts = np.array([barcode.polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, (255, 0, 255), 5)
            pts2 = barcode.rect
            cv2.putText(img, barcode_data, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    except Exception as e:
        print(f"Error decoding barcode: {e}")

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_box = (left, top, right, bottom)  # Define the face bounding box

        face_color = (0, 0, 255)  # Default color for unknown face (red)
        status_text = " "

        # Check overlap with Person and Hardhat/No-Hardhat boxes
        for person_box in person_boxes:
            intersection_area_person = calculate_intersection_area(face_box, person_box)
            if intersection_area_person > 0.1:
                for hardhat_box in hardhat_boxes:
                    intersection_area_hardhat_person = calculate_intersection_area(hardhat_box, person_box)
                    intersection_area_hardhat_face = calculate_intersection_area(hardhat_box, face_box)
                    if (intersection_area_hardhat_person > 0.1 or intersection_area_hardhat_face > 0.1):
                        if name != "Unknown":
                            if hardhat_box[4] == "Hardhat":
                                if barcode_data and name == barcode_data:
                                    face_color = (0, 255, 0)  # Green
                                    status_text = f"{name}, All Good!"
                                else:
                                    face_color = (0, 165, 255)  # Orange
                                    status_text = f"{name}, Wear Your Own Helmet!!"
                            else:
                                face_color = (0, 255, 255)  # Yellow
                                status_text = f"{name}, Please Wear Your Helmet"
                        else:
                            if hardhat_box[4] == "Hardhat":
                                if barcode_data:
                                    face_color = (255, 105, 180)  # Pink
                                    status_text = f"{name}, Guest User Alert!"
                            else:
                                if not barcode_data:    
                                    face_color = (0, 0, 255)  # Red
                                    status_text = f"{name}, Unknown User Alert!!"
                        break

        # Log the status_text to the database for each face detected
        log_status(status_text)

        # Draw rectangle and text for face box only
        cv2.rectangle(img, (left, top), (right, bottom), face_color, 2)
       #cv2.putText(img, name, (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, face_color, 2)
        cv2.putText(img, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

    return img
