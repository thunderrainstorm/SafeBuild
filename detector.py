import cv2
import math
import face_recognition
import os
from ultralytics import YOLO
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol

def calculate_intersection_area(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return intersection_area

def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            known_image = face_recognition.load_image_file(image_path)
            known_face_encoding = face_recognition.face_encodings(known_image)[0]
            known_face_encodings.append(known_face_encoding)
            known_face_names.append(os.path.splitext(filename)[0])
    return known_face_encodings, known_face_names

def detect_objects(img, model, known_face_encodings, known_face_names):
    classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']
    myColor = (0, 0, 255)

    results = model(img, stream=True)
    person_boxes = []
    hardhat_boxes = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ['Mask', 'NO-Mask', 'Safety Vest', 'NO-Safety Vest']:
                continue

            if conf > 0.5 and (currentClass == 'Hardhat' or currentClass == 'NO-Hardhat' or currentClass == 'Person'):
                if currentClass == 'NO-Hardhat':
                    myColor = (0, 0, 255)
                elif currentClass == 'Hardhat':
                    myColor = (0, 255, 0)
                else:
                    myColor = (255, 0, 0)

                if currentClass == 'Hardhat' or currentClass == 'NO-Hardhat':
                    hardhat_boxes.append((x1, y1, x2, y2, currentClass))
                if currentClass == 'Person':
                    person_boxes.append((x1, y1, x2, y2))

    face_locations = face_recognition.face_locations(img)
    face_encodings = face_recognition.face_encodings(img, face_locations)

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

        face_box = (left, top, right, bottom)

        face_color = (0, 0, 255)
        status_text = " "

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
                                    face_color = (0, 255, 0)
                                    status_text = "All Good!"
                                else:
                                    face_color = (0, 165, 255)
                                    status_text = "Wear Your Own Helmet!!"
                            else:
                                face_color = (0, 255, 255)
                                status_text = "Please Wear Your Helmet"
                        else:
                            if hardhat_box[4] == "Hardhat":
                                if barcode_data:
                                    face_color = (255, 105, 180)
                                    status_text = "Guest User Alert!"
                                else:
                                    face_color = (0, 0, 255)
                                    status_text = "Unknown User Alert!!"
                        break

        cv2.rectangle(img, face_box, face_color, 2)
        cv2.putText(img, name, (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, face_color, 2)
        cv2.putText(img, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2)

    return img
