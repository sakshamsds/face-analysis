import cv2
import csv
import math
import numpy as np
import face_recognition
from deepface import DeepFace    # The DeepFace library uses VGG-Face as the default model.


# Task 2: Face Detection
def get_faces(original_image, annotate=False):
    # load the pretrained haar cascade face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_scale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)    # get grayscale image
    faces_coordinates = face_cascade.detectMultiScale(
                gray_scale_image, 
                scaleFactor=1.1, 
                minNeighbors=4, 
                minSize=(30, 30))
    return faces_coordinates

def annotate_image(x, y, w, h, text, image):
    # draw rectangles around the faces
    # color = (203, 192, 255) if detected_gender == "Woman" else (230, 216, 173)  #bgr
    color = (0, 255, 0)
    cv2.rectangle(img = image, 
                  pt1=(x, y), 
                  pt2=(x+w, y+h), 
                  color=color, 
                  thickness=2)
    cv2.putText(img = image, 
                text = text,
                org = (x, y-5),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 1.0,
                color = color,
                thickness = 2)
    return

# Task 3 and 4: sentiment analysis and gender classification
def get_emotion_and_gender(face):
    analysis = DeepFace.analyze(img_path=face, 
                                actions=['emotion', 'gender'], 
                                enforce_detection=False, 
                                detector_backend='opencv')
    detected_emotion = analysis[0]['dominant_emotion']
    detected_gender = analysis[0]['dominant_gender']
    return (detected_emotion, detected_gender)

# Task 5: Face pose estimation
def get_angle(face):
    feature_locations = face_recognition.face_landmarks(face)
    if not feature_locations:
        return float('inf')
    else:
        # Extract the nose bridge and chin points
        nose_bridge = feature_locations[0]['nose_bridge']
        chin = feature_locations[0]['chin']
        # Calculate the angle using the nose bridge and chin points
        delta_x = nose_bridge[-1][0] - chin[0][0]
        delta_y = nose_bridge[-1][1] - chin[0][1]
        yaw = math.atan2(delta_y, delta_x) * 180 / math.pi
        return yaw

# Task 6: face embeddings
def get_embeddings(face):
    encoding = face_recognition.face_encodings(face)
    return np.empty(0) if not encoding else encoding[0]

def get_person_face_file_mapping():
    tsv_path = "./../data/congress.tsv"
    image_path = "./../data/"
    person_face_mapping = {}
    with open(tsv_path) as tsv:
        for line in csv.reader(tsv, dialect='excel-tab'):
            image_file_name = image_path + line[0].strip()
            name = line[1].strip()
            person_face_mapping[name] = image_file_name
    return person_face_mapping

def generate_encodings_for_known():
    person_face_encoding = {}
    person_face_mapping = get_person_face_file_mapping()
    for i, (name, path) in enumerate(person_face_mapping.items()):
        dateset_roi = cv2.imread(path)
        faces_coordinates = get_faces(dateset_roi, annotate=False)
        faces = []
        for (x, y, w, h) in faces_coordinates:
            face = dateset_roi[y:y+h, x:x+w]  
            faces.append(face)
        if faces:
            face = faces[0]
            if face.any():
                encoding = get_embeddings(face)
                if encoding.any():
                    print("known face encoded: {}/487".format(i))
                    person_face_encoding[name] = encoding
    return person_face_encoding

# Task 7: face recognition
def recognise_face(face, known_names, known_encodings, unknown_encoding):
    name = "Unknown"
    if not unknown_encoding.any():
        return name
    # match each roi face with the set of known encodings
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding) 
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_names[best_match_index]
    return name
    
