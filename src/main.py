# !pip3 install opencv-python
# !pip3 install deepface
# !pip3 install face_recognition

# Total runtime ~ 3-4 mins

import cv2
import json
import numpy as np
from util import get_emotion_and_gender, annotate_image, get_angle, get_embeddings, get_faces, generate_encodings_for_known, recognise_face


# Read the image
original_image = cv2.imread('./../count_faces.jpg')   
faces_coordinates = get_faces(original_image, annotate=True)     # the roi faces and their gender and emotions

bounding_boxes = []
angles = []
embeddings = []
detected_genders = []
detected_emotions = []
person_face_encoding = generate_encodings_for_known()
known_names = list(person_face_encoding.keys())
known_encodings = list(person_face_encoding.values())
face_names = []

annotated_faces_img = original_image.copy()
annotated_emotion_img = original_image.copy()
annotated_gender_img = original_image.copy()
annotated_pose_img = original_image.copy()
annotated_names_img = original_image.copy()

num_detected_faces = faces_coordinates.shape[0]

for i, (x, y, w, h) in enumerate(faces_coordinates):
    print("detected face: {}/{}".format(i+1, num_detected_faces))
    face = original_image[y:y+h, x:x+w]  
    (detected_emotion, detected_gender) = get_emotion_and_gender(face)    # detect gender and emotion
    detected_emotions.append(detected_emotion)
    detected_genders.append(detected_gender)
    unknown_encoding = get_embeddings(face)             # get encoding for unknown detected faces
    embeddings.append(unknown_encoding)
    yaw_angle = str(round(get_angle(face), 2))
    angles.append(yaw_angle)                      # get yaw angle for unknown detected faces
    name = recognise_face(face, known_names, known_encodings, unknown_encoding)
    face_names.append(name)
    cv2.imwrite("./results/detected_faces/{}_{}.png".format(name, i), face)    # save detected faces
    annotate_image(x, y, w, h, None, annotated_faces_img)   # annotate face
    annotate_image(x, y, w, h, detected_emotion, annotated_emotion_img)   # annotate emotion
    annotate_image(x, y, w, h, detected_gender, annotated_gender_img)   # annotate gender
    annotate_image(x, y, w, h, yaw_angle, annotated_pose_img)   # annotate face pose
    annotate_image(x, y, w, h, name, annotated_names_img)   # annotate recognised name
    box_dict = {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
    bounding_boxes.append(box_dict)


# save bounding boxes to json file
with open('./results/bounding_boxes.json', 'w') as f:      # clear the file
    pass
with open('./results/bounding_boxes.json', 'w') as f:      # save bounding_boxes 
    json.dump(bounding_boxes, f)
    
np.savez('./results/task_6_embeddings.npz', embeddings)
print("Number of faces detected: ", len(faces_coordinates))

cv2.imwrite('./results/task_2_count_faces_annotated_faces.png', annotated_faces_img)    # save the resulting annotated image
cv2.imwrite('./results/task_3_count_faces_annotated_emotions.png', annotated_emotion_img)    # save the resulting annotated image
cv2.imwrite('./results/task_4_count_faces_annotated_gender.png', annotated_gender_img)    # save the resulting annotated image
cv2.imwrite('./results/task_5_count_faces_annotated_pose.png', annotated_pose_img)    # save the resulting annotated image
cv2.imwrite('./results/task_7_count_faces_annotated_names.png', annotated_names_img)    # save the resulting annotated image


# from collections import Counter

# Counter(detected_emotions).keys() # equals to list(set(words))
# Counter(detected_emotions).values() # counts the elements' frequency

# Counter(detected_genders).keys() # equals to list(set(words))
# Counter(detected_genders).values() # counts the elements' frequency