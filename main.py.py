import numpy as np 
import face_recognition
import cv2
import os

path = 'person'
images = []
classNames = []
personlist = os.listdir(path)

print(personlist)

for cl in personlist:
    if cl.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
        curperson = cv2.imread(os.path.join(path, cl))
        images.append(curperson)
        classNames.append(os.path.splitext(cl)[0])

def find_encodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img, None, 1, "small")[0]
        encodeList.append(encode)
    return encodeList 

encodeListKnown = find_encodings(images)
print(encodeListKnown)

cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faceCurrentFrame = face_recognition.face_locations(imgS)
    encodeCurrentFrame = face_recognition.face_encodings(imgS, faceCurrentFrame)
    
    for encode_face, faceLoc in zip(encodeCurrentFrame, faceCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encode_face)
        faceDis = face_recognition.face_distance(encodeListKnown, encode_face)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
        else:
            name = "Unknown"
            
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Face Recognition', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break 
    elif key == ord('s'):
        cv2.imwrite('C:/videooooo.jpg', img)
