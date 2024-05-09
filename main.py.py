import numpy as np 
import face_recognition
import cv2
import os
path= 'person'
images = []
classNames=[]
personlist = os.listdir(path)
print(personlist)

for cl in personlist:
    curperson = cv2.imread(f'{path}/{cl}')
    images.append(curperson)
    classNames.append(os.path.splitext(cl)[0])
def findEncodeings (image):
    encodeList =[]
    for img in (images):

        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img,  None,  1, "small") [0]
        encodeList.append(encode)
        
    return encodeList 
encodeListKnown = findEncodeings(images)
print (encodeListKnown)
cap = cv2.VideoCapture(0)
while True:
    _,img = cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    faceCurentFrame = face_recognition.face_locations(imgS)
    encodeCurentFrame = face_recognition.face_encodings(imgS,faceCurentFrame)
    for encodeface , faceLoc in zip(encodeCurentFrame,faceCurentFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeface)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeface)
        print ( faceDis)
        matchIndex = np.argmin(faceDis)
        if matches [matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1 =faceLoc
            y1,x2,y2,x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        

    
    
    
    cv2.imshow('Face Recofnition',img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break 
    elif cv2.waitKey(1) == ord ('s'):
        cv2.imwrite('C//videooooo',cap)