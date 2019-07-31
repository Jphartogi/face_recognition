# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
"""


@author: Joshua Phartogi
"""
import numpy as np
import cv2
import pickle


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("result.yml")

labels = {"person_name":1}
with open("labels.pickle",'rb')as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    
while(True):
    ret,frame = cap.read()
     
    if ret is True:
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
               
        for(x,y,w,h) in faces:
            print(x,y,w,h)
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = frame[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.5,minNeighbors=5)
            for (ex,ey,ew,eh) in eyes:
                #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                pass
            id_,conf = recognizer.predict(roi_gray)
            #check the confidence level 
            if conf >= 30 and conf <= 85:
                font = cv2.FONT_HERSHEY_COMPLEX
                name = labels[id_]
                color = (255,255,0)
                stroke = 2
                cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
            
            else:
                font = cv2.FONT_HERSHEY_COMPLEX
                name = 'unknown'
                color = (255,255,0)
                stroke = 2
                cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
                
            end_point_x = x+w
            end_point_y = y+h
            color = (0,255,0)
            
            cv2.rectangle(frame,(x,y),(end_point_x,end_point_y),color)
            
        cv2.imshow('frame',frame)
        
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
