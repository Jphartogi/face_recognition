# -*- coding: utf-8 -*-
"""
@author: Joshua Phartogi
"""
import cv2
import os
from PIL import Image
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,"images")

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
x_train = []
y_labels = []

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).lower()
            #print(label,path)
            
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            
            id_ = label_ids[label]
            print('Labels that is trained:',label)
            #convert image into BW
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image,"uint8")
            
            faces = face_cascade.detectMultiScale(image_array,scaleFactor = 1.5, minNeighbors=5)
            
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle","wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("result.yml")

print('successfully create a trained yml of the image')