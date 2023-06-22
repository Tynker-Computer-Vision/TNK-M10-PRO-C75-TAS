
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential,load_model,Model
from keras.layers import Conv2D,MaxPool2D,Dense,Dropout,BatchNormalization,Flatten,Input
from sklearn.model_selection import train_test_split
from cvzone.FaceDetectionModule import FaceDetector

path = "Face-image-dataset"
images = []
ages = []
gender = []

detector = FaceDetector()

# Control number of image to be given i.e [0:10], for fast processing 
for img in os.listdir(path)[0:10]:
  try:
    if img!='.git':
      age = img.split("_")[0]
      genders = img.split("_")[1]
      img = cv2.imread(str(path)+"/"+str(img))
      img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      
      img, bbox = detector.findFaces(img, draw=False) 
      if bbox:
        X = bbox[0]['bbox'][0]
        Y = bbox[0]['bbox'][1]
        W = bbox[0]['bbox'][2]
        H = bbox[0]['bbox'][3]
        
        img = img[Y:Y+H, X:X+W]
        img = cv2.resize(img, (200, 200))
        
      images.append(img)
      ages.append(age)

  except Exception as e:
    print("Exception",e)

ages = np.array(ages,dtype=np.int64)
images = np.array(images)

# Split the images in Training and testing sets i.e. training_images, testing_images, training_ages, testing_ages 
training_images, testing_images, training_ages, testing_ages = train_test_split(images, ages, random_state=42)

print(":::::::::::::::::::::::::::::::::::::::::::::::::")
# Print the length of training and testing images datasets
print("Number of training images = ", len(training_images))
print("Number of testing images = ", len(testing_images))
print("Number of training ages = ", len(training_ages))
print("Number of testing ages = ", len(testing_images))
print(":::::::::::::::::::::::::::::::::::::::::::::::::")



