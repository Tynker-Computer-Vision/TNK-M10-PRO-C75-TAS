
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

training_images, testing_images, training_ages, testing_ages = train_test_split(images, ages, random_state=42)
print(":::::::::::::::::::::::::::::::::::::::::::::::::")
print("Number of training images = ", len(training_images))
print("Number of testing images = ", len(testing_images))
print("Number of training ages = ", len(training_ages))
print("Number of testing ages = ", len(testing_images))
print(":::::::::::::::::::::::::::::::::::::::::::::::::")

# Create a Sequential model and name it age_model
age_model = Sequential()

# Add Conv2D layer of 128, kernal size of 3, activation function as relu and input shape as (200,200,3)
age_model.add(Conv2D(128, kernel_size=3, activation='relu', input_shape=(200,200,3)))
# Add MaxPool2D with pool_size of 3 and strides of 2
age_model.add(MaxPool2D(pool_size=3, strides=2))


# Add Conv2D layer of 128, kernal size of 3, activation function as relu 
age_model.add(Conv2D(128, kernel_size=3, activation='relu'))
# Add MaxPool2D with pool_size of 3 and strides of 2
age_model.add(MaxPool2D(pool_size=3, strides=2))
              
# Add Conv2D layer of 256, kernal size of 3, activation function as relu
age_model.add(Conv2D(256, kernel_size=3, activation='relu'))
# Add MaxPool2D with pool_size of 3 and strides of 2
age_model.add(MaxPool2D(pool_size=3, strides=2))

# Add Conv2D layer of 512, kernal size of 3, activation function as relu 
age_model.add(Conv2D(512, kernel_size=3, activation='relu'))
# Add MaxPool2D with pool_size of 3 and strides of 2
age_model.add(MaxPool2D(pool_size=3, strides=2))

# Add Flatten() to the model
age_model.add(Flatten())
# Add Dropout of 0.2
age_model.add(Dropout(0.2))
# Add Dense with 512, activation as relu
age_model.add(Dense(512, activation='relu'))
# Add Dense with 1, activation as linear and name as 'age'
age_model.add(Dense(1, activation='linear', name='age'))
              
# Compile the model with optimizer 'adam and loss 'name'              
age_model.compile(optimizer='adam', loss='mae')

# Print model summary
print(age_model.summary()) 

# Fit the model and save the returned information in history variable
history = age_model.fit(training_images, training_ages, validation_data=(testing_images, testing_ages), epochs=10)

# Save the model to file named age_model_50epochs.h
age_model.save('age_model_50epochs.h5')


