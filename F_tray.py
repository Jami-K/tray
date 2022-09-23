from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt #visible output module

import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

class KF_Keras:
   def __init__(self, IMG_SIZE1, IMG_SIZE2):

      self.model = self.make_network(IMG_SIZE1, IMG_SIZE2)
      
      
   def make_network(self, IMG_SIZE1, IMG_SIZE2):
      model = Sequential()
      
      model.add(Conv2D(32, (2,2), input_shape=(IMG_SIZE1, IMG_SIZE2, 1), activation='relu'))
      model.add(MaxPool2D(pool_size=(2,2)))
      
      model.add(Conv2D(64, (2,2), activation='relu'))
      model.add(Conv2D(64, (2,2), activation='relu'))
      model.add(MaxPool2D(pool_size=(2,2)))
      model.add(Dropout(0.5))
      
      model.add(Conv2D(128, (2,2), activation='relu'))
      model.add(Conv2D(128, (2,2), activation='relu'))
      model.add(Conv2D(128, (2,2), activation='relu'))
      model.add(MaxPool2D(pool_size=(2,2)))
      model.add(Dropout(0.5))
      
      model.add(Conv2D(256, (2,2), activation='relu'))
      model.add(Conv2D(256, (2,2), activation='relu'))
      model.add(Conv2D(256, (2,2), activation='relu'))
      model.add(MaxPool2D(pool_size=(2,2)))
      model.add(Dropout(0.5))
      
      model.add(Flatten())
      model.add(Dense(units=512, activation='relu'))
      model.add(Dropout(0.5))
      model.add(Dense(units=512, activation='relu'))
      model.add(Dropout(0.5))
      
      model.add(Dense(units=2, activation='softmax'))
      
      print(model.summary())
      
      return model

   def return_model(self):
      return self.model

def Return_model_T(IMG_SIZE1, IMG_SIZE2):
   model = KF_Keras(IMG_SIZE1, IMG_SIZE2)
   return model.return_model()
   
def Img_Prepare(image):
   img_copy = np.uint8(image)
   img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
   img_blur = cv2.GaussianBlur(img_rgb, ksize=(3,3), sigmaX=0)
   img_canny = cv2.Canny(img_blur, 200, 200)
   img_rgb2 = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2RGB)
   images = cv2.resize(img_rgb2, (256, 256), interpolation=cv2.INTER_LINEAR)

   return images


if __name__ == "__main__":

   IMG_SIZE1, IMG_SIZE2 = 256, 256
   
   model = Return_model_T(IMG_SIZE1, IMG_SIZE2)
   
   model.compile(optimize='adadelta', loss='binary_crossentropy', metrics=['acc'])
   
   train_datagen = ImageDataGenerator(preprocessing_function=Img_Prepare,
                                    rescale = None,
                                    rotation_range=1, width_shift_range=0.01,
                                    height_shift_range=0.01, shear_range=0.01,
                                    horizontal_flip=False, vertical_flip=False,
                                    brightness_range=(0.2, 0.8),
                                    fill_mode='nearest')

   valid_datagen = ImageDataGenerator(preprocessing_function=Img_Prepare,
                                    rescale = None)

   training_set = train_datagen.flow_from_directory('Database/Sub',
                                                   target_size=(IMG_SIZE1, IMG_SIZE2),
                                                   color_mode='rgb',
                                                   batch_size=256, class_mode='input')

   valid_set = train_datagen.flow_from_directory('Database/SubSub',
                                                   target_size=(IMG_SIZE1, IMG_SIZE2),
                                                   color_mode='rgb',
                                                   batch_size=32, class_mode='input')

   es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
   mc = ModelCheckpoint('trayF.h5', monitor='val_loss', mode='min', save_best_only=True)

   model.fit_generator(training_set, epochs=1000, validation_data=valid_set, callbacks=[es,mc])
   
   model.save_weights('./trayF.h5')
   #model.load_weights('./trayF.h5')

   v_img = cv2.imread('./validation/11.jpg') #1 2 11 12
   v_prepare = Img_Prepare(v_img)
   v_resized = cv2.resize(v_prepare, (IMG_SIZE1, IMG_SIZE2), interpolation=cv2.INTER_LINEAR)
   v_reshape = v_resized.reshape((1,) + (IMG_SIZE1, IMG_SIZE2) + (3,))

   cv2.imshow('Test', v_resized)
   print(v_resized.shape)
   output_T = model.predict(v_resized)
   print(output_T)

   n_img = cv2.imread('./validation/1.jpg') #1 2 11 12
   n_prepare = Img_Prepare(n_img)
   n_resized = cv2.resize(n_prepare, (IMG_SIZE1, IMG_SIZE2), interpolation=cv2.INTER_LINEAR)
   n_reshape = n_resized.reshape((1,) + (IMG_SIZE1, IMG_SIZE2) + (3,))

   cv2.imshow('Valid', n_resized)
   print(n_resized.shape)
   output_N = model.predict(n_resized)
   print(output_N)

   m_img = cv2.imread('./validation/80.jpg') #1 2 11 12
   m_prepare = Img_Prepare(m_img)
   m_resized = cv2.resize(m_prepare, (IMG_SIZE1, IMG_SIZE2), interpolation=cv2.INTER_LINEAR)
   m_reshape = m_resized.reshape((1,) + (IMG_SIZE1, IMG_SIZE2) + (3,))

   cv2.imshow('Valid2', m_resized)
   print(m_resized.shape)
   output_M = model.predict(m_resized)
   print(output_N)
