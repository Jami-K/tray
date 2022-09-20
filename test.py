from tensorflow import keras
#from keras import layers
import keras.backend as K
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Dense, Lambda, BatchNormalization, Add, Flatten, Reshape
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt #visible output module

from glob import glob
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

class KK_Keras:
   def __init__(self, IMG_SIZE, latent_dim):

     gpus = tf.config.experimental.list_physical_devices('GPU')
     if gpus:
      try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
       print(e)

     strategy = tf.distribute.MirroredStrategy() #devices=["/gpu:0", "/gpu:1"]
     with strategy.scope():


      inp_data = Input(IMG_SIZE)

      x = self.EncoderUnit(inp_data, 3)
      x = self.EncoderUnit(x, 3, Residual=False)
      x = self.EncoderUnit(x, 16)
      x = self.EncoderUnit(x, 16, Pool=True, Batch_N=True, Residual=True)
      x = self.EncoderUnit(x, 32)
      x = self.EncoderUnit(x, 32, Pool=True, Batch_N=True, Residual=True)
      x = self.EncoderUnit(x, 64)
      x = self.EncoderUnit(x, 64, Pool=True, Batch_N=True, Residual=True)
      x = self.EncoderUnit(x, 128)
      #x = self.EncoderUnit(x, 128, Pool=True, Batch_N=True, Residual=True)
      #x = self.EncoderUnit(x, 256, Pool=True)
      x = Flatten()(x)
      x = Dense(16*16, activation='tanh')(x)

      self.z_mean = Dense(latent_dim)(x)
      self.z_sig = Dense(latent_dim)(x)
      self.enco_Data = Lambda(self.sampling, output_shape=(latent_dim,))([self.z_mean, self.z_sig])

      latent_inp = Input((latent_dim))
      x = Dense(16*16, activation='relu')(latent_inp)
      x = Reshape((16, 16, 1))(x)
      #x = self.DecoderUnit(x, 256)
      #x = self.DecoderUnit(x, 256, Residual=True)
      x = self.DecoderUnit(x, 128)
      x = self.DecoderUnit(x, 128, Sample=True, Residual=True)
      x = self.DecoderUnit(x, 64)
      x = self.DecoderUnit(x, 64, Sample=True, Residual=True)
      x = self.DecoderUnit(x, 32)
      x = self.DecoderUnit(x, 32, Sample=True, Residual=True)
      x = self.DecoderUnit(x, 16)
      x = self.DecoderUnit(x, 16, Sample=True, Residual=True)
      x = self.DecoderUnit(x, 3)
      x = self.DecoderUnit(x, 3, Residual=True)
      decoded = self.DecoderUnit(x, 3)

      self.encoder = Model(inp_data, [self.z_mean, self.z_sig, self.enco_Data], name='Encoder')
      self.decoder = Model(latent_inp, decoded, name='Decoder')
      self.vae = Model(inp_data, self.decoder(self.encoder(inp_data)[2]), name='VAE')

      self.vae.compile(optimizer='adadelta', loss='binary_crossentropy')
      #self.encoder.summary()
      #self.decoder.summary()
      self.vae.summary()

   def EncoderUnit(self, inp, filters, kernel_size=(3,3), Pool=False, activation=relu, Batch_N=False, Residual=False):
      x = Conv2D(filters, kernel_size, padding='same')(inp)
      if Batch_N:
         x = BatchNormalization()(x)
      x = activation(x)
      if Residual:
         x = Add()([inp,x])
      if Pool:
         x = MaxPool2D((2, 2))(x)
      return x

   def DecoderUnit(self, inp, filters, kernel_size=(3,3), Sample=False, activation=relu, Batch_N=False, Residual=False):
      x = Conv2D(filters, kernel_size, padding='same')(inp)
      if Batch_N:
         x = BatchNormalization()(x)
      x = activation(x)
      if Residual:
         x = Add()([inp,x])
      if Sample:
         x = UpSampling2D((2, 2))(x)
      return x
   
   def sampling(self, args):
      z_mean, z_sig = args
      batch = K.shape(z_mean)[0]
      dim = K.int_shape(z_mean)[1]
      # by default, random_normal has mean=0 and std=1.0
      epsilon = K.random_normal(shape=(batch, dim))
      return z_mean + z_sig * epsilon

   def FIT(self, train, test, es, mc, epochs):

      print("\n========== Model Study Starts ==========\n")
      self.vae.fit_generator(train, validation_data=test, epochs=epochs, callbacks=[es, mc])

   def predict(self, image):
      return self.vae.predict(image)

   def save_model(self, path):
      self.vae.save_weights(path)

   def load_model(self, path):
      self.vae.load_weights(path)

   def return_model(self):
      #return self.autoencoder
      return self.vae


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
   latent_size = 32
   epochs = 1000 # Replay the Learning Process
   batch_size = 32

   od = KK_Keras((IMG_SIZE1, IMG_SIZE2, 3), latent_size)

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
   mc = ModelCheckpoint('tray4.h5', monitor='val_loss', mode='min', save_best_only=True)

   #od.FIT(training_set, valid_set, es, mc, epochs=epochs)
   od.load_model('./tray4.h5')	
   #od.save_model('./tray4.h5')


   v_img = cv2.imread('./validation/11.jpg') #1 2 11 12
   v_prepare = Img_Prepare(v_img)
   v_resized = cv2.resize(v_prepare, (IMG_SIZE1, IMG_SIZE2), interpolation=cv2.INTER_LINEAR)
   v_reshape = v_resized.reshape((1,) + (IMG_SIZE1, IMG_SIZE2) + (3,))

   cv2.imshow('Test', v_resized)
   print(v_resized.shape)

   x_test_pred = od.predict(v_reshape/255)

   n_img = cv2.imread('./validation/1.jpg') #1 2 11 12
   n_prepare = Img_Prepare(n_img)
   n_resized = cv2.resize(n_prepare, (IMG_SIZE1, IMG_SIZE2), interpolation=cv2.INTER_LINEAR)
   n_reshape = n_resized.reshape((1,) + (IMG_SIZE1, IMG_SIZE2) + (3,))
   n_test_pred = od.predict(n_reshape/255)

   cv2.imshow('Valid', n_resized)
   print(n_resized.shape)

   m_img = cv2.imread('./validation/80.jpg') #1 2 11 12
   m_prepare = Img_Prepare(m_img)
   m_resized = cv2.resize(m_prepare, (IMG_SIZE1, IMG_SIZE2), interpolation=cv2.INTER_LINEAR)
   m_reshape = m_resized.reshape((1,) + (IMG_SIZE1, IMG_SIZE2) + (3,))
   m_test_pred = od.predict(m_reshape/255)

   cv2.imshow('Valid2', m_resized)
   print(m_resized.shape)


   test_mae_loss = np.mean(np.power(x_test_pred - v_reshape, 2), axis=1)
   test_mae_loss = test_mae_loss.reshape((-1))

   valid_mae_loss = np.mean(np.power(n_test_pred - n_reshape, 2), axis=1)
   valid_mae_loss = valid_mae_loss.reshape((-1))
   valid2_mae_loss = np.mean(np.power(m_test_pred - m_reshape, 2), axis=1)
   valid2_mae_loss = valid2_mae_loss.reshape((-1))


   plt.title("Graph")
   plt.hist(valid_mae_loss, color='blue', histtype='step', label='valid')
   plt.hist(valid2_mae_loss, color='green', histtype='step', label='valid2')
   plt.hist(test_mae_loss, color='red', histtype='step', label='test')
   plt.xlabel('mae')
   plt.legend()
   plt.show()
