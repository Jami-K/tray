import keras
#from keras import layers
import keras.backend as K
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Dense, Lambda, BatchNormalization, Add, Flatten, Reshape
from keras.activations import relu, sigmoid
from keras.models import Sequential, Model
from keras.utils.multi_gpu_utils import multi_gpu_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt #visible output module

from glob import glob
import cv2
import numpy as np

class KK_Keras:
   def __init__(self, IMG_SIZE, latent_dim):

      inp_data = Input(IMG_SIZE)

      x = self.EncoderUnit(inp_data, 3)
      x = self.EncoderUnit(x, 3, Residual=True)
      x = self.EncoderUnit(x, 16)
      x = self.EncoderUnit(x, 16, Pool=True, Batch_N=True, Residual=True)
      x = self.EncoderUnit(x, 32)
      x = self.EncoderUnit(x, 32, Pool=True, Batch_N=True, Residual=True)
      x = self.EncoderUnit(x, 64)
      x = self.EncoderUnit(x, 64, Pool=True, Batch_N=True, Residual=True)
      x = self.EncoderUnit(x, 128, Pool=True)
      x = Flatten()(x)
      x = Dense(16*16, activation='tanh')(x)

      self.z_mean = Dense(latent_dim)(x)
      self.z_sig = Dense(latent_dim)(x)
      self.enco_Data = Lambda(self.sampling, output_shape=(latent_dim,))([self.z_mean, self.z_sig])

      latent_inp = Input((latent_dim))
      x = Dense(16*16, activation='tanh')(latent_inp)
      x = Reshape((16, 16, 1))(x)
      x = self.DecoderUnit(x, 128)
      x = self.DecoderUnit(x, 128, Residual=True)
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

      #self.vae = multi_gpu_model(self.vae, gpus=4)
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

   def FIT(self, images, epochs, validation_split=0.2):

      print("\n========== Model Study Starts ==========\n")
      self.vae.fit(images, images, validation_split=validation_split, epochs=epochs,
         callbacks=[EarlyStopping(patience=15), ModelCheckpoint('./tray.h5', save_best_only=True)])

   def predict(self, image):
      return self.vae.predict(image)

   def save_model(self, path):
      self.vae.save_weights(path)

   def load_model(self, path):
      self.vae.load_weights(path)

   def return_model(self):
      #return self.autoencoder
      return self.vae


def img_to_np(fpaths, IMG_SIZE1, IMG_SIZE2):
   img_array = []
   for i in fpaths:
      try:
         image = cv2.imread(i).astype(np.float32) / 255.
         img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         img_resized = cv2.resize(img_rgb, (IMG_SIZE1, IMG_SIZE2), interpolation=cv2.INTER_LINEAR)
         #img_reshape = img_resized.reshape((1,) + img_resized.shape + (3,))
         img_array.append(np.asarray(img_resized))
      except:
         continue

   images = np.asarray(img_array)
   return images


if __name__ == "__main__":

   IMG_SIZE1, IMG_SIZE2 = 128, 128
   latent_size = 27
   epochs = 10000 # Replay the Learning Process
   #batch_size = 32

   od = KK_Keras((IMG_SIZE1, IMG_SIZE2, 3), latent_size)

   Img_path = glob('./Database/Train/*.jpg')
   train_img_list, val_img_list = train_test_split(Img_path, test_size=0.1, random_state=2021)

   x_train = img_to_np(train_img_list, IMG_SIZE1, IMG_SIZE2)
   x_val = img_to_np(val_img_list, IMG_SIZE1, IMG_SIZE2)
   database = img_to_np(Img_path, IMG_SIZE1, IMG_SIZE2)

   #grph = od.FIT(database, epochs=epochs)
   od.load_model('./tray.h5')	
   #od.save_model('./tray.h5')
  
   #plt.plot(grph.history["loss"], label = "Training Loss")
   #plt.plot(grph.history["val_loss"], label = "Validation Loss")
   #plt.legend()
   #plt.show()

   x_train_pred = od.predict(x_train)
   train_mae_loss = np.mean(np.power(x_train_pred - x_train, 2), axis=1)
   threshold = np.max(train_mae_loss) #0.24836096
   print("Reconstruction error threshold: ", threshold)
   
   print('==================================================')
   print("    Anomaly Detection - Model Study completed.")
   print('==================================================')

   #od.load_model('./tray.h5')

   Y_path = glob('./validation/dust.jpg') #blackdot dust hair normal 
   z_val = img_to_np(Y_path, IMG_SIZE1, IMG_SIZE2)

   
   x_test_pred = od.predict(z_val)
   test_mae_loss = np.mean(np.power(x_test_pred - z_val, 2), axis=1)
   test_mae_loss = test_mae_loss.reshape((-1))

   print(test_mae_loss)
   print(test_mae_loss.max())

   if test_mae_loss.max() < threshold:
      print("\nIt is Normal data\n")
   else:
      print("\nIt is Anomal data\n")


   plt.plot(test_mae_loss)
   plt.axhline(threshold, 0, len(test_mae_loss), color='gray', linestyle='solid', linewidth=2)
   plt.show()


'''
   decoded_imgs = od.predict(z_val)

   n = 1
   plt.figure(figsize=(10,2), dpi=100)
   for i in range(n):
      ax=plt.subplot(2, n, i+1)
      plt.imshow(z_val[i].reshape(IMG_SIZE1,IMG_SIZE2,3))
      plt.gray()
      ax.set_axis_off()

      ax = plt.subplot(2, n, i+1 + n)
      plt.imshow(decoded_imgs[i].reshape(IMG_SIZE1,IMG_SIZE2,3))
      plt.gray()
      ax.set_axis_off()
      
   plt.show()
'''
