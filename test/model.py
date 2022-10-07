import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Reshape, InputLayer, Flatten
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
from alibi_detect.utils.saving import save_detector, load_detector

# 1.OK    2.Reject

class LMS :
    def __init__(self):
        self.model = self.make_network()

    def make_network(self):
        encoding_dim = 1024  # Dimension of the bottleneck encoder vector.
        dense_dim = [8, 8, 512]  # Dimension of the last conv. output. This is used to work our way back in the decoder.

        encoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=(64, 64, 3)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
                Flatten(),
                Dense(encoding_dim, )
            ])
        #print(encoder_net.summary())

        decoder_net = tf.keras.Sequential(
            [
                InputLayer(input_shape=(encoding_dim,)),
                Dense(np.prod(dense_dim)),
                Reshape(target_shape=dense_dim),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
            ])
        #print(decoder_net.summary())

        #######################################################################
        # Define and train the outlier detector.

        latent_dim = 1024  # (Same as encoding dim. )

        # initialize outlier detector
        od = OutlierVAE(threshold=.015,  # threshold for outlier score above which the element is flagged as an outlier.
                        score_type='mse',  # use MSE of reconstruction error for outlier detection
                        encoder_net=encoder_net,  # can also pass VAE model instead
                        decoder_net=decoder_net,  # of separate encoder and decoder
                        latent_dim=latent_dim,
                        samples=4)
        return od

    def return_model(self):
        return self.model


def Return_model_L():
    model = LMS()
    return model.return_model()

def img_to_np(image_directory):
    img_list = os.listdir(image_directory)
    dataset=[]
    for i, image_name in enumerate(img_list):
        if (image_name.split('.')[1] == 'jpg'):
            image = cv2.imread(image_directory + '/' + image_name)
            image = cv2.cvtColor(image, cv2.BGR2RGB)
            image = cv2.resize(image, (64, 64))
            dataset.append(np.array(image))
        dataset = np.array(dataset)
    return dataset


if __name__ == "__main__":
    image_directory = 'Database/train/'
    SIZE = 64
    dataset = []  #Many ways to handle data, you can use pandas. Here, we are using a list format.

#    good_images = os.listdir(image_directory + 'ok/')
#    for i, image_name in enumerate(good_images):
#        if (image_name.split('.')[1] == 'jpg'):
#            image = cv2.imread(image_directory + 'ok/' + image_name)
#            image = Image.fromarray(image, 'RGB')
#            image = image.resize((SIZE, SIZE))
#            dataset.append(np.array(image))
#        dataset = np.array(dataset)

    #train = dataset[0:300]
    #test = dataset[300:374]

    #train = train.astype('float32') / 255.
    #test = test.astype('float32') / 255.

    #Let us also load bad images to verify our trained model.
    #bad_images = os.listdir(image_directory + 'reject')
    #bad_dataset=img_to_np(bad_images)

    #bad_dataset = bad_dataset.astype('float32') / 255.

    # train
    #from alibi_detect.models.tensorflow.losses import elbo #evidence lower bound loss

    od = Return_model_L()
    adam = tf.keras.optimizers.Adam(lr=1e-4)

    #od.fit(train, optimizer = adam, epochs=10, batch_size=4, verbose=True)

    #Check the threshold value. Should be the same as defined before.
    print("Current threshold value is: ", od.threshold)

    #infer_threshold Updates threshold by a value inferred from the percentage of
    #instances considered to be outliers in a sample of the dataset.
    #percentage of X considered to be normal based on the outlier score.
    #Here, we set it to 99%
    #od.infer_threshold(test, outlier_type='instance', threshold_perc=99.0)
    #print("Current threshold value is: ", od.threshold)

    # save the trained outlier detector
    #As mentioned in their documentation, save and load is having issues in python3.6 but works fine in 3.7
    #from alibi_detect.utils import save_detector, load_detector  #If this does not work, try the next line
    #save_detector(od, "tray_od_20epochs.h5")
    od = load_detector("tray_od_20epochs.h5")

    #Test our model on a bad image
    #img_num = 9
    #test_bad_image = bad_dataset[img_num].reshape(1, 64, 64, 3)
    #plt.imshow(test_bad_image[0])

    #test_bad_image_recon = od.vae(test_bad_image)
    #test_bad_image_recon = test_bad_image_recon.numpy()
    #plt.imshow(test_bad_image_recon[0])

    #test_bad_image_predict = od.predict(test_bad_image) #Returns a dictionary of data and metadata

    #Data dictionary contains the instance_score, feature_score, and whether it is an outlier or not.
    #Let u look at the values under the 'data' key in our output dictionary
    #bad_image_instance_score = test_bad_image_predict['data']['instance_score'][0]
    #print("The instance score is:", bad_image_instance_score)

    #bad_image_feature_score = test_bad_image_predict['data']['feature_score'][0]
    #plt.imshow(bad_image_feature_score[:,:,0])
    #print("Is this image an outlier (0 for NO and 1 for YES)?", test_bad_image_predict['data']['is_outlier'][0])

    A = []
    img = cv2.imread('./1.jpg')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_rgb, (SIZE, SIZE))
    A = img_resize.reshape((1,) + img_resize.shape)
    A = A.astype('float32') / 255.

    A_predict = od.predict(A)
    A_instance_score = A_predict['data']['instance_score'][0] #specific score?
    A_feature_score = A_predict['data']['feature_score'][0] #whole score?
    print("Instance is {} / Feature is {}".format(A_instance_score, A_feature_score))
    print("Is A an outlier (0 for NO and 1 for YES)?", A_predict['data']['is_outlier'][0])


    #You can also manually define the threshold based on your specific use case.
    od.threshold = 0.002
    print("Current threshld value is: ", od.threshold)

