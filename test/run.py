import tensorflow as tf
import cv2, os, hid, time
import numpy as np

#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
  # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
#  try:
#    tf.config.experimental.set_virtual_device_configuration(gpus[0],
#        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])
#    tf.config.experimental.set_virtual_device_configuration(gpus[1],
#        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])
#    tf.config.experimental.set_virtual_device_configuration(gpus[2],
#        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])
#    tf.config.experimental.set_virtual_device_configuration(gpus[3],
#        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15000)])
#  except RuntimeError as e:
    # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
#    print(e)

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Reshape, InputLayer, Flatten
from alibi_detect.od import OutlierVAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
from alibi_detect.utils.saving import save_detector, load_detector

from model import LMS
from relay import Relay
from pypylon import pylon
from multiprocessing import Process, Queue


class Main:
    def __init__(self, window_name, model, queue):
        self.converter = pylon.ImageFormatConverter()
        self.config = tf.compat.v1.ConfigProto()
        self.IMG_SIZE1 = 64
        self.IMG_SIZE2 = 64
        self.queue = queue
        self.save_folder = 'Reject_img'

        if window_name == 'TRAY':
            self.info = {'window_name': window_name, 'GPU_id': '1',
                         'camera_property': 'camera_settingN.pfs',
                         'camera_id': 0, 'IMG_SIZE1': self.IMG_SIZE1, 'IMG_SIZE2': self.IMG_SIZE2}

        self.load_camera()
        self.run()

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


    def predict(self): #이미지 변환 방법 구현
        A = []
        img_resize = cv2.resize(self.img, (self.IMG_SIZE1, self.IMG_SIZE2), interpolation=cv2.INTER_LINEAR)
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        A = img_resize.reshape((1,) + img_resize.shape)
        A = A.astype('float32') / 255.
        output = self.od.predict(A)
        answer = output['data']['is_outlier'][0]

        return answer

    def decide_ng(self, output):
        if output != []:
            answer = output
            print("Is img Outlier..?    {}".format(answer))
            if answer == 1 :
                error_data = ['Reject']
                self.queue.put(error_data)
                self.save_file(output, self.img)

    def save_file(self, output, img):
        dirname_reject = self.make_dir()

        name1 = str(strftime("%m-%d-%H-%M", localtime()))
        name2 = ".jpg"

        confidence = str(round(output[0][np.argmax(output[0])] * 100, 1))

        name_orig = str('[' + confidence + ']') + '_' + name1 + name2
        cv2.imwrite(os.path.join(dirname_reject, name_orig), img)

    def make_dir(self):
        dir_path = self.save_folder
        if os.path.exists(
                dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(strftime("%H", localtime()))):
            dirname_reject = dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(
                strftime("%H", localtime()))
        else:
            if os.path.exists(dir_path + "/" + str(strftime("%Y-%m-%d", localtime()))):
                try:
                    os.mkdir(
                        dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(
                            strftime("%H", localtime())))
                except:
                    pass

                dirname_reject = dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(
                    strftime("%H", localtime()))
            else:
                try:
                    os.mkdir(dir_path + "/" + str(strftime("%Y-%m-%d", localtime())))
                    os.mkdir(
                        dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(
                            strftime("%H", localtime())))
                except:
                    pass
                dirname_reject = dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(
                    strftime("%H", localtime()))
            print("\nThe New Folder For saving Rejected image is Maked...\n")

        return dirname_reject

    def show_img(self):
        img = self.img.copy()
        cv2.imshow(self.info['window_name'], img)
        cv2.moveWindow(self.info['window_name'], 300, 500)
        self.grabResult.Release()
        self.num += 1

    def run(self):
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        height = int(480 * .8)
        width = int(640 * .8)
        img = np.zeros((height, width, 1), np.uint8)
        img = cv2.putText(img, "Waiting signal", (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1, [255, 255, 255], 1)

        self.num = 0
        self.start = time.time()
        quit = 'on'

        while True:
            try:
                quit = self.queue.get(timeout=0.001)
                #print(quit)
            except:
                pass

            self.get_img()
            print("Picture Captured")
            output = self.predict()
            self.decide_ng(output)
            self.show_img()

            if cv2.waitKey(1) & 0xFF == 27:
                self.queue.put(None)
                break

            if quit == 'off':
                break

        self.cameras.StopGrabbing()
        cv2.destroyAllWindows()


maxCamerasToUse = 1
tlFactory = pylon.TlFactory.GetInstance()
devices = tlFactory.EnumerateDevices()

cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

for i, cam in enumerate(cameras):
    cam.Attach(tlFactory.CreateDevice(devices[0]))
cameras.Open()
pylon.FeaturePersistence.Load('./camera_setting.pfs', cam.GetNodeMap())
cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

if __name__ == "__main__":
    # 만약 save_foler가 없으면 폴더 만들기
    if not os.path.exists('Reject_img'):
        os.mkdir('Reject_img')

    L = LMS()
    model = L.return_model()
    model = load_detector("tray_od_20epochs.h5")
    print("Alibi-detect Model Loaded...")

    while cameras.IsGrabbing():
        try :
            grabResult = cameras.RetrieveResult(50000, pylon.TimeoutHandling_ThrowException)
            image_raw = converter.Convert(grabResult)
            img_raw = image_raw.GetArray()
            img_crop = img_raw[0:494, 0:494]
            img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
            img_resize = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
            A = img_resize.reshape((1,) + img_resize.shape)
            A = A.astype('float32') / 255.
            output = model.predict(A)
            answer = output['data']['instance_score'][0]
            is_outlier = output['data']['is_outlier'][0]
            total_num += 1
            if is_outlier == 1:
                Outlier_num += 1
        except:
            pass
        
        print("{}....{} / Outlier: {} / total: {}".format(answer,is_outlier, Outlier_num, total_num))

        try:
            cv2.imshow('example', img)
        except:
            print("No Image is Captured....")
            pass

        k = cv2.waitKey(1) & 0xFF

        if k == ord('r'):
          print("\nTotal : {} ... / Outlier : {} ...\n".format(toal_num, Outlier_num))
          print("Reset Completed...")
          print("=====================\n")
          total_num = 0
          Outlier_num = 0
        
        if k == 27:
            break
     
    cameras.Close()
    cv2.destroyAllWindows()

       
 
