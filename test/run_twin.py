import tensorflow as tf
import cv2, os, hid, time
import numpy as np
from multi_relay import Relay

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
from pypylon import pylon
from multiprocessing import Process, Queue


class Cam:
    def __init__(self, camera_num, camera_setting, Img_Q, Break_Q):
        self.camera_num = camera_num
        self.camera_setting = camera_setting
        self.Img_Q = Img_Q
        self.Break_Q = Break_Q

        self.load_camera()
        print(f'{self.camera_num} Camera is Ready!...')

        while True:
            self.get_img()
            self.put_Queue()

            if self.Break_Q = '1':
                break

        self.cameras.StopGrabbing()
        print(f'{self.camera_num} Camera is Power-OFF...')

    def load_camera(self):
        maxCamerasToUse = 1
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        self.cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

        for i, self.cam in enumerate(self.cameras):
            self.cam.Attach(tlFactory.CreateDevice(devices[0]))

        self.cameras.Open()
        pylon.FeaturePersistence.Load(self.camera_setting, cam.GetNodeMap())
        self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def get_img(self):
        try:
            sefl.grabResult = self.cameras.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
            image_raw = self.converter.Convert(self.grabResult)
            image_raw = image_raw.GetArray()
            img_crop = img_raw[0:494, 0:494]
            image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            img_resize = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
            self.img = img_resize
        except:
            pass

    def put_Queue(self):
        try:
            self.Img_Q.put(self.img)
        except:
            self.Img_Q.put("None")


if __name__ == "__main__":
    # 만약 save_foler가 없으면 폴더 만들기
    if not os.path.exists('Reject_img'):
        os.mkdir('Reject_img')

    total_num, Outlier_num, answer, is_outlier = 0, 0, 0, 0
    L = LMS()
    model = L.return_model()
    model = load_detector("tray_od_20epochs.h5")
    print("Alibi-detect Model Loaded...")

    Img_A, Break_A = Queue(), Queue()
    Img_B, Break_B = Queue(), Queue()

    LINE_A = Process(target=Cam, args=('0', Img_A, Break_A,))
    LINE_B = Process(target=Cam, args=('1', Img_B, Break_B,))

    dic = []
    for i in hid.enumerate():
        if i['produce_string'] == 'USBRelay2':
            dic.append(i['path'])
    print(dic)

    Relay_A = Relay(path=dic[0])
    Relay_B = Relay(path=dic[1])

    LINE_A.start()
    LINE_B.start()

    total_A, total_B, outlier_A, outlier_B = 0, 0, 0, 0
    
    while True:
        img_A = ''
        img_B = ''

        try:
            img_A = Img_A.get(timeout=0.000001)
            print('A Image Loaded...')
            if img_A is not None:
                total_A += 1
                img = cv2.resize(img_A, (64,64), interpolation=cv2.INTER_LINEAR)
                A = img_resize.reshape((1,) + img_resize.shape)
                A = A.astype('float32') / 255.
                output = model.predict(A)
                answer_A = output['data']['is_outlier'][0]
                if answer_A = 1:
                    outlier_A += 1
                    Relay_A.state(0, on=True)
                    sleep(0.01)
                    Relay_A.state(0, on=False)        
        except:
            pass


        try:
            img_B = Img_B.get(timeout=0.000001)
            print('B Image Loaded...')
            if img_B is not None:
                total_B += 1
                img = cv2.resize(img_B, (64,64), interpolation=cv2.INTER_LINEAR)
                B = img_resize.reshape((1,) + img_resize.shape)
                B = B.astype('float32') / 255.
                output = model.predict(B)
                answer_B = output['data']['is_outlier'][0]
                if answer_B = 1:
                    outlier_B += 1
                    Relay_B.state(0, on=True)
                    sleep(0.01)
                    Relay_B.state(0, on=False)
        except:
            pass

        cv2.imshow('A', Img_A)
        cv2.moveWindow('A', 300, 500)
        cv2.imshow('B', Img_B)
        cv2.moveWindow('B', 1180, 500)

        k = cv2.waitKey(1) & 0xFF

        if k == 114: #lowercase r
          print(f'[A-side] Total : {toal_A}... / Reject : {outlier_A}')
          print(f'[B-side] Total : {toal_B}... / Reject : {outlier_B}')
          print('====================')
          print{'\n Counter Reset!...')
          total_A, total_B, outlier_A, outlier_B = 0, 0, 0, 0

        if k == 27: #esc
            Break_A.put('1')
            Break_B.put('1')
            Reject_Q.put(None)
            break

    cv2.destroyAllWindows()
       
 

