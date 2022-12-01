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

class Tray:
    def __init__(self, line, camera_name, Relay_address, Queue):

      self.line = line
      self.camera_name = camera_name
      self.Relay_address = Relay_address
      self.Relay = Relay(path=self.Relay_address)
      self.Queue = Queue
      
      self.total_num = 0
      self.reject_num = 0

      while True:
          self.load_camera()
          self.load_model()
          self.get_img()
          self.predict()
      
          num_display = 'Total : ' + str(self.total_num) + ' / ' + ' Reject : ' + str(self.reject_num)
          self.img = cv2.putText(self.img, num_display, (0, 0), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 1)
          cv2.imshow(line, self.img)
          if self.line == 'A':
              cv2.moveWindow(line, 300, 500)
          else:
              cv2.moveWindow(line, 1100, 500)

          k = cv2.waitKey(0) & 0xFF
          
          if k == 27:
              self.Queue.put('OFF')
              break

      self.cameras.Close()
      cv2.destroyAllWindows()
              
    def load_camera(self):
        maxCamerasToUse = 1
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        self.cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

        for i, self.cam in enumerate(self.cameras):
            self.cam.Attach(tlFactory.CreateDevice(devices[0]))

        self.cameras.Open()
        pylon.FeaturePersistence.Load('./camera_settingN.pfs', self.cam.GetNodeMap())
        self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def load_model(self):
        L = LMS()
        self.model = L.return_model()
        self.model = load_detector("tray_od_20epochs.h5")
        print("Alibi-detect Model Loaded...")
              
    def get_img(self):    
        try:
            sefl.grabResult = self.cameras.RetrieveResult(500, pylon.TimeoutHandling_ThrowException)
            image_raw = self.converter.Convert(self.grabResult)
            image_raw = image_raw.GetArray()
            img_crop = img_raw[0:494, 0:494]
            image = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            img_resize = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
            self.img = img_resize
            self.total_num += 1
        except:
            print("self.img is NOT CAPTURED...1")
            pass
          
    def predict(self):
        try:
            img_resize = cv2.resize(self.img, (64,64), interpolation=cv2.INTER_LINEAR)
            I = img_resize.reshape((1,) + img_resize.shape)
            I = I.astype('float32') / 255.
            output = self.model.predict(I)
            answer = output['data']['is_outlier'][0]
            if answer == 1:
               self.reject_num += 1
               Relay.state(0, on=True)
               sleep(0.01)
               Relay.state(0, on=False)
        except:
            print("self.img is NOT CAPTURED...2")
            pass


if __name__ == "__main__":
    # 만약 save_foler가 없으면 폴더 만들기
    if not os.path.exists('Reject_img'):
        os.mkdir('Reject_img')

    dic = []
    for i in hid.enumerate():
        if i['product_string'] == 'USBRelay2':
            dic.append(i['path'])
    print(dic)

    main = str("Tray Detection")
    daemun = cv2.imread('./daemun.jpg')
    
    queueA, queueB = Queue(), Queue()
    
    queueA.put('ON')
    queueB.put('ON')

    LINE_A = Tray('A', '0', dic[0], queueA)

    LINE_A.start()
    
    while True:
        answer = ''     
        for i in range(queueA.qsize()):
            answer = queueA.get()
            if i == 0:
                #print(queueA.qsize())
                if answer == 'ON':
                    pass
                    
        if answer == 'OFF':
            queueB.put('OFF')
            break

    cv2.destroyAllWindows()
    
