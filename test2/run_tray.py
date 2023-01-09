import cv2, os, hid, time
import numpy as np
from detect import *

from pypylon import pylon
from multiprocessing import Process, Queue
from utils.augmentations import letterbox


class Main:
    def __init__(self, line, queue):

        self.queue = queue
        self.save_folder = 'Reject_img'
        self.Total_num = 0
        self.Reject_num = 0
        self.line = line

        if self.line == 'A':
            camera_num = 0
            self.Reject_limit = 97
            self.save_img_limit = 80
        elif self.line == 'B':
            camera_num = 1
            self.Reject_limit = 97
            self.save_img_limit = 80

        self.weights = './runs/train/exp5/weights/best.pt'
        self.yaml = './data/tray.yaml'
        
        self.load_network()
        self.load_camera(camera_num)
        self.Run()

    def load_network(self):
        device = select_device('')
        self.model = DetectMultiBackend(self.weights, device=device, dnn=False, data=self.yaml, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

    def load_camera(self, camera_num):
        """ 카메라 설정 """
        maxCamerasToUse = 1
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        self.cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
        for i, self.cam in enumerate(self.cameras):
            self.cam.Attach(tlFactory.CreateDevice(devices[camera_num]))
        self.cameras.Open()
        #pylon.FeaturePersistence.Load(self.camera_setting, self.cam.GetNodeMap(), True)
        self.cameras.Close()
        self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def Predict(self):
        #try:
            #이미지를 self.img에 할당하기
            grabResult = self.cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            image_raw = self.converter.Convert(grabResult)
            img_raw = image_raw.GetArray()
            image = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            self.img = image
            #self.img = cv2.imread('./tray.jpg')
                
            gn = torch.tensor(self.img.shape)[[1,0,1,0]]
            self.Total_num += 1
            #YOLO 프로그램을 불러와서 목표물 탐색하기
            windows, dt = [], (Profile(), Profile(), Profile())

            with dt[0]:
                img_temp = letterbox(self.img, (640, 640), stride=self.stride, auto=self.pt)[0]
                img_temp = img_temp.transpose((2, 0, 1))[::-1]
                img_temp = np.ascontiguousarray(img_temp)
                img_temp = torch.from_numpy(img_temp).to(self.model.device)
                img_temp = img_temp.float()
                img_temp /= 255.
                if len(img_temp.shape) == 3:
                    img_temp = img_temp[None]
                    #print(img_temp.shape)

            with dt[1]:
                pred = self.model(img_temp, augment=False, visualize=False)

            with dt[2]:
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=5)
                    
            for i, det in enumerate(pred):  # per image
                im0 = self.img.copy()
                if len(det):
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class

                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)  # label format
                        self.confi_int = round(conf.item() * 100)
                        #print(confi_int)

                        if cls.item() == 2 and self.save_img_limit < self.confi_int:
                            print("Image Saved...")
                            #self.save_file(conf, self.img)
                            if self.Reject_limit < self.confi_int:
                                self.Reject_num += 1
                                error_data = 1
                                print("Outlier Detected...")
                                #특정 릴레이에 신호를 보내도록 작성
        #except:
            #pass
   
    def Show_Img(self):
        try:
           img = self.img.copy()
           txt_confi = str(self.confi_int) + '%'
           cv2.putText(img, txt_confi, (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 2)
           cv2.imshow(self.line, img)
           if self.line == 'A':
               cv2.moveWindow(self.line, 300, 500)
           elif self.line == 'B':
               cv2.moveWindwo(self.line, 800, 500)
        except:
           print('No Image is Captured...')
           pass
   
    def Run(self):
        while True:
            self.Predict()
            self.Show_Img()
  
            k = cv2.waitKey(1) & 0xFF
        
            if k == 114: #lowercase r
                print("\nTotal : {} ... / Outlier : {} ...\n".format(self.Total_num, self.Reject_num))
                print("Reset Completed...")
                print("=====================\n")
                self.Total_num = 0
                self.Reject_num = 0
        
            if k == 113: #lowercase q
                break
                
        self.cameras.StopGrabbing()
        cv2.destroyAllWindows()


    def save_file(self, confidence, img):
        dirname_reject = self.make_dir()

        name1 = str(strftime("%m-%d-%H-%M", localtime()))
        name2 = ".jpg"

        confi = round(float(confidence), 1)
        name_orig = str('[' + confi + ']') + '_' + name1 + name2
        
        cv2.imwrite(os.path.join(dirname_reject, name_orig), img)


if __name__ == "__main__":
    # 만약 save_foler가 없으면 폴더 만들기
    if not os.path.exists('Reject_img'):
        os.mkdir('Reject_img')

    R_relay = Queue()

    Main('A', R_relay)

