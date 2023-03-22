import cv2, os, hid, time, psutil
import numpy as np
from detect import *

from relay import Relay
from pypylon import pylon
from multiprocessing import Process, Queue
from utils.augmentations import letterbox


class Main:
    def __init__(self, line, queue):

        self.queue = queue
        self.rr_img = np.zeros((200,200,1), np.uint8)
        self.Total_num = 0
        self.Reject_num = 0
        self.line = line

        self.weights = './weights/230321_best.pt'
        self.yaml = './data/tray.yaml'
        self.img_save_path = '/home/nongshim/바탕화면/Reject_Image'

        if self.line == 'A':
            camera_num = 1
            self.Reject_limit = 70
            self.save_img_limit = 60
            self.camera_setting = './camera_settingA.pfs'
        elif self.line == 'B':
            camera_num = 0
            self.Reject_limit = 100
            self.save_img_limit = 100
            self.camera_setting = './camera_settingN.pfs'
        
        self.load_network()
        self.load_camera(camera_num)
        self.Run()

    def load_network(self):
        """ YOLO 모델 선언 """
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
        pylon.FeaturePersistence.Load(self.camera_setting, self.cam.GetNodeMap(), True)
        #self.cameras.Close()
        self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    
    def make_dir(self):
        """ 폴더 생성 후 경로 반환 """
        dir_path = self.img_save_path
        
        if os.path.exists(dir_path + "/" + str(time.strftime("%Y-%m-%d",time.localtime()))):
            dirname_reject = dir_path + "/" + str(time.strftime("%Y-%m-%d", time.localtime()))
        else:
            try:
                os.mkdir(dir_path + "/" + str(time.strftime("%Y-%m-%d",time.localtime())))
            except:
                pass
            dirname_reject = dir_path + "/" + str(time.strftime("%Y-%m-%d",time.localtime()))
            print("\nThe New Folder For saving Rejected image is Maked...\n")
        
        return dirname_reject
     
    def save_file(self, img):
        """ 이미지 저장 """
        dirname_reject = self.make_dir()

        name1 = str(time.strftime("%m-%d-%H-%M", time.localtime()))
        name2 = ".jpg"
        confi = round(float(self.confi_int), 1)
        name_orig = str('[' + str(confi) + ']') + '_' + self.line + '_' + name1 + name2
        #print(name_orig)
        
        cv2.imwrite(os.path.join(dirname_reject, name_orig), img)


    def put_queue(self, error_data):
        self.queue.put(error_data)

    def Predict(self):
        try:
            """ 이미지를 self.img에 할당하기 """
            grabResult = self.cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            image_raw = self.converter.Convert(grabResult)
            img_raw = image_raw.GetArray()
            image = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            self.img = image
            #self.img = cv2.imread('./tray.jpg')

            gn = torch.tensor(self.img.shape)[[1,0,1,0]]
            self.Total_num += 1
            
            """ YOLO 프로그램을 불러와서 목표물 탐색하기 """
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
                        self.cls = cls.item()
                        self.confi_int = round(conf.item() * 100)
                        #print(line)

                        if cls.item() == 2 and self.save_img_limit < self.confi_int:
                            #print("Image Saved...{}".format(self.confi_int))
                            self.save_file(self.img)
                        if cls.item() == 2 and self.Reject_limit < self.confi_int:
                            self.Reject_num += 1
                            error_data = [self.line, 'reject']
                            print("{} : Outlier Detected...{}".format(self.line,self.confi_int)
                            #특정 릴레이에 신호를 보내도록 작성
                            self.put_queue(error_data)
                            img0 = self.img.copy()
                            r_time = "["+str(self.confi_int)+"%]"+str(time.strftime("%H:%M:%S",time.localtime()))
                            cv2.putText(img0, r_time, (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
                            self.rr_img = cv2.resize(img0, (400, 300), interpolation=cv2.INTER_LINEAR)
        except:
            #self.img = np.zeros((494,659,1),np.uint8)
            pass
   
    def Show_Img(self):
        """ 화면 내 이미지 출력 """
        try:
           img = self.img.copy()
           if self.cls == 1:
               cls = 'OK'
               txt_clr = (255,255,0)
           elif self.cls == 2:
               cls = 'Reject'
               txt_clr = (0,0,255)
           txt_confi = cls + ' : ' + str(self.confi_int) + '%'
           cv2.putText(img, txt_confi, (20,70), cv2.FONT_HERSHEY_SIMPLEX, 2, txt_clr, 2)
           total_txt = 'Total : ' + str(self.Total_num) + ' / Reject : ' + str(self.Reject_num)
           cv2.putText(img, total_txt, (20, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
           cv2.imshow(self.line, img)
           
           cv2.createTrackbar ('Reject', self.line, self.Reject_limit, 100, onChange)
           self.Reject_limit = cv2.getTrackbarPos('Reject', self.line)
           
           if self.line == 'A':
               cv2.moveWindow(self.line, 100, 500)
           elif self.line == 'B':
               cv2.moveWindow(self.line, 800, 500)
        except:
           #print('No Image is Captured...')
           pass

    def Show_RR(self):
        """ 최근 리젝트된 이미지 출력 """
        rr_name = str(self.line) + ' reject'
        try:
            cv2.imshow(rr_name, self.rr_img)
            if self.line == 'A':
                cv2.moveWindow(rr_name, 900, 100)
            elif self.line == 'B':
                cv2.moveWindow(rr_name, 1300, 100)
        except:
            pass
                
    def Run(self):
        """ 메인 프로그램 구동 """
        while True:

            try:
                quit = self.queue.get(timeout=0.001)
                #print(quit)
            except:
                pass
            
            self.Predict()
            self.Show_Img()
            self.Show_RR()
  
            k = cv2.waitKey(1) & 0xFF
        
            if k == 114: #lowercase r
                print("=================================")
                print("\n{} | Total : {} ... / Outlier : {} ...\n".format(self.line, self.Total_num, self.Reject_num))
                print("Reset Completed...")
                print("=================================\n")
                self.Total_num = 0
                self.Reject_num = 0
        
            if k == 113: #lowercase q
                self.queue.put('off')
                break
            
            if quit == 'off':
                break
                
        self.cameras.StopGrabbing()
        cv2.destroyAllWindows()

def onChange():
    pass

class Reject_sys:
    def __init__(self, Q):
        self.relay = Relay(idVendor=0x16c0, idProduct=0x05df)
        self.relay.state(0, False)
        
        self.relay_runtime = 0.15
        self.relay_A = 'off'
        self.relay_B = 'off'
        self.relay_A_time = time.time()
        self.relay_B_time = time.time()
        self.queue = Q
        self.Time_out = 0.001

        """ 1개 지나가는데 필요한 시간 """
        self.reject_need_time = int(0.050 / self.Time_out)
        
        """ 리젝트까지 필요한 시간 """
        self.standby_time_A = int(4.2 / self.Time_out)
        self.standby_time_B = int(4.0 / self.Time_out)
        
        self.T_A = [0] * (self.reject_need_time + self.standby_time_A)
        self.T_B = [0] * (self.reject_need_time + self.standby_time_B)

        #print(self.T_A)
        self.get_Queue()

    def get_Queue(self):
        initial_A = 0
        initial_B = 0
        while True:
            answer = [0]
            ng_question_A = 'ok'
            ng_question_B = 'ok'
            
            for i in range(self.queue.qsize()):
                answer_ = self.queue.get()
                if i == 0:
                    answer = answer_
            
            """ Queue에서 받은 내용을 할당 """
            if answer is not None:
                #print("Now answer is {}".format(answer[0]))
                if answer[0] == 'A' and answer[1] == 'reject':
                    #print("A Reject Queue Arrived...")
                    if self.relay_A =='off':
                        self.T_A = self.time_traveler('A', 0, 1)
                        #print('Reject Signal put...')
                        #print("*"*20)
                        ng_question_A = 'ng'
                    initial_A = 0
                    
                elif answer[0] == 'B' and answer[1] == 'reject':
                    #print("B Reject Queue Arrived...")
                    if self.relay_B =='off':
                        self.T_B = self.time_traveler('A', 0, 1)
                        #print('Reject Signal put...')
                        #print("*"*20)
                        ng_question_B = 'ng'
                    initial_B = 0
                   
            self.relay_off_cal_time()
      
            """ Queue에서 받지 못하였다면 할당 """
            if ng_question_A == 'ok':
                self.T_A = self.time_traveler('A', 0, 0)
                initial_A += 1
                
            if ng_question_B == 'ok':
                self.T_B = self.time_traveler('B', 0, 0)
                initial_B += 1
                                
            if initial_A == 100:
                initial_A = 0
            
            """ 가장 마지막 자리에 오면 신호 출력 """
            if self.T_A[len(self.T_A) - 1] == 1:
                self.state(1)
            if self.T_B[len(self.T_B) - 1] == 1:
                self.state(2)
                
            if answer is None:
                break

    def relay_off_cal_time(self):
        """ 릴레이가 켜져있다면, 일정 시간 이상 초과 되었을 시 릴레이 끄기 """
        if self.relay_A == 'on':
            if time.time() - self.relay_A_time > self.relay_runtime:
                self.relay.state(1, False)
                self.relay_A = 'off'
                
        if self.relay_B == 'on':
            if time.time() - self.relay_B_time > self.relay_runtime:
                self.relay.state(2, False)
                self.relay_B = 'off'

    def state(self, i):
        """ i 에 해당하는 릴레이가 꺼져 있다면, 가동하고 현재의 시간을 저장 """
        if i == 1 and self.relay_A == 'off':
            self.relay.state(1, True)
            self.relay_A = 'on'
            self.relay_A_time = time.time()
            
        if i == 2 and self.relay_B == 'off':
            self.relay.state(2, True)
            self.relay_B = 'on'
            self.relay_B_time = time.time()
        

    def time_traveler(self, line, location, pass_):
        """ 타임테이블 한칸씩 오른쪽으로 이동시키기 """
        if line == 'A':
            temp = self.T_A
        else:
            temp = self.T_B
        temp = temp[:-1]
        temp.insert(location, pass_)

        return temp


    def relay_off_cal_time(self):
        """ 릴레이가 켜져있는 상태이면, 일정 시간 이상이 초과 되었을 시 릴레이 끄기 """
        if self.relay_A == 'on':
            # print(time.time() - self.relay_A_time)
            if time.time() - self.relay_A_time > self.relay_runtime:
                self.relay.state(1, False)
                self.relay_A = 'off'

        if self.relay_B == 'on':
            # print(time.time() - self.relay_A_time)
            if time.time() - self.relay_B_time > self.relay_runtime:
                self.relay.state(2, False)
                self.relay_B = 'off'

    def state(self, i):
        """ i에 해당되는 릴레이가 꺼져 있으면 가동하고 현재의 시간을 저장 """
        if i == 1 and self.relay_A == 'off':
            self.relay.state(1, True)
            self.relay_A = 'on'
            self.relay_A_time = time.time()

        elif i == 2 and self.relay_B == 'off':
            self.relay.state(2, True)
            self.relay_B = 'on'
            self.relay_B_time = time.time()


if __name__ == "__main__":
    """ 만약 python이 두개 켜져있다면, 실행 종료 """
    Check = []
    for process in psutil.process_iter():
       if 'python' in process.name():
          Check.append(str(process.pid))

    queueA, queueB = Queue(), Queue()
    queueR = Queue()

    if len(Check) == 1:
        A = Process(target=Main, args=('A', queueA,))
        B = Process(target=Main, args=('B', queueB,))
        R = Process(target=Reject_sys, args=(queueR,))
        
        A.start()
        B.start()
        R.start()

        while True:
            answerA = ''
            answerB = ''
            for i in range(queueA.qsize()):
                answerA = queueA.get()
                if i == 0:
                    #print(queueA.qsize())
                    if answerA is not None:
                        queueR.put(answerA)
            
            for i in range(queueB.qsize()):
                answerB = queueB.get()
                if i == 0:
                    if answerB is not None:
                        queueR.put(answerB)
            
            if answerA == 'off':
                queueB.put('off')
                queueR.put(None)
                break
                
            if answerB == 'off':
                queueA.put('off')
                queueR.put(None)
                break
        
        #A.join()
        #B.join()
        #R.join()
