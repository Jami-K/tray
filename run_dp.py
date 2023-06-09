import cv2, os, hid, time, psutil
import numpy as np
import properties

from detect import *
from utils.augmentations import letterbox
from pypylon import pylon
from tqdm import tqdm
from relay import Relay
from multiprocessing import Process, Queue
from screeninfo import get_monitors


class dp_window:
    def __init__(self):
    
        # 각종 환경 변수를 선언한다
        window_info = 'Tray-Vision by NOKSAN | Q : Program off' #프로그램 이름
        self.start_switch = 0 #프로그램 동작 여부
        self.quit_switch = 0 #프로그램 종료 여부
        self.now = 'off' #프로그램 작동 여부

        global queueA, queueB, queueR, queueA_img, queueB_img, queueA_off, queueB_off
        queueA, queueB, queueR = Queue(), Queue(), Queue() # [라인명, 불량여부]
        queueA_img, queueB_img = Queue(), Queue()
        queueA_off, queueB_off = Queue(), Queue()

        self.x_max, self.y_max = self.screen_frame() #모니터 화면을 크기를 가져온다 (1920, 1080)
        self.y_max = int(self.y_max * 0.90) #상부 시작파일 바의 크기를 감안하여 축소함
        print("Window size is {}x{}...".format(self.x_max, self.y_max))
        self.total_num_A, self.total_num_B = 0, 0
        self.total_reject_A, self.total_reject_B = 0, 0

        self.reject_switch_A = 0 #A기 리젝트 미작동 여부
        self.reject_switch_B = 0 #B기 리젝트 미작동 여부
        
        #이미지 좌표 설정
        x_operateA, y_operateA, x_operateB, y_operateB = int(self.x_max*0.030), int(self.y_max*0.122), int(self.x_max*0.400), int(self.y_max*0.122) #리젝트 작동여부 좌표
        x_imgA, y_imgA, x_imgB, y_imgB = int(self.x_max*0.030), int(self.y_max*0.220), int(self.x_max*0.400), int(self.y_max*0.220) #카메라 사진 좌표
        x_outputA, y_outputA, x_outputB, y_outputB = int(self.x_max*0.030), int(self.y_max*0.740), int(self.x_max*0.400), int(self.y_max*0.740) #검사결과 좌표
        xa_tback, ya_tback, xb_tback, yb_tback = int(self.x_max*0.030), int(self.y_max*0.850), int(self.x_max*0.400), int(self.y_max*0.850) #현황판 좌표
        xa_total, ya_total, ya_reject, xb_total = int(self.x_max * 0.132), int(self.y_max*0.890), int(self.y_max*0.955), int(self.x_max * 0.506) #현황판 항목 좌표
        x_rimg, y_rimg, x_rtext, y_rtext = int(self.x_max*0.760), int(self.y_max*0.400), int(self.x_max*0.770), int(self.y_max*0.430) #최근 리젝트 이미지 좌표
        r_color = (255,255,0) #리젝트에 표기되는 글자 색상
        
        #이미지 크기 결정
        w_operate, h_operate = int(self.x_max*0.302), int(self.y_max*0.090) # (580, )
        w_img, h_img = int(self.x_max*0.302), int(self.y_max*0.508) # (580, 494)
        w_bar, h_bar = int(self.x_max*0.302), int(self.y_max*0.103) # (580, 100)
        w_total, h_total = int(self.x_max*0.302), int(self.y_max*0.140) # (580, 136)
        
        #배경에 들어갈 이미지 정의
        self.background = cv2.resize(cv2.imread('./display/background.png'), (self.x_max, self.y_max))
        self.A_operate = cv2.resize(cv2.imread('./display/A-run.png'), (w_operate, h_operate))
        self.B_operate = cv2.resize(cv2.imread('./display/B-run.png'), (w_operate, h_operate))
        self.A_stop = cv2.resize(cv2.imread('./display/A-stop.png'), (w_operate, h_operate))
        self.B_stop = cv2.resize(cv2.imread('./display/B-stop.png'), (w_operate, h_operate))
        self.img_ok = cv2.resize(cv2.imread('./display/OK.png'), (w_bar, h_bar))
        self.img_reject = cv2.resize(cv2.imread('./display/Reject.png'), (w_bar, h_bar))
        self.img_waiting = cv2.resize(cv2.imread('./display/waiting.png'), (w_bar, h_bar))
        self.img_total = cv2.resize(cv2.imread('./display/Total.png'), (w_total, h_total))
        self.final_window = self.background

        #멀티 프로세싱 선언
        #A = Process(target=Main, args=('A', queueA, queueA_img, queueA_off,))
        #B = Process(target=Main, args=('B', queueB, queueB_img, queueB_off,))
        #R = Process(target=Reject_sys, args=(queueR,))
    
        while True:
            #전체 수량 / 불량 수량 화면 내 표기
            self.total_counter(self.total_num_A, self.total_reject_A, xa_tback, ya_tback, xa_total, ya_total, xa_total, ya_reject)
            self.total_counter(self.total_num_B, self.total_reject_B, xb_tback, yb_tback, xb_total, ya_total, xb_total, ya_reject)
            
            #리젝트 가동 여부 화면 내 표기
            self.run_stop(self.A_operate, self.A_stop, self.B_operate, self.B_stop, x_operateA, y_operateA, x_operateB, y_operateB)
            
            cv2.imshow(window_info, self.final_window)
            cv2.moveWindow(window_info, 0, 0)
            
            cv2.setMouseCallback(window_info, self.mouseEvent)
            k = cv2.waitKey(1) & 0xFF
            
            if k == 113: # lowercase q
               self.end_mode()
               break
               
            if k == 114: # lowercase r
               self.reset_counter()
            
            if self.quit_switch == 1:
               print("Program Killed by Touch-User...{}".format(self.quit_switch))
               self.end_mode()
               break
            
            answerA, answerB = '', ''
            img_A , img_B = np.zeros((w_img,h_img,3), np.uint8), np.zeros((w_img,h_img,3), np.uint8)
            img_A_resent, img_B_resent = np.zeros((400,400,3), np.uint8), np.zeros((400,400,3), np.uint8)
            
            if self.start_switch == 1:
                if self.now == 'off':
                    self.now = 'on'
                    try:
                       print("="*35)
                       print(" Main Program will start Soon... ")
                       print("="*35)
                       A.start()
                       B.start()
                       R.start()
                    except:
                       print("="*35)
                       print(" Process is Already started... ")
                       print("="*35)
                       pass
            
            if self.now == 'on':
                for i in range(queueA.qsize()):
                    answerA = queueA.get()
                    if i == 0:
                        #print(queueA.qsize())
                        if answerA is not None:
                            #print(answerA)
                            if self.reject_switch_A == 0:
                                queueR.put(answerA)
                            if answerA[1] == 'ok':
                                self.total_num_A += 1
                            elif answerA[1] == 'reject':
                                self.total_num_A += 1
                                self.total_reject_A += 1
                
                for i in range(queueB.qsize()):
                    answerB = queueB.get()
                    if i == 0:
                        #print(queueB.qsize())
                        if answerB is not None and answerB[1] != 'none':
                            self.total_num_B += 1
                            if self.reject_switch_B == 0:
                                queueR.put(answerB)
                            if answerB[1] == 'reject':
                                self.total_reject_B += 1
                                    
                for i in range(queueA_img.qsize()):
                    img_A = queueA_img.get()
                    if i == 0:
                        #print(queueA_img.qsize())
                        if img_A is not None:
                            #print(img_A.shape)
                            imgA = cv2.resize(img_A, (w_img, h_img))
                            self.final_window = self.merge_image(self.final_window, imgA, x_imgA, y_imgA)
                
                for i in range(queueB_img.qsize()):
                    img_B = queueB_img.get()
                    if i == 0:
                        #print(queueB_img.qsize())
                        if img_B is not None:
                            imgB = cv2.resize(img_B, (w_img, h_img))
                            self.final_window = self.merge_image(self.final_window, imgB, x_imgB, y_imgB)
                try:
                    if answerA[1] == 'ok':
                        self.final_window = self.merge_image(self.final_window, self.img_ok, x_outputA, y_outputA)
                    elif answerA[1] == 'reject':
                        self.final_window = self.merge_image(self.final_window, self.img_reject, x_outputA, y_outputA)
                        self.show_rr(img_A, answerA, x_rimg, y_rimg, x_rtext, y_rtext, r_color)
                    elif answerA[1] == 'none':
                        self.final_window = self.merge_image(self.final_window, self.img_waiting, x_outputA, y_outputA)
                except:
                    #print("Error on loading A-line Detection...")
                    pass
                        
                try:
                    if answerB[1] == 'ok':
                        self.final_window = self.merge_image(self.final_window, self.img_ok, x_outputB, y_outputB)
                    elif answerB[1] == 'reject':
                        self.final_window = self.merge_image(self.final_window, self.img_reject, x_outputB, y_outputB)
                        self.show_rr(img_B, answerB, x_rimg, y_rimg, x_rtext, y_rtext, r_color)
                    elif answerB[1] == 'none':
                        self.final_window = self.merge_image(self.final_window, self.img_waiting, x_outputB, y_outputB)
                except:
                    #print("Error on loading B-line Detection...")
                    pass
                    
            if self.start_switch == 0 and self.now == 'on':
                #print("\nProcess will be off soon...\n")
                self.now = 'off'

        cv2.destroyAllWindows()
        A.terminate()
        B.terminate()
        R.terminate()
        A.join()
        B.join()
        R.join()

    def screen_frame(self):
        screen = get_monitors()[0]
        #print(screen)
        screen = str(screen).split(',')
        screen_width = int(screen[2].split('=')[1])
        screen_height = int(screen[3].split('=')[1])
        return screen_width, screen_height

    def mouseEvent(self, event, x, y, flags, param):
        x_ratio = round(x / self.x_max, 3)
        y_ratio = round(y / self.y_max, 3)

        if event == cv2.EVENT_FLAG_RBUTTON:
            print("Mouse Click! My X is {}, My Y is {}".format(x_ratio, y_ratio))
        if event == cv2.EVENT_FLAG_LBUTTON:
            if y_ratio >= 0.120 and y_ratio <= 0.199: # 각 라인 리젝트 여부 결정
                if x_ratio >= 0.030 and x_ratio <= 0.330:
                    if self.reject_switch_A == 0:
                        self.reject_switch_A = 1
                        print("A reject system is De-Activated...")
                    else:
                        self.reject_switch_A = 0
                        print("A reject system Run...")
                if x_ratio >= 0.400 and x_ratio <= 0.700:
                    if self.reject_switch_B == 0:
                        self.reject_switch_B = 1
                        print("B reject system is De-Activated...")
                    else:
                        self.reject_switch_B = 0
                        print("B reject system Run...")

            if x_ratio >= 0.000 and x_ratio <= 0.132:  # 검사 시작 버튼
                if y_ratio >= 0.019 and y_ratio <= 0.092:
                    if self.start_switch == 0:
                        self.start_switch = 1
                        print("start switch on")
                    else:
                        self.start_switch = 0
                        print("start switch off")

            if x_ratio >= 0.938 and x_ratio <= 0.978 : # 종료 버튼
               if y_ratio >= 0.016 and y_ratio <= 0.089 :
                  self.quit_switch = 1
                                    
            if x_ratio >= 0.917 and x_ratio <= 0.968 : # 리셋 버튼
                if y_ratio >= 0.167 and y_ratio <= 0.258 :
                    self.reset_counter()

    def merge_image(self, back, front, x, y): #back 이미지 내 x, y좌표에 front 이미지를 덮어씌운다
        if back.shape[2] == 3:
            back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
        if front.shape[2] == 3:
            front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)

        bh, bw = back.shape[:2]
        fh, fw = front.shape[:2]
        x1, x2 = max(x, 0), min(x + fw, bw)
        y1, y2 = max(y, 0), min(y + fh, bh)
        front_cropped = front[y1 - y:y2 - y, x1 - x:x2 - x]
        back_cropped = back[y1:y2, x1:x2]

        alpha_front = front_cropped[:, :, 3:4] / 255
        alpha_back = back_cropped[:, :, 3:4] /255

        result = back.copy()
        result[y1:y2, x1:x2, :3] = alpha_front * front_cropped[:, :, :3] + (1 - alpha_front) * back_cropped[:, :, :3]
        result[y1:y2, x1:x2, 3:4] = (alpha_front + alpha_back) / (1 + alpha_front * alpha_back) * 255

        result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        return result
       
    def total_counter(self, total_num, reject_num, x_back, y_back, x_total, y_total, x_reject, y_reject): #카운터 화면 내 표시
        self.final_window = self.merge_image(self.final_window, self.img_total, x_back, y_back)
        cv2.putText(self.final_window, str(total_num), (x_total, y_total), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(self.final_window, str(reject_num), (x_reject, y_reject), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
 
    def run_stop(self, img_runA, img_stopA, img_runB, img_stopB, xa, ya, xb, yb):
        if self.reject_switch_A == 0: # 리젝트 가동 중
            self.final_window = self.merge_image(self.final_window, img_runA, xa, ya)
        else:
            self.final_window = self.merge_image(self.final_window, img_stopA, xa, ya)
        if self.reject_switch_B == 0:
            self.final_window = self.merge_image(self.final_window, img_runB, xb, yb)
        else:
            self.final_window = self.merge_image(self.final_window, img_stopB, xb, yb)
        
 
    def show_rr(self, r_img, answer, xi, yi, xt, yt, color): #리젝트 이미지를 화면에 보여줌
        rr_img = cv2.resize(r_img, (250, 250))
        self.final_window = self.merge_image(self.final_window, rr_img, xi, yi)
        text = str(answer[0]) + " Line : " + str(answer[2]) + "%"
        cv2.putText(self.final_window, str(text), (xt, yt), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
    
    def reset_counter(self): #전체 / 불량 카운터 리셋
        print("====================================")
        print(" Line  |    Total    /    Reject    ")
        print("A-line | {} ea  /  {} ea".format(self.total_num_A, self.total_reject_A))
        print("B-line | {} ea  /  {} ea".format(self.total_num_B, self.total_reject_B))
        print("====================================")
        print("         Total Number Reset         ")
        print("====================================")
        self.total_num_A, self.total_num_B = 0, 0
        self.total_reject_A, self.total_reject_B = 0, 0
    
    def end_mode(self): #종료 모드 실행
        queueA_off.put('off')
        queueB_off.put('off')
        queueR.put(None)
    
class Main:
    def __init__(self, line, queue_detect, img_queue, off_queue):

        self.queue = queue_detect
        self.queue_img = img_queue
        self.queue_off = off_queue
        self.line = line

        self.weights = './weights/230321_best.pt'
        self.yaml = './data/tray.yaml'
        self.img_save_path = '/home/nongshim/바탕화면/Reject_Image'
        self.img = np.zeros((494,659,3),np.uint8)
        self.error_data = [self.line, 'none', 105]

        if self.line == 'A':
            attribute = properties.A
            self.Reject_limit = attribute['reject_limit']
            self.save_img_limit = attribute['save_img_limit']
            camera_num = attribute['cam_num']
            self.camera_setting = attribute['camera_setting']
        elif self.line == 'B':
            attribute = properties.B
            self.Reject_limit = attribute['reject_limit']
            self.save_img_limit = attribute['save_img_limit']
            camera_num = attribute['cam_num']
            self.camera_setting = attribute['camera_setting']
        
        self.load_network()
        self.load_camera(camera_num)
        self.Run()
        print("{} Line Process End...".format(self.line))

    def load_network(self): # YOLO 모델 선언
        device = select_device('')
        self.model = DetectMultiBackend(self.weights, device=device, dnn=False, data=self.yaml, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

    def load_camera(self, camera_num): # 카메라 설정
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

    def make_dir(self): # 폴더 생성 후 경로 반환
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
     
    def save_file(self, img): #이미지 저장
        dirname_reject = self.make_dir()
        name1 = str(time.strftime("%m-%d-%H-%M", time.localtime()))
        name2 = ".jpg"
        confi = round(float(self.confi_int), 1)
        name_orig = str('[' + str(confi) + ']') + '_' + self.line + '_' + name1 + name2
        #print(name_orig)
        
        cv2.imwrite(os.path.join(dirname_reject, name_orig), img)

    def put_queue(self, error_data):
        self.queue.put(error_data)

    def put_queue_img(self, img):
        self.queue_img.put(img)

    def Predict(self):
        try:
            # 이미지를 self.img에 할당하기
            grabResult = self.cameras.RetrieveResult(10000, pylon.TimeoutHandling_ThrowException)
            image_raw = self.converter.Convert(grabResult)
            img_raw = image_raw.GetArray()
            image = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            self.img = image
            #print("{} line Image grab...".format(self.line))

            gn = torch.tensor(self.img.shape)[[1,0,1,0]]
            
            # YOLO 프로그램을 불러와서 목표물 탐색하기
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
                            self.error_data = [self.line, 'reject', self.confi_int]
                            print("{} : Outlier Detected...{}".format(self.line,self.confi_int))
                        else:
                            self.error_data = [self.line, 'ok', 105]
                            
        except:
            self.img = np.zeros((494,659,3),np.uint8)
            self.error_data = [self.line, 'none', 105]
            pass
                
    def Run(self):
        # 메인 프로그램 구동
        while True:
            try:
                quit = self.queue_off.get(timeout=0.001)
                #print(quit)
            except:
                quit = 'on'
                pass
            
            self.Predict()
            self.put_queue_img(self.img)
            self.put_queue(self.error_data)
            
            if quit == 'off':
                break
        
        self.cameras.Close()
        self.cameras.StopGrabbing()


class Reject_sys:
    def __init__(self, Q):
        self.relay = Relay(idVendor=0x16c0, idProduct=0x05df)
        self.relay.state(0, False)
        
        attributeA = properties.A
        attributeB = properties.B
        self.relay_runtime_A = attributeA['relay_runtime']
        self.relay_runtime_B = attributeB['relay_runtime']
        self.relay_A = 'off'
        self.relay_B = 'off'
        self.relay_A_time = time.time()
        self.relay_B_time = time.time()
        self.queue = Q
        self.Time_out = 0.001

        """ 1개 지나가는데 필요한 시간 """
        self.reject_need_time = int(0.050 / self.Time_out)
        
        """ 리젝트까지 필요한 시간 """
        self.standby_time_A = int(attributeA['relay_delay'] / self.Time_out)
        self.standby_time_B = int(attributeB['relay_delay'] / self.Time_out)
        
        self.T_A = [0] * (self.reject_need_time + self.standby_time_A)
        self.T_B = [0] * (self.reject_need_time + self.standby_time_B)

        #print(self.T_A)
        self.get_Queue()
        print("Relay Process End...")

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
            if time.time() - self.relay_A_time > self.relay_runtime_A:
                self.relay.state(1, False)
                self.relay_A = 'off'
                
        if self.relay_B == 'on':
            if time.time() - self.relay_B_time > self.relay_runtime_B:
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
            if time.time() - self.relay_A_time > self.relay_runtime_A:
                self.relay.state(1, False)
                self.relay_A = 'off'

        if self.relay_B == 'on':
            # print(time.time() - self.relay_A_time)
            if time.time() - self.relay_B_time > self.relay_runtime_B:
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

    Check = []
    for process in psutil.process_iter():
       if 'python' in process.name():
          Check.append(str(process.pid))
          
    if len(Check) == 1:
        dp_window()
        print("트레이 비전검사 프로그램 종료")
        time.sleep(2)
    else:
        print("=========================")
        print("                         ")
        print("    중 복   실 행   방 지    ")
        print("                         ")
        print("=========================")
        time.sleep(2)
