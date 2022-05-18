import cv2, os, hid, time
import numpy as np
from screeninfo import get_monitors
from pypylon import pylon
from time import sleep, localtime, strftime
from multiprocessing import Process, Queue

from tray2 import KK_Keras
from relay import Relay

#Version Checking: 2022.05.18 Modified on Apple Mini

class dp_window:
    def __init__(self, camera_image, Q):

        """ "Network 변수 설정 """
        self.queue = Q
        self.error_data = []
        camera_num = 0
        self.window_name = '14-line'
        self.reject_limit = 0.015
        self.IMG_SIZE1, self.IMG_SIZE2 = 128, 128
        self.img_save_path = 'Reject_Images'
        self.camera_setting = './camera_settingN.pfs'

        """ 화면 출력 변수 설정 """
        self.pg_exit = 0
        self.total_num = 0
        self.reject_num = 0
        self.operate = 'off'
        self.operate_mode = 0
        self.decide_ng = 0
        self.x_max, self.y_max = self.screen_frame()
        self.y_max = round(self.x_max * 0.5)
        window_info = str("Nongshim Noksan Deep Learning Program : ESC = Quit")

        """ 이미지 크기/위치 정의 """
        self.cam_x_size = round(self.x_max*0.587)
        self.cam_y_size = round(self.y_max*0.717)
        bar_x_size = round(self.x_max * 0.31)
        bar_y_size = round(self.y_max * 0.13)
        menu_x = round(self.x_max*0.027)
        self.menu_x = round(self.x_max*0.027)
        numbering_y = round(self.y_max*0.188)
        setting_y = round(self.y_max*0.4054)
        decide_y = round(self.y_max*0.645)
        mode_y = round(self.y_max*0.779)
        set_x_size, set_y_size = round(self.x_max*0.313), round(self.y_max*0.205)
        total_x, total_y = round(self.x_max*0.200), round(self.y_max*0.255)
        reject_x, reject_y = round(self.x_max*0.200), round(self.y_max*0.343)
        limit_x, limit_y = round(self.x_max*0.200), round(self.y_max*0.480)

        """ 배경에 들어갈 이미지 정의 """
        self.img = cv2.resize(camera_image, (self.cam_x_size, self.cam_y_size))
        self.img_ok = cv2.resize(cv2.imread('./display/OK.png'), (bar_x_size, bar_y_size))
        self.img_reject = cv2.resize(cv2.imread('./display/Reject.png'), (bar_x_size, bar_y_size))
        self.setting = cv2.resize(cv2.imread('./display/setting.png'), (set_x_size, set_y_size))
        self.numbering = cv2.resize(cv2.imread('./display/number.png'), (set_x_size, set_y_size))
        self.mode = cv2.resize(cv2.imread('./display/now.png'), (bar_x_size, bar_y_size))
        self.mode2 = cv2.resize(cv2.imread('./display/standby.png'), (bar_x_size, bar_y_size))
        start_mode = cv2.resize(cv2.imread('./display/Main.png'), (self.x_max, self.y_max))
        self.result = start_mode.copy()

        self.result = self.merge_image(self.result, self.mode2, menu_x, mode_y)
        self.result = self.merge_image(self.result, self.img, round(self.x_max*0.37), round(self.y_max*0.188))
        self.result = self.merge_image(self.result, self.setting, menu_x, setting_y)
        self.result = self.merge_image(self.result, self.numbering, menu_x, numbering_y)

        self.load_network() #self.VAE로 할당됨
        self.load_camera(camera_num=camera_num)
        print("Keras Network Model Ready!!")

        while True:
            self.result = self.merge_image(self.result, self.img, round(self.x_max * 0.37), round(self.y_max * 0.188))

            if self.decide_ng == 0:
                self.result = self.merge_image(self.result, self.img_ok, menu_x, decide_y)
            elif self.decide_ng == 1:
                self.result = self.merge_image(self.result, self.img_reject, menu_x, decide_y)

            """ 텍스트 삽입문 들어가야할곳 - total&reject 숫자가 겹쳐지기 때문에 구문 수정이 필요함 """
            self.dp_count(self.numbering, menu_x, numbering_y, self.total_num, self.reject_num, total_x, total_y, reject_x, reject_y)
            cv2.putText(self.result, str(round(self.reject_limit * 1000)), (limit_x, limit_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 7)

            cv2.namedWindow(window_info, flags=cv2.WINDOW_AUTOSIZE)
            cv2.imshow(window_info, self.result)
            #cv2.moveWindow(window_info, 0, 0)

            cv2.setMouseCallback(window_info, self.mouseEvent)

            if self.operate == 'on':
                self.get_predict()
                self.img = cv2.resize(self.img, (self.cam_x_size, self.cam_y_size))
                if self.operate_mode == 1:
                    self.operate = 'off'
                    self.operate_mode = 0
                    self.img = camera_image

            if cv2.waitKey(1) & 0xFF == 27 or self.pg_exit == 1:
                self.queue.put(None)
                break

        self.cameras.StopGrabbing()
        cv2.destoryAllWindows()

    def dp_count(self, background, back_x, back_y, total, reject, t_x, t_y, r_x, r_j):
        self.merge_image(self.result, background, back_x, back_y)
        cv2.putText(self.result, str(total), (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 7)
        cv2.putText(self.result, str(reject), (r_x, r_y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 7)
        
    def screen_frame(self):
        screen = get_monitors()[0]
        print(screen)
        screen = str(screen).split(',')
        screen_width = int(screen[2].split('=')[1])
        screen_height = int(screen[3].split('=')[1])
        return screen_width, screen_height

    def mouseEvent(self, event, x, y, flags, param):
        x_ratio = round(x / self.x_max, 3)
        y_ratio = round(y / self.y_max, 3)

        if event == cv2.EVENT_FLAG_LBUTTON:
            print("Left Mouse Click! My X is {}, My Y is {}".format(x, y))
            print("Left Mouse Click! My X is {}, My Y is {}".format(x_ratio, y_ratio))

            if y_ratio >= 0.54 and y_ratio <= 0.57: # 리젝트 감도 조정 버튼
               if x_ratio >= 0.045 and x_ratio <= 0.166:
                   self.result = self.merge_image(self.result, self.setting, self.menu_x, setting_y)
                   self.reject_limit += 0.001
               if x_ratio >= 0.187 and x_ratio <= 0.308:
                   self.result = self.merge_image(self.result, self.setting, self.menu_x, setting_y)
                   self.reject_limit -= 0.001

            if x_ratio >= 0.282 and x_ratio <= 0.308: # total&reject 초기화 버튼
               if y_ratio >= 0.277 and y_ratio <= 0.323:
                   self.total_num = 0
                   self.reject_num = 0

            if x_ratio >= 0.023 and x_ratio <= 0.333:  # 검사 시작 버튼
                if y_ratio >= 0.783 and y_ratio <= 0.907:
                    if self.operate == 'off':
                        self.operate = 'on'
                        self.result = self.merge_image(self.result, self.mode, self.menu_x, mode_y) #2485 1552
                    else:
                        self.operate_mode = 1
                        self.result = self.merge_image(self.result, self.mode2, self.menu_x, mode_y)

            if x_ratio >= 0.908 and x_ratio <= 0.956 : # 종료 버튼
               if y_ratio >= 0.026 and y_ratio <= 0.110 :
                  print("Program Killed by Touch-User")
                  self.pg_exit = 1


    def merge_image(self, back, front, x, y):    #back이미지에 front이미지를 해당 좌표자리에 병합
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

    def load_camera(self, camera_num):     #카메라 설정 불러오기
        maxCamerasToUse = 1
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        self.cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))
        for i, self.cam in enumerate(self.cameras):
            self.cam.Attach(tlFactory.CreateDevice(devices[camera_num]))
        self.cameras.Open()
        pylon.FeaturePersistence.Load(self.camera_setting, self.cam.GetNodeMap(), True)
        self.cameras.Close()
        self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def load_network(self):    #신경망 구조 불러오기
       self.VAE = KK_Keras((self.IMG_SIZE1, self.IMG_SIZE2, 3), 32)

    def get_predict(self):    #카메라로부터 이미지를 불러와 예측값을 불러옴
        self.grabResult = self.cameras.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if self.grabResult.GrabSucceeded():
            image_raw = self.converter.Convert(self.grabResult)
            image = image_raw.GetArray()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.img = image
            self.total_num += 1

            k_img = self.img.astype(np.float32) / 255.
            k_resized = cv2.resize(k_img, (self.IMG_SIZE1, self.IMG_SIZE2), interpolation=cv2.INTER_LINEAR)
            k_reshape = k_resized.reshape((1,) + (self.IMG_SIZE1, self.IMG_SIZE2) + (3,))

            detect = self.VAE.predict(k_reshape)
            detect_mae_loss = np.mean(np.power(detect - k_resized, 2), axis=1)
            detect_mae_loss = detect_mae_loss.reshape((-1))

            print("Detected : {} // Limit : {}".format(round(detect_mae_loss.max(),2), self.reject_limit)) # 검사추정치 출력하기

            self.decide_ng = 0
            if detect_mae_loss.max() > self.reject_limit:
                #self.save_img(self.img)
                self.reject_num += 1
                self.decide_ng = 1
                error_data = ['Reject']
                self.put_queue(error_data)

    def put_queue(self, error_data):
        self.queue.put(error_data)

    def make_dir(self):
        dir_path = self.img_save_path

        if os.path.exists(dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(strftime("%H", localtime()))):
            dirname_reject = dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(strftime("%H", localtime()))
        else:
            if os.path.exists(dir_path + "/" + str(strftime("%Y-%m-%d", localtime()))):
                os.mkdir(dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(strftime("%H", localtime())))
                dirname_reject = dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(strftime("%H", localtime()))
            else:
                os.mkdir(dir_path + "/" + str(strftime("%Y-%m-%d", localtime())))
                os.mkdir(dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(strftime("%H", localtime())))
                dirname_reject = dir_path + "/" + str(strftime("%Y-%m-%d", localtime())) + "/" + str(strftime("%H", localtime()))
            print("\nThe New Folder For saving Rejected image is Maked...\n")
        return dirname_reject

    def save_img(self, img):
       dirname_reject = self.make_dir()
       name = str(strftime("%m-%d-%H-%M-%S", localtime())) + ".jpg"
       cv2.imwrite(os.path.join(dirname_reject, name), img)

class Reject_sys:
    def __init__(self, Q):
        self.relay = Relay(idVendor=0x16c0, idProduct=0x05df)
        self.relay.state(0, False)
        self.relay.h.close()
        
        self.relay_runtime = 0.05
        self.relay = 'off'
        self.relay_time = time.time()

        self.queue = Q
        self.Time_out = 0.001

        """ 제품 한개가 지나가는데 필요한 시간 """
        self.reject_need_time = int(0.135 / self.Time_out)
        
        """ 리젝트까지 필요한 시간 """
        self.standby_time = int(0.001 / self.Time_out)

        self.T = [0] * (self.reject_need_time + self.standby_time)

        print(self.T)

        self.get_Queue()


    def get_Queue(self):
        initial = 0
        while True:
            answer = [[0]]
            ng_question = 'ok'
            
            time.sleep(self.Time_out)
            
            for i in range(self.queue.qsize()):
                answer = self.queue.get()
                if i == 0:
                    answer = answer_
                    
            if answer is not None:
                if answer[0] == 'Reject':
                    if self.relay =='off':
                        self.T = self.time_traveler('A', 0, 1)
                        print('Reject Signal put...')
                        prtin("*"*20)
                        ng_question = 'ng'
                    initial = 0
            
            self.relay_off_cal_time()
            
            if ng_question == 'ok':
                self.T = self.time_traveler('A', 0, 0)
                initial += 1
                
            if initial == 100:
                initial = 0
            
            if self.T[len(self.T) - 1] == 1:
                self.state(1)
                
            if answer is None:
                break

    def time_traveler(self, line, location, passs):
        if line == 'A':
            temp = self.T
        temp = temp[:-1]
        temp.insert(location, passs)
        return temp

    def relay_off_cal_time(self):
        if self.relay == 'on':
            if time.time() - self.relay_time > self.relay_runtime:
                self.relay = Relay(idVendor=0x16c0, idProduct=0x05df)
                self.relay.state(1, False)
                self.relay = 'off'
                self.relay.h.close()

    def state(self, i):
        if i == 1 and self.relay == 'off':
            self.relay = Relay(idVendor=0x16c0, idProduct=0x05df)
            self.relay.state(1, True)
            self.relay = 'on'
            self.relay_time = time.time()
            self.relay.h.close()

def main():

    queueA, queueR = Queue(), Queue()
    image = cv2.imread('./display/Loading.png')

    A = Process(target=dp_window, args=(image, queueA,))
    R = Process(target=Reject_sys, args=(queueR,))

    A.start()
    R.start()

    while True:
        answer = ''
        
        for i in range(queueA.qsize()):
            answer = queueA.get()
            if i == 0:
                print(queueA.qsize())
                if answer is not None:
                    queueR.put(answer)
                    
        if answer is None:
            queueR.put(None)
            break


if __name__ == "__main__":
    main()
