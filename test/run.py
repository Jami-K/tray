from model import LMS, Return_model_L
from alibi_detect.utils.saving import load_detector

from relay import Relay
from pypylon import pylon
from multiprocessing import Process, Queue
import tensorflow as tf
import cv2, os, hid, time
import numpy as np

class Main:
    def __init__(self, window_name, queue):
        self.IMG_SIZE1 = 64
        self.IMG_SIZE2 = 64
        self.queue = queue
        self.save_folder = 'Reject_img'

        if window_name == 'TRAY':
            self.info = {'window_name': window_name, 'GPU_id': '1',
                         'camera_property': 'camera_settingN.pfs',
                         'camera_id': 0, 'IMG_SIZE1': self.IMG_SIZE1, 'IMG_SIZE2': self.IMG_SIZE2}

        print("Alibi Detect is ready...1")
        self.od = Return_model_L()
        self.od = load_detector("tray_od_20epochs.h5")
        print("Alibi Detect is ready...2")

        self.load_camera()
        self.run()

    def load_camera(self):
        maxCamerasToUse = 1
        tlFactory = pylon.TlFactory.GetInstance()
        devices = tlFactory.EnumerateDevices()

        self.cameras = pylon.InstantCameraArray(min(len(devices), maxCamerasToUse))

        for i, self.cam in enumerate(self.cameras):
            self.cam.Attach(tlFactory.CreateDevice(devices[self.info['camera_id']]))

        self.cameras.Open()
        pylon.FeaturePersistence.Load(self.info['camera_property'], self.cam.GetNodeMap(), True)
        # self.cameras.Close()

        self.cameras.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def get_img(self):
        try:
            self.grabResult = self.cameras.RetrieveResult(50000, pylon.TimeoutHandling_ThrowException)
            image_raw = self.converter.Convert(self.grabResult)
            img_raw = image_raw.GetArray()
            img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            self.img = img
        except:
            pass

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


class Reject:
    def __init__(self, queue1, queue2):
        self.relay = Relay(idVendor=0x16c0, idProduct=0x05df)
        self.relay.state(0, False)

        self.relay_B = 'off'
        self.relay_B_time = time.time()

        self.relay_A_time = time.time()
        self.relay_runtime = 0.05
        self.relay_A = 'off'
        self.Time_out = 0.001

        self.reject_need_time = int(0.010 / self.Time_out) #10
        self.standby_time = int(0.055 / self.Time_out) #55

        self.queue1 = queue1
        self.queue2 = queue2

        self.T_A = [0] * (self.reject_need_time + self.standby_time)
        self.T_B = [0] * (self.reject_need_time + self.standby_time)
        print(self.T_A)

        self.main()

    def main(self):
        """ Queue를 받아서 릴레이 가동을 위한 본 프로그램 """
        initial_A, initial_B = 0, 0
        while True:
            answer = ''
            ng_question_A = 'ok'
            ng_question_B = 'ok'
            try:
                answer = self.queue1.get(timeout=self.Time_out)
                if answer is not None:
                    if answer[0][0] == 'R':
                        #print('***** Reject Detected ****** ')
                        if self.relay_A == 'off':
                            self.T_A = self.time_traveler('A', 0, 1)
                            #print('A : Reject signal put...')
                            ng_question_A = 'ng'
                        initial_A = 0

            except:
                pass

            self.relay_off_cal_time()

            if ng_question_A == 'ok':
                self.T_A = self.time_traveler('A', 0, 0)
                initial_A += 1
            if ng_question_B == 'ok':
                self.T_B = self.time_traveler('B', 0, 0)
                initial_B += 1

            if self.T_A[len(self.T_A) - 1] == 1:
                self.state(1)
            if self.T_B[len(self.T_B) - 1] == 1:
                self.state(2)

            if answer is None:
                break

    def time_traveler(self, line, location, pass_):
        if line == 'A':
            temp = self.T_A
        else:
            temp = self.T_B
        temp = temp[:-1]
        temp.insert(location, pass_)

        return temp


    def relay_off_cal_time(self):
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
        if i == 1 and self.relay_A == 'off':
            self.relay.state(1, True)
            self.relay_A = 'on'
            self.relay_A_time = time.time()
            print("***** [[1]] Reject! *****")

        elif i == 2 and self.relay_B == 'off':
            self.relay.state(2, True)
            self.relay_B = 'on'
            self.relay_B_time = time.time()
            print("***** [[2]] Reject! *****")


if __name__ == "__main__":
    # 만약 save_foler가 없으면 폴더 만들기
    if not os.path.exists('Reject_img'):
        os.mkdir('Reject_img')

    Q1, Q2 = Queue(), Queue()
    relayqueue1 = Queue()
    relayqueue2 = Queue()

    p1 = Process(target=Main, args=('TRAY', Q1,))
    reject = Process(target=Reject, args=(relayqueue1, relayqueue2))

    p1.start()
    reject.start()

    while True:
        answer1, answer2 = '', ''
        try:
            answer1 = Q1.get(timeout=0.00001)
            if answer1 is not None:
                relayqueue1.put(answer1)
        except:
            pass

        try:
            answer2 = Q2.get(timeout=0.00001)
            if answer2 is not None:
                relayqueue2.put(answer2)
        except:
            pass

        if answer1 is None:
            Q2.put('off')
            relayqueue1.put(None)
            relayqueue2.put(None)
            break

        if answer2 is None:
            Q1.put('off')
            relayqueue1.put(None)
            relayqueue2.put(None)
            break
