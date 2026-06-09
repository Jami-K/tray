import cv2, os, shutil, time
import numpy as np
from datetime import datetime
from pypylon import pylon

class Camera:
    """ 카메라 활성화 """
    def __init__(self, camera_ip, camera_setting, camera_mode='VIDEO'):
        self.camera_ip = camera_ip
        self.camera_setting = camera_setting
        self.camera_mode = camera_mode
        self.load_camera()
        
    """ 카메라 설정 """
    def load_camera(self):
        maxCamerasToUse = 1
        devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        selectedDevice = None
        
        self.cam = None
        
        if len(devices) == 0:
            raise pylon.RuntimeException("\n카메라 네트워크 상태 또는 주소를 확인해주세요.")

        for device in devices:
            if device.GetIpAddress() == self.camera_ip:
                selectedDevice = device
                print('Camera_IP :', selectedDevice.GetIpAddress())
                break

        if selectedDevice is None:
            raise NameError(f"카메라 IP를 찾을 수 없습니다: {self.camera_ip}")

        if selectedDevice is not None:
            try:
                self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(selectedDevice))
                self.cam.Open()
                
            except Exception as e:
                raise NameError(f"카메라 Open() 오류 : {str(e)}")
            
            if self.camera_setting is not None:
                try:
                    pylon.FeaturePersistence.Load(self.camera_setting, self.cam.GetNodeMap(), True)
                except pylon.GenericException as e:
                    raise NameError(f"카메라 pfs 설정파일 오류 : \n{str(e)}")
            
            self.reset_trigger()
            
            if self.camera_mode == 'VIDEO':
                self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            elif self.camera_mode == 'TRIGGER':
                try:
                    self.cam.TriggerSelector.SetValue('FrameStart')
                    self.cam.TriggerSource.SetValue('Line1')
                    self.cam.TriggerActivation.SetValue('RisingEdge')
                    self.cam.TriggerMode.SetValue('On')
                    self.cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
                except pylon.GenericException as e:
                    raise NameError(f"카메라 트리거모드 설정 초기화 오류 : \n{str(e)}")

            try:
                self.converter = pylon.ImageFormatConverter()
                self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            except Exception as e:
                raise NameError(f"카메라 이미지 컨버터 설정 초기화 오류 : \n{str(e)}")   
            
    """ 이미지 생성 """
    def get_img(self, image_no):
        grab_on = 0 #카메라 인식 초기화
        grabResult = 0
        try:
            grabResult = self.cam.RetrieveResult(100, pylon.TimeoutHandling_Return)
            if grabResult.IsValid():
                if grabResult.GrabSucceeded():
                    image_raw = self.converter.Convert(grabResult).GetArray()
                    image_rgb = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
                    grab_on = 2
                    return image_raw, image_rgb, grabResult, grab_on
                else:
                    grab_on = 1  # grab 실패 (프레임 손상 등)
            else:
                if self.cam.IsCameraDeviceRemoved():
                    grab_on = 0  # 네트워크 고장으로 장치 제거됨
                else:
                    grab_on = 1  # 트리거 대기 중 (타임아웃) = 카메라 정상 연결
        except Exception:
            grab_on = 0  # 카메라 미연결 또는 심각한 오류
        return image_no, image_no, grabResult, grab_on

    def reset_trigger(self):
        self.cam.UserOutputValue.SetValue(False)
        
    def cam_trigger(self):
        self.cam.UserOutputValue.SetValue(True)
        time.sleep(0.3)
        self.cam.UserOutputValue.SetValue(False)        
    
    def destroy_cam(self):
        if self.cam is not None:
            try:
                if self.cam.IsGrabbing():
                    self.cam.StopGrabbing()
            except Exception:
                pass

            try:
                if self.cam.IsOpen():
                    self.cam.Close()
            except Exception:
                pass

            self.cam = None
