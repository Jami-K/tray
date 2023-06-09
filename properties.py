""" 변수 설정을 위한 파일입니다 """

#A기 설정값입니다
A = {
     'cam_num': 1,                               #카메라 번호
     'camera_setting': './camera_settingA.pfs',  #카메라 설정값
     'reject_limit': 90,     #리젝트 감도(0~100)
     'save_img_limit': 80,   #이미지 저장 감도(0~100)
     'relay_runtime': 0.5,   #릴레이 실행 시간(초)
     'relay_delay': 4.2      #릴레이 실행까지 걸리는 시간(ms)
}

#B기 설정값입니다
B = {
     'cam_num': 0,                               #카메라 번호
     'camera_setting': './camera_settingB.pfs',  #카메라 설정값
     'reject_limit': 90,     #리젝트 감도(0~100)
     'save_img_limit': 80,   #이미지 저장 감도(0~100)
     'relay_runtime': 0.5,   #릴레이 실행 시간(초)
     'relay_delay': 2.3      #릴레이 실행까지 걸리는 시간(ms)
}
