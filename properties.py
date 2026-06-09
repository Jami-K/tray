""" 변수 설정을 위한 파일입니다 """

#A기 설정값입니다
A = {
     'camera_ip': '192.168.70.1',                #카메라 IP 주소
     'camera_setting': './camera_settingA.pfs',  #카메라 설정값
     'reject_limit': 88,     #검사값 감도(0~100)
     'save_img_limit': 80,   #이미지 저장 감도(0~100)
     'relay_runtime': 0.1,   #릴레이 실행 시간(초)
     'relay_delay': 4.2,     #리젝트 신호까지 걸리는 지연 시간(초)
     'img_save_path': '/home/nongshim/바탕화면/Reject_Image'  #불량 이미지 저장 경로
}

#B기 설정값입니다
B = {
     'camera_ip': '192.168.80.1',                #카메라 IP 주소
     'camera_setting': './camera_settingB.pfs',  #카메라 설정값
     'reject_limit': 88,     #검사값 감도(0~100)
     'save_img_limit': 80,   #이미지 저장 감도(0~100)
     'relay_runtime': 0.1,   #릴레이 실행 시간(초)
     'relay_delay': 2.3,     #리젝트 신호까지 걸리는 지연 시간(초)
     'img_save_path': '/home/nongshim/바탕화면/Reject_Image'  #불량 이미지 저장 경로
}
