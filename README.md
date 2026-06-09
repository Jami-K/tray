[26.06.09] Patch

카메라 연결
load_camera() → img_grab.py의 Camera 클래스로 교체 (IP 기반 선택)
properties.py에서 camera_ip, camera_setting 로드
grabResult.Release() 추가 (메모리 누수 수정)
카메라 종료 순서 정상화 (destroy_cam())
카메라 재연결 자동화
시작 시 미연결 → _connect_camera_with_retry() 로 연결될 때까지 대기
구동 중 연결 끊김 → 30프레임 후 _reconnect_camera() 자동 복구
YOLO 모델은 메모리 유지, 카메라만 재연결
I/O 상태 UI 표시
우측 흰색 패널 하단에 Camera A / Camera B / Relay 상태 인디케이터 추가
초록(정상) / 노랑(재연결 중) / 빨강(오류) 색상 표시
버그 수정
time_traveler B 라인 오타 ('A' → 'B') — B 라인 리젝트 미동작 수정
relay_off_cal_time, state 중복 메서드 제거
relay is None guard 추가
properties.py 통합
cam_num → camera_ip
relay_runtime, relay_delay, img_save_path 하드코딩 제거
모든 설정값을 properties.py 한 곳에서 제어
이미지 저장 개선
폴더 구조: 날짜 → 라인 → 날짜 (Option B)
파일명: 분 단위 → ms 단위 (14-30-25-123.jpg)
365일 이전 폴더 자동 삭제 (프로그램 시작 시 1회)
make_dir() 간소화 (os.makedirs(exist_ok=True))
기타
initial_B 리셋 누락 수정
from properties import 통합으로 설정 일원화
