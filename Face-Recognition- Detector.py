from ultralytics import YOLO
import cv2
import cvzone
# import FaceRecognitionFunction
import math
import pymysql
from PIL import ImageFont, ImageDraw, Image

p_inf = []
player_info = ''

# 커서 생성 (데이터베이스와 상호작용, 데이터베이스 연결을 통해 SQL 쿼리를 실행하고 결과를 처리하기 위한 객체)
cursor = connection.cursor()

# 웹캠으로 인식된 선수와 이름을 비교하여 정보 출력
    for i in range(len(data)): # db에 저장되어 있는 컬럼의 개수 범위
        if classNames[cls] == data[i][1]: # classNames의 이름과 db -> data 컬럼의 이름과 동일하다면 정보 출력 하여라
            print("선수 이름:", data[i][0])
            print("선수 이름:", data[i][1])
            print("나이:", data[i][2])
            print("국적:", data[i][3])
            print("소속팀:", data[i][4])
            print("등번호:", data[i][5])
            print("포지션:", data[i][6])
            p_inf.append(data[i][0])
            p_inf.append(data[i][1])
            p_inf.append(data[i][2])
            p_inf.append(data[i][3])
            p_inf.append(data[i][4])
            p_inf.append(data[i][5])
            p_inf.append(data[i][6])

cap = cv2.VideoCapture(cv2.CAP_DSHOW+0) # for webcam
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("C:/Users/sky/Desktop/study/opencv/FootballPlayer_Face_Recognition_Project/runs/detect/train2/weights/best.pt") # YOLO 모델을 불러옵니다.
# results = model("C:/Users/sky/Desktop/study/opencv/FootballPlayer_Face_Recognition_Project/FootballPlayer_images", show=True) # 

classNames = [
        'AhnHyunBeom',
        'ChoGuesung',
        'CristianoRonaldo',
        'ErlingHaaland',
        'FrenkiedeJong',
        'HwangHeechan',
        'HwangUiJo',
        'JoHyeonwoo',
        'KarimBenzema',
        'KevinDeBruyne',
        'KimMinJae',
        'KimSeunggyu',
        'KylianMbappe',
        'LeeDongGyeong',
        'LeeSoonMin',
        'LionelMessi',
        'MoonSeonMin',
        'NGoloKante',
        'Neymar',
        'OhHyeonGyu',
        'ParkYongWoo',
        'SadioMane',
        'SonHeungMin',
        'YangHyunJun',
]

#print('result ', classNames)
while True:
    frame, img = cap.read() # 카메라 읽어옴
    results = model(img) # 카메라 or 동영상 Yolo 객체 인식 요청

    if frame:
        for r in results:
            boxes = r.boxes # 바운딩박스
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0] # 바운딩 박스 좌표 2곳
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # 좌표받은값이 int형이 아니여서 int형으로 변환
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h)) # 라운드 사각형 cornetRect                
                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100 # 정확도 표기 math.cell

                # Class Name
                cls = int(box.cls[0]) # Yolo에서 받아온 인덱스값      
                box = []          
                font = ImageFont.truetype('.\\fonts\\MaruBuri-Bold.ttf',15)
                cvzone.putTextRect(
                    img,
                    f"{classNames[cls]} {conf}", # cls에서 받아온 값을 classNames 인덱스 값에 맞춰서 반환
                    (max(0, x1), max(35, y1)), # -값이 나올수 있으니 0과 x1값중 높은 값으로 사용함 / 35보다 큰값이 나올수 있으니 35와 y1중 높은값을 사용함 / puTextRect에 넣을 좌표값을 구함 / 박스 크기맞춰서 최소 최대 값 지정
                    scale=1, # 확댁축소 0.5는 축소, 1은 원본 1.5 확대 원본대로 해라
                    thickness=1, # 텍스트 박스두깨
                )
                


    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])

            if conf > 0.5:
                text = p_print()  # 출력할 텍스트
                position = (int(x1), int(y1) - 10)  # 바운딩 박스의 위쪽 중앙 위치로 초기화

                # 바운딩 박스의 중앙 위치 계산
                bbox_center_x = int((x1 + x2) / 2)
                bbox_center_y = int((y1 + y2) / 2)

                # 선수 정보 텍스트 위치 계산 (바운딩 박스 옆쪽)
                text_x = bbox_center_x + 10  # 바운딩 박스의 오른쪽으로 이동
                text_y = bbox_center_y

                font = cv2.FONT_HERSHEY_SIMPLEX  # 폰트 설정
                font_scale = 1  # 폰트 크기
                font_color = (0, 255, 0)  # 텍스트 색상 (BGR 형식, 여기서는 녹색)
                thickness = 2  # 텍스트 두께

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # 선수 정보 텍스트 출력
                cv2.putText(frame, text, (int(x2), int(y1)), font, font_scale, font_color, thickness)
                #print("x1, y1의 값" + str(x1),str(y1))
                #print("text x1, y1의 값" + str(text_x),str(text_y))

    cv2.imshow('test',img)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()
