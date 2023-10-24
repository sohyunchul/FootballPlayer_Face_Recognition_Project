from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.pt')

# Training
 # data안에 있는 이미지가지고 학습
results = model.train(
    data = 'C:/Users/sky/Desktop/study/opencv/FootballPlayer_Face_Recognition_Project/opencvfootball-2/data.yaml',
    imgsz = 640, 
    epochs = 10, 
    batch = 8, 
    name = 'Football' 
)