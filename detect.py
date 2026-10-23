
from ultralytics import YOLO

# 1. 사전 학습된 YOLO 모델 불러오기
model = YOLO("yolo11n.pt")

# 2. 이미지에서 객체 탐지
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    conf=0.25,
    save=True,
    project="results",
    name="first_detection"
)

# 3. 탐지된 객체의 이름과 신뢰도 출력
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"객체: {model.names[class_id]}, "
            f"신뢰도: {confidence:.2f}"
        )
