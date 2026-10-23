
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

model = YOLO("yolo11n.pt")

image_dir = Path("data/frames")
images = sorted(image_dir.glob("*.jpg"))

thresholds = [0.25, 0.50, 0.75]

# COCO 클래스: 사람, 자전거, 자동차, 오토바이, 버스, 트럭
target_classes = [0, 1, 2, 3, 5, 7]

records = []

for threshold in thresholds:
    print(f"\n임계값: {threshold}")

    for image_path in images:
        results = model.predict(
            source=str(image_path),
            conf=threshold,
            classes=target_classes,
            verbose=False
        )

        result = results[0]

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            records.append({
                "image": image_path.name,
                "threshold": threshold,
                "class_name": model.names[class_id],
                "confidence": round(confidence, 4)
            })

        print(
            f"{image_path.name}: "
            f"{len(result.boxes)}개 탐지"
        )

Path("reports").mkdir(exist_ok=True)

df = pd.DataFrame(
    records,
    columns=[
        "image", "threshold",
        "class_name", "confidence"
    ]
)

df.to_csv(
    "reports/dataset_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n임계값별 탐지 결과")
print(df.groupby("threshold").size())
