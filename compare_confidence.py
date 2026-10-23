
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

# 분석 대상 이미지
image_path = "data/frames/frame_000010.jpg"

# YOLO 모델
model = YOLO("yolo11n.pt")

# 비교할 신뢰도 임계값
thresholds = [0.25, 0.50, 0.75]

records = []

for threshold in thresholds:
    results = model.predict(
        source=image_path,
        conf=threshold,
        save=True,
        project="experiments",
        name=f"conf_{threshold}",
        exist_ok=True,
        verbose=False
    )

    result = results[0]

    print(f"\n신뢰도 임계값: {threshold}")

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"  {model.names[class_id]}: "
            f"{confidence:.3f}"
        )

    records.append({
        "threshold": threshold,
        "detected_objects": len(result.boxes)
    })

# CSV 저장
Path("reports").mkdir(exist_ok=True)

df = pd.DataFrame(records)
df.to_csv(
    "reports/confidence_comparison.csv",
    index=False
)

print("\n실험 결과")
print(df)
