
from pathlib import Path
from ultralytics import YOLO

# 입력 이미지 폴더
input_dir = Path("data/frames")

# 데이터셋 저장 경로
output_dir = Path("dataset")
image_dir = output_dir / "images"
label_dir = output_dir / "labels"

image_dir.mkdir(parents=True, exist_ok=True)
label_dir.mkdir(parents=True, exist_ok=True)

# YOLO 모델 불러오기
model = YOLO("yolo11n.pt")

# 추출된 이미지 목록
images = sorted(input_dir.glob("*.jpg"))

print(f"처리할 이미지 수: {len(images)}")

for image_path in images:

    # 객체 탐지
    results = model.predict(
        source=str(image_path),
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # 이미지 복사
    import shutil
    shutil.copy2(
        image_path,
        image_dir / image_path.name
    )

    # YOLO 형식 라벨 파일 생성
    label_path = label_dir / f"{image_path.stem}.txt"

    with open(label_path, "w", encoding="utf-8") as f:
        for box in result.boxes:
            class_id = int(box.cls[0])

            # 정규화된 중심좌표, 너비, 높이
            x, y, w, h = box.xywhn[0].tolist()

            f.write(
                f"{class_id} {x:.6f} {y:.6f} "
                f"{w:.6f} {h:.6f}\n"
            )

    print(f"완료: {image_path.name}")

print("자동 라벨링 완료")
