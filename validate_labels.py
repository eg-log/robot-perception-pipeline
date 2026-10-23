
from pathlib import Path
import pandas as pd

image_dir = Path("dataset/images")
label_dir = Path("dataset/labels")

records = []
errors = []

images = sorted(image_dir.glob("*.jpg"))

for image_path in images:
    label_path = label_dir / f"{image_path.stem}.txt"

    if not label_path.exists():
        errors.append(f"라벨 파일 없음: {image_path.name}")
        continue

    lines = label_path.read_text(
        encoding="utf-8"
    ).splitlines()

    if not lines:
        records.append({
            "image": image_path.name,
            "object_count": 0
        })
        continue

    valid_count = 0

    for line_number, line in enumerate(lines, start=1):
        values = line.split()

        if len(values) != 5:
            errors.append(
                f"{label_path.name}:{line_number} 형식 오류"
            )
            continue

        try:
            class_id = int(values[0])
            x, y, w, h = map(float, values[1:])

            valid = (
                class_id >= 0
                and 0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
                and 0 <= x - w / 2
                and x + w / 2 <= 1
                and 0 <= y - h / 2
                and y + h / 2 <= 1
            )

            if not valid:
                errors.append(
                    f"{label_path.name}:{line_number} 좌표 오류"
                )
            else:
                valid_count += 1

        except ValueError:
            errors.append(
                f"{label_path.name}:{line_number} 숫자 변환 오류"
            )

    records.append({
        "image": image_path.name,
        "object_count": valid_count
    })

df = pd.DataFrame(
    records,
    columns=["image", "object_count"]
)

Path("reports").mkdir(exist_ok=True)

df.to_csv(
    "reports/label_statistics.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"전체 이미지: {len(images)}")
print(f"검증한 라벨 파일: {len(records)}")
print(f"정상 형식의 객체 수: {df['object_count'].sum()}")
print(f"발견한 형식·좌표 오류: {len(errors)}")

for error in errors:
    print("오류:", error)

print("통계 저장 완료")
