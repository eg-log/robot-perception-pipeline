
from pathlib import Path
import csv
from ultralytics import YOLO

# ==========================================
# 1. 경로 및 실험 설정
# ==========================================

BASE = Path(__file__).resolve().parent
IMAGE_DIR = BASE / "data" / "frames"
GT_DIR = BASE / "ground_truth" / "labels"
REPORT_DIR = BASE / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(BASE / "yolo11n.pt"))

thresholds = [0.25, 0.50, 0.75]
IOU_THRESHOLD = 0.5


# ==========================================
# 2. 좌표 변환 및 IoU 계산
# ==========================================

def to_xyxy(x, y, w, h):
    return (
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    )


def calculate_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = (
        max(0, x2 - x1) *
        max(0, y2 - y1)
    )

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


# ==========================================
# 3. 정답 라벨 읽기
# ==========================================

def load_ground_truth(path):
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        values = line.split()

        if not values:
            continue

        if len(values) != 5:
            raise ValueError(f"잘못된 라벨: {path}")

        class_id = int(values[0])

        # Make Sense에서 car 클래스 번호는 0
        if class_id != 0:
            raise ValueError(
                f"예상하지 못한 클래스: {path}"
            )

        x, y, w, h = map(float, values[1:])
        boxes.append(to_xyxy(x, y, w, h))

    return boxes


# ==========================================
# 4. 이미지 및 라벨 자동 확인
# ==========================================

images = sorted(IMAGE_DIR.glob("*.jpg"))

if not images:
    raise FileNotFoundError(
        f"분석할 이미지가 없습니다: {IMAGE_DIR}"
    )

total_images = len(images)

missing_labels = [
    image.name
    for image in images
    if not (GT_DIR / f"{image.stem}.txt").exists()
]

unlabeled_count = len(missing_labels)
labeled_count = total_images - unlabeled_count

print(f"\n전체 이미지: {total_images}장")
print(f"라벨 파일 존재: {labeled_count}장")
print(f"라벨 파일 없음: {unlabeled_count}장")

if unlabeled_count > 0:
    print("\n라벨 파일이 없는 이미지:")
    for name in missing_labels:
        print(f" - {name}")

    question = (
        f"\n라벨이 없는 {unlabeled_count}장에 "
        "정답으로 표시할 자동차가 없는 것을 "
        "확인했나요? (yes/no): "
    )
else:
    question = (
        f"\n전체 {total_images}장의 "
        "자동차 라벨링을 검토했나요? "
        "(yes/no): "
    )

answer = input(question)

if answer.strip().lower() != "yes":
    raise SystemExit(
        "정답 라벨 검토 후 다시 실행하십시오."
    )


# ==========================================
# 5. 임계값별 YOLO 모델 평가
# ==========================================

rows = []

for threshold in thresholds:
    tp = fp = fn = 0

    for image_path in images:
        gt = load_ground_truth(
            GT_DIR / f"{image_path.stem}.txt"
        )

        result = model.predict(
            source=str(image_path),
            conf=threshold,
            classes=[2],  # COCO 자동차 클래스
            verbose=False
        )[0]

        predictions = [
            to_xyxy(*box.xywhn[0].tolist())
            for box in result.boxes
        ]

        # 정답과 예측의 IoU 계산
        candidates = []

        for pi, pred in enumerate(predictions):
            for gi, truth in enumerate(gt):
                iou = calculate_iou(pred, truth)

                if iou >= IOU_THRESHOLD:
                    candidates.append((iou, pi, gi))

        # 일대일 매칭
        matched_pred = set()
        matched_gt = set()

        for iou, pi, gi in sorted(
            candidates,
            reverse=True
        ):
            if (
                pi not in matched_pred
                and gi not in matched_gt
            ):
                matched_pred.add(pi)
                matched_gt.add(gi)

        tp += len(matched_pred)
        fp += len(predictions) - len(matched_pred)
        fn += len(gt) - len(matched_gt)

    # 평가 지표 계산
    precision = (
        tp / (tp + fp)
        if tp + fp else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall else 0
    )

    rows.append({
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    })

    print(
        f"conf={threshold} | "
        f"TP={tp} FP={fp} FN={fn} | "
        f"Precision={precision:.3f}, "
        f"Recall={recall:.3f}, "
        f"F1={f1:.3f}"
    )


# ==========================================
# 6. CSV 결과 저장
# ==========================================

output = REPORT_DIR / "evaluation_results.csv"

with output.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=rows[0].keys()
    )

    writer.writeheader()
    writer.writerows(rows)

print(f"\n평가 결과 저장 완료: {output}")
