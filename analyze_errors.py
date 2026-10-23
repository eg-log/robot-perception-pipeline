
from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

IMAGE_DIR = BASE / "data" / "frames"
GT_DIR = BASE / "ground_truth" / "labels"
OUTPUT_DIR = BASE / "reports" / "error_images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(BASE / "yolo11n.pt"))

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5


def to_xyxy(x, y, w, h):
    return (
        x - w / 2, y - h / 2,
        x + w / 2, y + h / 2
    )


def calculate_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2-x1) * max(0, y2-y1)

    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


def load_gt(path):
    boxes = []

    if not path.exists():
        return boxes

    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue

        class_id = int(values[0])
        if class_id != 0:
            raise ValueError(f"예상하지 못한 클래스: {path}")

        x, y, w, h = map(float, values[1:])
        boxes.append(to_xyxy(x, y, w, h))

    return boxes


def draw_box(image, box, color, label):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box

    p1 = (int(x1 * width), int(y1 * height))
    p2 = (int(x2 * width), int(y2 * height))

    cv2.rectangle(image, p1, p2, color, 2)
    cv2.putText(
        image, label,
        (p1[0], max(20, p1[1] - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, color, 2
    )


records = []

images = sorted(IMAGE_DIR.glob("*.jpg"))

for image_path in images:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {image_path}")

    gt = load_gt(GT_DIR / f"{image_path.stem}.txt")

    result = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        classes=[2],
        verbose=False
    )[0]

    predictions = [
        to_xyxy(*box.xywhn[0].tolist())
        for box in result.boxes
    ]

    candidates = []

    for pi, pred in enumerate(predictions):
        for gi, truth in enumerate(gt):
            iou = calculate_iou(pred, truth)
            if iou >= IOU_THRESHOLD:
                candidates.append((iou, pi, gi))

    matched_pred = set()
    matched_gt = set()

    for iou, pi, gi in sorted(candidates, reverse=True):
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)

    tp = len(matched_pred)
    fp = len(predictions) - tp
    fn = len(gt) - len(matched_gt)

    records.append({
        "image": image_path.name,
        "TP": tp,
        "FP": fp,
        "FN": fn
    })

    # 오류가 있는 프레임만 이미지로 저장
    if fp > 0 or fn > 0:
        for pi, box in enumerate(predictions):
            if pi in matched_pred:
                draw_box(image, box, (0, 255, 0), "TP")
            else:
                draw_box(image, box, (0, 0, 255), "FP")

        for gi, box in enumerate(gt):
            if gi not in matched_gt:
                draw_box(image, box, (255, 0, 0), "FN")

        cv2.imwrite(
            str(OUTPUT_DIR / image_path.name),
            image
        )

        print(
            f"{image_path.name}: "
            f"TP={tp}, FP={fp}, FN={fn}"
        )


csv_path = BASE / "reports" / "error_analysis.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["image", "TP", "FP", "FN"]
    )
    writer.writeheader()
    writer.writerows(records)

print(f"\n오류 분석 완료: {csv_path}")
