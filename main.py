"""Video -> sampled frames -> YOLO pseudo-labels -> validation summary.

This pipeline does not modify manually annotated ground truth or previous reports.
Run from anywhere: python main.py --video data/videos/road.mp4
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent


def resolve_input(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BASE / path


def extract_frames(video: Path, frames_dir: Path, seconds: float) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("영상 FPS를 확인할 수 없습니다.")
    interval = max(1, round(fps * seconds))
    count, saved = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count % interval == 0:
                target = frames_dir / f"frame_{saved:06d}.jpg"
                if not cv2.imwrite(str(target), frame):
                    raise RuntimeError(f"이미지 저장 실패: {target}")
                saved += 1
            count += 1
    finally:
        cap.release()
    return saved, fps


def main() -> None:
    parser = argparse.ArgumentParser(description="영상 자동 라벨링 파이프라인")
    parser.add_argument("--video", default="data/videos/road.mp4")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--interval", type=float, default=1.0, help="추출 간격(초), 고정 FPS 가정")
    parser.add_argument("--out", default="pipeline_output")
    parser.add_argument("--car-only", action="store_true", help="자동차(car)만 라벨링")
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        parser.error("--conf는 0 초과 1 이하이어야 합니다.")
    if args.interval <= 0:
        parser.error("--interval은 0보다 커야 합니다.")

    video = resolve_input(args.video)
    if not video.is_file():
        raise FileNotFoundError(f"영상 파일 없음: {video}")
    out = resolve_input(args.out)
    # Separate output prevents clobbering source annotations/reports.
    if out.resolve() in (BASE.resolve(), video.parent.resolve()):
        raise ValueError("출력 경로를 프로젝트 루트나 입력 영상 폴더로 지정할 수 없습니다.")
    if out.exists():
        raise FileExistsError(f"출력 폴더가 이미 존재합니다: {out}\n--out 옵션으로 새 폴더명을 사용하십시오.")
    frames_dir = out / "frames"
    images_dir = out / "dataset" / "images"
    labels_dir = out / "dataset" / "labels"
    reports_dir = out / "reports"
    for folder in (frames_dir, images_dir, labels_dir, reports_dir):
        folder.mkdir(parents=True, exist_ok=True)

    count, fps = extract_frames(video, frames_dir, args.interval)
    if not count:
        raise RuntimeError("추출된 이미지가 없습니다.")
    model_path = resolve_input(args.model)
    model = YOLO(str(model_path if model_path.exists() else args.model))
    records = []
    total_objects = 0
    for frame in sorted(frames_dir.glob("*.jpg")):
        result = model.predict(source=str(frame), conf=args.conf,
                               classes=[2] if args.car_only else None,
                               verbose=False)[0]
        shutil.copy2(frame, images_dir / frame.name)
        label_path = labels_dir / f"{frame.stem}.txt"
        with label_path.open("w", encoding="utf-8") as f:
            for box in result.boxes:
                cls = int(box.cls[0]); x, y, w, h = box.xywhn[0].tolist()
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        n = len(result.boxes)
        total_objects += n
        records.append({"image": frame.name, "object_count": n})
        print(f"{frame.name}: {n}개")

    with (reports_dir / "label_statistics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "object_count"])
        writer.writeheader(); writer.writerows(records)
    print(f"완료: FPS {fps:.2f}, 이미지 {count}장, 자동 라벨 {total_objects}개")
    print(f"출력 경로: {out}")
    print("주의: 자동 라벨은 검수되지 않은 pseudo-label입니다. Precision/Recall은 별도 Ground Truth 평가가 필요합니다.")


if __name__ == "__main__":
    main()
