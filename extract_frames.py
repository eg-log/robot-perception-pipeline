
import cv2
from pathlib import Path

# 입력 영상과 출력 폴더
video_path = Path("data/videos/road.mp4")
output_dir = Path("data/frames")

output_dir.mkdir(parents=True, exist_ok=True)

# 영상 열기
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise FileNotFoundError(f"영상을 열 수 없습니다: {video_path}")

# 초당 프레임 수 확인
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"영상 FPS: {fps}")

if fps <= 0:
    cap.release()
    raise ValueError("영상 FPS 정보를 확인할 수 없습니다.")

# 1초마다 이미지 한 장 추출
frame_interval = max(1, round(fps))

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = output_dir / f"frame_{saved_count:06d}.jpg"

        cv2.imwrite(str(filename), frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"총 읽은 프레임: {frame_count}")
print(f"저장한 이미지: {saved_count}")
print(f"저장 위치: {output_dir.resolve()}")
