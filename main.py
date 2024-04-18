import sys
import cv2 as cv
import numpy as np


video_path = "video/resized.mp4"

cap = cv.VideoCapture(video_path)

frames = []
c = 0

frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# total number of frames in the video
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
cols = 20
print("Total frames:", total_frames)

resized_w = int(frame_width / 2)
resized_h = int(frame_height / 2)

print("Resized frame width:", resized_w, "Resized frame height:", resized_h)

for i in range(total_frames):
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break

    # resized_frame = cv.resize(frame, (resized_w, resized_h))
    frames.append(resized_frame)


print(len(frames))
mosaic_rows = (len(frames) // cols) + 1
print(mosaic_rows, cols)

mosaic = np.zeros((mosaic_rows * resized_h, cols * resized_w, 3), dtype=np.uint8)
print(mosaic.shape)

for i, frame in enumerate(frames):
    x = int((i % cols) * resized_w)
    y = int((i // cols) * resized_h)
    mosaic[y : y + resized_h, x : x + resized_w] = frame

cv.imwrite("output/mosaic.jpg", mosaic)
cv.imshow("Mosaic", mosaic)
