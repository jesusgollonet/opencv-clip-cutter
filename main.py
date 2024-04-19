import cv2 as cv
import numpy as np


video_path = "video/resized-long.mp4"

cap = cv.VideoCapture(video_path)

frames = []
bw_frames = []
bs_frames = []
c = 0

frame_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv.CAP_PROP_FPS))

movement_start_threshold = 30
movement_end_threshold = 20

movement_detected = False

print("fps:", fps)

# total number of frames in the video
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
cols = 20

bs = cv.createBackgroundSubtractorKNN()

non_white_ar = []

# store frames
for i in range(total_frames):
    # set the frame position
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break
    # threshold image
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, bw_frame = cv.threshold(gray_frame, 127, 255, cv.THRESH_BINARY_INV)
    fg_mask = bs.apply(frame)

    frames.append(frame)
    bw_frames.append(bw_frame)
    white_px = cv.countNonZero(fg_mask)
    if (not movement_detected) and white_px > movement_start_threshold:
        movement_detected = True
        print("Movement detected at frame:", i)
    if movement_detected and white_px < movement_end_threshold:
        movement_detected = False
        print("Movement ended at frame:", i)

    non_white_ar.append(white_px)
    cv.putText(
        fg_mask,
        white_px.__str__() + "-> " + movement_detected.__str__(),
        (10, 10),
        cv.FONT_HERSHEY_PLAIN,
        1,
        (255, 255, 255),
        2,
        cv.FILLED,
    )
    bs_frames.append(fg_mask)


# given the total number of frames and the number of columns
# calculate the number of rows for the mosaic
# note the +1 to account for the last row
mosaic_rows = (len(frames) // cols) + 1

# create a blank mosaic with the correct dimensions
mosaic = np.zeros((mosaic_rows * frame_h, cols * frame_w, 3), dtype=np.uint8)
bw_mosaic = np.zeros((mosaic_rows * frame_h, cols * frame_w), dtype=np.uint8)
bs_mosaic = np.zeros((mosaic_rows * frame_h, cols * frame_w), dtype=np.uint8)

for i, frame in enumerate(frames):
    x = int((i % cols) * frame_w)
    y = int((i // cols) * frame_h)
    mosaic[y : y + frame_h, x : x + frame_w] = frame
    bw_mosaic[y : y + frame_h, x : x + frame_w] = bw_frames[i]
    bs_mosaic[y : y + frame_h, x : x + frame_w] = bs_frames[i]

cv.imwrite("output/mosaic.jpg", mosaic)
cv.imwrite("output/mosaic_bw.jpg", bw_mosaic)
cv.imwrite("output/mosaic_bs.jpg", bs_mosaic)
cv.imshow("Mosaic", bs_mosaic)
cv.waitKey(0)
