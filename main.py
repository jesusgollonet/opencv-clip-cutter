import cv2 as cv
import numpy as np

video_path = "video/2024-02-22 21.35.35.mov.resized.mp4"

cap = cv.VideoCapture(video_path)
bs = cv.createBackgroundSubtractorKNN()


movement_start_threshold = 30
movement_end_threshold = 20

movement_detected = False

fps = int(cap.get(cv.CAP_PROP_FPS))
print("fps:", fps)

# total number of frames in the video
frames = []
bs_frames = []
non_white_ar = []

# store frames
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
for i in range(total_frames):
    # set the frame position
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        break

    fg_mask = bs.apply(frame)

    frames.append(frame)
    bs_frames.append(fg_mask)

    # count the number of white pixels in the mask
    white_px = cv.countNonZero(fg_mask)
    non_white_ar.append(white_px)

    if (not movement_detected) and white_px > movement_start_threshold:
        movement_detected = True
        print("Movement detected at frame:", i)
    if movement_detected and white_px < movement_end_threshold:
        movement_detected = False
        print("Movement ended at frame:", i)

    cv.putText(
        fg_mask,
        movement_detected.__str__(),
        (10, 10),
        cv.FONT_HERSHEY_PLAIN,
        1,
        (255, 255, 255),
        2,
        cv.FILLED,
    )


# create a blank mosaic with the correct dimensions


def make_mosaic(source_frames, cols):
    initial_frame = source_frames[0]
    # given the total number of frames and the number of columns
    # calculate the number of rows for the mosaic
    # note the +1 to account for the last row

    mosaic_rows = (len(source_frames) // cols) + 1
    frame_h, frame_w = initial_frame.shape[:2]
    # initial tuple will be (frame_h, frame_w, frame_channels) if it's a color image
    # and (frame_h, frame_w) if it's a grayscale image

    # Determine whether the image is grayscale or color
    if len(initial_frame.shape) == 2:
        # Grayscale image
        mosaic_shape = (mosaic_rows * frame_h, cols * frame_w)
    else:
        # Color image, use 3 for RGB channels
        frame_channels = initial_frame.shape[2]  # typically 3 for RGB
        mosaic_shape = (mosaic_rows * frame_h, cols * frame_w, frame_channels)

    mosaic = np.zeros(mosaic_shape, dtype=np.uint8)

    for i, frame in enumerate(source_frames):
        x = int((i % cols) * frame_w)
        y = int((i // cols) * frame_h)
        mosaic[y : y + frame_h, x : x + frame_w] = frame
    return mosaic


cv.imshow("Mosaic", make_mosaic(bs_frames, 20))
cv.waitKey(0)
cv.imshow("Mosaic", make_mosaic(frames, 50))
cv.waitKey(0)
