import cv2 as cv
import numpy as np
from cc_utils import video_utils as vu

source_path = "video/2024-01-13 08.38.31.mov"
target_path = "output/downscaled.mp4"
target_fps = 2
target_width = 128
print(vu.is_video_downscaled(source_path, target_width, target_fps))
if not vu.is_video_downscaled(source_path, target_width, target_fps):
    vu.downscale_video(source_path, target_path, target_width, target_fps)


cap = cv.VideoCapture(target_path)
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

segment_count = 0
# array to store start and end times of each segment
segment_times = []
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
        # store start time
        segment_times.append([i / fps])
    if movement_detected and white_px < movement_end_threshold:
        movement_detected = False
        print("Movement ended at frame:", i)
        # store end time
        segment_times[-1].append(i / fps)

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


print("segment_frames", segment_times)


def format_time(seconds):
    return (
        str(int(seconds // 3600)).zfill(2)
        + ":"
        + str(int((seconds % 3600) // 60)).zfill(2)
        + ":"
        + str(int(seconds % 60)).zfill(2)
    )


# filter out segments that hav e no end time
segment_times = [times for times in segment_times if len(times) == 2]
# write a list of segments to a text file in the format 00:00:00 - 00:00:00
# start_time duration
with open("output/segments.txt", "w") as f:
    for start, end in segment_times:
        # convert end to duration
        duration = end - start
        # convert start and end times to a string in the format 00:00:00
        f.write(f"{format_time(start)} {format_time(duration)}\n")


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


cv.imwrite("output/mosaic.png", make_mosaic(bs_frames, 20))
