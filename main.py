import cv2 as cv
from cc_utils import video_utils as vu
from cc_utils import metadata_utils as mu

source_path = "video/2024-01-13 08.38.31.mov"
mu.scaffold(source_path)
# get full path without file extension

# metadata path is the same as the source path without extension

metadata_path = source_path.split(".")[0]
mu.create_folder("output")
# create folder for video


target_path = "video/downscaled.mp4"
clip_cutter_input_video = source_path
target_fps = 2
target_width = 128

if not vu.is_video_downscaled(source_path, target_width, target_fps):
    vu.downscale_video(source_path, target_path, target_width, target_fps)
    clip_cutter_input_video = target_path


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


# filter out segments that hav e no end time
segment_times = [times for times in segment_times if len(times) == 2]
# write a list of segments to a text file in the format 00:00:00 - 00:00:00
# start_time duration
for segment_count, (start, end) in enumerate(segment_times):
    # convert end to duration
    duration = end - start
    vu.cut_video_segment(
        source_path,
        f"output/segment_{segment_count}.mp4",
        vu.format_time(start),
        vu.format_time(duration),
    )

cv.imwrite("output/bs_mosaic.png", vu.make_mosaic(bs_frames, 20))
cv.imwrite("output/mosaic.png", vu.make_mosaic(frames, 20))
