import subprocess
import json
import numpy as np


def is_video_downscaled(video_path, target_width, target_fps):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,width,height",
            "-print_format",
            "json",
            video_path,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        return False
    str = result.stdout.decode("utf-8")
    data = json.loads(str)

    video_width = data["streams"][0]["width"]
    video_fps = data["streams"][0]["r_frame_rate"]
    print(convert_fps_to_int(video_fps))
    return video_width == target_width and video_fps == target_fps


def downscale_video(source_path, target_path, target_width, target_fps):
    result = subprocess.run(
        [
            "ffmpeg",
            "-i",
            source_path,
            "-vf",
            f"fps={target_fps},scale={target_width}:-1",
            "-c:v",
            "libx264",
            "-crf",
            "17",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ac",
            "2",
            target_path,
        ],
    )
    return result.returncode == 0


def cut_video_segment(source_path, target_path, start_time, duration):
    # ffmpeg -nostdin -i "$input" -ss "$start" -t "$duration"  -c:v libx264 -c:a aac -metadata:s:v rotate="0" "output_${start//[:]/-}_to_${duration//[:]/-}.mp4"
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-i",
            source_path,
            "-ss",
            start_time,
            "-t",
            duration,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-metadata:s:v",
            "rotate=0",
            target_path,
        ],
    )
    return result.returncode == 0


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


def convert_fps_to_int(fps):
    num, denom = fps.split("/")
    return int(int(num) / int(denom))


def format_time(seconds):
    return (
        str(int(seconds // 3600)).zfill(2)
        + ":"
        + str(int((seconds % 3600) // 60)).zfill(2)
        + ":"
        + str(int(seconds % 60)).zfill(2)
    )
