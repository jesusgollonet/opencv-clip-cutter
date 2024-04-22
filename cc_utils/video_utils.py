import subprocess
import json


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
    print(result.stdout)
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


def convert_fps_to_int(fps):
    num, denom = fps.split("/")
    return int(int(num) / int(denom))
