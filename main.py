import argparse
import os
import sys

import cv2 as cv
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cc_utils import video_utils as vu

DOWNSCALE_WIDTH = 128
DOWNSCALE_FPS = 2
MOVEMENT_START_THRESHOLD = 30
MOVEMENT_END_THRESHOLD = 20


THUMBNAIL_COUNT = 20


def build_motion_signal(video_path):
    cap = cv.VideoCapture(video_path)
    bs = cv.createBackgroundSubtractorKNN()
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    sample_every = max(1, total_frames // THUMBNAIL_COUNT)

    signal = []
    thumbnails = []  # list of (time_s, fg_mask_rgb)
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        fg_mask = bs.apply(frame)
        signal.append(cv.countNonZero(fg_mask))
        if i % sample_every == 0 and len(thumbnails) < THUMBNAIL_COUNT:
            mask_rgb = cv.cvtColor(fg_mask, cv.COLOR_GRAY2RGB)
            thumbnails.append((i / fps, mask_rgb))

    cap.release()
    return np.array(signal, dtype=float), fps, thumbnails


def detect_segments(signal, fps):
    segments = []
    in_segment = False
    start_frame = None

    for i, val in enumerate(signal):
        if not in_segment and val > MOVEMENT_START_THRESHOLD:
            in_segment = True
            start_frame = i
        elif in_segment and val < MOVEMENT_END_THRESHOLD:
            in_segment = False
            segments.append((start_frame / fps, i / fps))

    return segments


def get_downscaled_path(source_path):
    base = os.path.splitext(source_path)[0]
    return f"{base}.downscaled.mp4"


def ensure_downscaled(source_path):
    downscaled = get_downscaled_path(source_path)
    if not os.path.exists(downscaled):
        print("Downscaling video for analysis...")
        vu.downscale_video(source_path, downscaled, DOWNSCALE_WIDTH, DOWNSCALE_FPS)
    return downscaled


def save_plot(signal, segments, fps, thumbnails, output_path):
    times = np.arange(len(signal)) / fps
    duration = times[-1]
    n = len(thumbnails)

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, n, height_ratios=[3, 1], hspace=0.08)

    ax_signal = fig.add_subplot(gs[0, :])
    ax_signal.plot(times, signal, color="steelblue", linewidth=0.8, label="motion signal")

    labeled = False
    for start, end in segments:
        ax_signal.axvspan(
            start,
            end,
            alpha=0.3,
            color="orange",
            label="detected segment" if not labeled else "_",
        )
        labeled = True

    ax_signal.axhline(
        MOVEMENT_START_THRESHOLD,
        color="red",
        linewidth=1,
        linestyle="--",
        label=f"start threshold ({MOVEMENT_START_THRESHOLD})",
    )
    ax_signal.axhline(
        MOVEMENT_END_THRESHOLD,
        color="salmon",
        linewidth=1,
        linestyle=":",
        label=f"end threshold ({MOVEMENT_END_THRESHOLD})",
    )

    ax_signal.set_xlim(0, duration)
    ax_signal.set_ylabel("white pixels")
    ax_signal.set_title(f"Motion signal — {len(segments)} segment(s) detected")
    ax_signal.set_xticks([])
    ax_signal.legend(loc="upper right")

    for i, (t, mask_rgb) in enumerate(thumbnails):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(mask_rgb, cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{t:.0f}s", fontsize=8)
        # highlight border if inside a detected segment
        in_seg = any(start <= t <= end for start, end in segments)
        color = "orange" if in_seg else "#cccccc"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cut clips from a session video using motion detection."
    )
    parser.add_argument("video_path", help="Path to the source video file")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a detection plot instead of cutting clips",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: video not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    analysis_video = ensure_downscaled(args.video_path)

    print("Analyzing motion...")
    signal, fps, thumbnails = build_motion_signal(analysis_video)
    segments = detect_segments(signal, fps)
    print(f"Detected {len(segments)} segment(s)")

    if args.plot:
        plot_path = os.path.splitext(args.video_path)[0] + ".plot.png"
        save_plot(signal, segments, fps, thumbnails, plot_path)
    else:
        print("Clip cutting not yet implemented — use --plot to inspect detection.")


if __name__ == "__main__":
    main()
