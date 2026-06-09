import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opencv-clip-cutter-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/opencv-clip-cutter-cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import cv2 as cv
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cc_utils import video_utils as vu

DOWNSCALE_WIDTH = 128
DOWNSCALE_FPS = 2

SMOOTH_SIGMA = 1.5
SENSITIVITY = 0.4  # MAD units above the median noise floor
END_THRESHOLD_RATIO = 0.05
SEGMENT_START_PADDING = 0.0
SEGMENT_END_PADDING = 1.0
MIN_CLIP_DURATION = 1.0  # seconds
MIN_GAP = 1.0  # seconds between segments
WARMUP_SECONDS = 2  # KNN needs a few seconds to stabilize; excluded from stats

THUMBNAIL_COUNT = 20


def build_motion_signal(video_path):
    cap = cv.VideoCapture(video_path)
    bs = cv.createBackgroundSubtractorKNN()
    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    sample_every = max(1, total_frames // THUMBNAIL_COUNT)

    signal = []
    thumbnails = []
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


def detect_segments(signal, fps, sensitivity=SENSITIVITY, min_clip_duration=MIN_CLIP_DURATION, min_gap=MIN_GAP):
    smoothed = gaussian_filter1d(signal, sigma=SMOOTH_SIGMA)

    # Exclude warmup frames so KNN stabilisation doesn't inflate the baseline.
    # Use median + k*MAD (robust to outlier spikes) instead of mean + k*std.
    warmup = int(WARMUP_SECONDS * fps)
    stable = smoothed[warmup:] if len(smoothed) > warmup * 2 else smoothed
    median = np.median(stable)
    mad = np.median(np.abs(stable - median))
    robust_std = 1.4826 * mad  # scale MAD to be comparable to std
    start_threshold = median + sensitivity * robust_std
    end_threshold = median + sensitivity * END_THRESHOLD_RATIO * robust_std

    min_gap_frames = max(1, int(min_gap * fps))
    min_duration_frames = max(1, int(min_clip_duration * fps))

    peaks, _ = find_peaks(smoothed, height=start_threshold, distance=min_gap_frames)

    raw_segments = []
    for peak in peaks:
        left = peak
        while left > 0 and smoothed[left - 1] > end_threshold:
            left -= 1
        right = peak
        while right < len(smoothed) - 1 and smoothed[right + 1] > end_threshold:
            right += 1

        if (right - left) >= min_duration_frames:
            max_time = (len(smoothed) - 1) / fps
            start = max(0.0, left / fps - SEGMENT_START_PADDING)
            end = min(max_time, right / fps + SEGMENT_END_PADDING)
            raw_segments.append((start, end))

    # merge segments closer than min_gap
    segments = []
    for start, end in raw_segments:
        if segments and start - segments[-1][1] < min_gap:
            segments[-1] = (segments[-1][0], max(segments[-1][1], end))
        else:
            segments.append((start, end))

    return segments, smoothed, start_threshold, end_threshold


def get_downscaled_path(source_path):
    base = os.path.splitext(source_path)[0]
    return f"{base}.downscaled.mp4"


def ensure_downscaled(source_path, quiet=False):
    downscaled = get_downscaled_path(source_path)
    if not os.path.exists(downscaled):
        if not quiet:
            print("Downscaling video for analysis...")
        vu.downscale_video(source_path, downscaled, DOWNSCALE_WIDTH, DOWNSCALE_FPS, quiet=quiet)
    return downscaled


def save_plot(signal, smoothed, segments, fps, thumbnails, output_path, sensitivity, min_clip_duration, start_threshold, end_threshold):
    times = np.arange(len(signal)) / fps
    duration = times[-1]
    n = len(thumbnails)

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, n, height_ratios=[3, 1], hspace=0.08)

    ax = fig.add_subplot(gs[0, :])

    ax.plot(times, signal, color="steelblue", linewidth=0.6, alpha=0.35, label="raw signal")
    ax.plot(times, smoothed, color="steelblue", linewidth=1.2, label="smoothed signal")

    labeled_seg = False
    labeled_start = False
    labeled_end = False
    for start, end in segments:
        ax.axvspan(start, end, alpha=0.2, color="orange", label="detected segment" if not labeled_seg else "_")
        ax.axvline(start, color="orange", linewidth=1.2, linestyle="-", label="segment boundary" if not labeled_start else "_")
        ax.axvline(end, color="orange", linewidth=1.2, linestyle="-", label="_")
        labeled_seg = True
        labeled_start = True

    ax.axhline(start_threshold, color="red", linewidth=1, linestyle="--", label=f"start threshold ({start_threshold:.0f}px, sensitivity={sensitivity})")
    ax.axhline(end_threshold, color="salmon", linewidth=1, linestyle=":", label=f"end threshold ({end_threshold:.0f}px)")

    ax.set_xlim(0, duration)
    ax.set_ylabel("white pixels")
    ax.set_title(f"Motion signal — {len(segments)} segment(s) detected  |  sensitivity={sensitivity}  min_clip={min_clip_duration}s")
    ax.set_xticks([])
    ax.legend(loc="upper right", fontsize=8)

    for i, (t, mask_rgb) in enumerate(thumbnails):
        ax_thumb = fig.add_subplot(gs[1, i])
        ax_thumb.imshow(mask_rgb, cmap="gray", vmin=0, vmax=255)
        ax_thumb.set_xticks([])
        ax_thumb.set_yticks([])
        ax_thumb.set_xlabel(f"{t:.0f}s", fontsize=8)
        in_seg = any(start <= t <= end for start, end in segments)
        color = "orange" if in_seg else "#cccccc"
        for spine in ax_thumb.spines.values():
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
    parser.add_argument("--plot", action="store_true", help="Save a detection plot instead of cutting clips")
    parser.add_argument("--sensitivity", type=float, default=SENSITIVITY, help=f"Std deviations above noise floor to start a segment (default: {SENSITIVITY})")
    parser.add_argument("--min-clip-duration", type=float, default=MIN_CLIP_DURATION, dest="min_clip_duration", help=f"Minimum clip length in seconds (default: {MIN_CLIP_DURATION})")
    parser.add_argument("--min-gap", type=float, default=MIN_GAP, dest="min_gap", help=f"Minimum gap between segments in seconds (default: {MIN_GAP})")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: video not found: {args.video_path}", file=sys.stderr)
        sys.exit(1)

    analysis_video = ensure_downscaled(args.video_path)

    print("Analyzing motion...")
    signal, fps, thumbnails = build_motion_signal(analysis_video)
    segments, smoothed, start_threshold, end_threshold = detect_segments(signal, fps, args.sensitivity, args.min_clip_duration, args.min_gap)
    print(f"Detected {len(segments)} segment(s)")

    if args.plot:
        plot_path = os.path.splitext(args.video_path)[0] + ".plot.png"
        save_plot(signal, smoothed, segments, fps, thumbnails, plot_path, args.sensitivity, args.min_clip_duration, start_threshold, end_threshold)
    else:
        print("Clip cutting not yet implemented — use --plot to inspect detection.")


if __name__ == "__main__":
    main()
