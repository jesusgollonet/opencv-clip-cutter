import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opencv-clip-cutter-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/opencv-clip-cutter-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    name: str = ""

    @property
    def duration(self):
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    duration: float
    frame_count: int | None


def parse_fps(value):
    if "/" not in value:
        return float(value)

    numerator, denominator = value.split("/", 1)
    denominator_float = float(denominator)
    if denominator_float == 0:
        raise ValueError(f"Invalid ffprobe frame rate: {value}")
    return float(numerator) / denominator_float


def get_video_metadata(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=r_frame_rate,duration,nb_frames",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout.decode("utf-8"))
    stream = payload["streams"][0]
    return VideoMetadata(
        fps=parse_fps(stream["r_frame_rate"]),
        duration=float(stream["duration"]),
        frame_count=int(stream["nb_frames"]) if stream.get("nb_frames") else None,
    )


def normalize_header(value):
    return value.strip().lower().replace(" ", "_")


def read_losslesscut_csv(csv_path, metadata, label_units):
    with open(csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no CSV header")

        header_map = {normalize_header(name): name for name in reader.fieldnames}
        start_key = header_map.get("start") or header_map.get("segment_start")
        end_key = header_map.get("end") or header_map.get("segment_end")
        name_key = header_map.get("name") or header_map.get("segment_name")

        if not start_key or not end_key:
            raise ValueError(
                f"{csv_path} must include Start/End columns or segment start/segment end columns"
            )

        raw_segments = []
        for row in reader:
            start_text = row[start_key].strip()
            end_text = row[end_key].strip()
            if not start_text or not end_text:
                continue
            raw_segments.append(
                Segment(
                    start=float(start_text),
                    end=float(end_text),
                    name=row.get(name_key, "").strip() if name_key else "",
                )
            )

    if not raw_segments:
        return []

    units = label_units
    if units == "auto":
        max_end = max(segment.end for segment in raw_segments)
        units = "seconds" if max_end <= metadata.duration + 1.0 else "frames"

    if units == "frames":
        return [
            Segment(segment.start / metadata.fps, segment.end / metadata.fps, segment.name)
            for segment in raw_segments
        ]
    if units == "seconds":
        return raw_segments

    raise ValueError(f"Unsupported label units: {label_units}")


def iou(a, b):
    intersection = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    if union <= 0:
        return 0.0
    return intersection / union


def match_segments(expected, detected, threshold):
    candidates = []
    for expected_index, expected_segment in enumerate(expected):
        for detected_index, detected_segment in enumerate(detected):
            score = iou(expected_segment, detected_segment)
            if score >= threshold:
                candidates.append((score, expected_index, detected_index))

    matches = []
    used_expected = set()
    used_detected = set()
    for score, expected_index, detected_index in sorted(candidates, reverse=True):
        if expected_index in used_expected or detected_index in used_detected:
            continue
        used_expected.add(expected_index)
        used_detected.add(detected_index)
        matches.append((expected_index, detected_index, score))

    missed = [index for index in range(len(expected)) if index not in used_expected]
    false_positives = [index for index in range(len(detected)) if index not in used_detected]
    return matches, missed, false_positives


def find_label_sets(path):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() == ".csv":
            video_path = path.with_suffix("")
            if video_path.exists() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                return [(video_path, path)]
            raise ValueError(f"Could not infer video path for label file: {path}")
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            csv_path = Path(str(path) + ".csv")
            if csv_path.exists():
                return [(path, csv_path)]
            raise ValueError(f"Missing label CSV next to video: {csv_path}")
        raise ValueError(f"Unsupported input file: {path}")

    label_sets = []
    for csv_path in sorted(path.rglob("*.csv")):
        video_path = Path(str(csv_path)[: -len(".csv")])
        if video_path.exists() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
            label_sets.append((video_path, csv_path))
    return label_sets


def run_detection(video_path, sensitivity, min_clip_duration, min_gap):
    from main import build_motion_signal, detect_segments, ensure_downscaled

    analysis_video = ensure_downscaled(str(video_path), quiet=True)
    signal, fps, _ = build_motion_signal(analysis_video)
    segments, _, _, _ = detect_segments(
        signal,
        fps,
        sensitivity=sensitivity,
        min_clip_duration=min_clip_duration,
        min_gap=min_gap,
    )
    return [Segment(start, end) for start, end in segments]


def format_segment(segment):
    return f"{segment.start:7.2f}s-{segment.end:7.2f}s"


def evaluate_video(video_path, csv_path, args):
    metadata = get_video_metadata(video_path)
    expected = read_losslesscut_csv(csv_path, metadata, args.label_units)
    detected = run_detection(
        video_path,
        sensitivity=args.sensitivity,
        min_clip_duration=args.min_clip_duration,
        min_gap=args.min_gap,
    )
    matches, missed, false_positives = match_segments(expected, detected, args.iou_threshold)
    average_iou = sum(score for _, _, score in matches) / len(matches) if matches else 0.0

    print(f"\n{video_path}")
    print(f"  labels:          {csv_path}")
    print(f"  manual segments: {len(expected)}")
    print(f"  detected:        {len(detected)}")
    print(f"  matched:         {len(matches)}")
    print(f"  missed:          {len(missed)}")
    print(f"  false positives: {len(false_positives)}")
    print(f"  average IoU:     {average_iou:.3f}")

    if args.details:
        for expected_index, detected_index, score in sorted(matches):
            print(
                "  match:           "
                f"manual #{expected_index + 1} {format_segment(expected[expected_index])} "
                f"<-> detected #{detected_index + 1} {format_segment(detected[detected_index])} "
                f"IoU={score:.3f}"
            )
        for expected_index in missed:
            print(f"  missed:          manual #{expected_index + 1} {format_segment(expected[expected_index])}")
        for detected_index in false_positives:
            print(f"  false positive:  detected #{detected_index + 1} {format_segment(detected[detected_index])}")

    return {
        "manual": len(expected),
        "detected": len(detected),
        "matched": len(matches),
        "missed": len(missed),
        "false_positives": len(false_positives),
        "average_iou": average_iou,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate detected motion segments against LosslessCut CSV labels."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="video/pre-segmented",
        help="Video file, LosslessCut CSV file, or directory to scan (default: video/pre-segmented)",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Minimum IoU for a match (default: 0.5)")
    parser.add_argument(
        "--label-units",
        choices=("auto", "frames", "seconds"),
        default="auto",
        help="Units used by the label CSV Start/End columns (default: auto)",
    )
    parser.add_argument("--sensitivity", type=float, default=0.4, help="Detector sensitivity (default: 0.4)")
    parser.add_argument(
        "--min-clip-duration",
        type=float,
        default=1.0,
        dest="min_clip_duration",
        help="Detector minimum clip duration in seconds (default: 1.0)",
    )
    parser.add_argument("--min-gap", type=float, default=1.0, dest="min_gap", help="Detector minimum gap in seconds (default: 1.0)")
    parser.add_argument("--details", action="store_true", help="Print per-segment matches, misses, and false positives")
    args = parser.parse_args()

    label_sets = find_label_sets(args.path)
    if not label_sets:
        print(f"No video/CSV label pairs found under {args.path}", file=sys.stderr)
        return 1

    totals = {
        "manual": 0,
        "detected": 0,
        "matched": 0,
        "missed": 0,
        "false_positives": 0,
        "weighted_iou": 0.0,
    }
    for video_path, csv_path in label_sets:
        result = evaluate_video(video_path, csv_path, args)
        totals["manual"] += result["manual"]
        totals["detected"] += result["detected"]
        totals["matched"] += result["matched"]
        totals["missed"] += result["missed"]
        totals["false_positives"] += result["false_positives"]
        totals["weighted_iou"] += result["average_iou"] * result["matched"]

    precision = totals["matched"] / totals["detected"] if totals["detected"] else 0.0
    recall = totals["matched"] / totals["manual"] if totals["manual"] else 0.0
    average_iou = totals["weighted_iou"] / totals["matched"] if totals["matched"] else 0.0

    print("\nSummary")
    print(f"  videos:          {len(label_sets)}")
    print(f"  manual segments: {totals['manual']}")
    print(f"  detected:        {totals['detected']}")
    print(f"  matched:         {totals['matched']}")
    print(f"  missed:          {totals['missed']}")
    print(f"  false positives: {totals['false_positives']}")
    print(f"  precision:       {precision:.3f}")
    print(f"  recall:          {recall:.3f}")
    print(f"  average IoU:     {average_iou:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
