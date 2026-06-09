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
    label_fps: float
    duration: float
    frame_count: int | None


def parse_fps(value):
    if "/" not in value:
        return float(value)

    numerator, denominator = value.split("/", 1)
    denominator_float = float(denominator)
    if denominator_float == 0:
        return 0.0
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
            "stream=r_frame_rate,avg_frame_rate,duration,nb_frames",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout.decode("utf-8"))
    stream = payload["streams"][0]
    fps = parse_fps(stream["r_frame_rate"])
    average_fps = parse_fps(stream.get("avg_frame_rate", "0/0"))
    duration = float(stream["duration"])
    frame_count = int(stream["nb_frames"]) if stream.get("nb_frames") else None
    label_fps = average_fps if average_fps > 0 else fps
    if frame_count and duration > 0:
        label_fps = frame_count / duration
    return VideoMetadata(
        fps=fps,
        label_fps=label_fps,
        duration=duration,
        frame_count=frame_count,
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
            Segment(segment.start / metadata.label_fps, segment.end / metadata.label_fps, segment.name)
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


def expand_segment(segment, tolerance):
    return Segment(segment.start - tolerance, segment.end + tolerance, segment.name)


def overlap_duration(a, b):
    return max(0.0, min(a.end, b.end) - max(a.start, b.start))


def duration_similarity(a, b):
    longer = max(a.duration, b.duration)
    if longer <= 0:
        return 0.0
    return min(a.duration, b.duration) / longer


def relaxed_duration_score(expected, detected, boundary_tolerance):
    tolerant_expected = expand_segment(expected, boundary_tolerance)
    overlap = overlap_duration(tolerant_expected, detected)
    shorter = min(expected.duration, detected.duration)
    if overlap <= 0 or shorter <= 0:
        return 0.0

    overlap_score = min(1.0, overlap / shorter)
    return min(overlap_score, duration_similarity(expected, detected))


def segment_score(expected, detected, metric, boundary_tolerance):
    if metric == "iou":
        return iou(expected, detected)
    if metric == "duration":
        return relaxed_duration_score(expected, detected, boundary_tolerance)
    raise ValueError(f"Unsupported match metric: {metric}")


def match_segments(expected, detected, threshold, metric, boundary_tolerance):
    candidates = []
    for expected_index, expected_segment in enumerate(expected):
        for detected_index, detected_segment in enumerate(detected):
            score = segment_score(expected_segment, detected_segment, metric, boundary_tolerance)
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
    matches, missed, false_positives = match_segments(
        expected,
        detected,
        args.match_threshold,
        args.match_metric,
        args.boundary_tolerance,
    )
    average_score = sum(score for _, _, score in matches) / len(matches) if matches else 0.0
    score_label = "IoU" if args.match_metric == "iou" else "score"

    print(f"\n{video_path}")
    print(f"  labels:          {csv_path}")
    print(f"  match metric:    {args.match_metric}")
    if args.match_metric == "duration":
        print(f"  boundary tol.:   {args.boundary_tolerance:.2f}s")
    print(f"  manual segments: {len(expected)}")
    print(f"  detected:        {len(detected)}")
    print(f"  matched:         {len(matches)}")
    print(f"  missed:          {len(missed)}")
    print(f"  false positives: {len(false_positives)}")
    print(f"  average {score_label}:   {average_score:.3f}")

    if args.details:
        for expected_index, detected_index, score in sorted(matches):
            print(
                "  match:           "
                f"manual #{expected_index + 1} {format_segment(expected[expected_index])} "
                f"<-> detected #{detected_index + 1} {format_segment(detected[detected_index])} "
                f"{score_label}={score:.3f}"
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
        "average_score": average_score,
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
    parser.add_argument(
        "--match-metric",
        choices=("iou", "duration"),
        default="iou",
        help="Segment matching metric: strict interval IoU or relaxed duration similarity (default: iou)",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=None,
        help="Minimum score for a match (default: 0.5 for iou, 0.7 for duration)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="Deprecated alias for --match-threshold",
    )
    parser.add_argument(
        "--boundary-tolerance",
        type=float,
        default=2.0,
        help="Seconds to expand manual labels when using --match-metric duration (default: 2.0)",
    )
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
    if args.match_threshold is None:
        args.match_threshold = args.iou_threshold
    if args.match_threshold is None:
        args.match_threshold = 0.5 if args.match_metric == "iou" else 0.7
    if args.boundary_tolerance < 0:
        parser.error("--boundary-tolerance must be non-negative")

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
        "weighted_score": 0.0,
    }
    for video_path, csv_path in label_sets:
        result = evaluate_video(video_path, csv_path, args)
        totals["manual"] += result["manual"]
        totals["detected"] += result["detected"]
        totals["matched"] += result["matched"]
        totals["missed"] += result["missed"]
        totals["false_positives"] += result["false_positives"]
        totals["weighted_score"] += result["average_score"] * result["matched"]

    precision = totals["matched"] / totals["detected"] if totals["detected"] else 0.0
    recall = totals["matched"] / totals["manual"] if totals["manual"] else 0.0
    average_score = totals["weighted_score"] / totals["matched"] if totals["matched"] else 0.0
    score_label = "IoU" if args.match_metric == "iou" else "score"

    print("\nSummary")
    print(f"  videos:          {len(label_sets)}")
    print(f"  match metric:    {args.match_metric}")
    if args.match_metric == "duration":
        print(f"  boundary tol.:   {args.boundary_tolerance:.2f}s")
    print(f"  match threshold: {args.match_threshold:.3f}")
    print(f"  manual segments: {totals['manual']}")
    print(f"  detected:        {totals['detected']}")
    print(f"  matched:         {totals['matched']}")
    print(f"  missed:          {totals['missed']}")
    print(f"  false positives: {totals['false_positives']}")
    print(f"  precision:       {precision:.3f}")
    print(f"  recall:          {recall:.3f}")
    print(f"  average {score_label}:   {average_score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
