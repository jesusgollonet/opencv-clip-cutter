import cv2 as cv
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from cc_utils.video_metadata import VideoMetadata
from cc_utils import video_utils as vu

DOWNSCALE_WIDTH = 128
DOWNSCALE_FPS = 2

st.set_page_config(page_title="Clip Cutter", layout="wide")
st.title("OpenCV Clip Cutter")

# --- Sidebar ---
with st.sidebar:
    st.header("Video")
    source_path = st.text_input("Path", placeholder="path/to/video.mp4")

    st.header("Algorithm")
    algorithm = st.selectbox("Background subtractor", ["KNN", "MOG2"])

    st.header("Thresholds")
    start_thresh = st.slider("Movement start (px)", 1, 500, 30)
    end_thresh = st.slider("Movement end (px)", 1, 500, 20)

    run = st.button("Run", type="primary", use_container_width=True)


# --- Cached processing ---

@st.cache_data(show_spinner="Running background subtraction…")
def detect_movement(
    source_path: str, algorithm: str, start_thresh: int, end_thresh: int
):
    vm = VideoMetadata(source_path)
    downscaled = Path(vm.metadata_path) / "downscaled.mp4"
    if not downscaled.exists():
        vu.downscale_video(source_path, str(downscaled), DOWNSCALE_WIDTH, DOWNSCALE_FPS)

    bs = (
        cv.createBackgroundSubtractorKNN()
        if algorithm == "KNN"
        else cv.createBackgroundSubtractorMOG2()
    )

    cap = cv.VideoCapture(str(downscaled))
    frames, bs_frames, white_counts = [], [], []
    segment_times = []
    in_segment = False
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fg = bs.apply(frame)
        white_px = cv.countNonZero(fg)
        white_counts.append(white_px)
        frames.append(frame.copy())
        bs_frames.append(fg.copy())

        if not in_segment and white_px >= start_thresh:
            in_segment = True
            segment_times.append([frame_idx / DOWNSCALE_FPS])
        elif in_segment and white_px < end_thresh:
            in_segment = False
            segment_times[-1].append(frame_idx / DOWNSCALE_FPS)

        frame_idx += 1

    cap.release()
    return frames, bs_frames, white_counts, [s for s in segment_times if len(s) == 2]


@st.cache_data(show_spinner="Cutting segments…")
def cut_segments(source_path: str, segment_times: list) -> list[str]:
    vm = VideoMetadata(source_path)
    paths = []
    for i, (start, end) in enumerate(segment_times):
        out = Path(vm.metadata_path) / f"segment_{i}.mp4"
        if out.exists():
            out.unlink()
        vu.cut_video_segment(
            source_path, str(out), vu.format_time(start), vu.format_time(end - start)
        )
        paths.append(str(out))
    return paths


# --- Main area ---

if run:
    if not source_path:
        st.warning("Enter a video path.")
        st.stop()
    if not Path(source_path).exists():
        st.error(f"File not found: {source_path}")
        st.stop()

    frames, bs_frames, white_counts, segment_times = detect_movement(
        source_path, algorithm, start_thresh, end_thresh
    )

    tab_plot, tab_mosaic, tab_clips = st.tabs(["Detection", "Mosaic", "Clips"])

    with tab_plot:
        fig, ax = plt.subplots(figsize=(12, 3))
        xs = [i / DOWNSCALE_FPS for i in range(len(white_counts))]
        ax.plot(xs, white_counts, linewidth=0.8, color="steelblue")
        ax.axhline(
            start_thresh, color="green", linestyle="--", linewidth=0.8,
            label=f"start ({start_thresh} px)",
        )
        ax.axhline(
            end_thresh, color="red", linestyle="--", linewidth=0.8,
            label=f"end ({end_thresh} px)",
        )
        for start, end in segment_times:
            ax.axvspan(start, end, alpha=0.15, color="yellow")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("White pixels")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

        st.subheader(f"{len(segment_times)} segment(s) detected")
        for i, (start, end) in enumerate(segment_times):
            st.write(f"**{i}** — {vu.format_time(start)} → {vu.format_time(end)} ({end - start:.1f}s)")

    with tab_mosaic:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original frames")
            mosaic = vu.make_mosaic(frames, cols=8)
            st.image(cv.cvtColor(mosaic, cv.COLOR_BGR2RGB), use_container_width=True)
        with col2:
            st.subheader("Background subtraction")
            bs_mosaic = vu.make_mosaic(bs_frames, cols=8)
            st.image(bs_mosaic, use_container_width=True)

    with tab_clips:
        if not segment_times:
            st.info("No segments detected. Try lowering the thresholds.")
        else:
            paths = cut_segments(source_path, [list(s) for s in segment_times])
            for i, path in enumerate(paths):
                st.subheader(f"Segment {i} — {vu.format_time(segment_times[i][0])} → {vu.format_time(segment_times[i][1])}")
                st.video(path)
