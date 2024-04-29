import cv2 as cv
import numpy as np

# initialize vars
videos = [
    "video/2024-01-13 08.38.31.mov",
    "video/2024-01-13 08.41.02.mov",
    "video/2024-01-13 08.54.35.mov",
    "video/2024-02-22 21.35.35.mov",
    "video/2024-02-22 22.06.36.mov",
    "video/2024-02-23 11.59.55.mov",
    "video/test.mov",
]

cap = cv.VideoCapture(videos[6])
frame_shape = (640, 360)
frame_w = frame_shape[0]
frame_h = frame_shape[1]
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
bgSub = cv.createBackgroundSubtractorMOG2()


def text(frame, text, coords):
    cv.putText(frame, text, coords, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


graph_img = np.zeros((frame_h, frame_w * 2, 3), np.uint8)
rect = cv.rectangle(graph_img, (0, 0), (frame_w * 2, frame_h), (0, 0, 0), -1)
c = 0

white_px_pct_ar = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv.resize(frame, frame_shape)
    sub = bgSub.apply(resized)
    white_px = cv.countNonZero(sub)
    white_px_pct = white_px / (frame_w * frame_h) * 100
    print(white_px_pct)

    # make lower values more visible + truncate to 100
    white_px_pct_scaled = min(int(white_px_pct * 20), frame_h)
    white_px_pct_ar.append(white_px_pct_scaled)

    mean = np.mean(white_px_pct_ar)
    median = np.median(white_px_pct_ar)
    std = np.std(white_px_pct_ar)

    # from here it's only about displaying
    sub = cv.cvtColor(sub, cv.COLOR_GRAY2BGR)
    graph_pct = c / total_frames
    rect_x = int(graph_pct * frame_w * 2)
    frame_pct = cv.rectangle(
        graph_img,
        (rect_x, frame_h),
        (rect_x, frame_h - int(white_px_pct_scaled)),
        (255, 255, 255),
        -1,
    )

    c += 1
    combined = cv.hconcat([resized, sub])
    combined = cv.vconcat([combined, rect])

    text(combined, "Original", (10, 20))
    text(combined, "Bg", (frame_w + 10, 20))
    text(combined, f"White px %: {white_px_pct:.2f}%", (frame_w + 10, 40))
    text(combined, f"White px over time %: {white_px_pct:.2f}%", (10, frame_h + 10))
    text(combined, f"mean %: {mean:.2f}%", (10, frame_h + 30))
    cv.line(
        combined,
        (0, (frame_h + frame_h) - int(mean)),
        (frame_w * 2, (frame_h + frame_h) - int(mean)),
        (255, 255, 255),
        2,
    )
    cv.line(
        combined,
        (0, (frame_h + frame_h) - int(median)),
        (frame_w * 2, (frame_h + frame_h) - int(median)),
        (0, 255, 255),
        2,
    )
    cv.line(
        combined,
        (0, (frame_h + frame_h) - int(std)),
        (frame_w * 2, (frame_h + frame_h) - int(std)),
        (255, 0, 0),
        2,
    )
    # move text to random points on every frame
    cv.imshow("Video", combined)

    if cv.waitKey(1) == ord("q"):
        break
