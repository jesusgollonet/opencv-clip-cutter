import cv2 as cv
import numpy as np

img = cv.imread("output/mosaic.jpg")
cap = cv.VideoCapture("video/2024-02-22 21.35.35.mov")

bgSub = cv.createBackgroundSubtractorMOG2()
frame_shape = (640, 360)
w = frame_shape[0]
h = frame_shape[1]

total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))


def text(frame, text, coords):
    cv.putText(frame, text, coords, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


rect_img = np.zeros((100, w * 2, 3), np.uint8)
rect = cv.rectangle(rect_img, (0, 0), (w * 2, 100), (0, 0, 0), -1)
c = 0

white_px_pct_ar = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv.resize(frame, frame_shape)
    sub = bgSub.apply(resized)
    white_px = cv.countNonZero(sub)
    white_px_pct = white_px / (w * h) * 100

    white_px_pct_scaled = min(int(white_px_pct * 10), 100)
    white_px_pct_ar.append(white_px_pct_scaled)

    mean = np.mean(white_px_pct_ar)
    median = np.median(white_px_pct_ar)
    std = np.std(white_px_pct_ar)

    # from here it's only about displaying
    sub = cv.cvtColor(sub, cv.COLOR_GRAY2BGR)
    graph_pct = c / total_frames
    rect_x = int(graph_pct * w * 2)
    frame_pct = cv.rectangle(
        rect_img,
        (rect_x, 100),
        (rect_x, 100 - int(white_px_pct_scaled)),
        (255, 255, 255),
        -1,
    )

    c += 1
    combined = cv.hconcat([resized, sub])
    combined = cv.vconcat([combined, rect])

    text(combined, "Original", (10, 20))
    text(combined, "Bg", (w + 10, 20))
    text(combined, f"White px %: {white_px_pct:.2f}%", (w + 10, 40))
    text(combined, f"White px over time %: {white_px_pct:.2f}%", (10, h + 10))
    text(combined, f"mean %: {mean:.2f}%", (10, h + 30))
    cv.line(
        combined,
        (0, (h + 100) - int(mean)),
        (w * 2, (h + 100) - int(mean)),
        (255, 255, 255),
        2,
    )
    cv.line(
        combined,
        (0, (h + 100) - int(median)),
        (w * 2, (h + 100) - int(median)),
        (0, 255, 255),
        2,
    )
    cv.line(
        combined,
        (0, (h + 100) - int(std)),
        (w * 2, (h + 100) - int(std)),
        (255, 0, 0),
        2,
    )
    # move text to random points on every frame
    cv.imshow("Video", combined)
    if cv.waitKey(1) == ord("q"):
        break
