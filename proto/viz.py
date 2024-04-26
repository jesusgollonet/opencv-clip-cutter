import cv2 as cv

img = cv.imread("output/mosaic.jpg")
cap = cv.VideoCapture("video/2024-01-13 08.38.31.mov")

bgSub = cv.createBackgroundSubtractorMOG2()
frame_shape = (640, 360)
w = frame_shape[0]
h = frame_shape[1]


def text(frame, text, coords):
    cv.putText(frame, text, coords, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv.resize(frame, frame_shape)
    sub = bgSub.apply(resized)
    white_px = cv.countNonZero(sub)
    sub = cv.cvtColor(sub, cv.COLOR_GRAY2BGR)
    combined = cv.hconcat([resized, sub])
    text(combined, "Original", (10, 20))
    text(combined, "Bg", (w + 10, 20))
    text(combined, f"White px: {white_px}", (w + 10, 40))
    cv.imshow("Video", combined)
    if cv.waitKey(1) == ord("q"):
        break
