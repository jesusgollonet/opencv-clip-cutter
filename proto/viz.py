import cv2 as cv

img = cv.imread("output/mosaic.jpg")
cap = cv.VideoCapture("video/2024-01-13 08.38.31.mov")

bgSub = cv.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv.resize(frame, (640, 360), interpolation=cv.INTER_AREA)
    sub = bgSub.apply(resized)
    cv.imshow("Video", resized)
    cv.imshow("Sub", sub)
    if cv.waitKey(1) == ord("q"):
        break
