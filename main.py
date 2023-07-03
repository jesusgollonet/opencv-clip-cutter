import sys
import cv2 as cv

cap = cv.VideoCapture(0)

cv.namedWindow("Video")

if not cap.isOpened():
    print("error opening video")
    sys.exit(1)

while (cap.isOpened()):
     ret, frame = cap.read()
     if ret is True:
         cv.imshow("Video", frame)

         if cv.waitKey(25) & 0xFF == ord('q'):
             break
     else:    
         break

cap.release()
cv.destroyAllWindows()
