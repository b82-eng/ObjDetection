import cv2 as cv
import time
from matplotlib import pyplot as plt
from windowcapture import list_window_names

list_window_names()

cap = cv.VideoCapture(1) # Camera 1 (Integrated webcam)
cap2 = cv.VideoCapture(0) # Camera 2 (Logitech webcam)

while(1):
    grabbed, frame = cap.read() # Read frame from webcam
    grabbed2, frame2 = cap2.read() # Read frame from Logitech
    #print(grabbed)
    #print(frame)
    cv.imshow('frame',frame) # Display frame from webcam
    cv.imshow('frame2', frame2) # Display frame from Logitech

    # press 'q' with the output window focused to exit
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break

    #time.sleep(5)

cap.release()
cap2.release()