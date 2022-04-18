import cv2
import numpy as np
from matplotlib import pyplot as plt
print(cv2.__version__)


## Camera Object
# webcam = cv2.VideoCapture(0)
camLeft = cv2.VideoCapture(0)
camRight = cv2.VideoCapture(1)

# Ensure at least one camera can be opened 
if not (camRight.isOpened() | camLeft.isOpened()):
    print("Could not open a video device")

## Infinite Loop
while(True):
        
    # Capture frame-by-frame    
    ret, frame = camRight.read() 
    ret2, frame2 = camLeft.read()    

    # Dispaly the individual camera frames
    cv2.imshow('frame', frame)
    cv2.imshow('frame2', frame2)


    # Combine the two frames, 50% weight to each to visualize the disparity and perform manual coarse alignment 
    dst = cv2.addWeighted(frame, 0.5, frame2, 0.5,0)
    cv2.imshow('dst',dst)

 

    # Waits for a user input (the letter 'q') to quit the application    
    if cv2.waitKey(1) & 0xFF == ord('q'):    
        break


# Release the cameras and close windows
camLeft.release() 
camRight.release()
cv2.destroyAllWindows()