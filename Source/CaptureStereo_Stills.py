import cv2
import numpy
print(cv2.__version__)


## Camera Object
camRight = cv2.VideoCapture(0)
camLeft = cv2.VideoCapture(1)  


## Infinite Loop
while(True):
        
    # Capture frame-by-frame     
    ret, frameRight = camRight.read() 
    ret2, frameLeft = camLeft.read()    

    cv2.imshow('frameRight', frameRight)
    cv2.imshow('frameLeft', frameLeft)

 # Waits for a user input (the letter 'q') to quit the application    
    if cv2.waitKey(1) & 0xFF == ord('c'):  
        cv2.imwrite('leftFrame.jpg', frameLeft)
        cv2.imwrite('rightFrame.jpg', frameRight)

    # Waits for a user input (the letter 'q') to quit the application    
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break
   


# Release the cameras and close windows
camLeft.release() 
camRight.release()
cv2.destroyAllWindows()