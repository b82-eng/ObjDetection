import numpy as np
import cv2 as cv
from time import time
from windowcapture import WindowCaputure
from windowcapture import list_window_names
#from vision import Vision
#from hsvfilter import HsvFilter
#from edgefilter import EdgeFilter
from PIL import ImageGrab



list_window_names()
#input the window that you want to capture
Wincap = WindowCaputure('Settings')

# Adding bookmarks in order to get FPS 
loop_time = time()

# Adding this to capture screenshots of the video
while(True):
    

    screenshot = Wincap.get_screenshot()

    cv.imshow('Computer Vision', screenshot)

    #print('FPS {}'.format(1/ (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        break


print('Done.')



'''
# We want to enable the web cam feed (either a built in webcam or an external webcam)
# '0' is needed for a built in web cam
# '1' or '-1' can be used for the external web cam ID

IMcap = cv.VideoCapture(1)

IMcap.set(3, 640) # Set the width to 640
IMcap.set(4, 480) # Set the height to 480

'''