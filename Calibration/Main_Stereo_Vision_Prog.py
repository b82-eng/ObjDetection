# Stereovision Cone Detector Project
# ECE 595
# Nathaniel Brown, Benjamin Edwards, Jon Brito

import logging
import logging.config
import time
from tkinter import Y
import numpy as np
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from openpyxl import Workbook
from sklearn.preprocessing import normalize

from utils import cv_utils
from utils import operations as ops
from utils import tf_utils

FROZEN_GRAPH_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.pb'
OUTPUT_WINDOW_WIDTH = 640
CROP_WIDTH = CROP_HEIGHT = 600
CROP_STEP_HORIZONTAL = CROP_STEP_VERTICAL = 600 - 15  # No cone bigger than 20px
SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5

# TensorFlow inits
detection_graph = tf_utils.load_model(FROZEN_GRAPH_PATH) # Read-in cone detection model for TensorFlow

# Filtering
kernel= np.ones((3,3),np.uint8)
global counterdist
counterdist = 86 # starting cm distance

# Function to calculate object distance on mouse click event
def coords_mouse_disp(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print('Mouse click coords: ' + str(x) + ' ' + str(y))
        #print x,y,disp[y,x],filteredImg[y,x]
        average=0
        for u in range (-1,2):
            for v in range (-1,2):
                average += disp[y+u,x+v]
        average=average/9 # 3x3 matrix size = 9
        #Distance= -593.97*average**(3) + 1506.8*average**(2) - 1373.1*average + 522.06
        Distance= -2480.0*average**(3) + 3903.8*average**(2) - 2030.7*average + 432.33
        #Distance= np.around(Distance*0.01,decimals=2)
        Distance= np.around(Distance,decimals=2)
        print('Distance at clicked point: '+ str(Distance)+' cm')
        print('Distance Bf/d: '+ str((10.47*0.4)/average) )
        
# This section is uncommented if one needs to recalibrate due to changes in the camera setup
#        global counterdist
#        ws.append([counterdist, average])
#        print('Measure at '+str(counterdist)+' cm, the dispasrity is ' + str(average))
#        if (counterdist <= 85):
#            counterdist += 3
#        elif(counterdist <= 120):
#            counterdist += 5
#        else:
#            counterdist += 10
#        print('Next distance to measure: '+str(counterdist)+'cm')

#wb=Workbook()
#ws=wb.active  

### Distortion calibraion procedure ###

# Termination criteria
criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo= (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

# Arrays to store object points and image points from all images
objpoints= []   # 3d points in real world space
imgpointsR= []   # 2d points in image plane
imgpointsL= []

# Start calibration from the camera
print('Starting camera calibration procedure... ')
# Call all saved images
for i in range(0,30):   # 68 calibration images provided
    print('Processing image: #' + str(i+1))
    t= str(i)
    ChessImaR= cv2.imread('chessboard-R'+t+'.png',0)    # Right side
    ChessImaL= cv2.imread('chessboard-L'+t+'.png',0)    # Left side
    retR, cornersR = cv2.findChessboardCorners(ChessImaR,
                                               (7,7),None)  # Define the number of chees corners we are looking for
    retL, cornersL = cv2.findChessboardCorners(ChessImaL,
                                               (7,7),None)  # Left side
    if (True == retR) & (True == retL):
        objpoints.append(objp)
        cv2.cornerSubPix(ChessImaR,cornersR,(11,11),(-1,-1),criteria)
        cv2.cornerSubPix(ChessImaL,cornersL,(11,11),(-1,-1),criteria)
        imgpointsR.append(cornersR)
        imgpointsL.append(cornersL)

# Determine the new values for different parameters
#   Right Side
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints,
                                                        imgpointsR,
                                                        ChessImaR.shape[::-1],None,None)
hR,wR= ChessImaR.shape[:2]
OmtxR, roiR= cv2.getOptimalNewCameraMatrix(mtxR,distR,
                                                   (wR,hR),1,(wR,hR))

#   Left Side
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints,
                                                        imgpointsL,
                                                        ChessImaL.shape[::-1],None,None)
hL,wL= ChessImaL.shape[:2]
OmtxL, roiL= cv2.getOptimalNewCameraMatrix(mtxL,distL,(wL,hL),1,(wL,hL))

print('Camera calibration procedure completed.')

### Stereovision calibration procedure ###

# StereoCalibrate function
flags = 0
flags |= cv2.CALIB_FIX_INTRINSIC
#flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
#flags |= cv2.CALIB_USE_INTRINSIC_GUESS
#flags |= cv2.CALIB_FIX_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_ASPECT_RATIO
#flags |= cv2.CALIB_ZERO_TANGENT_DIST
#flags |= cv2.CALIB_RATIONAL_MODEL
#flags |= cv2.CALIB_SAME_FOCAL_LENGTH
#flags |= cv2.CALIB_FIX_K3
#flags |= cv2.CALIB_FIX_K4
#flags |= cv2.CALIB_FIX_K5
retS, MLS, dLS, MRS, dRS, R, T, E, F= cv2.stereoCalibrate(objpoints,
                                                          imgpointsL,
                                                          imgpointsR,
                                                          mtxL,
                                                          distL,
                                                          mtxR,
                                                          distR,
                                                          ChessImaR.shape[::-1],
                                                          criteria_stereo,
                                                          flags)

# StereoRectify function
rectify_scale= 0 # if 0 image croped, if 1 image nor croped
RL, RR, PL, PR, Q, roiL, roiR= cv2.stereoRectify(MLS, dLS, MRS, dRS,
                                                 ChessImaR.shape[::-1], R, T,
                                                 rectify_scale,(0,0))  # last paramater is alpha, if 0= croped, if 1= not croped
# initUndistortRectifyMap function
Left_Stereo_Map= cv2.initUndistortRectifyMap(MLS, dLS, RL, PL,
                                             ChessImaR.shape[::-1], cv2.CV_16SC2)   # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map= cv2.initUndistortRectifyMap(MRS, dRS, RR, PR,
                                              ChessImaR.shape[::-1], cv2.CV_16SC2)

### Stereovision parameters ###

# Create StereoSGBM and prepare all parameters
window_size = 3
min_disp = 2
num_disp = 130-min_disp
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    disp12MaxDiff = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2)

# Used for the filtered image
stereoR=cv2.ximgproc.createRightMatcher(stereo) # Create another stereo for right this time

# WLS FILTER Parameters
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

### Begin distance measuring and cone detection loop ###

# Call the two cameras
CamR= cv2.VideoCapture(0)   # 0 = Left camera, 1 = right camera
CamL= cv2.VideoCapture(1)

with tf.Session(graph=detection_graph) as sess:
    while True:
        retR, frameR= CamR.read() # Reads in a single frame from each camera
        retL, frameL= CamL.read()

        # Image denoising and rectification
        frameL = cv2.fastNlMeansDenoising(frameL,None,20,7,21)
        Left_nice= cv2.remap(frameL,Left_Stereo_Map[0],Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        frameR = cv2.fastNlMeansDenoising(frameR,None,20,7,21)
        Right_nice= cv2.remap(frameR,Right_Stereo_Map[0],Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

        ##    # Draw Red lines
        ##    for line in range(0, int(Right_nice.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
        ##        Left_nice[line*20,:]= (0,0,255)
        ##        Right_nice[line*20,:]= (0,0,255)
        ##
        ##    for line in range(0, int(frameR.shape[0]/20)): # Draw the Lines on the images Then numer of line is defines by the image Size/20
        ##        frameL[line*20,:]= (0,255,0)
        ##        frameR[line*20,:]= (0,255,0)    
            
        # Show the Undistorted images
        #cv2.imshow('Both Images', np.hstack([Left_nice, Right_nice]))
        #cv2.imshow('Normal', np.hstack([frameL, frameR]))

        # Convert from color(BGR) to gray
        grayR= cv2.cvtColor(Right_nice,cv2.COLOR_BGR2GRAY)
        grayL= cv2.cvtColor(Left_nice,cv2.COLOR_BGR2GRAY)

        # Compute the 2 images for the Depth_image
        disp= stereo.compute(grayL,grayR)#.astype(np.float32)/ 16
        dispL= disp
        dispR= stereoR.compute(grayR,grayL)
        dispL= np.int16(dispL)
        dispR= np.int16(dispR)

        # Using the WLS filter
        filteredImg= wls_filter.filter(dispL,grayL,None,dispR)
        filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        #cv2.imshow('Disparity Map', filteredImg)
        filteredImg = np.uint8(filteredImg)
        #cv2.imshow('Disparity Map', filteredImg)
        disp= ((disp.astype(np.float32)/ 16)-min_disp)/num_disp # Calculation allowing us to have 0 for the most distant object able to detect

        # Resize the image for faster executions
        #dispR= cv2.resize(disp,None,fx=0.7, fy=0.7, interpolation = cv2.INTER_AREA)

        # Filtering the Results with a closing filter
        #closing= cv2.morphologyEx(disp,cv2.MORPH_CLOSE, kernel) # Apply an morphological filter for closing little "black" holes in the picture(Remove noise) 

        # Colors map
        #dispc= (closing-closing.min())*255
        #dispC= dispc.astype(np.uint8)                                   # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
        #disp_Color= cv2.applyColorMap(dispC,cv2.COLORMAP_OCEAN)         # Change the Color of the Picture into an Ocean Color_Map
        filt_Color= cv2.applyColorMap(filteredImg,cv2.COLORMAP_OCEAN)    # Applies colormap to filtered image as it makes it easier for a human to see the contrast

        # Show the result for the Depth_image
        #cv2.imshow('Disparity', disp)
        #cv2.imshow('Closing',closing)
        #cv2.imshow('Color Depth',disp_Color)
        cv2.imshow('Filtered Color Depth',filt_Color) # Output window with disparity map for user to click

        frame = Left_nice # Uses rectified image from left camera for cone detection
        
        tic = time.time()

        # crops are images as ndarrays of shape
        # (number_crops, CROP_HEIGHT, CROP_WIDTH, 3)
        # crop coordinates are the ymin, xmin, ymax, xmax coordinates in
        #  the original image
        crops, crops_coordinates = ops.extract_crops(
            frame, CROP_HEIGHT, CROP_WIDTH,
            CROP_STEP_VERTICAL, CROP_STEP_VERTICAL)


        # Uncomment this if you also uncommented the two lines before
        #  creating the TF session.
        #crops = np.array([crops[0]])
        #crops_coordinates = [crops_coordinates[0]]

        detection_dict = tf_utils.run_inference_for_batch(crops, sess)

        # The detection boxes obtained are relative to each crop. Get
        # boxes relative to the original image
        # IMPORTANT! The boxes coordinates are in the following order:
        # (ymin, xmin, ymax, xmax)
        boxes = []
        for box_absolute, boxes_relative in zip(
                crops_coordinates, detection_dict['detection_boxes']):
            boxes.extend(ops.get_absolute_boxes(
                box_absolute,
                boxes_relative[np.any(boxes_relative, axis=1)]))
        if boxes:
            boxes = np.vstack(boxes)

        # Remove overlapping boxes
        boxes = ops.non_max_suppression_fast(
            boxes, NON_MAX_SUPPRESSION_THRESHOLD)

        # Get scores to display them on top of each detection
        boxes_scores = detection_dict['detection_scores']
        boxes_scores = boxes_scores[np.nonzero(boxes_scores)]

        conesfnd = 0
        boxMidX = []
        boxMidY = []

        for box, score in zip(boxes, boxes_scores):
            if score > SCORE_THRESHOLD:
                conesfnd += 1
                ymin, xmin, ymax, xmax = box
                color_detected_rgb = cv_utils.predominant_rgb_color(
                    frame, ymin, xmin, ymax, xmax)
                text = '#' + str(conesfnd) + ' ' +  '{:.2f}'.format(score)
                print('X,Y Coords Calculated: ' + str(xmax-xmin) + ' ' + str(ymax-ymin))
                cv_utils.add_rectangle_with_text(
                    frame, ymin, xmin, ymax, xmax,
                    color_detected_rgb, text)
                # Calculates location specs for each detected cone
                boxMidX.append(xmax-xmin)
                boxMidY.append(ymax-ymin)


        #if OUTPUT_WINDOW_WIDTH:
        #    frame = cv_utils.resize_width_keeping_aspect_ratio(
        #        frame, OUTPUT_WINDOW_WIDTH)

        cv2.imshow('TF Detection Result (Left Camera)', frame)
        cv2.waitKey(1)
        #processed_images += 1

        toc = time.time()
        processing_time_ms = (toc - tic) * 100
        logging.debug(
            'Detected {} objects in {} images in {:.2f} ms'.format(
                len(boxes), len(crops), processing_time_ms))
        
        if conesfnd > 0:
            print(str(conesfnd) + ' cones detected. Finding distances...')
            
            # Get locations of each cone detection
            for j in range(conesfnd):
                x = boxMidX[j]*2
                y = boxMidY[j]

                average = 0
                for u in range (-1,2):
                    for v in range (-1,2):
                        average += disp[y+u,x+u]
                average = average / 9 # 3x3 matrix size = 9
                Distance = -2480.0*average**(3) + 3903.8*average**(2) - 2030.7*average + 432.33
                #Distance= np.around(Distance*0.01,decimals=2)
                Distance= np.around(Distance,decimals=2)
                print('Distance of cone ' + str(j+1) + ': '+ str(Distance)+' cm')
                print('Distance Bf/d: '+ str((10.47*0.4)/average) )
        else:
            print('No cones found.')

        # Mouse click
        cv2.setMouseCallback("Filtered Color Depth",coords_mouse_disp,filt_Color)
        
        # End the Programme
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    
# Save excel
wb.save("data4.xlsx")

# Release the Cameras
CamR.release()
CamL.release()
cv2.destroyAllWindows()
