from telnetlib import NOP
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

ret2, leftFrame = camLeft.read()
ret2, rightFrame = camRight.read()  
while((not leftFrame.any()) | (not rightFrame.any())):
    ret2, leftFrame = camLeft.read()
    ret2, rightFrame = camRight.read() 

## Infinite Loop
while(True):

    # Capture frame-by-frame    
    ret, rightFrame = camRight.read() 
    ret2, leftFrame = camLeft.read()    

    #   converting to gray-scale
    leftFrame = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
    rightFrame = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)

    # Dispaly the individual camera frames
    cv2.imshow('frame', rightFrame)
    cv2.imshow('frame2', leftFrame)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(leftFrame, None)
    kp2, des2 = sift.detectAndCompute(rightFrame, None)

    # Visualize keypoints
    #imgSift = cv2.drawKeypoints(
    #    leftFrame, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow("SIFT Keypoints", imgSift)



    # Match keypoints in both images
    # Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2) # Descriptors 1 and 2 based on SIFT

    # Keep good matches: calculate distinctive image features
    # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
    # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            # Keep this keypoint pair
            matchesMask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    # Draw the keypoint matches between both pictures
    # Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    draw_params = dict(matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask=matchesMask[300:500],
                    flags=cv2.DrawMatchesFlags_DEFAULT)

    keypoint_matches = cv2.drawMatchesKnn(
        leftFrame, kp1, rightFrame, kp2, matches[300:500], None, **draw_params)
    cv2.imshow("Keypoint matches", keypoint_matches)





    # ------------------------------------------------------------
    # STEREO RECTIFICATION

    # Calculate the fundamental matrix for the cameras
    # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # We select only inlier points
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]

    # Stereo rectification (uncalibrated variant)
    # Adapted from: https://stackoverflow.com/a/62607343
    h1, w1 = leftFrame.shape
    h2, w2 = rightFrame.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
    )

    # Undistort (rectify) the images and save them
    # Adapted from: https://stackoverflow.com/a/62607343
    img1_rectified = cv2.warpPerspective(leftFrame, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(rightFrame, H2, (w2, h2))


    ##
    ## CV Disparity Map
    # ------------------------------------------------------------
    # CALCULATE DISPARITY (DEPTH MAP)
    # Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
    # and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

    # StereoSGBM Parameter explanations:
    # https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

    # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    block_size = 11
    min_disp = -128
    max_disp = 128
    # Maximum disparity minus minimum disparity. The value is always greater than zero.
    # In the current implementation, this parameter must be divisible by 16.
    num_disp = max_disp - min_disp
    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
    # Normally, a value within the 5-15 range is good enough
    uniquenessRatio = 5
    # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
    # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleWindowSize = 200
    # Maximum disparity variation within each connected component.
    # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    # Normally, 1 or 2 is good enough.
    speckleRange = 2
    disp12MaxDiff = 0

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )

    #stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

    disparity = stereo.compute(img1_rectified, img2_rectified)
    
    # Normalize the values to a range from 0..255 for a grayscale image
    disparity_norm = cv2.normalize(disparity, disparity, alpha=255,
                                beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_norm = np.uint8(disparity_norm)
    cv2.imshow("Disparity", disparity_norm)
  

    ## -- ABOVE IS Work In Progress -- ##
    if cv2.waitKey(1) & 0xFF == ord('q'):    
        break


# Release the cameras and close windows
camLeft.release() 
camRight.release()
cv2.destroyAllWindows()