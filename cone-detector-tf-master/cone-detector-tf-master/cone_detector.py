import logging
import logging.config
import time

from telnetlib import NOP
import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils import cv_utils
from utils import operations as ops
from utils import tf_utils

logging.config.fileConfig('logging.ini')

#VIDEO_PATH = 'testdata/sample_video.mp4'
#VIDEO_PATH = 'testdata/autocross_test.mp4'
#VIDEO_PATH2 = 'testdata/autocross_test.mp4'
FROZEN_GRAPH_PATH = 'models/ssd_mobilenet_v1/frozen_inference_graph.pb'

OUTPUT_WINDOW_WIDTH = 640  # Use None to use the original size of the image
DETECT_EVERY_N_SECONDS = None  # Use None to perform detection for each frame
#DETECT_EVERY_N_SECONDS = 0.1  # Use None to perform detection for each frame
#DETECT_EVERY_N_SECONDS = 2  # Use None to perform detection for each frame

# TUNE ME
CROP_WIDTH = CROP_HEIGHT = 600
CROP_STEP_HORIZONTAL = CROP_STEP_VERTICAL = 600 - 15  # no cone bigger than 20px
SCORE_THRESHOLD = 0.5
NON_MAX_SUPPRESSION_THRESHOLD = 0.5


def main():
    #tf.config.list_physical_devices("GPU") # Shows what devices are available to TensorFlow
    
    # Read TensorFlow graph
    detection_graph = tf_utils.load_model(FROZEN_GRAPH_PATH)

    # Read video from disk and count frames
    #cap = cv2.VideoCapture(VIDEO_PATH)
    #cap = cv2.VideoCapture(0)
    #cap2 = cv2.VideoCapture(VIDEO_PATH)
    #cap2 = cv2.VideoCapture(1)

    #fps = cap.get(cv2.CAP_PROP_FPS)
    #fps2 = cap2.get(cv2.CAP_PROP_FPS)
    #fps = cap.set(cv2.CAP_PROP_FPS, 30)
    #fps2 = cap2.set(cv2.CAP_PROP_FPS, 30)
    #cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    #cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    camLeft = cv2.VideoCapture(0)
    camRight = cv2.VideoCapture(1)

    # Ensure at least one camera can be opened 
    if not (camRight.isOpened() | camLeft.isOpened()):
        print("Could not open a video device")

   
    #CROP_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #CROP_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    
    #ret, frame = leftRead
    #ret2, frame2 = rightRead

    with tf.Session(graph=detection_graph) as sess:

        processed_images = 0
        while (True):
            
            ret, frame = camLeft.read()
            ret2, frame2 = camRight.read()

            while((not frame.any()) | (not frame2.any())):
                ret, frame = camLeft.read()
                ret2, frame2 = camRight.read()
            

            leftRead = ret, frame
            rightRead = ret2, frame2
            '''if DETECT_EVERY_N_SECONDS:
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        processed_images * fps * DETECT_EVERY_N_SECONDS)
                cap2.set(cv2.CAP_PROP_POS_FRAMES,
                        processed_images * fps2 * DETECT_EVERY_N_SECONDS)
            '''
            
            
            #ret, frame = cap.read() # Read in next frame
            #if (ret):
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

            for box, score in zip(boxes, boxes_scores):
                if score > SCORE_THRESHOLD:
                    ymin, xmin, ymax, xmax = box
                    color_detected_rgb = cv_utils.predominant_rgb_color(
                        frame, ymin, xmin, ymax, xmax)
                    text = '{:.2f}'.format(score)
                    cv_utils.add_rectangle_with_text(
                        frame, ymin, xmin, ymax, xmax,
                        color_detected_rgb, text)

            if OUTPUT_WINDOW_WIDTH:
                frame = cv_utils.resize_width_keeping_aspect_ratio(
                    frame, OUTPUT_WINDOW_WIDTH)

            cv2.imshow('Detection result', frame)
            cv2.waitKey(1)
            processed_images += 1

            toc = time.time()
            processing_time_ms = (toc - tic) * 100
            logging.debug(
                'Detected {} objects in {} images in {:.2f} ms'.format(
                    len(boxes), len(crops), processing_time_ms))
            
            #ret2, frame2 = cap2.read() # Read in next frame
            #if (ret2):
            tic = time.time()

            # crops are images as ndarrays of shape
            # (number_crops, CROP_HEIGHT, CROP_WIDTH, 3)
            # crop coordinates are the ymin, xmin, ymax, xmax coordinates in
            #  the original image
            crops2, crops_coordinates2 = ops.extract_crops(
                frame2, CROP_HEIGHT, CROP_WIDTH,
                CROP_STEP_VERTICAL, CROP_STEP_VERTICAL)

            # Uncomment this if you also uncommented the two lines before
            #  creating the TF session.
            #crops = np.array([crops[0]])
            #crops_coordinates = [crops_coordinates[0]]

            detection_dict2 = tf_utils.run_inference_for_batch(crops2, sess)

            # The detection boxes obtained are relative to each crop. Get
            # boxes relative to the original image
            # IMPORTANT! The boxes coordinates are in the following order:
            # (ymin, xmin, ymax, xmax)
            boxes2 = []
            for box_absolute, boxes_relative in zip(
                    crops_coordinates2, detection_dict2['detection_boxes']):
                boxes2.extend(ops.get_absolute_boxes(
                    box_absolute,
                    boxes_relative[np.any(boxes_relative, axis=1)]))
            if boxes2:
                boxes2 = np.vstack(boxes2)

            # Remove overlapping boxes
            boxes2 = ops.non_max_suppression_fast(
                boxes2, NON_MAX_SUPPRESSION_THRESHOLD)

            # Get scores to display them on top of each detection
            boxes_scores2 = detection_dict2['detection_scores']
            boxes_scores2 = boxes_scores2[np.nonzero(boxes_scores2)]

            for box, score in zip(boxes2, boxes_scores2):
                if score > SCORE_THRESHOLD:
                    ymin, xmin, ymax, xmax = box
                    color_detected_rgb = cv_utils.predominant_rgb_color(
                        frame2, ymin, xmin, ymax, xmax)
                    text = '{:.2f}'.format(score)
                    cv_utils.add_rectangle_with_text(
                        frame2, ymin, xmin, ymax, xmax,
                        color_detected_rgb, text)
                        
            if OUTPUT_WINDOW_WIDTH:
                frame2 = cv_utils.resize_width_keeping_aspect_ratio(
                    frame2, OUTPUT_WINDOW_WIDTH)

            cv2.imshow('Detection result 2', frame2)
            cv2.waitKey(1)
            processed_images += 1

            toc = time.time()
            processing_time_ms = (toc - tic) * 100
            logging.debug(
                'Detected {} objects in {} images in {:.2f} ms'.format(
                    len(boxes), len(crops), processing_time_ms))
            
            # Capture frame-by-frame    
            #ret, rightFrame = camRight.read() 
            #ret2, leftFrame = camLeft.read()
            
            ret, rightFrame = rightRead
            ret2, leftFrame = leftRead
            
            #   converting to gray-scale
            leftFrame = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
            rightFrame = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)

            '''kernel = np.ones((5,5),np.float32)/25
            leftFrame = cv2.filter2D(leftFrame,-1,kernel)
            rightFrame = cv2.filter2D(rightFrame,-1,kernel)'''

            leftFrame = cv2.fastNlMeansDenoising(leftFrame,None,20,7,21)
            rightFrame = cv2.fastNlMeansDenoising(rightFrame,None,20,7,21)

            '''leftFrame = cv2.GaussianBlur(leftFrame,(5,5),0) # Gaussian filtering attempt
            rightFrame = cv2.GaussianBlur(rightFrame,(5,5),0)'''

            # Dispaly the individual camera frames
            #cv2.imshow('frame', rightFrame)
            #cv2.imshow('frame2', leftFrame)

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
            # LMEDS is Least Median of Squres
            fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

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

            cv2.imshow('img1 Rectified', img1_rectified)
            cv2.imshow('img2 Rectified', img2_rectified)


            ##
            ## CV Disparity Map
            # ------------------------------------------------------------
            # CALCULATE DISPARITY (DEPTH MAP)
            # Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
            # and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

            # StereoSGBM Parameter explanations:
            # https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

            # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
            block_size = 12
            min_disp = -16
            max_disp = 128
            # Maximum disparity minus minimum disparity. The value is always greater than zero.
            # In the current implementation, this parameter must be divisible by 16.
            num_disp = max_disp - min_disp
            # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
            # Normally, a value within the 5-15 range is good enough
            uniquenessRatio = 15
            # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
            # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
            speckleWindowSize = 150
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
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):    
                break


if __name__ == '__main__':
    main()
