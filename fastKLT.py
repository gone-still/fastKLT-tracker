# File        :   fastKLT.py (fastKLT-tracker Python Version)
# Version     :   1.1.3
# Description :   Implements Zana Zakaryaie's FAST-KLT tracker originally written in C++
# Date:       :   May 19, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   Creative Commons CC0

import numpy as np
import cv2
import heapq


# Shows an image
def showImage(imageName, inputImage, delay=0):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(delay)


class FastKLT:
    # Class attribute
    trackerName = "FAST-KLT Tracker"

    # Creates the tracker object and sets its parameters:
    def __init__(self, paramTuple):
        # Unpack the parameters tuple:
        (maxPts, tgrid, fastThreshold, shrinkRat, KltWindowSize, ransacThreshold, id) = paramTuple

        # Initialize the tracker parameters:
        self.maxTotalKeypoints = maxPts
        self.grid = tgrid
        self.fastThresh = fastThreshold
        self.shrinkRatio = shrinkRat
        self.kltWinSize = KltWindowSize
        self.ransacThresh = ransacThreshold

        # Tracker ID:
        self.trackerID = id

        # Debug flag:
        self.verbose = False
        # Show grid flag:
        self.setGrid = False

        # "default" member variables:
        self.prevGrayFrame = np.zeros((1, 1, 3), np.uint8)  # empty mat/numpy array
        self.prevCorners = []  # list of float pairs
        self.prevKeyPoints = []  # list of float pairs

        # Parameters for lucas-kanade optical flow:
        self.lkParams = dict(winSize=self.kltWinSize,
                             maxLevel=3,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    # Debug info setter:
    def setVerbose(self, verbose):
        self.verbose = verbose
        print("FastKLT(" + str(self.getTrackerId()) + ")>> Verbose set: " + str(verbose))

    # Show grid setter:
    def showGrid(self, gridFlag):
        self.setGrid = gridFlag
        print("FastKLT(" + str(self.getTrackerId()) + ")>> Show Grid set to: " + str(gridFlag))

    # Tracker ID getter:
    def getTrackerId(self):
        return self.trackerID

    # Receives a list of "points". "points" is a numpy array, output is a rect tuple:
    @staticmethod
    def boundingRect2f(points):
        # Numpy array to list:
        (h1, w1, c1) = points.shape
        points = list(map(tuple, points.reshape((h1, c1))))

        # numpy array to list:
        x = [tup[0] for tup in points]
        y = [tup[1] for tup in points]

        # Get the max x,y and min x, y coordinates for
        # new bounding rect construction:
        minX = min(x)
        minY = min(y)
        maxX = max(x)
        maxY = max(y)

        # Get the new width and height:
        width = maxX - minX
        height = maxY - minY

        # (min_x, min_y, width, height);
        outTuple = (minX, minY, width, height)

        return outTuple

    # Keeps the strongest keypoints detected by FAST:
    @staticmethod
    def keepStrongest(N, keypoints):
        # Set output keypoints:
        outKeyPoints = keypoints

        # Get total keep points:
        totalKeyPoints = len(keypoints)
        # print("totalKeyPoints: "+str(totalKeyPoints))

        # Check if the total number of objects is bigger than the N requested
        # largest objects:

        if totalKeyPoints > N:
            # Get the N largest keypoints based on the "response" attribute:
            outKeyPoints = heapq.nlargest(N, keypoints, lambda o: o.response)

        # Done:
        return outKeyPoints

    # Computes the shrank dimensions of the "object" (object is a tuple of dimensions):
    @staticmethod
    def shrinkRect(inputTuple, shrinkRatio):
        # Get input's dimensions:
        (xInput, yInput, wInput, hInput) = inputTuple

        # Set new dimensions:
        deltaSizeWidth = wInput * shrinkRatio
        deltaSizeHeight = hInput * shrinkRatio

        # Set the offsets:
        offsetX = 0.5 * deltaSizeWidth
        offsetY = 0.5 * deltaSizeHeight

        # Subtract the height & width:
        outputWidth = wInput - deltaSizeWidth
        outputHeight = hInput - deltaSizeHeight

        # Add Offset:
        outputX = xInput + offsetX
        outputY = yInput + offsetY

        # Set the output tuple:
        outputTuple = (int(outputX), int(outputY), int(outputWidth), int(outputHeight))

        return outputTuple

    # Returns a rectangle that is the intersection of "a" and "b", both tuples that represent
    # a rectangle as (x,y,w,h)
    @staticmethod
    def findIntersection(a, dimTuple):
        # Construct the "b" rectangle:
        (h, w) = dimTuple
        # (h, w) = self.prevGrayFrame.shape[:2]
        b = (0, 0, w, h)

        # Find intersection between rectangles:
        xIntersection = max(a[0], b[0])
        yIntersection = max(a[1], b[1])
        wIntersection = min(a[0] + a[2], b[0] + b[2]) - xIntersection
        hIntersection = min(a[1] + a[3], b[1] + b[3]) - yIntersection

        # Check for valid dimensions:
        if wIntersection < 0 or hIntersection < 0:
            outTuple = (0, 0, 0, 0)
        else:
            outTuple = (xIntersection, yIntersection, wIntersection, hIntersection)

        # Return the rectangle/tuple:
        return outTuple

    # Initializes the tracker:
    def initTracker(self, frame, rectTuple):
        # Get the shape of the frame:
        (h, w, c) = frame.shape
        # Check channels:
        if c == 3:
            self.prevGrayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self.prevGrayFrame = frame

        # Set the shrunk rectangle (?)
        shrankObj = self.shrinkRect(rectTuple, self.shrinkRatio)
        self.prevKeyPoints = self.detectGridFASTpoints(self.prevGrayFrame, shrankObj)

        # Unpack the rect tuple:
        (x, y, w, h) = rectTuple

        # Create the numpy array of 32-bit floats:
        self.prevCorners = np.zeros(shape=(4, 2), dtype=np.float32)
        self.prevCorners[0] = (x, y)
        self.prevCorners[1] = (x + w, y)
        self.prevCorners[2] = (x + w, y + h)
        self.prevCorners[3] = (x, y + h)

        # Format numpy array to Opencv's "InputArray" type:
        self.prevCorners = np.expand_dims(self.prevCorners, axis=1)

        if self.verbose:
            print("FastKLT(initTracker)>> Tracker Initialized.")

    # Runs the FAST keypoint detector:
    def detectGridFASTpoints(self, image, rectTuple):
        # set the initial keypoints:
        keypoints = []

        # Unpack the "object"
        # (outputX, outputY, outputWidth, outputHeight)
        (xObject, yObject, wObject, hObject) = rectTuple

        if xObject < 0 or yObject < 0 or wObject < 0 or hObject < 0:
            return keypoints

        # cropped = img[start_row:end_row, start_col:end_col]
        # img(Range(start_row, end_row), Range(start_col, end_col))

        # Crop the image using the object's dimensions:
        image = image[yObject:yObject + hObject, xObject:xObject + wObject]
        # # showImage("image 1", image)

        gridRows = self.grid[0]  # get grid width
        gridCols = self.grid[1]  # get grid height

        # Prepare the keypoints list of size equal to maxTotalKeypoints:
        # keypoints = [None] * maxTotalKeypoints

        # set the max per cell dimensions:
        maxPerCell = int(self.maxTotalKeypoints / (gridRows * gridCols))

        # Get the image dimensions:
        (imageRows, imageCols) = image.shape[:2]

        # Local image copy:
        imageCopy = image.copy()

        for i in range(gridRows):
            # Compute the "row range":
            rowRange = ((i * imageRows) / gridRows, ((i + 1) * imageRows) / gridRows)

            for j in range(gridCols):
                # Compute the "col range":
                colRange = ((j * imageCols) / gridCols, ((j + 1) * imageCols) / gridCols)
                # Crop the image:
                subImage = image[int(rowRange[0]):int(rowRange[1]), int(colRange[0]):int(colRange[1])]
                # showImage("Subimage", subImage)

                # Show the local image:
                cv2.rectangle(imageCopy, (int(colRange[0]), int(rowRange[0])), (int(colRange[1]), int(rowRange[1])),
                              (0, 0, 0), 1)
                if self.setGrid:
                    showImage("Grid", imageCopy)

                # Call to FAST:
                fast = cv2.FastFeatureDetector_create(threshold=self.fastThresh)
                subKeyPoints = fast.detect(subImage, None)

                # Call to keepStrongest, which keeps only the strongest keypoints computed by FAST:
                subKeyPoints = self.keepStrongest(maxPerCell, subKeyPoints)

                # Let's see WTF I am doing:
                subImage2 = cv2.drawKeypoints(subImage, subKeyPoints, None, color=(255, 0, 0))
                if self.setGrid:
                    showImage("SubKeypoints", subImage2)

                # Get the total subKeyPoint:
                totalSubKeyPoints = len(subKeyPoints)

                if self.verbose:
                    print("fastKLT>> Region: " + str(i) + "," + str(j) + "-> Got: " + str(
                        totalSubKeyPoints) + " Key Points.")

                for k in range(totalSubKeyPoints):
                    # Get current subKeyPoint:
                    currentSubKEyPoint = subKeyPoints[k]
                    # Compute x and y:
                    x = currentSubKEyPoint.pt[0] + colRange[0] + xObject
                    y = currentSubKEyPoint.pt[1] + rowRange[0] + yObject
                    # Into the list:
                    keypoints.append((x, y))
                    # keypoints[k] = (x, y)

        return keypoints

    # Tracks the keypoints:
    def trackerTrack(self, img1, img2, points1):
        # Set output parameters:
        points1Out = points1

        # Reformat points into a float numpy array for use as
        # OpenCV's "InputArray":
        points1 = np.float32(points1).reshape(-1, 1, 2)
        # Compute Optical Flow:
        points2Out, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, points1, None, **self.lkParams)

        # Numpy array to list conversion:
        (h1, w1, c1) = points2Out.shape
        points2Out = list(map(tuple, points2Out.reshape((h1, c1))))

        # Mask those points that weren't correctly tracked:

        # points1Out2 = []
        # points2Out2 = []
        #
        # if points2Out is not None:
        #
        #     for i in range(len(points2Out)):
        #         (x, y) = points2Out[i]
        #         s = status[i]
        #
        #         if s == 1:
        #             if x > 0 and y >= 0:
        #                 points1Out2.append(points1Out[i])
        #                 points2Out2.append(points2Out[i])
        #             else:
        #                 print("got negatives")
        #         else:
        #             print("got invalid status")

        points1Out = [points1Out[i] for i in range(len(points1Out)) if status[i]]
        points2Out = [points2Out[i] for i in range(len(points2Out)) if status[i]]

        # points1Out = points1Out2
        # points2Out = points2Out2

        # Compute the probability "factor"
        len1 = len(points1Out)
        len2 = len(status)
        factor = ((1.0 * len1) / len2)

        # Prepare the output tuple:
        outTuple = (factor, points1Out, points2Out)

        if self.verbose:
            print("FastKLT(trackerTrack)>> Tracking...")

        # Done:
        return outTuple

    # Updates the tracker:
    def updateTracker(self, frame):
        # Default value for output:
        rectOutput = (int(0), int(0), int(0), int(0))

        # Get total of previous keypoints:
        listLen = len(self.prevKeyPoints)
        if self.verbose:
            print("FastKLT(trackerTrack)>> Total Past Keypoints: " + str(listLen))
        # Check if there are enough points:
        if listLen < 5:
            if self.verbose:
                print("FastKLT(trackerTrack)>> Not enough keypoints! Got: " + str(listLen))
            return (False, rectOutput)

        # Channel check:
        (h, w, c) = frame.shape
        if c == 3:
            currFrameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            currFrameGray = frame

        (prob, self.prevKeyPoints, currKeypoints) = self.trackerTrack(self.prevGrayFrame, currFrameGray,
                                                                      self.prevKeyPoints)

        # Few points tracked?
        if prob < 0.6:
            return (False, rectOutput)

        # Convert lists to numpy arrays:
        self.prevKeyPoints = np.float32(self.prevKeyPoints).reshape(-1, 1, 2)
        currKeypoints = np.float32(currKeypoints).reshape(-1, 1, 2)

        # self.prevKeyPoints, currKeypoints, None, cv2.RANSAC, self.ransacThresh
        (geometry, inliers) = cv2.estimateAffinePartial2D(self.prevKeyPoints, currKeypoints, None,
                                                          cv2.RANSAC, self.ransacThresh,
                                                          2000, 0.99, 10)

        # showImage("geo", geometry)

        if geometry is not None:
            currCorners = cv2.transform(self.prevCorners, geometry)
            # # showImage("currCorners", currCorners)
            # currCorners is a numpy array:
            rectOutput = self.boundingRect2f(currCorners)
            tempOutput = rectOutput
            # tempOutput is a rect tuple: (x, y, w, h)
            (ox, oy, ow, oh) = tempOutput
            # Check if wrong estimation:
            if ow < 0 or oh < 0:
                return (False, rectOutput)

            # Handle the box partially outside the frame:
            shrankObj = self.shrinkRect(tempOutput, self.shrinkRatio)
            (xshrankObj, yshrankObj, wshrankObj, hshrankObj) = shrankObj

            intersection = self.findIntersection(shrankObj, self.prevGrayFrame.shape[:2])
            # intersection is a rect tuple:
            (xIntersection, yIntersection, wIntersection, hIntersection) = intersection

            # area (width*height) of the rectangle:
            intersectionArea = wIntersection * hIntersection
            shrankObjArea = wshrankObj * hshrankObj
            factor = intersectionArea / shrankObjArea

            if factor < 0.5:
                return (False, rectOutput)
            else:
                shrankObj = intersection

            # Update keypoints:
            self.prevKeyPoints = self.detectGridFASTpoints(currFrameGray, shrankObj)
            # Store the previous image:
            self.prevGrayFrame = currFrameGray.copy()
            # Store the previous corners:
            self.prevCorners = currCorners

            if self.verbose:
                print("FastKLT(updateTracker)>> Updated Tracker...")

            # Done:
            return (True, rectOutput)
