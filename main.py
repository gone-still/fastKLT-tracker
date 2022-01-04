# File        :   main.oy (fastKLT-tracker Python Version)
# Version     :   1.0.0
# Description :   Implements Zana Zakaryaie's FAST-KLT tracker originally written in C++
# Date:       :   Jan 03, 2022
# Author      :   Ricardo Acevedo-Avila
# License     :   _Not for commercial use_

import cv2
import numpy as np
import heapq


# Shows an image
def showImage(imageName, inputImage, delay=0):
    cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
    cv2.imshow(imageName, inputImage)
    cv2.waitKey(delay)


# Writes a png image to disk:
def writeImage(imagePath, inputImage):
    imagePath = imagePath + ".png"
    cv2.imwrite(imagePath, inputImage, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print("Wrote Image: " + imagePath)


# Rotates an array:
def rotateArray(inputArray):
    tempArray = inputArray.copy()
    listEnd = len(inputArray)
    for i in range(listEnd):
        e = inputArray[(listEnd - 1) - i]
        tempArray[i] = e

    return tempArray


# Receives a list of points
# points is a numpy array,
# output is a rect tuple
def boundigRect2f(points):
    # Numpy array to list:
    (h1, w1, c1) = points.shape
    points = list(map(tuple, points.reshape((h1, c1))))

    # numpy array to list:
    x = [tup[0] for tup in points]
    y = [tup[1] for tup in points]

    minX = min(x)
    minY = min(y)
    maxX = max(x)
    maxY = max(y)

    width = maxX - minX
    height = maxY - minY

    # (min_x, min_y, width, height);
    outTuple = (minX, minY, width, height)
    return outTuple


# Returns a rectangle that is the intersection of "a"
# and "b", both tuples that represent a rectangle
# as (x,y,w,h)
def findIntersection(a):
    # Construct the "b" rectangle:
    (h, w) = prevGrayFrame.shape[:2]
    b = (0, 0, w, h)

    # Find intersection:
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


# Computes the shrank dimensions of the "object" (object is a tuple of dimensions):
def shrinkRect(inputTuple):
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


# Keeps the strongest keypoints detected by FAST (?)
def keepStrongest(N, keypoints):
    # Set output keypoints:
    outKeyPoints = keypoints

    # Get total keep points:
    totalKeyPoints = len(keypoints)

    # Check if the total number of objects is bigger than the N requested
    # largest objects:

    if totalKeyPoints > N:
        # Get the N largest keypoints based on the "response" attribute:
        outKeyPoints = heapq.nlargest(N, keypoints, lambda o: o.response)

    # Done:
    return outKeyPoints


# Runs the FAST keypoint detector:
def detectGridFASTpoints(image, object):
    # set the initial keypoints:
    keypoints = []

    # Unpack the "object"
    # (outputX, outputY, outputWidth, outputHeight)
    (xObject, yObject, wObject, hObject) = object

    if xObject < 0 or yObject < 0 or wObject < 0 or hObject < 0:
        return keypoints

    # cropped = img[start_row:end_row, start_col:end_col]
    # img(Range(start_row, end_row), Range(start_col, end_col))

    # Crop the image using the object's dimensions:
    image = image[yObject:yObject + hObject, xObject:xObject + wObject]
    # showImage("image 1", image)

    gridRows = grid[0]  # get grid width
    gridCols = grid[1]  # get grid height

    # Prepare the keypoints list of size equal to maxTotalKeypoints:
    # keypoints = [None] * maxTotalKeypoints

    maxPerCell = int(maxTotalKeypoints / (gridRows * gridCols))

    # Get the image dimensions:
    (imageRows, imageCols) = image.shape[:2]

    for i in range(gridRows):
        # Compute the "row range":
        rowRange = ((i * imageRows) / gridRows, ((i + 1) * imageRows) / gridRows)

        for j in range(gridCols):
            # Compute the "col range":
            colRange = ((j * imageCols) / gridCols, ((j + 1) * imageCols) / gridCols)
            # Crop the image:
            subImage = image[int(rowRange[0]):int(rowRange[1]), int(colRange[0]):int(colRange[1])]
            # showImage("Subimage", subImage)

            # Call to FAST:
            fast = cv2.FastFeatureDetector_create(threshold=fastThreshold)
            subKeyPoints = fast.detect(subImage, None)

            # Let's see WTF I am doing:
            subImage2 = cv2.drawKeypoints(subImage, subKeyPoints, None, color=(255, 0, 0))
            # showImage("SubKeypoints", subImage2)

            # Call to keepStrongest, which I guess keeps only the
            # strongest keypoints computed by FAST, IDFK:
            subKeyPoints = keepStrongest(maxPerCell, subKeyPoints)

            # Get the total subKeyPoint:
            totalSubKeyPoints = len(subKeyPoints)

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
def trackerTrack(img1, img2, points1):
    # Set output parameters:
    points1Out = points1

    # Parameters for lucas-kanade optical flow:
    lkParams = dict(winSize=kltWinSize,
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    points1 = np.float32(points1).reshape(-1, 1, 2)
    points2Out, status, err = cv2.calcOpticalFlowPyrLK(img1, img2, points1, None, **lkParams)

    # Numpy array to list:
    (h1, w1, c1) = points2Out.shape
    points2Out = list(map(tuple, points2Out.reshape((h1, c1))))

    # Mask those points that weren't correctly tracked:
    if points2Out is not None:
        points1Out = [points1Out[i] for i in range(len(points1Out)) if status[i]]
        points2Out = [points2Out[i] for i in range(len(points2Out)) if status[i]]

    # Getting rid of points for which the KLT tracking failed
    # or those who have gone outside the frame:
    # indexCorrection = 0
    # for i in range(len(status)):
    #     newIndex = i - indexCorrection
    #     pt = points2Out[newIndex]
    #
    #     if status[i] == 0 or pt[0] < 0 or pt[1] < 0:
    #         points1Out.pop(newIndex)
    #         points2Out.pop(newIndex)
    #         indexCorrection = indexCorrection + 1

    # Compute the "factor"
    len1 = len(points1Out)
    len2 = len(status)
    factor = ((1.0 * len1) / len2)

    # Prepare the output tuple:
    outTuple = (factor, points1Out, points2Out)

    # Done:
    return outTuple


# Sets the tracker parameters:
def setTrackerParams(maxPts, tgrid, fastThreshold, shrinkRat, KltWindowSize, ransacThreshold):
    global maxTotalKeypoints
    maxTotalKeypoints = maxPts

    global grid
    grid = tgrid

    global fastThresh
    fastThresh = fastThreshold

    global shrinkRatio
    shrinkRatio = shrinkRat

    global kltWinSize
    kltWinSize = KltWindowSize

    global ransacThresh
    ransacThresh = ransacThreshold


# Initializes the tracker:
def initTracker(frame, object):
    # Get the shape of the frame:
    (h, w, c) = frame.shape
    # Check channels:
    global prevGrayFrame
    if c == 3:
        prevGrayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        prevGrayFrame = frame

    # Set the shrunk rectangle (?)
    shrinkedObj = shrinkRect(object)
    global prevKeyPoints
    prevKeyPoints = detectGridFASTpoints(prevGrayFrame, shrinkedObj)

    (x, y, w, h) = object

    global prevCorners
    # prevCorners = [None] * 4
    # Create the numpy array of 32 floats:
    prevCorners = np.zeros(shape=(4, 2), dtype=np.float32)
    prevCorners[0] = (x, y)
    prevCorners[1] = (x + w, y)
    prevCorners[2] = (x + w, y + h)
    prevCorners[3] = (x, y + h)

    prevCorners = np.expand_dims(prevCorners, axis=1)
    # prevCorners = np.float32(prevCorners).reshape(-1, 1, 2)


# Updates the tracker:
def updateTracker(frame):
    # Default value for output:
    rectOutput = (int(0), int(0), int(0), int(0))

    global prevKeyPoints
    listLen = len(prevKeyPoints)

    # Check if there are enough points:
    if listLen < 10:
        return (False, rectOutput)

    # Channel check:
    (h, w, c) = frame.shape
    if c == 3:
        currFrameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        currFrameGray = frame

    global prevGrayFrame
    (prob, prevKeyPoints, currKeypoints) = trackerTrack(prevGrayFrame, currFrameGray, prevKeyPoints)

    # Few points tracked?
    if prob < 0.6:
        return (False, rectOutput)

    # Convert lists to numpy arrays:
    prevKeyPoints = np.float32(prevKeyPoints).reshape(-1, 1, 2)
    currKeypoints = np.float32(currKeypoints).reshape(-1, 1, 2)
    (geometry, inliers) = cv2.estimateAffinePartial2D(prevKeyPoints, currKeypoints, None, cv2.RANSAC, ransacThresh)

    global prevCorners

    if geometry is not None:
        currCorners = cv2.transform(prevCorners, geometry)
        # currCorners is a numpy array:
        rectOutput = boundigRect2f(currCorners)
        tempOutput = rectOutput
        # tempOutput is a rect tuple: (x, y, w, h)
        (ox, oy, ow, oh) = tempOutput
        # Check if wrong estimation:
        if ow < 0 or oh < 0:
            return (False, rectOutput)

        # Handle the box partially outside the frame:
        shrinkedObj = shrinkRect(tempOutput)
        (xShrinkedObj, yShrinkedObj, wShrinkedObj, hShrinkedObj) = shrinkedObj

        intersection = findIntersection(shrinkedObj)
        # intersection is a rect tuple:
        (xIntersection, yIntersection, wIntersection, hIntersection) = intersection

        # area (width*height) of the rectangle:
        intersectionArea = wIntersection * hIntersection
        shrinkedObjArea = wShrinkedObj * hShrinkedObj
        factor = intersectionArea / shrinkedObjArea

        if factor < 0.5:
            return (False, rectOutput)
        else:
            shrinkedObj = intersection

        # Update keypoints:
        prevKeyPoints = detectGridFASTpoints(currFrameGray, shrinkedObj)
        # Store the previous image:
        prevGrayFrame = currFrameGray.copy()
        # Store the previous corners:
        prevCorners = currCorners

        # Done:
        return (True, rectOutput)


# Set the file paths and names:
filePath = "D://opencvImages//faceModel//"
outPath = filePath + "out//"
caffeConfigFile = filePath + "deploy.prototxt"
caffeWeightsFile = filePath + "res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Load the caffe model & weights:
net = cv2.dnn.readNetFromCaffe(caffeConfigFile, caffeWeightsFile)

# Set the video device:
# videoDevice = cv2.VideoCapture(0, cv2.CAP_DSHOW)
videoDevice = cv2.VideoCapture(filePath + "Haddock.mp4")

# Variables:
faceLocations = []
inWidth = 300
inHeight = 300
inScaleFactor = 1.0
confidenceThreshold = 0.5
meanVal = (104.0, 177.0, 123.0)

doDetection = True

# Tracker Default Params:
maxTotalKeypoints = 0
grid = (0, 0)  # grid = (gridWidth, gridHeight)
fastThresh = 0
shrinkRatio = 0.0
kltWinSize = (0, 0)  # kltWinSize = (kltWinSizeWidth, kltWinSizeHeight)
ransacThresh = 0.0

# the "empty mat"
prevGrayFrame = np.zeros((1, 1, 3), np.uint8)
prevKeyPoints = []  # vector of float pairs
prevCorners = []  # vector of float pairs

# Tracker New Params:
maxFeatures = 100
fastThreshold = 5
nRows = 4
nCols = 5
kltWindowSize = 11
shrinkRatio = 0.1
ransacThreshold = 0.9

# Out variables:
frameCounter = 0

# Set tracker parameters:
setTrackerParams(maxFeatures, (nRows, nCols), fastThreshold, shrinkRatio, (kltWindowSize, kltWindowSize),
                 ransacThreshold)

# Check if device is opened:
while videoDevice.isOpened():

    # Get video device frame:
    success, frame = videoDevice.read()

    if success:

        # Extract frame size:
        (frameHeight, frameWidth) = frame.shape[:2]

        showImage("Input Frame", frame, 10)

        # Check detection flag:
        if doDetection:

            # Construct an input blob for the input image
            inputBlob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), meanVal, False, False)
            net.setInput(inputBlob)
            detections = net.forward()

            # loop over the detections
            totalDetections = detections.shape[2]
            print("totalDetections: " + str(totalDetections))

            for i in range(0, totalDetections):

                # Extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > confidenceThreshold:
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    box = detections[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                    (startX, startY, endX, endY) = box.astype("int")

                    #  Set bounding box data to tracker:
                    # Send the frame, and the bounding rect as a tuple with (x,y,w,h)
                    initTracker(frame, (startX, startY, endX - startX, endY - startY))

                    # draw the bounding box of the face along with the associated
                    # probability
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                    # Set detection flag:
                    doDetection = False

                    showImage("Detected Face", frame, 10)

        else:

            print("Updating Tracker...")
            # Update the tracker:
            status, trackedObj = updateTracker(frame)

            if status:
                # Draw rectangle:
                (startX, startY, endX, endY) = trackedObj
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(startX), int(startY)), (int(startX + endX), int(startY + endY)), color, 2)
            else:
                doDetection = True

        showImage("Processed Frame", frame, 10)

        # Write Result:
        # outName = outPath + "tracked-" + str(frameCounter)
        # writeImage(outName, frame)
        # frameCounter = frameCounter + 1
    else:
        break

# When everything done, release the capture device:
videoDevice.release()
cv2.destroyAllWindows()
print("Video Device closed")
