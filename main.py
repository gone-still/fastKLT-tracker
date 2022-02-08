# File        :   main.py (Example use of the fastKLT-tracker)
# Version     :   1.0.2
# Description :   Implements Zana Zakaryaie's FAST-KLT tracker originally written in C++
# Date:       :   Feb 07, 2022
# Author      :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# License     :   Creative Commons CC0

import cv2
import numpy as np
from fastKLT import FastKLT


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


# Set the file paths and names:
filePath = "D://opencvImages//faceModel//"
outPath = filePath + "out//"
caffeConfigFile = filePath + "deploy.prototxt"
caffeWeightsFile = filePath + "res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Load the caffe model & weights:
net = cv2.dnn.readNetFromCaffe(caffeConfigFile, caffeWeightsFile)

# Set the video device:
videoDevice = cv2.VideoCapture(filePath + "Haddock.mp4")

# Main Variables:
faceLocations = []
inWidth = 300
inHeight = 300
inScaleFactor = 1.0
confidenceThreshold = 0.5
meanVal = (104.0, 177.0, 123.0)
doDetection = True
frameCounter = 0

# Tracker  Parameters:
maxFeatures = 100
fastThreshold = 5
nRows = 4
nCols = 5
kltWindowSize = 11
shrinkRatio = 0.1
ransacThreshold = 0.9
trackerId = 1

# Set tracker parameters:
parametersTuple = [maxFeatures, (nRows, nCols), fastThreshold, shrinkRatio, (kltWindowSize, kltWindowSize),
                   ransacThreshold, trackerId]

# Create the tracker with parameters:
tracker = FastKLT(parametersTuple)

# Enable debug information:
tracker.setVerbose(False)

# Check if device is opened:
while videoDevice.isOpened():

    # Get video device frame:
    success, frame = videoDevice.read()

    if success:

        # Extract frame size:
        (frameHeight, frameWidth) = frame.shape[:2]
        # Show the raw, input frame:
        showImage("Input Frame", frame, 10)

        # Check detection flag:
        if doDetection:

            # Construct an input blob for the input image
            inputBlob = cv2.dnn.blobFromImage(frame, inScaleFactor, (inWidth, inHeight), meanVal, False, False)
            net.setInput(inputBlob)
            detections = net.forward()

            # loop over the detections
            totalDetections = detections.shape[2]
            # print("totalDetections: " + str(totalDetections))

            for i in range(0, totalDetections):

                # Extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > confidenceThreshold:
                    # Compute the (x, y)-coordinates of the bounding box for the
                    # object
                    box = detections[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                    (startX, startY, endX, endY) = box.astype("int")

                    print((startX, startY, endX, endY))

                    # Set bounding box data to tracker:
                    # Send the frame, and the bounding rect as a tuple with (x,y,w,h)
                    tracker.initTracker(frame, (startX, startY, endX - startX, endY - startY))

                    # Draw the bounding box of the face along with the associated
                    # probability
                    text = "{:.2f}%".format(confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    color = (0, 0, 255)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                    # Set detection flag:
                    doDetection = False
                    # Show the initial detection:
                    showImage("Detected Face", frame, 10)

        else:

            # Update the tracker:
            status, trackedObj = tracker.updateTracker(frame)

            if status:
                # Draw rectangle:
                (startX, startY, endX, endY) = trackedObj
                color = (0, 255, 0)
                cv2.rectangle(frame, (int(startX), int(startY)), (int(startX + endX), int(startY + endY)), color, 2)
            else:
                doDetection = True

        # Show the tracked face:
        textX = 10
        textY = 30
        org = (textX, textY)
        font = cv2.FONT_HERSHEY_SIMPLEX

        color = (0, 255, 0)
        frameString = "Frame: " + str(frameCounter)
        cv2.putText(frame, frameString, org, font, 1, color, 1, cv2.LINE_AA)
        showImage("Processed Frame", frame, 0)

        # Write Result:
        # outName = outPath + "tracked-" + str(frameCounter)
        # writeImage(outName, frame)

        frameCounter += 1
    else:
        break

# Release the capture device:
videoDevice.release()
cv2.destroyAllWindows()
print("Video Device closed")
