'''
Created on Jul 19, 2019

@author: Alexander
'''

import cv2
import numpy as np
import os

# Playing video from file:
cap = cv2.VideoCapture('zion_fastbreak.mp4')

currentFrame = 0
# Capture frame-by-frame
ret, frame = cap.read()

while frame is not None:
    # Saves image of the current frame in jpg file
    name = os.path.join("output2", str(currentFrame) + '.jpg')
    if currentFrame % 10 == 0:
        print('Creating...' + name)
    cv2.imwrite(name, frame)
    #print(len(frame))
    # To stop duplicate images
    currentFrame += 1
    ret, frame = cap.read()

# When everything done, release the capture
print(currentFrame, "frames")
cap.release()
cv2.destroyAllWindows()
