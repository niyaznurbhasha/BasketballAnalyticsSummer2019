'''
Created on Jul 24, 2019

@author: Alexander
'''
import json, math, numpy

def getCorners(fname):
    f = open(fname)
    fileText = f.read()
    d = json.loads(fileText)
    pts = d["objects"][0]['points']['exterior']
    
    # Find top right point (far point on baseline)
    topRight = [0,math.inf]
    for pair in pts:
        if pair[1] < topRight[1]:
            topRight = pair
    
    # Find bottom right point (close point on baseline)
    bottomRight = [0,0]
    for pair in pts:
        if pair[0] > bottomRight[0]:
            bottomRight = pair
    
    # Find top left point (same plane as point on baseline)
    topLeft = [math.inf,0]
    for pair in pts:
        if pair[0] < topLeft[0]:
            topLeft = pair
    
    return numpy.float32([topRight, bottomRight, topLeft])

if __name__ == '__main__':
    print(getCorners("ball_1.png.json"))