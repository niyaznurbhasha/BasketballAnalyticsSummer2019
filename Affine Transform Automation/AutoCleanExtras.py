'''
Created on Jun 18, 2019

@author: alexb
'''
import cv2, json, os
import numpy as np
import matplotlib.pyplot as plt
from CleanJSON import cleanJSON
from GetPaintCorners import getCorners

imgfile = 'ball_1.png'
img = cv2.imread(os.path.join("img_clean", imgfile))
rows,cols,ch = img.shape
#print(rows,cols,ch)

#plt.imshow(img)
#plt.show()

pts1 = getCorners(os.path.join("ann_clean", imgfile + ".json"))
pts2 = np.float32([[cols,380],[cols,620],[cols-380,380]])
## 1 ft --> 20 pixels

M = cv2.getAffineTransform(pts1,pts2)
#print(M)

#Vector to project is bottom left corner 
# vectorToProject = np.float32([[1013, 667, 1]])  ## [x, y, 1]
# X = vectorToProject.transpose()
# projection = np.matmul(M,X)
# print(projection)

rightBound = cols+30  # x
topBound = -20  # y
bottomBound = 1250  # y (TRIAL AND ERROR?)

points = [11, 14, 19, 20, 21, 22, 23, 24]

fname = os.path.join('logs', imgfile.split(".")[0] + '_keypoints.json')
d = cleanJSON(fname)

newDict = {'people': []}
for num, thisPerson in enumerate(d['people']):
    add = False
    for pt in points:
        dataForPoint = thisPerson['Keypoint ' + str(pt)]
        dataForPoint[2] = 1
        X = np.float32(dataForPoint).transpose()
        projection = np.matmul(M,X)
        if projection[0] < rightBound and projection[1] > topBound and projection[1] < bottomBound:
            add = True
            break
    if add:
        thisPerson['Old #'] = num
        newDict['people'].append(thisPerson)
        
print([x['Old #'] for x in newDict['people']])
#print([x['label'] for x in newDict['people']])
print(len(newDict['people']))

#newFilename = fname[:-5] + '_bounded' + fname[-5:]
#with open(newFilename, 'w') as fp:
#        json.dump(newDict, fp)

# dst = cv2.warpAffine(img,M,(cols,rows))
#      
# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()
