'''
Created on Jun 18, 2019

@author: alexb
'''
import cv2, json
import numpy as np
import matplotlib.pyplot as plt
from CleanJSON import cleanJSON

img = cv2.imread('syracuse_duke_002_rendered.png')
rows,cols,ch = img.shape
print(rows,cols,ch)

#plt.imshow(img)
#plt.show()
#pts1 = np.float32([[567,544],[360,772],[1196,817]])

pts1 = np.float32([[657,347],[157,627],[999,687]])
pts2 = np.float32([[0,0],[0,310],[190,310]])
## pt 1 to 2: 19 ft, pt 2 to 3: 19 ft
# 
M = cv2.getAffineTransform(pts1,pts2)
#print(M)

# Vector to project is bottom left corner 
# vectorToProject = np.float32([[1279, 613, 1]])  ## [x, y, 1]
# X = vectorToProject.transpose()
# projection = np.matmul(M,X)
#print(projection)

leftBound = -30  # x
topBound = -20  # y
bottomBound = 650  # y

points = [11, 14, 19, 20, 21, 22, 23, 24]

fname = 'ball_1_keypoints.json'
d = cleanJSON(fname)

newDict = {'people': []}
for num, thisPerson in enumerate(d['people']):
    add = False
    for pt in points:
        dataForPoint = thisPerson['Keypoint ' + str(pt)]
        dataForPoint[2] = 1
        X = np.float32(dataForPoint).transpose()
        projection = np.matmul(M,X)
        if projection[0] > leftBound and projection[1] > topBound and projection[1] < bottomBound:
            add = True
            break
    if add:
        thisPerson['Old #'] = num
        newDict['people'].append(thisPerson)
        
print([x['Old #'] for x in newDict['people']])
#print([x['label'] for x in newDict['people']])
print(len(newDict['people']))

newFilename = fname[:-5] + '_bounded' + fname[-5:]
with open(newFilename, 'w') as fp:
        json.dump(newDict, fp)

# dst = cv2.warpAffine(img,M,(cols,rows))
#      
# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()
