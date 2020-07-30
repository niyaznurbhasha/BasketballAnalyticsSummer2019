'''
Created on Jun 18, 2019

@author: alexb
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('ball_1.png')
rows,cols,ch = img.shape
print(rows,cols,ch)

#plt.imshow(img)
#plt.show()

pts1 = np.float32([[882,333],[1047,409],[528,412]])
pts2 = np.float32([[cols,380],[cols,620],[cols-380,380]])
## pt 1 to 2: 19 ft, pt 2 to 3: 19 ft
# 
M = cv2.getAffineTransform(pts1,pts2)
#print(M)
 
#vectorToProject = np.float32([[521.6, 366.8, 1]])  ## [x, y, 1]
#X = vectorToProject.transpose()
#projection = np.matmul(M,X)
#print(projection)



dst = cv2.warpAffine(img,M,(cols,rows))
     
plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()
