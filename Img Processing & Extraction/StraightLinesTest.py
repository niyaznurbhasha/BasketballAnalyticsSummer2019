'''
Created on Jul 2, 2019

@author: alexb
'''
import cv2, os.path, math
import numpy as np
import matplotlib.pyplot as plt

filepath = os.path.join('output_small', 'crop_11.png')
img = cv2.imread(filepath)
rows,cols,ch = img.shape
#print(rows,cols)
scale = rows/37
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,450,500,apertureSize = 3)
#print(len(edges))
#print(38 * (rows*cols)/scale)
lines = cv2.HoughLines(edges,1,np.pi, int(15))# * 0.46*(rows*cols)/scale)) ## last param: num pixels match
#print("# lines:", len(lines))
# Draw the lines
if lines is not None:
    print(len(lines))
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)

#cv2.imwrite('houghlines.png',img)

#
# filepath = os.path.join('output_small', 'crop_11.png')
# img = cv2.imread(filepath)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# edges = cv2.Canny(gray,50,150,apertureSize = 3)
# lines = cv2.HoughLines(edges,1,np.pi/180,10)
# minLineLength = 10
# maxLineGap = 10
# lines = cv2.HoughLinesP(edges,1,np.pi,10,minLineLength,maxLineGap)
# if lines is not None:
#     print(len(lines))
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#         cv2.line(img, pt1, pt2, (0,255,0), 2, cv2.LINE_AA)

#cv2.imwrite('houghlines2.jpg',img)
plt.imshow(edges)
plt.show()
