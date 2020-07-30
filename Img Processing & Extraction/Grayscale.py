'''
Created on Jul 8, 2019

@author: alexb
'''
import cv2, os.path, math
import numpy as np
#import matplotlib.pyplot as plt
from Sharpen import sharpen

## READ FILE
filepath = os.path.join('output_small', 'crop_11.png')

for file in os.listdir('output_small'):
    filepath = os.path.join('output_small', file)
    img = cv2.imread(filepath)
    
    rows,cols,ch = img.shape
    #print(rows,cols)
    
    ## CONVERT TO GRAYSCAPE AND EQUALIZE
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    #print(gray)
    #edges = cv2.Canny(gray,50,150,apertureSize = 3)
    
    ## RESIZE IMAGE
    dim = (50, 75)
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    
    ## SHARPEN AND OUPUT
    final = sharpen(resized)
    f2 = os.path.join("output_gray", file)
    cv2.imwrite(f2,final)
    
    # cv2.imshow("Sharpen", edges)
    # cv2.waitKey(0)
    
    ## EDGES AND HOUGHLINES
    edges = cv2.Canny(final,450,500,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi, int(31))
    #print("# lines:", len(lines))
    # Draw the lines
    if lines is not None:
        print(filepath, len(lines))
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            #cv2.line(img, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)
    
    #cv2.imwrite('houghlines.png',img)


# plt.imshow(gray)
# plt.show()
