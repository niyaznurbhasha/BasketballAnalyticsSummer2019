'''
Created on Jul 9, 2019

@author: Student
'''
import cv2, os.path, math
import numpy as np

for file in os.listdir('output_gray'):
    filepath = os.path.join('output_gray', file)
    src0 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Introduce consistency in width
    # const_width = 300
    # aspect = float(src.shape[0]) / float(src.shape[1])
    # src = cv2.resize(src, (const_width, int(const_width * aspect)))
    
    src = cv2.GaussianBlur(src0, (7,7), 0)
    
    # Apply gabor kernel to identify vertical edges
    g_kernel = cv2.getGaborKernel((4,4), 4, 0, 5, 0.5, 0, ktype=cv2.CV_32F)
    gabor = cv2.filter2D(src, cv2.CV_8UC3, g_kernel)
    
    # Visual the gabor kernel
    h, w = g_kernel.shape[:2]
    g_kernel = cv2.resize(g_kernel, (20*w, 20*h), interpolation=cv2.INTER_CUBIC)
    
    #cv2.imshow('src', src0)
    #cv2.imshow('gabor', gabor)  # gabor is just black
    #cv2.imshow('gabor kernel', g_kernel)
    #cv2.waitKey(0)
    
    f2 = os.path.join("output_gabor", file)
    cv2.imwrite(f2,gabor)
    
    ## HOUGHLINES
    edges = cv2.Canny(gabor,800,900,apertureSize = 3)
    lines = cv2.HoughLines(edges,1,np.pi, int(24))
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
    