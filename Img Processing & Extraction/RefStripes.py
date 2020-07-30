'''
Created on Jul 10, 2019

@author: Student
'''
import cv2, os.path, math, scipy.stats
import numpy as np

d = {}
for file in os.listdir('output_gabor'):
    filepath = os.path.join('output_gabor', file)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    count = 0
    for i in range(img.shape[1]):
        col = img[:,i]
        #print(col)
        #print(np.array(scipy.stats.mode(col)))
        if np.median(col) == np.array(scipy.stats.mode(col))[0][0]:
            count += 1
    d[file] = count

s = sorted(d.items(), key = lambda x: -x[1])
for x in s:
    print(x[0], x[1])