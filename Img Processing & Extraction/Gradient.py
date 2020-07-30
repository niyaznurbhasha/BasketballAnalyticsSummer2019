'''
Created on Jul 9, 2019

@author: Student
'''
import cv2, os.path, math
import numpy as np
from sklearn.cluster import KMeans

d = {}
for file in os.listdir('output_gabor'):
    filepath = os.path.join('output_gabor', file)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    cols = []
    
    for i in range(img.shape[1] - 1):
        col0 = img[:,i]
        col1 = img[:,i+1]
        cols.append(abs(col1 - col0))
        #print(img)
        #print(abs(img[:,1] - img[:,0]))
        #print(abs(img[:,2] - img[:,1]))
    
    gradient = np.column_stack(tuple(cols))
    d[file] = gradient

s = sorted(d.items(), key = lambda x: -x[1].sum())
for x in s:
    print(x[0], x[1].sum())

## CLUSTERING
# gradVectors = []
# labels = [0,0,0,1,0,1,0,0,0,0,0,0]
# for file, gradient in d.items():
#     gradVectors.append(gradient.flatten())
# X = np.array(gradVectors)
# 
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
# #print(kmeans.cluster_centers_)
# print(kmeans.labels_)
    
    #cv2.imshow("img", img)
    #cv2.waitKey(0)