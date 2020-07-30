'''
Created on Jul 10, 2019

@author: Student
'''
import cv2, os.path, math, scipy.stats
import numpy as np
from Sharpen import sharpen
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

MAX_POSITIVES = False
BLUR = True

d = {}

## GRAYSCALE & HOUGHLINES
for file in os.listdir('output_small'):
    filepath = os.path.join('output_small', file)
    img = cv2.imread(filepath)
    d[file] = []
    
    rows,cols,ch = img.shape   
     
    ## CONVERT TO GRAYSCAPE AND EQUALIZE
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    ## RESIZE IMAGE
    dim = (50, 75)
    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    
    ## SHARPEN AND OUPUT
    final = sharpen(resized)
    f2 = os.path.join("output_gray", file)
    cv2.imwrite(f2,final)
    
    if not MAX_POSITIVES:
        ## EDGES AND HOUGHLINES
        edges = cv2.Canny(final,450,500,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi, int(27))
    
        if lines is None:
            d[file].append(0)
        else:
            d[file].append(len(lines))
    
## GABOR
for file in os.listdir('output_gray'):
    filepath = os.path.join('output_gray', file)
    src0 = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if BLUR:
        src0 = cv2.GaussianBlur(src0, (7,7), 0)
    
    # Apply gabor kernel to identify vertical edges
    g_kernel = cv2.getGaborKernel((4,4), 4, 0, 5, 0.5, 0, ktype=cv2.CV_32F)
    gabor = cv2.filter2D(src0, cv2.CV_8UC3, g_kernel)
    
    # Visual the gabor kernel
    h, w = g_kernel.shape[:2]
    g_kernel = cv2.resize(g_kernel, (20*w, 20*h), interpolation=cv2.INTER_CUBIC)
    
    # Write file
    f2 = os.path.join("output_gabor", file)
    cv2.imwrite(f2,gabor)
    
## COLUMN SIMILARITY
for file in os.listdir('output_gabor'):
    filepath = os.path.join('output_gabor', file)
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    
    count = 0
    for i in range(img.shape[1]):
        col = img[:,i]
        if np.median(col) == np.array(scipy.stats.mode(col))[0][0]:
            count += 1
    
    if MAX_POSITIVES:
        ## EDGES AND HOUGHLINES
        edges = cv2.Canny(img,550,600,apertureSize = 3)
        lines = cv2.HoughLines(edges,1,np.pi, int(27))
    
        if lines is None:
            d[file].append(0)
        else:
            d[file].append(len(lines))
    
    d[file].append(count)

## CREATE FEATURES & CLUSTER
s = sorted(d.items())
#s = sorted(d.items(), key = lambda x: int(x[0].split("_")[1].split(".")[0]))
features = []
for pair in s:
    features.append(np.array(pair[1]))

X = np.array(features)
print(X)
kmeans = KMeans(n_clusters=2, random_state=10).fit(X)

#realLabels = [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]
#print(kmeans.cluster_centers_)
print(kmeans.labels_)
#print(kmeans.labels_ == realLabels)
print(kmeans.inertia_)

x = [v[0] for v in X]
y = [v[1] for v in X]
fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x,y)
ax.scatter([v[0] for v in kmeans.cluster_centers_], [v[1] for v in kmeans.cluster_centers_], color = 'red', marker = "*")
plt.show()
