'''
Created on Jun 24, 2019

@author: alexb
'''
# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os, cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

#import utils

# import the necessary packages
import numpy as np
import cv2
 
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
 
    # return the histogram
    return hist

def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
 
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
    
    # return the bar chart
    return bar

if __name__ == '__main__':
    topVectors = []
    l = []
    for file in os.listdir('output_players'):
        l.append(file.split("_")[1])
        filepath = os.path.join('output_players', file)
        print(filepath)

        image = cv2.imread(filepath)
        clusters = 3
         
        #load the image and convert it from BGR to RGB so that
        #we can dispaly it with matplotlib
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  ## RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  ## LAB color space
           
    #     # show our image
    #     plt.figure()
    #     plt.axis("off")
    #     plt.imshow(image)
    #     #plt.show()
          
        # reshape the image to be a list of pixels
        image = image.reshape((image.shape[0] * image.shape[1], 3))
         
        # cluster the pixel intensities
        clt = KMeans(n_clusters = clusters)
        clt.fit(image)
         
    #     # build a histogram of clusters and then create a figure
    #     # representing the number of pixels labeled to each color
    #     hist = centroid_histogram(clt)
    #     bar = plot_colors(hist, clt.cluster_centers_)
        
        centers = clt.cluster_centers_
        lst = list(clt.labels_)
        counts = [lst.count(0), lst.count(1), lst.count(2)]
        #print(clt.cluster_centers_)
        #print(counts)
        
        maxVectorIndex = counts.index(max(counts))
        maxVector = clt.cluster_centers_[maxVectorIndex]
        topVectors.append(np.delete(maxVector,0))
    
    #     # show our color bart
    #     plt.figure()
    #     plt.axis("off")
    #     plt.imshow(bar)
    #     plt.show()
    
    realLabels = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
    #realLabelsWithRefs = [0, 1, 1, 2, 0, 2, 1, 0, 0, 1, 0, 1]
    X = np.array(topVectors)
    print(X)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    #print(kmeans.cluster_centers_)
    print(l)
    print(kmeans.labels_)
    print(kmeans.labels_ == realLabels)
    print(kmeans.inertia_)
    
    x = [v[0] for v in topVectors]
    y = [v[1] for v in topVectors]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    colorV = ['goldenrod', 'blue', 'black']
    #ax.scatter(x, y)
    #plt.xlabel("Blue (-) vs. Yellow (+)")
    #plt.ylabel("Dark (-) vs. Light (+)")
    #plt.title("LAB Space")
    ax.scatter(x, y, c=[colorV[l] for l in kmeans.labels_], marker = "o")
#     ax.scatter([kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[1][0]], 
#                [kmeans.cluster_centers_[0][2], kmeans.cluster_centers_[1][2]],
#                color = 'black', marker = "*")
    plt.show()
    
    
