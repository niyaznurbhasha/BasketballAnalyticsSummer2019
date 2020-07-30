'''
Created on Jun 24, 2019

@author: alexb
'''
# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os.path, cv2
import numpy as np
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
    # load the image and convert it from BGR to RGB so that
    # we can display it with matplotlib
    # image 1 good for Duke, 8 good for Kentucky, 7 is outlier for Duke
    image = cv2.imread(os.path.join('output_players', 'crop_1.png'))
    #image = cv2.imread('output.png', -1)
    clusters = 3
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     
    # show our image
    plt.figure()
    plt.axis("off")
    plt.imshow(image)
    #plt.show()

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    #print(len(image))
    #image = [row for row in image if row[3] != 0]
    #print(len(image))
    #image = np.float32([row for row in image if row != [255,255,255,0]])
    #print(image)
    # cluster the pixel intensities
    clt = KMeans(n_clusters = clusters)
    clt.fit(image)
    
    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)
    
    print(clt.cluster_centers_)
    lst = list(clt.labels_)
    print(lst.count(0), lst.count(1), lst.count(2))
    
    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()