'''
Created on Jul 16, 2019

@author: Student
'''
import matplotlib.pyplot as plt
import os.path, cv2
import numpy as np

d = {}
for file in os.listdir('output_players'):
    filepath = os.path.join('output_players', file)
    image = cv2.imread(filepath)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    rows, cols, n = img.shape
    
    #b = [31,41,100]
    b = [255,255,255]
    
    num = 0
    for i in range(rows):
        for j in range(cols):
            v = img[i][j]
            dist = np.linalg.norm(v-b)
            if dist < 200: ## 150 for blue 
                num += 1
                img[i][j] = [255,0,0]
            #print(dist)
    
    d[file] = num/(rows*cols)
    #f = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    f = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    #cv2.imshow("", f)
    #cv2.waitKey()
    fn = os.path.join("test", file)
    cv2.imwrite(fn, f)
    
s = sorted(d.items(), key = lambda x: -x[1])
for x in s:
    print(x[0], x[1])

#f = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
# nm = os.path.join("test", file)
# f = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imwrite(nm, f)
    
    # cv2.imshow("", f)
    # cv2.waitKey()
    #  
    # cv2.imwrite("test.png", f)
