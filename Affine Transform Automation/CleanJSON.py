'''
Created on Jun 20, 2019

@author: alexb
'''
import json, os

def cleanJSON(filename, write = False):
    f = open(filename)
    fileText = f.read()
    d = json.loads(fileText)
    newDict = {'people': []}
    for personDict in d['people']:
        newPerson = {}
        for i in range(0, len(personDict['pose_keypoints_2d']), 3):
            thisLst = personDict['pose_keypoints_2d']
            newPerson['Keypoint ' + str(i//3)] = [thisLst[i], thisLst[i+1], thisLst[i+2]]
        newDict['people'].append(newPerson)
    #print(filename)
    if write:
        newFilename = os.path.join("keypoints_rendered_clean", 'duke_' + filename.split("_")[2] + '_clean.json')
        with open(newFilename, 'w') as fp:
            json.dump(newDict, fp)
    return newDict

if __name__ == '__main__':
    pass
