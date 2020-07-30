'''
Created on Jun 20, 2019

@author: alexb
'''
import json

def cleanJSON(filename):
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
        
    newFilename = filename[:-5] + '_clean' + filename[-5:]
    with open(newFilename, 'w') as fp:
        json.dump(newDict, fp)
#         lst = []
#         newDict['people'].append(lst)
#         oldKeys = ['face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d', 'pose_keypoints_3d', \
#                    'face_keypoints_3d', 'hand_left_keypoints_3d', 'hand_right_keypoints_3d']
#         for k in oldKeys:
#             personDict.pop(k)
#     print(d['people'])

if __name__ == '__main__':
    cleanJSON("duke_000000000000_keypoints.json")
