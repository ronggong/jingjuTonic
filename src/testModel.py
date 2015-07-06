'''
For testing jingju tonic model, erhuang and xipi
'''

import BozkurtEstimation as be
import utilFunctionsRong as UFR
import numpy as np

def testModel(testSet, groundtruth, pitchMethod, modelName, counter):
    # erhuang train, test
    pitchTrackFolder = "/home/rgong/MTG/tonic/ModeTonicRecognition-master/pitchTrack"

    b = be.BozkurtEstimation()
        
    for testID in testSet:
         pitchTrackFilename = pitchTrackFolder + '/'+ pitchMethod +'/' + testID + '.txt'
         pitchList = np.loadtxt(pitchTrackFilename)
         print 'testing ', pitchTrackFilename
         tResult =  b.estimate(pitchList, mode_names=[], mode_name = modelName, est_tonic=True, est_mode=False, rank = 3, distance_method="euclidean", metric='pcd')
         
         # try octave error
         if tResult[0] > 500 and tResult[0] < 1000:
             estimateTonic = tResult[0] / 2.0
         if tResult[0] >= 1000:
             estimateTonic = tResult[0] / 4.0

         
         if abs(estimateTonic - groundtruth[testID][0]) < 10:
             counter += 1

         print 'test result: ', estimateTonic, ' groundtruth: ', groundtruth[testID][0]
         print 'correct counter: ', counter

    return counter

if __name__ == '__main__':
    groundtruthFilename = './JingjuZhiXinGong.csv'
    groundtruth = UFR.readJingjuZhiXinTonicGroundtruth(groundtruthFilename) # groundtruth 0: tonic, 1: erhuang=1 or xipi=2

    erhuangRID = []
    xipiRID = []
    for rid in groundtruth:
        if groundtruth[rid][1] == 1:
            erhuangRID.append(rid)
        else:
            xipiRID.append(rid)
    
    testModel(erhuangRID, groundtruth, 'makamMethod', 'erhuang', 0)
    testModel(xipiRID, groundtruth, 'makamMethod', 'xipi', 0)

