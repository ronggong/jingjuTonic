# -*- coding: utf-8 -*-

import sys
sys.path.append("/home/rgong/MTG/tonic/rong")

import pitch as makamPitch
import PitchDistribution as p_d
import ModeFunctions as mf
import BozkurtEstimation as be
import matplotlib.pyplot as pl
import jingjuRecordingIDreader as jrIDreader
import utilFunctionsRong as UFR
import numpy as np
import tonic

def makamPitchExtract(filename, pitchTrackFolder, recordingID):
    makam = makamPitch.PitchExtractMakam()
    makam.setup()
    pitches = makam.run(filename)

    pitchList = []
    for pitch in pitches:
        pitchList.append(pitch[1])

    outputFilename = pitchTrackFolder + '/makamMethod/' + recordingID + '.txt'
    np.savetxt(outputFilename, pitchList)
    print 'pitch save at: ', outputFilename

def essentaiPitch(audioFilename, segFilename, pitchTrackFolder, recordingID):
    pitchTrackFilename = pitchTrackFolder + '/essentiaMethod/' + recordingID + '.txt'

    fs = 44100
    singingAudio = tonic.partExtraction(audioFilename, segFilename, fs, 'V')
    tonic.melodyExtraction(singingAudio, guessUnvoiced = True, frameSize = 2048, hopSize = 128, minFrequency = 20, maxFrequency = 20000, savePitch = True, filename = pitchTrackFilename)

def trainTest(trainSet, testSet, groundtruth, pitchMethod, counter):
    # erhuang train, test
    b = be.BozkurtEstimation()
    trainFilepathList = []
    trainRefTonicList = []
    
    for trainID in trainSet:
        try:
            pitchTrackFilename = pitchTrackFolder + '/'+ pitchMethod +'/' + trainID
            trainFilepathList.append(pitchTrackFilename)
            trainRefTonicList.append(groundtruth[trainID][0])
        except:
            pass

    print 'training pcd ... ...'
    jingju_pcd = b.train('jingju', trainFilepathList, trainRefTonicList, metric='pcd')
        
    for testID in testSet:
         pitchTrackFilename = pitchTrackFolder + '/'+ pitchMethod +'/' + testID + '.txt'
         pitchList = np.loadtxt(pitchTrackFilename)
         print 'testing ', pitchTrackFilename
         tResult =  b.estimate(pitchList, mode_names=[], mode_name='jingju', est_tonic=True, est_mode=False, rank = 3, distance_method="euclidean", metric='pcd')
         
         # try octave error
         for ii in range(1,5):
             if abs(tResult[0]/ii - groundtruth[testID][0]) < 10:
                 counter += 1
                 break

         print 'test result: ', tResult, ' groundtruth: ', groundtruth[testID][0]
         print 'correct counter: ', counter

    return counter

jingjuFolder = '/home/rgong/Music/jingjuzhixin'
segFolder = "/home/rgong/MTG/segmentation"
pitchTrackFolder = "/home/rgong/MTG/tonic/ModeTonicRecognition-master/pitchTrack"
jingjuRecordingIDs, jingjuRecordingPaths = jrIDreader.jingjuRecordingIDreader(jingjuFolder)

segPaths = []
for fileName in jingjuRecordingPaths:
    segFileName = segFolder + fileName[len(jingjuFolder):]
    lastPathPart = segFileName.split('/')[-1]
    segFileName = segFileName.replace(lastPathPart,"")
    lastPathPart = 'output_' + lastPathPart[:-4]
    lastPathPart = lastPathPart.replace(" ", "")
    segFileName = segFileName + lastPathPart + '/segAnnotationVJP.txt'
    segPaths.append(segFileName)

groundtruthFilename = './JingjuZhiXinGong.csv'
groundtruth = UFR.readJingjuZhiXinTonicGroundtruth(groundtruthFilename) # groundtruth 0: tonic, 1: erhuang=1 or xipi=2

'''melody extract'''
#for rid in groundtruth:
#    index = jingjuRecordingIDs.index(rid)
#    jingjuRecordingPaths[index]
#    essentaiPitch(jingjuRecordingPaths[index], segPaths[index], pitchTrackFolder, rid)
#    makamPitchExtract(jingjuRecordingPaths[index], pitchTrackFolder, rid)

'''split recordingIDs into train and test'''
erhuangRID = []
xipiRID = []
for rid in groundtruth:
    if groundtruth[rid][1] == 1:
        erhuangRID.append(rid)
    else:
        xipiRID.append(rid)

erhuangRID15len = int(len(erhuangRID)/5)
xipiRID15len = int(len(xipiRID)/5)

erhuangID5folds = []
xipiID5folds = []

erhuangID5folds.append(erhuangRID[:erhuangRID15len])
xipiID5folds.append(xipiRID[:xipiRID15len])

for ii in range(1,4):
    erhuangID5folds.append(erhuangRID[ii*erhuangRID15len + 1: (ii+1)*erhuangRID15len])
    xipiID5folds.append(xipiRID[ii*xipiRID15len + 1: (ii+1)*xipiRID15len])

erhuangID5folds.append(erhuangRID[4*erhuangRID15len + 1:])
xipiID5folds.append(xipiRID[4*xipiRID15len + 1:])

tonicCorrectNumErhuangMakam = 0
tonicCorrectNumXipiMakam = 0

tonicCorrectNumErhuangEssentia = 0
tonicCorrectNumXipiEssentia = 0

for ii in range(0,5):
    erhuangTestSet = erhuangID5folds[ii]
    erhuangTrainSet = [x for x in erhuangRID if x not in erhuangTestSet]
    xipiTestSet = xipiID5folds[ii]
    xipiTrainSet = [x for x in xipiRID if x not in xipiTestSet]
    
    # erhuang train, test
    tonicCorrectNumErhuangMakam = trainTest(erhuangTrainSet, erhuangTestSet, groundtruth, 'makamMethod', tonicCorrectNumErhuangMakam)
    
    # xipi train, test
    tonicCorrectNumXipiMakam = trainTest(xipiTrainSet, xipiTestSet, groundtruth, 'makamMethod', tonicCorrectNumXipiMakam)

    # erhuang train, test
    tonicCorrectNumErhuangEssentia = trainTest(erhuangTrainSet, erhuangTestSet, groundtruth, 'essentiaMethod', tonicCorrectNumErhuangEssentia)
    
    # xipi train, test
    tonicCorrectNumXipiEssentia = trainTest(xipiTrainSet, xipiTestSet, groundtruth, 'essentiaMethod', tonicCorrectNumXipiEssentia)

print tonicCorrectNumErhuangMakam, tonicCorrectNumXipiMakam, tonicCorrectNumErhuangEssentia, tonicCorrectNumXipiEssentia

accuracyErhuangMakam = tonicCorrectNumErhuangMakam / float(len(erhuangRID))
accuracyXipiMakam = tonicCorrectNumXipiMakam / float(len(xipiRID))
accuracyMakam = (tonicCorrectNumErhuangMakam + tonicCorrectNumXipiMakam) / float(len(erhuangRID) + len(xipiRID))

accuracyErhuangEssentia = tonicCorrectNumErhuangEssentia / float(len(erhuangRID))
accuracyXipiEssentia = tonicCorrectNumXipiEssentia / float(len(xipiRID))
accuracyEssentia = (tonicCorrectNumErhuangEssentia + tonicCorrectNumXipiEssentia) / float(len(erhuangRID) + len(xipiRID))

print accuracyErhuangMakam, accuracyXipiMakam, accuracyMakam,accuracyErhuangEssentia, accuracyXipiEssentia, accuracyEssentia


