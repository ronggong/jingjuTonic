'''
for training jingju tonic model: erhuang and xipi
'''

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

def trainModel(trainSet, groundtruth, pitchMethod, modelName):
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

    print 'training pcd ... ...', modelName
    jingju_pcd = b.train(modelName, trainFilepathList, trainRefTonicList, metric='pcd')

    return

jingjuFolder = '/home/rgong/Music/jingjuzhixin' # folder of jing ju zhi xin audios
segFolder = "/home/rgong/MTG/segmentation" # folder of segmentation file, not been used
pitchTrackFolder = "/home/rgong/MTG/tonic/ModeTonicRecognition-master/pitchTrack" # folder of the pre-calculated pitch tracks
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

groundtruthFilename = './JingjuZhiXinGong.csv' # jing ju zhi xin ground truth file
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

'''train model'''
trainModel(erhuangRID, groundtruth, 'makamMethod', 'erhuang')
trainModel(xipiRID, groundtruth, 'makamMethod', 'xipi')

