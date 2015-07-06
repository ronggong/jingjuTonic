# -*- coding: utf-8 -*-

import sys
sys.path.append("./src/")

import pitch as makamPitch
import BozkurtEstimation as be
import utilFunctionsRong as uf

def makamPitchExtract(filename, outputfilename = 'output_pitchtrack.txt', savePitchtrack = False):
    '''
    extract pitch track by sertan's makam method
    '''
    makam = makamPitch.PitchExtractMakam()
    makam.setup()
    pitches = makam.run(filename)

    pitchList = []
    for pitch in pitches:
        pitchList.append(pitch[1])

    if savePitchtrack == True:
        np.savetxt(outputFilename, pitchList)
        print 'pitch save at: ', outputFilename

    return pitchList

def jingjuTonic(filename, modelName = 'erhuang', outputfilename = 'output_pitchtrack.txt', savePitchtrack = False):
    '''
    calculate jingju Tonic
    '''

    # extract pitch track
    pitchList = makamPitchExtract(filename, outputfilename = 'output_pitchtrack.txt', savePitchtrack = False)

    b = be.BozkurtEstimation()
    
    # estimation
    tResult =  b.estimate(pitchList, mode_names=[], mode_name = modelName, est_tonic=True, est_mode=False, rank = 3, distance_method="euclidean", metric='pcd')

    # try octave error
    if tResult[0] > 500 and tResult[0] < 1000:
        estimateTonic = tResult[0] / 2.0
    if tResult[0] >= 1000:
        estimateTonic = tResult[0] / 4.0

    estimateTonic = estimateTonic.item() # convert numpy float type to native python float

    # convert hz to cents
    tonicCents = uf.hz2cents(estimateTonic)

    pitchStr = uf.cents2pitch(tonicCents)
    
    print 'estimated Tonic in Hz: ', estimateTonic, ' ', pitchStr

    return pitchStr

if __name__ == '__main__':
    filename = './01 白蛇传：青妹慢举龙泉宝剑.mp3' # erhuang
    jingjuTonic(filename, modelName = 'erhuang', outputfilename = 'output_pitchtrack.txt', savePitchtrack = False)
