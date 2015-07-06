import sys, csv, codecs, cStringIO
import numpy as np
import math
import intonation
import essentia.standard as ess
import matplotlib.font_manager as fm
from scipy.signal import freqz
from scipy.fftpack import fft, ifft, fftshift
from pylab import title as drawTitle
from pylab import show, savefig

droidTitle = fm.FontProperties(fname='font/DroidSansFallback.ttf', size = 16)

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def findMinDisFrameNum(t, vec_len, hop, fs):
    '''we find the minimum distance frame between t and a frame in vector,
    the principle is the distance decreases then increases'''
    oldDis = sys.float_info.max
    for p in range(1, vec_len):
        dis = abs(t - (p + 0.5) * hop / fs)
        if oldDis < dis:
            return p
        oldDis = dis
        
def meanValueRejectZero(vec, startFrameNum, endFrameNum):
    '''vector is a list'''
    vec = vec[startFrameNum:endFrameNum]
    out = vecRejectValue(vec, threshold = 0)
    return np.mean(out)
    
def stdValueRejectZero(vec, startFrameNum, endFrameNum):
    '''vector is a list'''
    vec = vec[startFrameNum:endFrameNum]
    out = vecRejectValue(vec, threshold = 0)
    return np.std(out)

def vecRejectValue(vec, vecRef = [], threshold = 0):
    out = []
    if len(vecRef) == 0:
        vecRef = vec
    for e in range(len(vecRef)):
        if vecRef[e] > threshold:
            out.append(vec[e])
    return out
    
def readSyllableMrk(syllableFilename):
    '''read syllable marker file'''
    inFile = codecs.open(syllableFilename, 'r', 'utf-8')
    
    title = None
    startMrk = []
    endMrk = []
    syl = []
    
    for line in inFile:
        fields = line.split()
        if len(fields) == 0:
            continue
        if not isfloat(fields[0]):
            #this line is title
            title = line.strip() # remove \n
        else:
            startMrk.append(float(fields[0]))
            endMrk.append(float(fields[1]))
            syl.append(fields[2])
            
    return (title, startMrk, endMrk, syl)

def readMelodiaPitch(inputFile):
    '''read syllable marker file'''
    inFile = open(inputFile, 'r')
    
    timeStamps = []
    pitch = []
    newPoints = [] # new point marker
    for line in inFile:
        fields = line.split()
        timeStamps.append(float(fields[0]))
        pitch.append(float(fields[1]))
        
        if len(fields) > 2:
            newPoints.append(fields[2])
        else:
            newPoints.append('')
            
    return (timeStamps, pitch, newPoints)

def readZhuGroundtruth(inputFile):
    inFile = open(inputFile, 'r')
    
    rDic = {}
    for line in inFile:
        fields = line.split()
        rDic[fields[0]] = float(fields[1])
            
    return rDic

def readJingjuZhiXinTonicGroundtruth(inputFile):
    tonicDict = {}

    with open(inputFile) as f:	
        for line in f:
            line = line[:-1]
            strs = line.split(',')
            if strs[0] != 'MBID' and strs[2] != '' and len(strs) == 4 and int(strs[3]) != 3:
                tonicDict[strs[0]] = (float(strs[2]), int(strs[3]))
    return tonicDict
    
def readYileSegmentation(inputFile):
    '''read Yile's segmentation file'''
    inFile = open(inputFile, 'r')
    
    startTime = []
    dur = []
    segNum = []
    segMarker = []
    
    for line in inFile:
        fields = line.split()
        if len(fields) == 4:
            startTime.append(float(fields[0]))
            dur.append(float(fields[2]))
            segNum.append(int(fields[1]))
            segMarker.append(fields[3])
    return (startTime, dur, segNum, segMarker)
        
def hz2cents(hz, tuning = 0):
    '''convert Hz to cents
    input: float num in Hz
    output: float num in cents
    
    if tuning is 0, 0 cents is C5'''
    assert type(hz) == float
    # cents = 1200 * np.log2(hz/(440 * pow(2,(0.25 + tuning))))
    
    tonic = 261.626
    cents = 1200*np.log2(1.0*hz/tonic)
    if math.isinf(cents):
        cents = -1.0e+04
    return cents
    
def hz2centsRafa(timeVec, pitchInHz, tonic=261.626, plotHisto = False, saveHisto = False, title = None):

    # with open(document, 'r') as f:
#         data = f.readlines()
# 
#     data2 = []
#     for i in range(len(data)):
#         x = []
#         time = float(data[i].split('\t')[0])
#         x.append(time)
#         value = float(data[i].split('\t')[1].rstrip('\r\n'))
#         x.append(value)
#         data2.append(x)

    cents = [-10000]*len(pitchInHz)
    for i in xrange(len(pitchInHz)):
        if pitchInHz[i] > 0:
            cents[i] = 1200*np.log2(1.0*pitchInHz[i]/tonic)
    data = zip(timeVec, cents)
    data_hist = np.array(data)

    pitch_obj = intonation.Pitch(data_hist[:, 0], data_hist[:, 1])
    #print data_hist[:,0], data_hist[:,1]
    rec_obj = intonation.Recording(pitch_obj)
    rec_obj.compute_hist()
    
    # draw this histogram
    rec_obj.histogram.plot()
    if not isinstance(title, unicode):
        title = title.decode('utf8') # decode the ascii to utf8
    if plotHisto == True:
        drawTitle(title, fontproperties=droidTitle)
        show()
    if saveHisto == True:
        drawTitle(title, fontproperties=droidTitle)
        savefig(title[:-4] + '-singingHisto.png', dpi=150, bbox_inches='tight')
        
    rec_obj.histogram.get_peaks()
    peaks = rec_obj.histogram.peaks
    return peaks['peaks']
    
def autolabelBar(rects, ax):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%.2f'%float(height),
                ha='center', va='bottom')

def lpcEnvelope(audioSamples, npts, order):
    '''npts is even number'''
    lpc = ess.LPC(order = order)
    lpcCoeffs = lpc(audioSamples)
    frequencyResponse = fft(lpcCoeffs[0], npts) 
    return frequencyResponse[:npts/2]
    
def spectralSlope(spec,frameSize,fs,xlim):
    startHz = xlim[0]
    endHz = xlim[1]
    freqRes = fs/float(frameSize)/2
    startP = np.round(startHz/freqRes)
    endP = np.round(endHz/freqRes)
    
    freqBins = np.arange(spec.shape[0])*freqRes
    xvals = freqBins[startP:endP]
    yvals = spec[startP:endP]
    
    a, b = np.polyfit(xvals, yvals, 1)
    # a is slope
    return (a, b)
    
class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") if isinstance(s, basestring) else s for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)

def pitch2letter(pitch):
    letters = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
    letter = letters[pitch]
    return letter
    
def cents2pitch(cents, regDefault = 4):
    intPart = int(cents/100.00)
    remPart = cents%100.00
    if cents >= 0:
        if remPart > 50:
            intPart += 1
            remPart = remPart - 100
    else:
        if remPart > 50:
            remPart = remPart - 100
        else:
            intPart -= 1
    
    regAug = int(intPart/12) # register augmentation
    pitch = (intPart%12)
    #print pitch
    
    pitchLetter = pitch2letter(pitch)
    reg = regDefault + regAug
    
    if remPart >= 0:
        returnStr = pitchLetter + str(reg) + ' + ' + str(round(remPart,2)) + ' cents'
    else:
        returnStr = pitchLetter + str(reg) + ' - ' + str(round(abs(remPart),2)) + ' cents'

    return returnStr
    
