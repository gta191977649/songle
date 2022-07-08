import math

import numpy as np
import librosa
from utils import *

def loadChroma(filename):
    y, sr = librosa.load(filename, duration=15)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma
def calcSimilarity(chroma):
    R = []
    T = len(chroma[0])
    v = np.array(chroma).T
    for t in range(0,T-1):
        for l in range(0,t-1):
            s = []
            for i in range(0,12):
                m1 = max(v[t])
                m2 = max(v[t-1])
                # be aware of NAN!
                v1 = v[t] / m1
                v2 = v[t-1] / m2
                r = 1 - abs(v1-v2) / math.sqrt(12)
                s.append(r)


    print(s)





if __name__ == '__main__':
    chroma = loadChroma("./1.mp3")
    calcSimilarity(chroma)