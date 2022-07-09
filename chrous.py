import math
import numpy as np
import librosa
import utils as helper

def extractChroma(filename):
    y, sr = librosa.load(filename, duration=5)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    # Convert Chroma matrix to [ [time] => [Chroma..], ... ] forms align the data
    return chroma.T

def similarity(t,l,chroma):
    # structure with article
    length = chroma.shape[0]
    lag = l
    if lag < 0 or lag > length: raise Exception("lag must satifiy (0 <= l <= t)")

    # compute v1
    max_t1 = max(chroma[t])
    vt1 = chroma[t]
    v1 = vt1 / max_t1 if max_t1 > 0 else vt1

    # compute v2
    max_t2 = max(chroma[t-l])
    vt2 = chroma[t-l]
    v2 = vt2 / max_t2 if max_t2 > 0 else vt2
    return 1 - np.linalg.norm(v1-v2) / math.sqrt(12)

def calcSimilarity(chroma):
    # Note: similarity is the r(t,l) in article
    length = chroma.shape[0]
    r = np.empty(shape=(length,length))
    for t in range(0,length):
        for lag in range(0,t):
            r[t,lag] = similarity(t,lag,chroma)
    return r

def normalizeSimilarity(r):
    length = r.shape[0]
    r_norm = np.empty(shape=(length,length))
    lag_size = 15
    for frame in range(0,length-1):
        dir = {"left":0,"right":0,"up":0,"down":0,"upRight":0,"downRight":0}
        for lag in range(0,frame):
            for tau in range(1,lag_size):
                if frame-tau >= 1: dir["left"] = dir["left"] + r[frame-tau,lag] / lag_size
                if frame+tau < length: dir["right"] = dir["right"] + r[frame+tau,lag] / lag_size
                if lag+tau <= frame: dir["up"] = dir["up"] + r[frame,lag+tau] / lag_size
                if lag-tau >= 0: dir["down"] = dir["down"] + r[frame,lag-tau] / lag_size
                if frame+tau < length and lag + tau < length: dir["upRight"] = dir["upRight"] + r[frame+tau,lag+tau]/lag_size
                if frame-tau >= 0 and lag - tau >=0: dir["downRight"] = dir["downRight"] + r[frame-tau,lag-tau]/lag_size
            # find max mean. Step 2
            max_val = max([dir["left"],dir["right"],dir["up"],dir["down"],dir["upRight"],dir["downRight"]])
            min_val = min([dir["left"],dir["right"],dir["up"],dir["down"],dir["upRight"],dir["downRight"]])
            if max_val == dir["left"] or max_val == dir["right"]:
                r_norm[frame,lag] = r[frame,lag] - min_val
            else:
                r_norm[frame,lag] = r[frame,lag] - max_val
    return r_norm

def findPossibleLineSegements(r_norm):
    length = r.shape[0]
    r_all = np.empty(shape=(length,length))
    for frame in range(0,length-1):
        for lag in range(0,frame):
            tmp = 0
            for tau in range(lag,frame):
                tmp = tmp + r_norm[tau,lag]
            r_all[frame,lag] = tmp / (frame - lag + 1)
    return r_all
def smoothDifferential(r_all,t,l):
    k_size = 4
    r = 0
    for w in range(-k_size,k_size):
        # Check lag offest Boundary
        if l+w <= t and l+w > 0:
            r = r + w * r_all[t,l+w]

    return r

def pickUpPeaks(r_all):
    length = r_all.shape[0]
    for frame in range(0,length):
        smooth_diff =[]
        for lag in range(0, frame):
            diff = smoothDifferential(r_all,frame,lag)
            smooth_diff.append(diff)
        peaks_indices = np.where(np.diff(np.sign(smooth_diff)))[0]



        #print("a",frame)
        #print(len(smooth_diff))

if __name__ == '__main__':
    chroma = extractChroma("1.mp3")
    r = calcSimilarity(chroma)
    r_norm = normalizeSimilarity(r)
    peaks = pickUpPeaks(r_norm)

    spline = helper.b_spline()

    helper.horiFilter(r_norm,spline)
    #print(peaks)
    #print(r_all.shape)
    #print(r_all)

