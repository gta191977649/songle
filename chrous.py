import math
import numpy as np
import librosa
import soundfile as sf
import utils as helper
import config as CONSTANT
from skimage.filters import try_all_threshold, threshold_otsu
import debug as debug
import matplotlib.pyplot as plt


class Chrous:
    def __init__(self, file):
        self.file = file

    def extractChroma(self, filename):
        y, sr = librosa.load(filename)
        sf.write("out.wav",y,sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        # Convert Chroma matrix to [ [time] => [Chroma..], ... ] forms align the data
        return chroma.T

    def similarity(self, t, l, chroma):
        # structure with article
        length = chroma.shape[0]
        lag = l
        if lag < 0 or lag > length: raise Exception("lag must satifiy (0 <= l <= t)")

        # compute v1
        max_t1 = max(chroma[t])
        vt1 = chroma[t]
        v1 = vt1 / max_t1 if max_t1 > 0 else vt1

        # compute v2
        max_t2 = max(chroma[t - l])
        vt2 = chroma[t - l]
        v2 = vt2 / max_t2 if max_t2 > 0 else vt2
        return 1 - np.linalg.norm(v1 - v2) / math.sqrt(12)

    def calcSimilarity(self, chroma):
        # Note: similarity is the r(t,l) in article
        length = chroma.shape[0]
        r = np.zeros(shape=(length, length))
        for t in range(0, length):
            for lag in range(0, t):
                r[t, lag] = self.similarity(t, lag, chroma)
        return r
    def calculate_r_all(self,r_norm):
        T = r_norm.shape[0]
        r_all = np.zeros(shape=(T, T))
        for t in range(0,T):
            for lag in range(0,t):
                temp = 0
                for tau in range(lag,t):
                    temp = temp + r_norm[tau,lag]

                r_all[t,lag] = temp / (t-lag+1)
        return r_all
    def normalizeSimilarity(self, r):
        length = r.shape[0]
        r_norm = np.zeros(shape=(length, length))
        lag_size = 15
        for frame in range(0, length - 1):
            dir = {"left": 0, "right": 0, "up": 0, "down": 0, "upRight": 0, "downRight": 0}
            for lag in range(0, frame):
                for tau in range(0, lag_size):
                    if frame - tau >= 1: dir["left"] = dir["left"] + r[frame - tau, lag] / lag_size
                    if frame + tau < length: dir["right"] = dir["right"] + r[frame + tau, lag] / lag_size
                    if lag + tau <= frame: dir["up"] = dir["up"] + r[frame, lag + tau] / lag_size
                    if lag - tau >= 0: dir["down"] = dir["down"] + r[frame, lag - tau] / lag_size
                    if frame + tau < length and lag + tau < length: dir["upRight"] = dir["upRight"] + r[
                        frame + tau, lag + tau] / lag_size
                    if frame - tau >= 0 and lag - tau >= 0: dir["downRight"] = dir["downRight"] + r[
                        frame - tau, lag - tau] / lag_size
                # find max mean. Step 2
                max_val = max([dir["left"], dir["right"], dir["up"], dir["down"], dir["upRight"], dir["downRight"]])
                min_val = min([dir["left"], dir["right"], dir["up"], dir["down"], dir["upRight"], dir["downRight"]])
                if max_val == dir["left"] or max_val == dir["right"]:
                    r_norm[frame, lag] = r[frame, lag] - min_val
                else:
                    r_norm[frame, lag] = r[frame, lag] - max_val
        return r_norm

    def smoothed(self, r_all, t, l):
        k_size = 4
        r = 0
        for w in range(-k_size, k_size):
            # Check lag offest Boundary
            if l + w <= t and l + w > 0:
                r = r + w * r_all[t, l + w]
        return r

    def movingAverageFilter(self, r):
        filter = helper.b_spline()
        r_all = r - helper.horiFilter(helper.vertFilter(r, filter), filter)
        return r_all

    # Old approach use change sign
    def pickpPeaks_old(self, r_all):
        peaks = []
        # apply b-spline filter
        r_all = self.movingAverageFilter(r_all)
        length = r_all.shape[0]
        for frame in range(0, length):
            smooth_diff = []
            for lag in range(0, frame):
                # apply smooth differential
                diff = self.smoothed(r_all, frame, lag)
                smooth_diff.append(diff)

            peaks_indices = np.where(np.diff(np.sign(smooth_diff)))[0].tolist()
            # print(peaks_indices)
            peaks.append(peaks_indices)
        return peaks

    def findSegements(self, r_all):
        #frame_length = CONSTANT.FRAME_LEN * CONSTANT.THRESHOLD_LEN * 1000
        frame_length = 300

        #  === [ Find Threshold ] ===
        # Doing discriminantCriterion
        # Transpose r_all matrix
        r_t = r_all.T
        r_t = self.removeDiagonal(r_t)
        r_merged = r_all + r_t
        threshold = self.discriminantCriterion(r_merged)
        T = r_all.shape[0]
        print("Singal Length:",T)
        print("Threshold:",threshold)
        r_debug = np.zeros(shape=(T, T))
        found_segements = []
        for frame in range(0, T):
            segement = []
            for lag in range(0,frame):
                # Check threshold
                similarity = r_all[frame,lag]
                #similarity = self.smoothed(r_all, frame, lag)
                if similarity > threshold:
                    r_debug[frame,lag] = 1
                    segement.append({
                        "time":lag,
                        "similarity":similarity
                    })
                # ensure buffered continues time is more than 6.4s in minium
                elif len(segement) > 0 and len(segement) < frame_length:
                    segement.clear()
            if len(segement) > frame_length:
                found_segements.append({
                    "start":segement[0]["time"],
                    "end":segement[len(segement)-1]["time"],
                })
            # clear buffer in the very last end
            segement.clear()
        print(r_debug)
        debug.plot(r_debug)
        print("Found Segements: {}".format(len(found_segements)))
        for s in found_segements:
            print(s)
        return found_segements
    def discriminantCriterion(self, r_all):
        thresh = threshold_otsu(r_all.ravel())
        # out = r_all > thresh
        # from PIL import Image
        # pil_image = Image.fromarray(out)
        # pil_image.show()
        # print(threshold)
        # fig.show()
        return thresh

    def removeDiagonal(self, r_all):
        T = r_all.shape[0]
        for frame in range(0, T):
            r_all[frame, frame] = 0
        return r_all

    def detect(self):
        # 1. Extract feature
        chroma_vec = self.extractChroma(self.file)
        # 2. Calculate similarity between chroma vectors
        r = self.calcSimilarity(chroma_vec)
        # 3. List repeated sections
        r_norm = self.normalizeSimilarity(r)
        r_all = self.calculate_r_all(r_norm)

        debug.plot(r_all)
        # r_peaks = self.pickpPeaks(r_norm)

        self.findSegements(r_norm)
        # print(r_threshold)
        #debug.plot(r_norm)
        # r_f = self.movingAverageFilter(r_norm)
        # print(r_f)
        # print(r_threshold)
        # Debug ...
        # print(r_norm)
        # debug.checkNan(r_norm)

        # debug.plotHorilFilter(r_norm)
        # debug.plotVertFilter(r_norm)


if __name__ == '__main__':
    chrous = Chrous("1.mp3")
    chrous.detect()
