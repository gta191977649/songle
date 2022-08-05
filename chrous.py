import math
import numpy as np
import librosa
import soundfile as sf
import utils as helper
import config as CONF
from skimage.filters import try_all_threshold, threshold_otsu
import debug as debug
import matplotlib.pyplot as plt
import json
import filewriter as file

class Chrous:
    def __init__(self, file,len = None):
        self.file = file
        self.len = len
        self.chroma,self.audio = self.extractChroma(file)

    def extractChroma(self, filename):
        y, sr = librosa.load(filename,mono=True, duration=self.len)
        #sf.write("out.wav", y, sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        # Convert Chroma matrix to [ [time] => [Chroma..], ... ] forms align the data
        return chroma.T,y

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

    def calculate_r_all(self, r_norm):
        T = r_norm.shape[0]
        r_all = np.zeros(shape=(T, T))
        for t in range(0, T):
            for lag in range(0, t):
                temp = 0
                for tau in range(lag, t):
                    temp = temp + r_norm[tau, lag]

                r_all[t, lag] = temp / (t - lag + 1)
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
        for w in range(-k_size, k_size+1):
            # Check lag offest Boundary
            if l + w <= t and l + w > 0:
                r = r + w * r_all[t, l + w]
        return r

    def smoothDifferntial(self, r):
        T = r.shape[0]
        out = np.zeros(shape=(T, T))
        for frame in range(0, T):
            for lag in range(0, frame):
                out[frame, lag] = self.smoothed(r, frame, lag)
        return out

    def applyMovingAverageFilter(self, r):
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

    # r_all: the simaritly after normalization
    # r: the initial simaritly obtain by calculaye similarity function
    def findSegements(self, r_all, r):
        FRAME_LENGTH = librosa.time_to_frames(CONF.FRAME_TIME)
        print("MINIMUM FRAME LEN: {} in {}s".format(FRAME_LENGTH,CONF.FRAME_TIME))
        # frame_length = CONSTANT.FRAME_LEN * CONSTANT.THRESHOLD_LEN * 1000
        # 1. Obtain smooth r_all
        #r_smooth = self.smoothDifferntial(r_all)
        # debug.plot(r_smooth)
        # 2. Apply moving average filter (b-spline)

        r_filtered = self.applyMovingAverageFilter(r)
        print("MAX",np.max(r),"MIN",np.min(r))
        print("MAX",np.max(r_filtered),"MIN",np.min(r_filtered))
        # 3. Find Threshold based on r_filtered
        r_trans = r_filtered.T
        threshold = self.findThreshold(r_filtered + r_trans)
        # print("max",np.max(r_filtered.flatten()),"min",np.min(r_filtered.flatten()))

        print("THRESHOLD", threshold)
        T = len(r_filtered)
        r_thr = np.zeros(shape=(T,T))

        # Check points
        for lag in range(0, T):
            for frame in range(0, lag):
                s = r_filtered[lag, frame]
                if s > threshold:
                    r_thr[lag,frame] = 1
                else:
                    r_thr[lag, frame] = 0

        # find the segements only > FRAME_LENGTH
        print("FILTER SEGMENTS > ",FRAME_LENGTH)
        segements = {}
        for lag in range(0, T):
            head_ptr = -1
            for current_ptr in range(lag,T):
                s = r_thr[current_ptr, lag]
                if s == 1:
                    if head_ptr == -1:
                        head_ptr = current_ptr
                elif head_ptr != -1: # only check when head pointer is being set
                    # This is the section what we want
                    if current_ptr - head_ptr >= FRAME_LENGTH:
                        s_2 = [head_ptr, current_ptr]
                        s_1 = [head_ptr-lag,current_ptr-lag]

                        print(s_1,"->", s_2)
                        print("LAG", lag, "FOUND SECTION LEN", current_ptr - head_ptr)
                        file.outputSectionWav(self.audio,lag,s_1[0],s_1[1])
                        file.outputSectionWav(self.audio,lag,s_2[0],s_2[1])
                    else: #otherwise, removed the array set to 0
                        for ptr in range(head_ptr,current_ptr):
                            r_thr[ptr,lag] = 0
                    # reset head_pointer
                    head_ptr = -1
        debug.plot(r_thr)

    # def findSegements(self, r_all, r):
    #     FRAME_LENGTH = librosa.time_to_frames(CONF.FRAME_TIME)
    #     print("MINIMUM FRAME LEN: {} in {}s".format(FRAME_LENGTH,CONF.FRAME_TIME))
    #     # frame_length = CONSTANT.FRAME_LEN * CONSTANT.THRESHOLD_LEN * 1000
    #     # 1. Obtain smooth r_all
    #     r_smooth = self.smoothDifferntial(r_all)
    #     # debug.plot(r_smooth)
    #     # 2. Apply moving average filter (b-spline)
    #     r_filtered = self.applyMovingAverageFilter(r)
    #     print("MAX",np.max(r_filtered),"MIN",np.min(r_filtered))
    #     # 3. Find Threshold based on r_filtered
    #     threshold = self.findThreshold(r_filtered)
    #     # print("max",np.max(r_filtered.flatten()),"min",np.min(r_filtered.flatten()))
    #
    #     print("THRESHOLD", threshold)
    #     T = len(r_filtered)
    #     r_thr = np.zeros(shape=(T,T))
    #
    #     # Check points
    #     segements = {}
    #     for lag in range(0, T):
    #         segement_buffer = []
    #         for frame in range(0, lag):
    #             #print(frame, lag)
    #             s = r_filtered[lag, frame]
    #             if s > threshold: # append to buffer, and ensure it's continues values
    #                 segement_buffer.append({
    #                     "lag":lag,
    #                     "frame":frame,
    #                 })
    #                 # High light it anyway
    #                 r_thr[lag, frame] = 1
    #             elif s < threshold or frame == lag-1:
    #                 # Drop if the segement if is less than the frame target len
    #                 if len(segement_buffer) < FRAME_LENGTH:
    #                     segement_buffer.clear()
    #                 else:
    #                     if not lag in segements:
    #                         segements[lag] = {
    #                             "sections": [],
    #                         }
    #                     ts = librosa.frames_to_time(segement_buffer[0]["frame"])
    #                     td = librosa.frames_to_time(segement_buffer[len(segement_buffer)-1]["frame"])
    #                     segements[lag]["sections"].append({
    #                         "start": ts,
    #                         "end": td,
    #                     })
    #                     segement_buffer.clear()
    #                 r_thr[lag, frame] = 0
    #     debug.plot(r_thr)
    #     return segements



    # Apply Discriminant Criterion (Find threshold)
    def findThreshold(self, r_all):
        thresh = threshold_otsu(r_all.flatten())
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
        #chroma_vec = self.extractChroma(self.file)
        chroma_vec = self.chroma
        # 2. Calculate similarity between chroma vectors
        r = self.calcSimilarity(chroma_vec)
        # 3. List repeated sections
        r_norm = self.normalizeSimilarity(r)
        # 3.1 Calc R_all
        r_all = self.calculate_r_all(r_norm)

        segements = self.findSegements(r_all, r)
        # r_peaks = self.pickpPeaks(r_norm)

        # self.findSegements(r_norm)
        # print(r_threshold)
        # debug.plot(r_norm)
        # r_f = self.movingAverageFilter(r_norm)
        # print(r_f)
        # print(r_threshold)
        # Debug ...
        # print(r_norm)
        # debug.checkNan(r_norm)

        # debug.plotHorilFilter(r_norm)
        # debug.plotVertFilter(r_norm)

        return segements
if __name__ == '__main__':
    sample_length = 30
    chrous = Chrous("marigorudo.mp3",sample_length)
    segments = chrous.detect()

    # Write segments to file
    # y, sr = librosa.load("marigorudo.mp3", mono=True, duration=sample_length)
    # for lag in segments:
    #     s_len = len(segments[lag]["sections"])
    #     s = segments[lag]["sections"]
    #     idx = 0
    #     if s_len >= 2:
    #         for section in s:
    #             #Start frame
    #             f_s = round(librosa.time_to_frames(section["start"])) * sr
    #             f_e = round(librosa.time_to_frames(section["end"])) * sr
    #             data = np.array(y[f_s:f_e])
    #             filename = "./segement/out_{}_{}.wav".format(lag,idx)
    #             sf.write(filename,data, sr)
    #             idx = idx + 1
    #             print(filename)
    #             print(section)

    #print(segments)
