import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def b_spline():
    b_size = 25
    bfilt = np.ones(b_size)
    #spline = signal.fftconvolve(bfilt, signal.fftconvolve(bfilt, bfilt,mode='same'),mode='same')
    spline = bfilt
    spline = spline/sum(spline)
    return spline

def horiFilter(img,filter):
    L = len(filter)
    T = len(img)
    halfL = int((L-1)/2)
    out = np.empty(shape=(T,T))
    for t in range(0,T-1):
        for l in range(0,t):
            temp = 0
            for tau in range(-halfL,halfL):
                if t+tau >= 0 and t+tau <= T-1:
                    temp = temp+img[t+tau,l]*filter[tau+halfL]
                if t+tau < 0 or t+tau > T-1:
                    temp = temp + img[t-tau,l]*filter[tau+halfL]
            out[t,l] = temp
    return out

def vertFilter(img,filter):
    L = len(filter)
    T = len(img)
    halfL = int((L - 1) / 2)
    out = np.empty(shape=(T, T))
    for t in range(0,T-1):
        for l in range(0,t):
            temp = 0
            for tau in range(-halfL, halfL):
                if t+tau >= 0 and t+tau <= T - 1:
                    temp = temp + img[t,l+tau] * filter[tau+halfL]
                if t+tau < 0 or t+tau > T - 1:
                    temp = temp + img[t,l-tau] * filter[tau+halfL]
            out[t,l] = temp
    return out