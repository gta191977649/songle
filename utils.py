import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def b_spline(b_size = 5):
    #b_size
    bfilt = np.ones(b_size)
    spline = signal.fftconvolve(bfilt, signal.fftconvolve(bfilt, bfilt,mode='same'),mode='same')
    #spline = bfilt
    spline = spline/sum(spline)
    return spline

def horiFilter(img,filter):
    L = len(filter)
    #T = len(img)
    T = img.shape[1]
    halfL = int((L-1)/2)
    out = np.zeros(shape=(T,T))
    new_array = np.zeros(shape=(2 * halfL + img.shape[0], img.shape[1]))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if (i < L):
                new_array[i, j] = img[L - 1 + i, j]
            if (i > L + img.shape[0]):
                new_array[i, j] = img[L + img.shape[0] - 1 - i, j]
            if (i >= L and i <= L + img.shape[0]):
                new_array[i, j] = img[i, j]

    for t in range(halfL-1,T-1+halfL):
        for l in range(0,t-halfL+1):
            temp = 0
            for tau in range(-halfL,halfL):
                temp = temp + new_array[t+tau, l] * filter[tau + halfL]
                #if t+tau >= l and t+tau <= T-1:
                #    temp = temp + img[t+tau,l]*filter[tau+halfL]
                #if t+tau < l:
                #    print("l",l,"t",t,"tau",tau)
                #    temp = temp + img[2*l-(t+tau), l]*filter[tau+halfL]
                #if t+tau > T-1:
                #    temp = temp + img[t+tau-2*(T-1), l] * filter[tau + halfL]
            out[t-halfL+1,l] = temp
    return out

def vertFilter(img,filter):
    L = len(filter)
    T = len(img)
    halfL = int((L - 1) / 2)

    # T = len(img)
    out = np.zeros(shape=(T, T))
    new_array = np.zeros(shape=(img.shape[0], 2*halfL + img.shape[1]))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            if (j < L):
                new_array[i, j] = img[i, j+L-1]
            if (j > L + img.shape[0]):
                new_array[i, j] = img[i, L + img.shape[0] - 1 - j]
            if (j >= L and j <= L + img.shape[0]):
                new_array[i, j] = img[i, j]

    #for l in range(halfL-1,T-1+halfL):
        #for t in range(0,l-halfL+1):
    for t in range(0,T):
        for l in range(halfL-1,t):
            temp = 0
            for tau in range(-halfL, halfL):
                temp = temp + new_array[t, l+tau] * filter[tau + halfL]
            out[t,l-halfL+1] = temp
    return out