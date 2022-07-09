import numpy as np

def b_spline():
    b_size = 201
    bfilt = np.empty(shape=(b_size))
    bfilt.fill(1)
    spline = np.convolve(bfilt, np.convolve(bfilt, bfilt))
    spline = spline / max(spline)
    return spline

def horiFilter(img,filter):
    L = len(filter)
    T = len(img)
    halfL = int((L-1)/2)
    out = np.zeros(len(img))

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

