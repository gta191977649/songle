import utils as helper
import matplotlib.pyplot as plt
import math

def plot(out):
    checkNan(out)
    plt.imshow(out,interpolation='none')
    plt.show()
    # x = []
    # y = []
    # for time, frame_val in enumerate(out):
    #
    #     for val in frame_val:
    #         print(val)
    #         x.append(time)
    #         y.append(val)
    # print("LEN: X:",len(x),"LEN: Y:",len(y))
    # plt.pause(2)
    # plt.scatter(x, y, s=0.5)
    # plt.show()
def plotConv(r_norm):
    spline = helper.b_spline()
    out = r_norm - helper.horiFilter(helper.vertFilter(r_norm, spline), spline)

    plot(out)

def plotHorilFilter(r_norm):
    spline = helper.b_spline()
    out = helper.horiFilter(r_norm, spline)
    plot(out)

def plotVertFilter(r_norm):
    spline = helper.b_spline()
    out = helper.vertFilter(r_norm, spline)
    plot(out)

def plotFilter():
    spline = helper.b_spline()
    plt.plot(spline)
    plt.show()

def checkNan(r):
    n_of_nan = 0
    for t,frame in enumerate(r):
        for idx,val in enumerate(frame):
            if math.isnan(val):
                print("FRAME:{},INDEX:{} is NAN!".format(t,idx))
                n_of_nan = n_of_nan + 1

    print("--------------")
    print("N OF NANs:",n_of_nan)
    print("NAN CHECK ENDS")