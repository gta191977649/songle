import numpy as np
import soundfile as sf


def outputSectionWav(audio, lag, start_frame, end_frame, sr=22050):
    f_s = start_frame * sr
    f_e = end_frame * sr
    print(len(audio),f_s,f_e)
    data = np.array(audio[f_s:f_e])
    filename = "./segement/out_{}_{}-{}.wav".format(lag, start_frame, end_frame)
    sf.write(filename, data, sr)
    print(filename)
