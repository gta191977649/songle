import numpy as np
import soundfile as sf
import librosa

def outputSectionWav(audio, lag, start_frame, end_frame, sr=22050):
    f_s = librosa.frames_to_samples(start_frame)
    f_e = librosa.frames_to_samples(end_frame)

    #print(len(audio),f_s,f_e)
    data = np.array(audio[f_s:f_e])
    filename = "./segement/out_{}_{}-{}.wav".format(lag, start_frame, end_frame)
    sf.write(filename, data, sr)
   # print(filename)

