from PIL import Image
import numpy as np
import utils as helper
import soundfile as sf
import librosa


y, sr = librosa.load("marigorudo.mp3", duration=10)

print(y.shape)
chroma = librosa.feature.chroma_stft(y=y,sr=sr)
print(chroma.shape)
print(round(librosa.frames_to_time(chroma.shape[1])) * sr)
print(librosa.frames_to_samples(chroma.shape[1]))
# y =  np.array(y[0:20000])
# sf.write("out.wav",y, sr)
frame = librosa.time_to_frames(10)
print(frame)

a = [1,2,3]
print(a[-1])