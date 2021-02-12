import audioread
import sounddevice as sd
import struct
import wave
import contextlib
import sys
import glob
from scipy.io import wavfile as wav
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import pandas as pd
import librosa
import wave
from scipy.io import wavfile as wav
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
out='/Users/MCA/Desktop/ANUJA B/sample/wav/'
inpath=input("PATH  :   ")
# s=input("PATH  :   ")
# l= r'/Users/MCA/Desktop/ANUJA B/sample/'
# inpath=l+s
# print(inpath)
ext = (inpath.split('/')[-1]).split('.')[-1]
# print(ext)    
#------------------------convert start -------------------------------------
filename = inpath
if(ext=='3gp'):        
    with audioread.audio_open(filename) as f:
#         print('Input file: %i channels at %i Hz; %.1f seconds.' %
#               (f.channels, f.samplerate, f.duration),
#               file=sys.stderr)
#         print('Backend:', str(type(f).__module__).split('.')[1],
#               file=sys.stderr)
        filename_new = out + filename.split('/')[-1]
        with contextlib.closing(wave.open(filename_new + '.wav', 'w')) as of:
            of.setnchannels(f.channels)
            of.setframerate(f.samplerate)
            of.setsampwidth(2)
            for buf in f:
                of.writeframes(buf)
            print(of)
elif(ext=='caf'):
    with audioread.audio_open(filename) as f:
#         print('Input file: %i channels at %i Hz; %.1f seconds.' %
#               (f.channels, f.samplerate, f.duration),
#               file=sys.stderr)
#         print('Backend:', str(type(f).__module__).split('.')[1],
#               file=sys.stderr)
        filename_new = out + filename.split('/')[-1]
        with contextlib.closing(wave.open(filename_new + '.wav', 'w')) as of:
            of.setnchannels(f.channels)
            of.setframerate(f.samplerate)
            of.setsampwidth(2)
            for buf in f:
                of.writeframes(buf)
elif(ext=='mp4'):
    with audioread.audio_open(filename) as f:
#         print('Input file: %i channels at %i Hz; %.1f seconds.' %
#               (f.channels, f.samplerate, f.duration),
#               file=sys.stderr)
#         print('Backend:', str(type(f).__module__).split('.')[1],
#               file=sys.stderr)
        filename_new = out + filename.split('/')[-1]
        with contextlib.closing(wave.open(filename_new + '.wav', 'w')) as of:
            of.setnchannels(f.channels)
            of.setframerate(f.samplerate)
            of.setsampwidth(2)
            for buf in f:
                of.writeframes(buf)
else:
    filename_new = filename
#-------------------------------end convert---------------------------------
    


# if(ext=='3gp'):
#     filename = inpath:
#     with audioread.audio_open(filename) as f:
#         print('Input file: %i channels at %i Hz; %.1f seconds.' %
#               (f.channels, f.samplerate, f.duration),
#               file=sys.stderr)
#         print('Backend:', str(type(f).__module__).split('.')[1],
#               file=sys.stderr)
#         filename_new = out_path + filename.split('\\')[-1]
#         with contextlib.closing(wave.open(filename_new + '.wav', 'w')) as of:
#             of.setnchannels(f.channels)
#             of.setframerate(f.samplerate)
#             of.setsampwidth(2)
#             for buf in f:
#                 of.writeframes(buf)

# inpath="/Users/MCA/Desktop/ANUJA B/sample/"
# f = inpath + 'hungry.wav'
# rate, wav_sample = wav.read(f)

#-----------------------------Feature Extraction----------------------------------------------------
f=filename_new+'.wav'
rate, wav_sample = wav.read(f)
if rate == 16000:
        wav_sample = wav_sample[:16000*5]
        wav_sample = wav_sample.astype(float)
        S1 = librosa.feature.melspectrogram(y=wav_sample[::2], sr=8000)
#         f, t, S1 = spectrogram(wav_sample[::2], fs=8000)
        #tmp = (10 * np.log10(S1)).ravel()
        tmp=S1.ravel()
        tmp = tmp/max(abs(tmp))
        tmp=[tmp]
        S2 = librosa.feature.melspectrogram(y=wav_sample[1::2], sr=8000)
#         f, t, S2 = spectrogram(wav_sample[1::2], fs=8000)
        #tmp = (10 * np.log10(S2)).ravel()
        tmp=S2.ravel()
        tmp = tmp/max(abs(tmp))
#         print(tmp.shape)
        tmp=[tmp]
elif rate == 8000:
        wav_sample = wav_sample[:8000*5]
        wav_sample = wav_sample.astype(float)
        S = librosa.feature.melspectrogram(y=wav_sample, sr=8000)
#         f, t, S = spectrogram(wav_sample, fs=8000)        
        #tmp = (10 * np.log10(S)).ravel()
        tmp=S.ravel()
        tmp = tmp/max(abs(tmp))
#         print(tmp.shape)
        tmp=[tmp]

    
# ------------------PREDICTION ------------------------------------------------------------
model_clone = joblib.load('my_model.pkl')
model_clone_pca = joblib.load('my_model_pca.pkl')
tmp1 = model_clone_pca.transform(tmp)
result = model_clone.predict(tmp1)
print (type(result[0]))
result = list(result[0])
if result == [1,0,0,0,0,0,0,0,0]:
    r = 'hungry'
elif result == [0,1,0,0,0,0,0,0,0]:
    r = 'needs burping'
elif result == [0,0,1,0,0,0,0,0,0]:
    r = 'belly pain'
elif result == [0,0,0,1,0,0,0,0,0]:
    r = 'discomfort'
elif result == [0,0,0,0,1,0,0,0,0] :
    r =  'tired'      
elif result == [0,0,0,0,0,1,0,0,0]:
    r = 'lonely'
elif result == [0,0,0,0,0,0,1,0,0]:
    r = 'cold/hot'
elif result == [0,0,0,0,0,0,0,1,0]:
    r = 'scared'
elif result == [0,0,0,0,0,0,0,0,1]:
    r = 'don\'t know'
else:
    r = 'no worries'
print(r)