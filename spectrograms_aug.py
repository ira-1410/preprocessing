import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

pathname = "/Users/ashna/Desktop/raw data/heavy metal/Master of Puppets (Remastered) [E0ozmU9cJDg].wav"
audio, sample_rate = librosa.load(pathname, sr=None, mono=False)

for x in [0, 1]:

    mel_spec = librosa.feature.melspectrogram(y=audio[x], sr=sample_rate)
    log_mel_spec = librosa.power_to_db(mel_spec)

    # display as plot
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(log_mel_spec, x_axis="time", y_axis="mel", sr=sample_rate, fmax=20000, cmap='magma')
    plt.axis('off')

    # save the spectrogram as png
    save_path = "/Users/ashna/Desktop/original" + str(x+1) + ".png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)






timeshift = np.roll(audio, random.randint(360000, 1080000), axis=1)

for x in [0, 1]:
    mel_spec = librosa.feature.melspectrogram(y=timeshift[x], sr=sample_rate)
    log_mel_spec = librosa.power_to_db(mel_spec)

    # display as plot
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(log_mel_spec, x_axis="time", y_axis="mel", sr=sample_rate, fmax=20000, cmap='magma')
    plt.axis('off')

    # save the spectrogram as png
    save_path = "/Users/ashna/Desktop/timeshift" + str(x+1) + ".png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
