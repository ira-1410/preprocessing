import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# untouched spectrograms for the train & test set
# run the program twice, once for ~/Desktop/test then for ~/Desktop/training_norm

# load the song
file= pd.read_csv('/Users/ashna/Desktop/train.csv')
names_file = pd.DataFrame(file)

for index, row in names_file.iterrows():
    pathname = "/Users/ashna/Desktop/raw data/" + row['label'] +"/" + row['names']

    audio, sample_rate = librosa.load(pathname, sr=None, mono=False)

    for x in [0, 1]:
        # spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio[x], sr=sample_rate)
        log_mel_spec = librosa.power_to_db(mel_spec)

        # display as plot
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(log_mel_spec, x_axis="time", y_axis="mel", sr=sample_rate, fmax=20000, cmap='magma')
        # plt.colorbar(format="%+2.f dB")
        # plt.colorbar()
        plt.axis('off')

        # save the spectrogram as png
        save_path = "/Users/ashna/Desktop/training_norm/" + row['label']+ "/" + row['names'][:-4] + str(x + 1) + ".png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(row['names'] + " saved successfully to " + row['label'])

