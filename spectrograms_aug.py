import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# spectrograms with data augmentation done (time shift and masking)
# saved in a seperate folder


def spec_augment(og_spec, freq_max_percent=0.15, time_max_percent=0.2):
    num_of_frames, num_of_freqs = og_spec.shape

    augmented_melspec = og_spec.copy()

    # Frequency masking
    freq_mask_percentage = random.uniform(0.0, freq_max_percent)
    num_freqs_to_mask = int(freq_mask_percentage * num_of_freqs)
    f0 = int(np.random.uniform(low=0.0, high=(num_of_freqs - num_freqs_to_mask)))

    augmented_melspec[:, f0:(f0 + num_freqs_to_mask)] = 0

    # Time masking
    time_percentage = random.uniform(0.0, time_max_percent)
    num_frames_to_mask = int(time_percentage * num_of_frames)
    t0 = int(np.random.uniform(low=0.0, high=(num_of_frames - num_frames_to_mask)))

    augmented_melspec[t0:(t0 + num_frames_to_mask), :] = 0

    return augmented_melspec




file= pd.read_csv('/Users/ashna/Desktop/train.csv')
names_file = pd.DataFrame(file)

for index, row in names_file.iterrows():
    pathname = "/Users/ashna/Desktop/raw data/" + row['label'] +"/" + row['names']
    audio, sample_rate = librosa.load(pathname, sr=None, mono=False)
    timeshift = np.roll(audio, random.randint(360000, 1080000), axis=1)

    for x in [0, 1]:
        mel_spec = librosa.feature.melspectrogram(y=timeshift[x], sr=sample_rate)
        log_mel_spec = librosa.power_to_db(mel_spec)
        final_ = spec_augment(log_mel_spec)


        # display as plot
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(final_, x_axis="time", y_axis="mel", sr=sample_rate, fmax=20000, cmap='magma')
        plt.axis('off')

        # save the spectrogram as png
        save_path = "/Users/ashna/Desktop/training_aug/" + row['label']+ "/" + row['names'][:-4] + str(x + 1) + " aug.png"
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

