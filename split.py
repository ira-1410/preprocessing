import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import model_selection

dataframe = pd.read_csv("/Users/ashna/Desktop/sheet.csv")
train, test = sklearn.model_selection.train_test_split(dataframe, train_size=0.8)
train.to_csv("/Users/ashna/Desktop/train.csv")
test.to_csv("/Users/ashna/Desktop/test.csv")


#pathname= "/Users/ashna/Desktop/raw data/heavy metal/Cannibal Corpse - Hammer Smashed Face (OFFICIAL) [vlgiWBCbCJk].wav"

#audio, sample_rate = librosa.load(pathname, sr=None, mono=True)



