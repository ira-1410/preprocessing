import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn import model_selection

dataframe = pd.read_csv("~/Desktop/sheet.csv")
train, test = sklearn.model_selection.train_test_split(dataframe, train_size=0.8)
train.to_csv("~/Desktop/train.csv")
test.to_csv("~/Desktop/test.csv")
