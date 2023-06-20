import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.show()

import seaborn as sns
import nltk
from nltk.corpus import stopwords

import string

import warnings
warnings.filterwarnings('ignore')

import re

df = pd.read_csv("./spam.csv")

print(df.head)

print(df.Label.value_counts())

print("There are {} rows and {} columns are present in the Data Set".format(df.shape[0],df.shape[1]))

print(df.info())

print(df.types)

print(df.describe())

print(df.groupby('Label').describe().T)

print(df.isnull().sum())