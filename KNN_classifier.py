#importing libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#loading the dataset

df=pd.read_excel("./content/speech_data.xlsx")
df.head()

