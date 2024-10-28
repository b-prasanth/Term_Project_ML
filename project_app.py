import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
import math
import func as fn

#Setting up pandas display settings
pd.options.mode.copy_on_write = True
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.3f}'.format

#import dataset
raw_dataset=pd.read_csv('2018_03.csv')

#dropping missing observations
cleaned_dataset=raw_dataset.dropna(axis=0,how='any')

#Standardizing the dataset
fn.standardized(cleaned_dataset)