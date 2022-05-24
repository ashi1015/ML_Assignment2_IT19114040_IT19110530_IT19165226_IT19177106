# Import Basic Libraries
import numpy as numpy #used for working with arrays
import pandas as pandas #used for analyze data
import seaborn as seaborn #used for drawing attractive and informative statistical graphics
import matplotlib.pyplot as plt #used for makes some change to a figure

# From SKLearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler 
from sklearn.svm import SVC # "Support vector classifier"  

st_x= StandardScaler()

# Loading Dataset
loan_tarin_df4 = pandas.read_csv("train.csv")
type(loan_tarin_df4)

# Checking Data
loan_tarin_df4.head()

# Shape of dataset
loan_tarin_df4.shape

# Describing Dataset
loan_tarin_df4.describe()

# Count the number of missing values in each column
loan_tarin_df4.isnull().sum()