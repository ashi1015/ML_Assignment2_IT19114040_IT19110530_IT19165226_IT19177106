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

# Dropping missing value rows
loan_tarin_df4 = loan_tarin_df4.dropna()

# Final Dataset Shape
loan_tarin_df4.shape

# Dependent Column Values
loan_tarin_df4['Dependents'].value_counts()

# Replacing 3+ values with 4
loan_tarin_df4 = loan_tarin_df4.replace(to_replace = '3+', value = 4)

# Marial Status and Loan Status
seaborn.countplot(x='Married', hue='Loan_Status', data = loan_tarin_df4)

# Education and Loan Status
seaborn.countplot(x='Education', hue='Loan_Status', data = loan_tarin_df4)

# Gender and Loan Status
seaborn.countplot(x='Gender', hue='Loan_Status', data = loan_tarin_df4)

# Gender and Loan Status
seaborn.countplot(x='Property_Area', hue='Loan_Status', data = loan_tarin_df4)

numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
print(numerical_columns)
fig,axes = plt.subplots(1,3,figsize=(17,5))
for idx,cat_col in enumerate(numerical_columns):
    seaborn.boxplot(y=cat_col,data=loan_tarin_df4,x='Loan_Status',ax=axes[idx])

print(loan_tarin_df4[numerical_columns].describe())
plt.subplots_adjust(hspace=1)