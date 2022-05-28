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

# Data Labeling
loan_tarin_df4.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
loan_tarin_df4.replace({"Married": {'No': 0, 'Yes': 1}}, inplace=True)
loan_tarin_df4.replace({"Self_Employed": {'No': 0, 'Yes': 1}}, inplace=True)
loan_tarin_df4.replace({"Education": {'Not Graduate': 0, 'Graduate': 1}}, inplace=True)
loan_tarin_df4.replace({"Gender": {'Male': 0, 'Female': 1}}, inplace=True)
loan_tarin_df4.replace({"Property_Area": {'Rural': 0, 'Semiurban': 1, 'Urban': 2}}, inplace=True)

# Checking Data
loan_tarin_df4.head()

# Separating Data and Label
X = loan_tarin_df4.drop(columns=['Loan_ID', 'Loan_Status'], axis = 1)
Y = loan_tarin_df4['Loan_Status']
X, Y

# Spliting Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 3)

# Checking the Shape
X.shape, X_train.shape, X_test.shape

# Checking the Shape
Y.shape, Y_train.shape, Y_test.shape

# Classifying the data 
classifier = svm.SVC(kernel = 'linear')

# Training data of the data set
classifier.fit(X_train, Y_train)