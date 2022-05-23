# Import Libraries 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import graphviz 
from sklearn import tree
from sklearn.metrics import confusion_matrix
import numpy as np

# Importing the dataset
training_df = pd.read_csv('train.csv')
training_df.info()

# Counting the number of numerical columns and categorical columns

# Drop the Load ID column in the dataset
training_df = training_df.drop(columns=['Loan_ID']) 

numerical_data_col_array = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
print(numerical_data_col_array)

categorical_data_col_array = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area','Credit_History','Loan_Amount_Term']
print(categorical_data_col_array)