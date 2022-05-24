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

# Import libraries for data visualization

fig,axes = plt.subplots(4,2,figsize=(12,15))
for idx,cat_col in enumerate(categorical_data_col_array):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=training_df,hue='Loan_Status',ax=axes[row,col])

plt.subplots_adjust(hspace=1)

fig,axes = plt.subplots(1,3,figsize=(17,5))
for idx,cat_col in enumerate(numerical_data_col_array):
    sns.boxplot(y=cat_col,data=training_df,x='Loan_Status',ax=axes[idx])

print(training_df[numerical_data_col_array].describe())
plt.subplots_adjust(hspace=1)

#Data Preprocessing

# Categrical Features Encoding

# Converting the categorical variables into dummy variables
# drop_first = True -> reduces the extra column created when creating the dummy variables.(reduces the correlation among the dummy variables)
encoded_training_df = pd.get_dummies(training_df, drop_first = True)
encoded_training_df.head()

# Splitting features and target varibles

# Drop the Load_Status_Y column in the dataset & Assign the encoded categorical features
x_val = encoded_training_df.drop(columns='Loan_Status_Y')

# Assign the Load_Status_Y column to y_val
y_val = encoded_training_df['Loan_Status_Y']

# Splitting the datset into training and testing data
X_training, X_testing, y_training, y_testing = train_test_split(x_val, y_val, test_size = 0.2, stratify = y_val, 
                                                                random_state = 42)
																