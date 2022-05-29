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
X_training, X_testing, y_training, y_testing = train_test_split(x_val, y_val, test_size = 0.2, stratify = y_val, random_state = 42)

# Handling the missing values 
impute = SimpleImputer(strategy='mean')
impute_training = impute.fit(X_training)
X_training = impute_training.transform(X_training)
X_testing_impute = impute_training.transform(X_testing)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score

decision_tree_classification = DecisionTreeClassifier()
decision_tree_classification.fit(X_training, y_training)

y_prediction = decision_tree_classification.predict(X_training)

print("Accuracy of the training dataset : ", accuracy_score(y_training, y_prediction))
print("F1 Score of the training dataset : ", f1_score(y_training, y_prediction))

print("Validation Mean F1 Score: ", cross_val_score(decision_tree_classification, X_training, y_training, cv = 5, scoring = 'f1_macro').mean())
print("Validation Mean Accuracy: ", cross_val_score(decision_tree_classification, X_training, y_training, cv = 5, scoring = 'accuracy').mean())

dt_training_accuracy = []
dt_val_accuracy = []
dt_training_f1 = []
dt_val_f1 = []
dt_tree_depths = []

for depth in range(1,20):
    decision_tree_classification = DecisionTreeClassifier(max_depth=depth)
    decision_tree_classification.fit(X_training, y_training)
    y_training_prediction = decision_tree_classification.predict(X_training)

    dt_training_acc = accuracy_score(y_training, y_training_prediction)
    dt_train_f1 = f1_score(y_training, y_training_prediction)
    dt_val_mean_f1 = cross_val_score(decision_tree_classification, X_training, y_training, cv = 5, scoring = 'f1_macro').mean()
    dt_val_mean_accuracy = cross_val_score(decision_tree_classification, X_training, y_training, cv = 5, scoring = 'accuracy').mean()
    
    dt_training_accuracy.append(dt_training_acc)
    dt_val_accuracy.append(dt_val_mean_accuracy)
    dt_training_f1.append(dt_train_f1)
    dt_val_f1.append(dt_val_mean_f1)
    dt_tree_depths.append(depth)
    
dt_Tuning_Max_depth = {"Training Accuracy": dt_training_accuracy, "Validation Accuracy": dt_val_accuracy, "Training F1": dt_training_f1, "Validation F1":dt_val_f1, "Max_Depth": dt_tree_depths }
dt_Tuning_Max_depth_df = pd.DataFrame.from_dict(dt_Tuning_Max_depth)

dt_plot_df = dt_Tuning_Max_depth_df.melt('Max_Depth',var_name='Metrics',value_name="Values")
fig,ax = plt.subplots(figsize=(15,5))
sns.pointplot(x="Max_Depth", y="Values",hue="Metrics", data=dt_plot_df,ax=ax)

# Decision Tree Visualization

decision_tree_classification = tree.DecisionTreeClassifier(max_depth = 3)
decision_tree_classification.fit(X_training, y_training) 
dot_data = tree.export_graphviz(decision_tree_classification, feature_names = x_val.columns.tolist()) #dot_data
graph = graphviz.Source(dot_data)
graph

# Accuracy & Confussion matrix
decision_tree_classification = DecisionTreeClassifier(max_depth=3, min_samples_leaf = 35)
decision_tree_classification.fit(X_training, y_training)
y_training_prediction = decision_tree_classification.predict(X_testing_impute)
print("Decision Tree Test Accuracy: ",accuracy_score(y_testing, y_training_prediction))
print("Decision TreeTest F1 Score: ",f1_score(y_testing, y_training_prediction))
print("")
print("Confusion Matrix on Test Data - Decision Tree")
pd.crosstab(y_testing, y_training_prediction, rownames=['True'], colnames=['Predicted'], margins=True)