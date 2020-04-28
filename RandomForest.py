# loading reQuired libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

#Importing the dataset
df = pd.read_csv("Downloads/churn.csv")
df
# Checking the shape of the data
df.shape
df.dtypes

df.head()
df_desc = df.describe()
df_desc
# Checking the variable type count in df
df.dtypes.value_counts()
# Getting the uniQue Values
for val in df:
    print(val, " ", df[val].unique().shape)

# dealing with UniQue, or same value columns.
df.drop("customerID", axis=1, inplace=True)
df.shape

# Converting the senior citizen column.
lst =[]
for val in df.SeniorCitizen:
    x = "Yes" if(val > 0.5) else "NO"
    lst.append(x)
df["SeniorCitizen"] = lst

# Checking the single value value domination
quasi_constant_feat = []
for feature in df.columns:
    dominant = (df[feature].value_counts() / np.float(len(df))).sort_values(ascending=False).values[0]
    if dominant > 0.90:
        quasi_constant_feat.append(feature)

print(quasi_constant_feat)


# Null Value analysis & treatment.
df.isnull().any()
# df.isnull()
df.isnull().sum()

# Bar plot for tg variable
sns.catplot(x="Churn", kind="count", data=df)

#Separating the target varaible
Y = df.iloc[:,-1]
Y.shape
df.drop("Churn", axis=1, inplace=True)
df.shape
# Get numrical and categorial column separate
df_num = df.select_dtypes(include=['int64', 'float64'])
df_factor = df.select_dtypes(include=['object'])

# Rescaling Numerical Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
for val in df:
    if(df[val].dtypes in ['int64', 'float64']):
        df[[val]] = scaler.fit_transform(df[[val]])


# Convert the Categorical Data into dummy variabless'''

df = pd.get_dummies(df, drop_first=False)
# df.columns
# df.shape
# df.dtypes

# Converting categorical variable into factor.
lst = df_num.columns
for val in df:
    if(val not in lst):
        df[val] = df[val].astype("object")


df.shape
df.columns

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df, Y, random_state = 42,test_size = 0.3)

# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestClassifier
# create regressor object
regressor = RandomForestClassifier(n_estimators = 100, random_state = 0)
# fit the regressor with x and y data
regressor.fit(x_train, y_train)
# Making prediction.
y_pred = regressor.predict(x_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)*100)
print(metrics.confusion_matrix(y_test, y_pred))

# save confusion matrix and slice into four pieces---- deep diving into confusion matrix
# Converting the categorical output into numerical output
lst_test =[]
for val in y_test:
    x = 1 if(val == "Yes") else 0
    lst_test.append(x)

lst_pred =[]
for val in y_pred:
    x = 1 if(val == "Yes") else 0
    lst_pred.append(x)

confusion = metrics.confusion_matrix(y_test, y_pred)
print(confusion)
# [row, column]
TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))  # Accuracy by calculation
print(metrics.accuracy_score(lst_test, lst_pred))  # Confusion maytrix

classification_error = (FP + FN) / float(TP + TN + FP + FN)  # Error
print(classification_error * 100)
print(1 - metrics.accuracy_score(lst_test, lst_pred))

sensitivity = TP / float(FN + TP)
print(sensitivity)
print(metrics.recall_score(lst_test, lst_pred))

specificity = TN / (TN + FP)
print(specificity)

false_positive_rate = FP / float(TN + FP)
print(false_positive_rate)
print(1 - specificity)

precision = TP / float(TP + FP)
print(precision)
print(metrics.precision_score(lst_test, lst_pred))

"""Receiver Operating Characteristic (ROC)"""
# IMPORTANT: first argument is true values, second argument is predicted values
# roc_curve returns 3 objects fpr, tpr, thresholds
# fpr: false positive rate
# tpr: true positive rate
fpr, tpr, thresholds = metrics.roc_curve(lst_test, lst_pred)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

"""AUC - Area under Curve"""

# AUC is the percentage of the ROC plot that is underneath the curve:
# IMPORTANT: first argument is true values, second argument is predicted probabilities
print(metrics.roc_auc_score(lst_test, lst_pred))

# F1 Score FORMULA
F1 = 2 * (precision * sensitivity) / (precision + sensitivity)



# Variable importance in RandomForest
importances = regressor.feature_importances_
indices = np.argsort(importances)
plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices[:10])), df.columns)
plt.xlabel('Relative Importance')


# plotting confusion matrix in case of multiclass classifier
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(y_test, y_pred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label');


















#-----------------------------adaboost------------

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

bdt_real = AdaBoostClassifier(
DecisionTreeClassifier(max_depth=2),
n_estimators=40,
learning_rate=1)
bdt_discrete = AdaBoostClassifier(
DecisionTreeClassifier(max_depth=2),
n_estimators=40,
learning_rate=1.5,
algorithm="SAMME")

bdt_real.fit(x_train, y_train)
bdt_discrete.fit(x_train, y_train)
real_test_errors = []
discrete_test_errors = []

for real_test_predict, discrete_train_predict in zip( bdt_real.staged_predict(x_test), bdt_discrete.staged_predict(x_test)):
    real_test_errors.append(    1. - accuracy_score(real_test_predict, y_test))
    discrete_test_errors.append(
        1. - accuracy_score(discrete_train_predict, y_test))
n_trees_discrete = len(bdt_discrete)
n_trees_real = len(bdt_real)
# Boosting might terminate early, but the following arrays are always
# n_estimators long. We crop them to the actual number of trees here:
real_estimators = bdt_real.estimators_
real_ns = bdt_real.n_classes_

discrete_estimator_errors = bdt_discrete.estimator_errors_[:n_trees_discrete]
real_estimator_errors = bdt_real.estimator_errors_[:n_trees_real]
discrete_estimator_weights = bdt_discrete.estimator_weights_[:n_trees_discrete]

plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.plot(range(1, n_trees_discrete + 1),
     discrete_test_errors, c='black', label='SAMME')


plt.plot(range(1, n_trees_real + 1),
     real_test_errors, c='black',
     linestyle='dashed', label='SAMME.R')
plt.legend()
plt.ylim(0.18, 0.62)
plt.ylabel('Test Error')
plt.xlabel('Number of Trees')


plt.subplot(132)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_errors,
     "b", label='SAMME', alpha=.5)
plt.plot(range(1, n_trees_real + 1), real_estimator_errors,
     "r", label='SAMME.R', alpha=.5)
plt.legend()
plt.ylabel('Error')
plt.xlabel('Number of Trees')
plt.ylim((.2,
     max(real_estimator_errors.max(),
         discrete_estimator_errors.max()) * 1.2))
plt.xlim((-20, len(bdt_discrete) + 20))


plt.subplot(133)
plt.plot(range(1, n_trees_discrete + 1), discrete_estimator_weights,
     "b", label='SAMME')
plt.legend()
plt.ylabel('Weight')
plt.xlabel('Number of Trees')
plt.ylim((0, discrete_estimator_weights.max() * 1.2))
plt.xlim((-20, n_trees_discrete + 20))
# prevent overlapping y-axis labels
plt.subplots_adjust(wspace=0.25)
plt.show()