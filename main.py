import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import scikitplot as skplt



#Read dataset
dataset = pd.read_csv('diabetes.csv')
print(dataset.head())
print(dataset.info())
print(dataset.describe())

#pre-processing
print("Checking NULL values:",dataset.isnull().any())

#EDA
"""Data visualization"""

"""heat map"""
corr = dataset.corr(method='pearson')
mask = np.triu(np.ones_like(corr, dtype=np.bool)) 
fig = plt.subplots(figsize=(25, 15))
sns.heatmap(dataset.corr(), annot=True,fmt='.2f',mask=mask)
plt.show()

"""Box plot"""
data1=dataset.drop('Outcome',axis=1)
data1.plot(kind='box', subplots=True, layout=(4,4), sharex=False,sharey=False ,figsize =(15,15))
plt.show()

"""pie graph"""
dataset['Outcome'].value_counts().plot(kind='pie',colors=['Brown', 'Green'],autopct='%1.1f%%',figsize=(9,9))
plt.show()

#model selection
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state =9)


'''RANDOM FOREST'''

#Create a Gaussian Classifier
rf_clf=RandomForestClassifier(n_estimators=100)
rf_clf.fit(x_train,y_train)

rf_ypred=rf_clf.predict(x_test)
print('\n')
print("------Accuracy------")
rf=accuracy_score(y_test, rf_ypred)*100
RF=('RANDOM FOREST Accuracy:',accuracy_score(y_test, rf_ypred)*100,'%')
print(RF)
print('\n')
print("------Classification Report------")
print(classification_report(rf_ypred,y_test))
print('\n')
print('Confusion_matrix')
rf_cm = confusion_matrix(y_test, rf_ypred)
print(rf_cm)
print('\n')
tn = rf_cm[0][0]
fp = rf_cm[0][1]
fn = rf_cm[1][0]
tp = rf_cm[1][1]
Total_TP_FP=rf_cm[0][0]+rf_cm[0][1]
Total_FN_TN=rf_cm[1][0]+rf_cm[1][1]
specificity = tn / (tn+fp)
rf_specificity=format(specificity,'.3f')
print('RF_specificity:',rf_specificity)
print()

plt.figure()
skplt.estimators.plot_learning_curve(RandomForestClassifier(), x_train, y_train,
                                     cv=7, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Random Forest Digits Classification Learning Curve");

plt.figure()                                   
sns.heatmap(confusion_matrix(y_test,rf_ypred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

import pickle
pickle.dump(rf_clf, open('model.pkl', 'wb'))

'''Logistic Regression'''
print('Logistics Regression')

'''Analysis Report'''
print()
model = LogisticRegression()
model.fit(x_train, y_train)
lg_ypred = model.predict(x_test)
print()
print("------Accuracy------")
LR=accuracy_score(y_test,lg_ypred)*100
lr=('LOGISTIC REGRESSION Accuracy:',accuracy_score(y_test,lg_ypred)*100,'%')
print(lr)
print("------Classification Report------")
print(classification_report(lg_ypred,y_test))
print('\n')
print('Confusion_matrix')
lr_cm = confusion_matrix(y_test, lg_ypred)
print(lr_cm)
print('\n')
tn = lr_cm[0][0]
fp = lr_cm[0][1]
fn = lr_cm[1][0]
tp = lr_cm[1][1]
Total_TP_FP=lr_cm[0][0]+lr_cm[0][1]
Total_FN_TN=lr_cm[1][0]+lr_cm[1][1]
specificity = tn / (tn+fp)
lr_specificity=format(specificity,'.3f')
print('LR_specificity:',lr_specificity)
print()
plt.figure()
skplt.estimators.plot_learning_curve(LogisticRegression() , x_train, y_train,
                                     cv=7, shuffle=True, scoring="accuracy",
                                     n_jobs=-1, figsize=(6,4), title_fontsize="large", text_fontsize="large",
                                     title="Logistic Regression Digits Classification Learning Curve");

plt.figure()
sns.heatmap(confusion_matrix(y_test,lg_ypred),annot = True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()