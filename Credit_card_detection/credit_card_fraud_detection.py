#import librery
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#load the dataset
dataset=pd.read_csv("creditcard.csv")

dataset=dataset.sample(frac=0.1,random_state=1)
#Determine number of fraud cases in dataset
fraud=dataset[dataset['Class']==1]
valid=dataset[dataset['Class']==0]
outlier_fraction=len(fraud)/float(len(valid))
#Get all the columns from DataFrame
columns=dataset.columns.tolist()
#Filter the columns to remove data we do want
columns=[c for c in columns if c not in ['Class']]
#store the variable to predict the output
target='Class'

X=dataset[columns]
y=dataset[target]

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
#define the random state
state=1

classifiers={
        "Isolation_Forest":IsolationForest(max_samples=len(X),
                    contamination=outlier_fraction,random_state=state),
        "Local_outlier_Fractor":LocalOutlierFactor(n_neighbors=20,
                                                   contamination=outlier_fraction)
        }
#fit the model
n_outliers=len(fraud)
for i, (clf_name,clf) in enumerate(classifiers.items()):
    if clf_name=="Local_outlier_Fractor":
        y_pred=clf.fit_predict(X)
        scores_pred=clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred=clf.predict(X)

    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
    
    n_errors=(y_pred!=y).sum()
    
    print('{}:{}'.format(clf_name,n_errors))
    print(accuracy_score(y,y_pred))
    print(classification_report(y,y_pred))



















