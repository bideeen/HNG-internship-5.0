#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn import datasets

# Load dataset
wine = datasets.load_wine()

#print all the features names
print("Features : ", wine.feature_names)

#print the target names
print("target : ", wine.target_names)

# print feature shape

wine.data.shape


# print the first 5 rows
print(wine.data[:5,:])


# X = Features 
# y = target


X = wine.data
y = wine.target

# Train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)

# Import the naive_bayes from sklearn
from sklearn.naive_bayes import GaussianNB

# create Gaussian Classifier
gnb = GaussianNB()

# Train your model
gnb.fit(X_train, y_train)

# Predict our test values
y_pred = gnb.predict(X_test)
y_pred[:3]

# for model accuracy import sklearn.metrics
from sklearn import metrics

# Model Accuracy
print("Accuracy : ", metrics.accuracy_score(y_test,y_pred))
