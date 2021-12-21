# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:33:44 2020

@author: lijj
"""
import numpy as np
import csv
from matplotlib import pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV



# for some sets of input parameters, SVM will be slow to converge.  We will terminate early.
# This code will suppress warnings.
import warnings
from sklearn.exceptions import ConvergenceWarning 
warnings.simplefilter("ignore", ConvergenceWarning)


# read data
pathtest='lab4-test.csv'
pathtrain='lab4-train.csv'
def read(path):
    with open(path,'r',encoding='UTF-8') as csvfile:
        reader=csv.reader(csvfile)
        rows=[row for row in reader]
        header=rows[0]
        data=np.array(rows[1:])
        data=data.astype(np.float)
    #return data
    return header,data
    
header,testdata=read(pathtest)
header,traindata=read(pathtrain)
Xtrain=traindata[:,:4]
ytrain=traindata[:,4]
Xtest=testdata[:,:4]
ytest=testdata[:,4]

#print (Xtrain[:5,:])
#print(ytrain[:5])


#standardize data
scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


# =============================================================================
#task1 ,Random Forest (RF) and AdaBoost
#random forest



















#task2 ,decision tree, cnn, logistic regression, navive baysionbayes
#decision tree
# Necessary imports 
from scipy.stats import randint 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import GridSearchCV

# Creating the hyperparameter grid  
param_dist = {"max_depth": [None, 1,2,3,4,5,6,7,8,9], 
              "max_features": randint(1, 4), 
              "min_samples_leaf": randint(1, 9), 
              "criterion": ["gini", "entropy"]} 
  
# Instantiating Decision Tree classifier 
tree = DecisionTreeClassifier() 
# Instantiating RandomizedSearchCV object 
tree_cv = RandomizedSearchCV(tree, param_dist, cv = 5) 
tree_cv.fit(Xtrain, ytrain) 
# Print the tuned parameters and score 
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_)) 
print("Best score is {}".format(tree_cv.best_score_)) 
print("Best test score{}".format(tree_cv.score(Xtest,ytest)))
testpredictions=tree_cv.predict(Xtest)
test_conf_mat = confusion_matrix(ytest, testpredictions)
print("Test confusion Matrix".format(test_conf_mat))
#print(test_conf_mat)

































