# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 17:33:44 2020

@author: lijj
"""
import numpy as np
import csv
from matplotlib import pyplot as plt

from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB

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

# =============================================================================
#task1 ,Random Forest (RF) and AdaBoost


depth=[1,2,3,4,5,6,7,8,9,10]
numtreerange=[1,5,10,25,50,100,200]
trainacc=np.zeros(10)
testacc=np.zeros(10)

#for i in range (5):   #4a
for i in range (10):     #4b 
     tempscore=0
     temp_numtree=0
     for j in range(7):
    
          # we create an instance of Neighbours Classifier and fit the data.
          #clf=SVC(C=Cvals[i],kernel='poly',degree=p,gamma=gamma,coef0=1.0,shrinking=True,probability=False,max_iter=max_iter)
          #clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=depth[i])
          clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[j],
                                     max_features=None,criterion='gini',max_depth=depth[i])  #bagging
          clf.fit(Xtrain, ytrain)
      
      
          # Plot the training points with the mesh
          #if (tempscore<clf.score(Xtrain,ytrain)):
          #    tempscore=clf.score(Xtrain,ytrain)
          if (tempscore<clf.score(Xtest,ytest)):
              tempscore=clf.score(Xtest,ytest)
              temp_numtree=numtreerange[j]
              
              
     
     clf=RandomForestClassifier(bootstrap=True,n_estimators=temp_numtree, max_features=None,criterion='gini',max_depth=depth[i])  #
    
     clf.fit(Xtrain, ytrain)
      

     #Report training and testing accuracies
     # print('Working on k=%i'%(n_neighbors))
     trainacc[i] =clf.score(Xtrain,ytrain) 
     testacc[i] = clf.score(Xtest,ytest) 
     print(temp_numtree)
print(trainacc)
print(testacc)

plt.title('depth-accuracy training data')
plt.plot(depth,trainacc)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()


plt.title('depth-accuracy test data')
plt.plot(depth,testacc)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()


clf=RandomForestClassifier(bootstrap=True,n_estimators=5,
                                     max_features=None,criterion='gini',max_depth=10)  #bagging
clf.fit(Xtrain,ytrain)
clf.fit(Xtest,ytest)


trainpredictions=clf.predict(Xtrain)
testpredictions=clf.predict(Xtest)
train_conf_mat = confusion_matrix(ytrain, trainpredictions)
test_conf_mat = confusion_matrix(ytest, testpredictions)
print(train_conf_mat)
print(test_conf_mat)
#print(trainpredictions)


#===============================================================
#adaboost
learnraterange=np.logspace(-3,0,15,base=10) #numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0
depth=[1,2,3,4,5,6,7,8,9,10]
numtreerange=[1,5,10,25,50,100,200]
trainacc=np.zeros(10)
testacc=np.zeros(10)

for i in range (10):     #4b 
     tempscore=0
     temp_numtree=0
     temp_learningrate=0
     for j in range(7):
         for k in range(15):
    
              # we create an instance of Neighbours Classifier and fit the data.
              #clf=SVC(C=Cvals[i],kernel='poly',degree=p,gamma=gamma,coef0=1.0,shrinking=True,probability=False,max_iter=max_iter)
              #clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=depth[i])
              #clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[j],max_features=None,criterion='gini',max_depth=depth[i])  #bagging
              clf=GradientBoostingClassifier(learning_rate=learnraterange[k],n_estimators=numtreerange[j],max_depth=depth[i])
              clf.fit(Xtrain, ytrain)
          
             
              if (tempscore<clf.score(Xtrain,ytrain)):
                  tempscore=clf.score(Xtrain,ytrain)
                  temp_numtree=numtreerange[j]
                  temp_learningrate=learnraterange[k]
              
              
     print(temp_learningrate)
     print(temp_numtree)
     clf=GradientBoostingClassifier(learning_rate=temp_learningrate,n_estimators=temp_numtree,max_depth=depth[i])
     clf.fit(Xtrain, ytrain)
      
     #Report training and testing accuracies
     # print('Working on k=%i'%(n_neighbors))
     trainacc[i] =clf.score(Xtrain,ytrain) 
     testacc[i] = clf.score(Xtest,ytest) 
          
print(trainacc)
print(testacc)

plt.title('depth-accuracy training data')
plt.plot(depth,trainacc)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()


plt.title('depth-accuracy test data')
plt.plot(depth,testacc)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()


clf=GradientBoostingClassifier(learning_rate=0.37276,n_estimators=10,max_depth=8)
clf.fit(Xtrain, ytrain)

trainpredictions=clf.predict(Xtrain)
testpredictions=clf.predict(Xtest)
train_conf_mat = confusion_matrix(ytrain, trainpredictions)
test_conf_mat = confusion_matrix(ytest, testpredictions)
print(train_conf_mat)
print(test_conf_mat)
#print(trainpredictions)




#task2==================================
#Neural Network (NN)

#standardize data
scaler = StandardScaler()
scaler.fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)


noderange = range(5,100,5) #[ 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95]
alpharange = np.logspace(-6,0,4)
learnrate = np.logspace(-2,-0.5,4)

trainacc=np.zeros(19)
testacc=np.zeros(19)
count=0
# =============================================================================
#for i in range (5):   #4a
for i in noderange:     #4b  
     tempscore=0
     temp_alpha=0
     temp_trainscore=0
     temp_learningrate=0
     for j in learnrate:
         for k in alpharange:
         
              # we create an instance of Neighbours Classifier and fit the data.
              #clf=SVC(C=Cvals[i],kernel='poly',degree=p,gamma=gamma,coef0=1.0,shrinking=True,probability=False,max_iter=max_iter)
              #clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=depth[i])
              #clf=RandomForestClassifier(bootstrap=True,n_estimators=numtreerange[j], max_features='sqrt',criterion='gini',max_depth=depth[i])  #random forest
              #clf=GradientBoostingClassifier(learning_rate=learnraterange[k],n_estimators=numtreerange[j],max_depth=depth[i])
              clf=MLPClassifier(hidden_layer_sizes=(i),alpha=k,learning_rate_init=j,activation='relu', solver='sgd',
                  learning_rate='adaptive',max_iter=200)
              clf.fit(Xtrain, ytrain)
              
          
              #Z = clf.predict(Xtrain, ytrain)
              #Z = clf.predict(Xtrain)
              #print(Z)
              # Put the result into a color plot
              #Z = Z.reshape(x1mesh.shape)
          
              # Plot the training points with the mesh
              #print (depth[i])
              if (tempscore<clf.score(Xtrain,ytrain)):
                  tempscore=clf.score(Xtrain,ytrain)
                  temp_trainscore=clf.score(Xtrain,ytrain)
                  temp_learningrate=j
                  temp_alpha=k
          
          #plt.show()
          #Report training and testing accuracies
          # print('Working on k=%i'%(n_neighbors))
     #trainacc[i] =clf.score(Xtrain,ytrain) 
     #testacc[i] = clf.score(Xtest,ytest) 
     trainacc[count] = temp_trainscore
     testacc[count] = tempscore
     count+=1
     print(temp_learningrate)
     print(temp_alpha)


print(trainacc)
print(testacc)

plt.title('nodenum-accuracy training data')
plt.plot(noderange,trainacc)
plt.xlabel('nodenum')
plt.ylabel('Accuracy')
plt.show()


plt.title('nodenum-accuracy test data')
plt.plot(noderange,testacc)
plt.xlabel('nodenum')
plt.ylabel('Accuracy')
plt.show()

clf=MLPClassifier(hidden_layer_sizes=(30),alpha=0.000001,learning_rate_init=0.3162,activation='relu', solver='sgd',
                  learning_rate='adaptive',max_iter=200)
clf.fit(Xtrain, ytrain)

trainpredictions=clf.predict(Xtrain)
testpredictions=clf.predict(Xtest)
train_conf_mat = confusion_matrix(ytrain, trainpredictions)
test_conf_mat = confusion_matrix(ytest, testpredictions)
print(train_conf_mat)
print(test_conf_mat)
print(clf.score(Xtest,ytest))
#print(trainpredictions)




#task2 ,decision tree, cnn, logistic regression, navive baysionbayes
#decision tree
depth=[1,2,3,4,5,6,7,8,9,10]
trainacc=np.zeros(10)
testacc=np.zeros(10)
# =============================================================================
#for i in range (5):   #4a
for i in range (10):     #4b   
          # we create an instance of Neighbours Classifier and fit the data.
          #clf=SVC(C=Cvals[i],kernel='poly',degree=p,gamma=gamma,coef0=1.0,shrinking=True,probability=False,max_iter=max_iter)
          clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=depth[i])
          clf.fit(Xtrain, ytrain)
      
        
          trainacc[i] =clf.score(Xtrain,ytrain) 
          testacc[i] = clf.score(Xtest,ytest) 
          
print(trainacc)
print(testacc)

plt.title('depth-accuracy training data')
plt.plot(depth,trainacc)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()


plt.title('depth-accuracy test data')
plt.plot(depth,testacc)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()


clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=10)
clf.fit(Xtrain, ytrain)

trainpredictions=clf.predict(Xtrain)
testpredictions=clf.predict(Xtest)
train_conf_mat = confusion_matrix(ytrain, trainpredictions)
test_conf_mat = confusion_matrix(ytest, testpredictions)
print(train_conf_mat)
print(test_conf_mat)
print(clf.score(Xtest,ytest))




#logistic regression



penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
solver = ['liblinear', 'saga']

param_grid = dict(penalty=penalty,
                  C=C,
                  class_weight=class_weight,
                  solver=solver)
logistic=LogisticRegression()
grid = GridSearchCV(estimator=logistic,
                    param_grid=param_grid)
clf= grid.fit(Xtrain, ytrain)

print('Best Score: ', clf.best_score_)
print('Best Params: ', clf.best_params_)

#clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=10)
#clf.fit(Xtrain, ytrain)

trainpredictions=clf.predict(Xtrain)
testpredictions=clf.predict(Xtest)
train_conf_mat = confusion_matrix(ytrain, trainpredictions)
test_conf_mat = confusion_matrix(ytest, testpredictions)
print(train_conf_mat)
print(test_conf_mat)
print(clf.score(Xtest,ytest))



#naive bayes


clf=GaussianNB()
clf.fit(Xtrain,ytrain)

#print('trainScore: ', clf.score)

#clf=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=10)
#clf.fit(Xtrain, ytrain)

trainpredictions=clf.predict(Xtrain)
testpredictions=clf.predict(Xtest)
train_conf_mat = confusion_matrix(ytrain, trainpredictions)
test_conf_mat = confusion_matrix(ytest, testpredictions)
print(train_conf_mat)
print(test_conf_mat)
print(clf.score(Xtest,ytest))


#ensemble vote
from sklearn.ensemble import VotingClassifier


clf1=MLPClassifier(hidden_layer_sizes=(30),alpha=0.000001,learning_rate_init=0.3162,activation='relu', solver='sgd',
                  learning_rate='adaptive',max_iter=200) #cnn
clf2=DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=10) #decision tree
clf3=LogisticRegression(C=0.01,class_weight={1: 0.4, 0: 0.6},penalty='l1',solver='liblinear') #logistic regression
clf4=GaussianNB() #naive bayes

eclf = VotingClassifier(
    estimators=[('cnn', clf1), ('dt', clf2), ('logictic', clf3),('NB',clf4)],
    voting='hard')
eclf.fit(Xtrain,ytrain)
clf2.fit(Xtrain,ytrain)

trainpredictions=eclf.predict(Xtrain)
testpredictions=eclf.predict(Xtest)
train_conf_mat = confusion_matrix(ytrain, trainpredictions)
test_conf_mat = confusion_matrix(ytest, testpredictions)
print(train_conf_mat)
print(test_conf_mat)
print(eclf.score(Xtest,ytest))




from sklearn.model_selection import cross_val_score


for tclf, label in zip([clf1, clf2, clf3, clf4, eclf], ['neuro network','Decision tree','Logistic Regression',  'naive Bayes', 'Ensemble']):
    scores = cross_val_score(tclf, Xtest, ytest, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


#weighted vote
eclf = VotingClassifier(
    estimators=[('cnn', clf1), ('dt', clf2), ('logictic', clf3),('NB',clf4)],
    voting='soft',weights=[2,2,1,4])

eclf.fit(Xtrain,ytrain)
clf2.fit(Xtrain,ytrain)

trainpredictions=eclf.predict(Xtrain)
testpredictions=eclf.predict(Xtest)
train_conf_mat = confusion_matrix(ytrain, trainpredictions)
test_conf_mat = confusion_matrix(ytest, testpredictions)
print(train_conf_mat)
print(test_conf_mat)
print(eclf.score(Xtest,ytest))




