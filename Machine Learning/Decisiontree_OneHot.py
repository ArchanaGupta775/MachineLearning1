# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 12:33:26 2018

@author: ARCHANA
"""
#DecissionTree and Predict methods are very important in this example. This is the real starting/building of ML
#Here we will be playing with more columns. However DecisionTreeClassifier algorithm works only on numeric/continuous data/columns
#Henceforth we need to convert  catogerical columns to dummy columns
#This technique is called one-hot encoding

import os
import pandas as pd
from sklearn import tree
import io
import pydot #if we need to use any external .exe files.... Here we are using dot.exe

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.getcwd()

os.chdir("F:\DataScience\DataS\Titanic")
titanic_train=pd.read_csv("titanic_train.csv")
titanic_train.shape
titanic_train.info()

#Transformation of non numneric cloumns
#There is an exception with the pclass. Though it's coinncidentally is a number but it's a classification but not a number.
#titanic_train1 = titanic_train[['Pclass', 'Sex', 'Embarked', 'Fare']]

#Convert categoric to One hot encoding using get_dummies
titanic_train1=pd.get_dummies(titanic_train,columns=['Pclass','Sex','Embarked'])
titanic_train1.shape
titanic_train1.info()
#now the drop non numerical columns where we will not be applying logic. Something like we will not apply logic on names, passengerID ticket id etc...
X_train = titanic_train1.drop(['PassengerId','Age','Cabin','Ticket', 'Name','Survived'],1) 
Y_train = titanic_train['Survived']

dt = tree.DecisionTreeClassifier()
dt.fit(X_train,Y_train)
dt.score(X_train,Y_train)
dt.feature_importances_

#visualize the decission tree
dot_data = io.StringIO()  

tree.export_graphviz(dt, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("DT-AugmentedColumns.pdf")



titanic_test=pd.read_csv("titanic_test.csv")
titanic_test.shape
titanic_test.info()

titanic_test.Fare[titanic_test['Fare'].isnull()] = titanic_test['Fare'].mean()
titanic_test1=pd.get_dummies(titanic_test,columns=['Pclass','Sex','Embarked'])
X_test = titanic_test1.drop(['PassengerId','Age','Cabin','Ticket', 'Name'], 1)
X_test.shape
X_test.info()


titanic_test['Survived'] = dt.predict(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId', 'Survived'], index=False)