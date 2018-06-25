# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 08:24:08 2018

@author: ARCHANA
"""
import os
import pandas as pd
from sklearn import tree
import io
import pydot 


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.getcwd()

os.chdir("F:\\DataScience\\DataS\\Titanic")
os.getcwd()

titanic_train=pd.read_csv("titanic_train.csv")
titanic_train.shape
titanic_train.info()
titanic_train.describe()

titanic_test=pd.read_csv("titanic_test.csv")
titanic_test.shape
titanic_test.info()
titanic_test.describe

#EDA
X_titanic_train = titanic_train[['Pclass', 'Parch']]
y_titanic_train = titanic_train['Survived']

#build the decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(X_titanic_train, y_titanic_train)
dt.feature_importances_
dt.score(X_titanic_train, y_titanic_train)
type(dt)
#visualize the decission tree
dot_data = io.StringIO() 
tree.export_graphviz(dt, out_file = dot_data, feature_names = X_titanic_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue()) #[0] 
graph[0].write_pdf("DS-DT.pdf")

#Predict the outcome using decision tree
#titanic_test = pd.read_csv("titanic_test.csv")
X_test = titanic_test[['Pclass', 'Parch']]
titanic_test['Survived'] = dt.predict(X_test)
titanic_test.to_csv("submission.csv", columns=['PassengerId','Survived'], index=False)
