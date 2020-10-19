import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
iris = sns.load_dataset('iris')

X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
print('Iris SVC Model Pre-GridSearchCV: \n', classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred), '\n')

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
grid_y_pred = grid.predict(X_test)
print('Iris SVC Model With GridSearchCV: \n', classification_report(y_test,grid_y_pred))
print(confusion_matrix(y_test,grid_y_pred), '\n')

