import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix

cancer = load_breast_cancer()
df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
#print(df_feat.info())
df_target = pd.DataFrame(cancer['target'], columns=['Cancer'])
#print(df_feat.head())
X_train, X_test, y_train, y_test = train_test_split(df_feat, np.ravel(df_target), test_size=0.30, random_state=101)
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('SVC Classification Pre-GridSearch: \n', classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred), '\n')
# Improve the model with GridSearchCV 
param_grid = {'C': [1.4, 1.5, 1.6, 1.7], 'gamma': [0.0001, .0002], 'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=1)
grid.fit(X_train,y_train)
print(grid.best_params_)
print(grid.best_estimator_)
grid_y_pred = grid.predict(X_test)
print('SVC Classification With GridSearch: \n', classification_report(y_test, grid_y_pred))
print(confusion_matrix(y_test, grid_y_pred), '\n')