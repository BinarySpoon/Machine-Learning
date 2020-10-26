# libraries used --> 
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from sklearn.datasets import make_classification

# models -->
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# load data -->
cancer = load_breast_cancer()

# feature names (total 30) -->
column = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
'mean smoothness', 'mean compactness', 'mean concavity',
'mean concave points', 'mean symmetry', 'mean fractal dimension',
'radius error', 'texture error', 'perimeter error', 'area error',
'smoothness error', 'compactness error', 'concavity error',
'concave points error', 'symmetry error', 'fractal dimension error',
'worst radius', 'worst texture', 'worst perimeter', 'worst area',
'worst smoothness', 'worst compactness', 'worst concavity',
'worst concave points', 'worst symmetry', 'worst fractal dimension',
'target']

# creating dataframe -->
data = pd.DataFrame(columns=column, index=pd.RangeIndex(start=0,stop=569,step=1))
data['target'] = cancer['target']
data[column[:30]] = cancer['data']

# train test split -->
labels = cancerdf['target']
data.drop('target',axis=1,inplace=1)
features = data
X_train, X_test, y_train, y_test = train_test_split(features,labels,random_state=0)

# logistic regression -->
lgr = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
lgr.fit(X_train,y_train)

# Support Vector Classifier -->
svc = make_pipeline(StandardScaler(), SVC(random_state=0, gamma='auto'))
svc.fit(X_train,y_train)
    
# k-nearest neighbours -->
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn.fit(X_train,y_train)
    
# Random Forest -->
rft = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=2, random_state=0))
rft.fit(X_train,y_train)

# accuracy --> 
accuracy_lgr = lgr.score(X_test,y_test)
accuracy_svc = svc.score(X_test,y_test)
accuracy_knn = knn.score(X_test,y_test)
accuracy_rft = rft.score(X_test,y_test)
   
# print out results -->
print('Logistic Regression: {}'.format(accuracy_lgr))
print('Support Vector Classifier: {}'.format(accuracy_svc))
print('Knearest neighbors: {}'.format(accuracy_knn))
print('Random Forests: {}'.format(accuracy_rft))
    
