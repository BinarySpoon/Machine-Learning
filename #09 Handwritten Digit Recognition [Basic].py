'''
Handwritten Number Recognition -->
---------------------------------
Tested on four most common models
1. Logistic Regression,
2. Random Forest,
3. Support Vector Classifier,
4. K-nearest Neighbors,
By Akash R. Patel
---------------------------------
'''

# libraries used -->
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.datasets import load_digits

# Importing Data -->
digits = load_digits()

# Display Number Of Images -->
print("Image data shape: ", digits.data.shape)
print()
print("Label data shape: ", digits.target.shape)

# Show images -->
plt.figure(figsize=(20,8))
for index, (image, label) in enumerate(zip(digits.data[0:16], digits.target[0:16])):
    plt.subplot(2,8, index+1)
    plt.imshow(np.reshape(image,(8,8)), cmap=plt.cm.gray)
    plt.title('Labeled %i\n' % label, fontsize = 20)

'''
1. create a blank fig.
2. run a loop calling the first 16 entries in digits.data, digits.target as "image" and "label" respectively.
3. create a subplot inside the blank figure of 2 rows and 8 cols.
4. place images in them (8,8) long and gray.
Algo credit: 'https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a'
'''

# Train Test Split -->
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


# The four step process
#1 import the model you wanna use -->
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#2 Make an instance of the model -->
lgr = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
svc = make_pipeline(StandardScaler(), SVC(random_state=0,gamma='auto'))
rf = make_pipeline(StandardScaler(), RandomForestClassifier(max_depth=2,random_state=0))
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))

#3 Training the model on the data (fitting) -->
lgr.fit(x_train, y_train)
svc.fit(x_train,y_train)
rf.fit(x_train,y_train)
knn.fit(x_train,y_train)

#4 Predict for new data (test) -->
lr_predict = lgr.predict(x_test)
svc_predict = svc.predict(x_test)
rf_predict = rf.predict(x_test)
knn_predict = knn.predict(x_test)

#5 Measuring performance (accuracy) -->
lr_score = lgr.score(x_test, y_test)
svc_score = svc.score(x_test,y_test)
rf_score = rf.score(x_test,y_test)
knn_score = knn.score(x_test,y_test)

print('Accuracy Score -->')
print('------------------')
print()
print('Logistic Regression: ', lr_score)
print()
print('Support Vector Machines: ', svc_score)
print()
print('Random Forrest: ', rf_score)
print()
print('K Nearest Neighbors: ', knn_score)
print('------------------')

#6 Heatmaps -->
lr_cm = metrics.confusion_matrix(y_test, lr_predict) # cm = confusion matrix, so sklearn has a built in for that.
svc_cm = metrics.confusion_matrix(y_test, svc_predict)
rf_cm = metrics.confusion_matrix(y_test, rf_predict)
knn_cm = metrics.confusion_matrix(y_test, knn_predict)

# Logistic regression
plt.figure(figsize=(9,9))
sns.heatmap(lr_cm, annot=True, fmt='.3f', linewidths=.5, square=True, cmap='Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Logistic Regression: {0}'.format(lr_score)
plt.title(all_sample_title, size=15);

# Support Vector Machines -->
plt.figure(figsize=(9,9))
sns.heatmap(svc_cm, annot=True, fmt='.3f', linewidths=.5, square=True, cmap='Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Support Vector Machines: {0}'.format(svc_score)
plt.title(all_sample_title, size=15);

# Random Forest Classifier -->
plt.figure(figsize=(9,9))
sns.heatmap(rf_cm, annot=True, fmt='.3f', linewidths=.5, square=True, cmap='Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Random Forest: {0}'.format(svc_score)
plt.title(all_sample_title, size=15);

# K-Nearest Neighbors -->
plt.figure(figsize=(9,9))
sns.heatmap(knn_cm, annot=True, fmt='.3f', linewidths=.5, square=True, cmap='Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'K-Nearest Neighbors: {0}'.format(svc_score)
plt.title(all_sample_title, size=15);

# Printout plots -->
plt.show()
