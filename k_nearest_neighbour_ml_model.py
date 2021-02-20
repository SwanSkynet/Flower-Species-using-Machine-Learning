#Here I imported some lilbraries

import numpy as np
from sklearn.datasets import load_iris
iris_dataset = load_iris()

#Looking at different keys of iris_dataset.

print("Keys in iris_dataset = \n {} ".format(iris_dataset.keys()))
print("Description of iris_dataset : \n {}".format(iris_dataset['DESCR']))
print("Target names of iris_dataset : \n {}".format(iris_dataset['target_names']))
print("Feature names in iris_dataset : \n {}".format(iris_dataset['feature_names']))
print("Type of data : {}".format(type(iris_dataset['data'])))
print("shape of data : {}".format(iris_dataset['data'].shape))
print("5 rows of data : {}".format(iris_dataset['data'][:5]))
print("Type of target : {}".format(type(iris_dataset['target'])))
print("Shape of target: \n {}".format(iris_dataset['target'].shape))
print("Target : \n {}".format(iris_dataset['target']))

#splitting dataset into train and test datasets.

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],random_state=0)

#Looking at the shapes of X_train,X_test,y_train,y_test

print("X_train shape : {}".format(X_train.shape))
print("X_test shape : {}".format(X_test.shape))
print("y_train shape : {}".format(y_train.shape))
print("y_test shape : {}".format(y_test.shape))

 #implementing ml model using k nearst neighbour.
 
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)
 
 # Training the ml model.
 
knn.fit(X_train,y_train)
 
 #tesing the ml model using a single dataset.

X_new= np.array([[5,2,3,1]])
print("X_new shape : {}".format(X_new.shape))

#making the predictions

prediction =knn.predict(X_new)
print("Prediction : {}".format(iris_dataset['target_names'][prediction]))

y_predict=knn.predict(X_test)
print("y_Prediction : {}".format(iris_dataset['target_names'][y_predict]))
print("Test Score = {} ".format(np.mean(y_predict==y_test)))  






