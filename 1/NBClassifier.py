#EECS 658 Assignment 1
#Create ML 'Hello World' program
#Outputs:
#Prints out the overall accuracy of the classifier.
#Prints out the confusion matrix.
#Prints out the P, R, and F1 score for each of the 3 varieties of iris.
#Author: Aidan Bowen
#Date: 8-31-2023


# load libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

#load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length',
'petal-width', 'class']
dataset = read_csv(url, names=names)

# print(dataset.shape) 

# (150, 5)

print(dataset.head(20)) #print the dataset


#Create Arrays for Features and Classes
array = dataset.values
X = array[:,0:4] #contains flower features (petal length, etc..)
y = array[:,4] #contains flower names

#split the data into 2 folds
x_train_fold1, x_test_fold1, y_train_fold1, y_test_fold1 = train_test_split(X, y,
test_size=0.50, random_state=1)

x_train_fold2, x_test_fold2, y_train_fold2, y_test_fold2 = train_test_split(X, y,
test_size=0.50, random_state=1)


model = GaussianNB()     #create the model
model.fit(x_train_fold1, y_train_fold1) #first fold training
pred1 = model.predict(x_test_fold2) #first fold testing
model.fit(x_train_fold2, y_train_fold2) #second fold training
pred2 = model.predict(x_test_fold1) #second fold testing

classes=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


# concatenates the test data
actual_array = np.concatenate([y_test_fold1, y_test_fold2])     #true array
prediction_array = np.concatenate([pred1, pred2])                   #prediction array


#prints using the built in functions for accuracy score, confusion matrices, and classification reports
print("Accuracy:")
print(accuracy_score(actual_array, prediction_array))
print('Confusion Matrix:')
print(confusion_matrix(actual_array, prediction_array))
print('Classification Report:')
print(classification_report(actual_array, prediction_array, target_names=classes))