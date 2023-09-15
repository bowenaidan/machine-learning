#EECS 658 Assignment 2
#Create ML Model Comparison program
#Outputs:
#Prints out the overall accuracy of each classifier.
#Prints out the confusion matrix for each classifier.
#Author: Aidan Bowen
#Date: 9-14-2023


# load libraries
from pandas import read_csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np

#load dataset
url = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length',
'petal-width', 'class']
dataset = read_csv(url, names=names)

#Create Arrays for Features and Classes
array = dataset.values
X = array[:,0:4] #contains flower features (petal length, etc..)
y = array[:,4] #contains flower names

#create test and training folds
X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, y,
test_size=0.50, random_state=1)



#Algorithm for regression models
def regModel(name, model):
#Fit and transform data sets according to the regression degree
    #Encode for each class
    #Use encoded training and validation values for prediction on linear regression
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y)
    #Split Data into 2 Folds for Training and Test
    X_Fold1, X_Fold2, y_Fold1, y_Fold2 = train_test_split(X, encoder.transform(y), test_size=0.50, random_state=1)

    poly_reg = None
    if (name == "Linear Regression"):
        poly_reg = PolynomialFeatures(degree=1)
    elif(name == "2 Degree Polynomial Regression"):
        poly_reg = PolynomialFeatures(degree=2)
    elif(name == "3 Degree Polynomial Regression"):
        poly_reg = PolynomialFeatures(degree=3)

    X_Poly1 = poly_reg.fit_transform(X_Fold1)
    X_Poly2 = poly_reg.fit_transform(X_Fold2)
    model.fit(X_Poly1, y_Fold1) #first fold training
    pred1 = model.predict(X_Poly2).round() #first fold testing
    pred1 = np.where(pred1 >= 3.0, 2.0, pred1)
    pred1 = np.where(pred1 <= -1.0, 0.0, pred1)

    model.fit(X_Poly2, y_Fold2) #second fold training
    pred2 = model.predict(X_Poly1).round() #second fold testing
    pred2 = np.where(pred2 >= 3.0, 2.0, pred2)
    pred2 = np.where(pred2 <= -1.0, 0.0, pred2)

    actual = np.concatenate([y_Fold2, y_Fold1])
    predicted = np.concatenate([pred1, pred2])

    # next few lines print the model name, accuracy score, and confusion matrix
    print('%s' % name)
    accuracy = accuracy_score(actual, predicted)
    print('Accuracy Score: ' + str(round(accuracy,3)))
    print('Confusion Matrix: ')
    print(confusion_matrix(actual, predicted))
    print("")


def gaussianModel():

    model = GaussianNB()     #create the model
    model.fit(X_Fold1, y_Fold1) #first fold training
    pred1 = model.predict(X_Fold2) #first fold testing
    model.fit(X_Fold2, y_Fold2) #second fold training
    pred2 = model.predict(X_Fold1) #second fold testing

    # concatenates the test data
    actual_array = np.concatenate([y_Fold2, y_Fold1])     #true array
    prediction_array = np.concatenate([pred1, pred2])     #prediction array

    #prints using the built in functions for accuracy score, confusion matrices, and classification reports
    print("Naive Bayesian Model")
    accuracy = accuracy_score(actual_array, prediction_array)
    print('Accuracy Score: ' + str(round(accuracy,3)))
    print('Confusion Matrix:')
    print(confusion_matrix(actual_array, prediction_array))
    print("")


def knnModel():

    # Create the model
    model = KNeighborsClassifier()

    model.fit(X_Fold1, y_Fold1) #first fold training
    pred1 = model.predict(X_Fold2) #first fold testing
    model.fit(X_Fold2, y_Fold2) #second fold training
    pred2 = model.predict(X_Fold1) #second fold testing

    # concatenates the test data
    actual_array = np.concatenate([y_Fold2, y_Fold1])     #true array
    prediction_array = np.concatenate([pred1, pred2])     #prediction array

    # next few lines print the model name, accuracy score, and confusion matrix
    print("K Nearest Neighbors Model")
    accuracy = accuracy_score(actual_array, prediction_array)
    print('Accuracy Score: ' + str(round(accuracy,3)))
    print('Confusion Matrix:')
    print(confusion_matrix(actual_array, prediction_array))
    print("")
    

def lda():

    # Create the model
    model = LinearDiscriminantAnalysis()

    model.fit(X_Fold1, y_Fold1) #first fold training
    pred1 = model.predict(X_Fold2) #first fold testing
    model.fit(X_Fold2, y_Fold2) #second fold training
    pred2 = model.predict(X_Fold1) #second fold testing

    # concatenates the test data
    actual_array = np.concatenate([y_Fold2, y_Fold1])     #true array
    prediction_array = np.concatenate([pred1, pred2])     #prediction array

    # next few lines print the model name, accuracy score, and confusion matrix
    print("Linear Discriminant Analysis Model")
    accuracy = accuracy_score(actual_array, prediction_array)
    print('Accuracy Score: ' + str(round(accuracy,3)))
    print('Confusion Matrix:')
    print(confusion_matrix(actual_array, prediction_array))
    print("")

def qda():

    # Create the model
    model = QuadraticDiscriminantAnalysis()

    model.fit(X_Fold1, y_Fold1) #first fold training
    pred1 = model.predict(X_Fold2) #first fold testing
    model.fit(X_Fold2, y_Fold2) #second fold training
    pred2 = model.predict(X_Fold1) #second fold testing

    # concatenates the test data
    actual_array = np.concatenate([y_Fold2, y_Fold1])     #true array
    prediction_array = np.concatenate([pred1, pred2])     #prediction array

    # next few lines print the model name, accuracy score, and confusion matrix
    print("Quadratic Discriminant Analysis Model")
    accuracy = accuracy_score(actual_array, prediction_array)
    print('Accuracy Score: ' + str(round(accuracy,3)))
    print('Confusion Matrix:')
    print(confusion_matrix(actual_array, prediction_array))
    print("")


#set up models
models = []
models.append(('Linear Regression', LinearRegression()))
models.append(('2 Degree Polynomial Regression', LinearRegression()))
models.append(('3 Degree Polynomial Regression', LinearRegression()))
#run models
for name, model in models:
    regModel(name, model)
gaussianModel()
knnModel()
lda()
qda()