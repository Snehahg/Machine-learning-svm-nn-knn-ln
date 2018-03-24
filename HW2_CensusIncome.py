# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:04:49 2018


"""


import pandas as pd
import sys
from sklearn.neural_network import MLPClassifier
from sklearn import svm
#import matplotlib.pyplot as plt
import cnm_plot
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from imblearn.over_sampling import SMOTE


headerFromFile = pd.read_csv(sys.argv[1], sep=':', header=None)

headers = headerFromFile.iloc[:,0:1]
lastRow = pd.DataFrame(["Income"])

if (len(headers) is 14):
    headers = pd.concat([headers, lastRow])

#print(headers)

data = pd.read_csv(sys.argv[2], header=None)
test = pd.read_csv(sys.argv[3], header=None)


data.columns = headers
test.columns = headers
#print(data.iloc[0:5,:-1])
#print(data.describe())

#print(len(data))
#print(headers.iloc[:-1,:])

#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)

# Neural Network

# Here we configure the architecure. These are hidden layers only
# The function will automatically create input nodes (one for each variable) and 
# one output node (for the target value)


mlp = MLPClassifier(hidden_layer_sizes=(5,10,5), max_iter=200)

#train_data = pd.get_dummies(data, columns=list(headers.values.flatten()))
#train_data = pd.get_dummies(data.iloc[:,:-1], columns=headers.iloc[:-1,:])
train_data = pd.get_dummies(data.iloc[:,:-1])
train_target = pd.get_dummies(data.iloc[:,-1])

test_data = pd.get_dummies(test.iloc[:,:-1])
test_target = pd.get_dummies(test.iloc[:,-1])

for i, col in enumerate(train_target.columns.tolist(), 1):
    train_target.loc[:, col] *= i
    #print(i)

train_target_svm = train_target.sum(axis=1)

for i, col in enumerate(test_target.columns.tolist(), 1):
    test_target.loc[:, col] *= i
    #print(i)

test_target_cumulative = test_target.sum(axis=1)




sm = SMOTE(random_state=12, ratio = "minority")
x_train_res, y_train_res = sm.fit_sample(train_data, train_target_svm)
#print(test_data.iloc[0:2, :])
#print(test_target.iloc[0:2, :])
#print(list(test_target.columns.values))

data_dummy_columns = list(train_data.columns.values)
#print(data_dummy_columns[0:3])
test_dummy_columns = list(test_data.columns.values)
#print(test_data.iloc[79:83,79:83])
for i in data_dummy_columns:
    if i not in test_dummy_columns:
        #print(data_dummy_columns.index(i))
       
        test_data.insert(data_dummy_columns.index(i), i, 0)
#print(data_dummy_columns)
#print(test_dummy_columns)

# Fit the model ... learn the weights

mlp.fit(x_train_res, y_train_res)

predictions = mlp.predict(test_data)

#print(predictions[0:4])
# Check the confusion matrix ...

predictions = pd.DataFrame(predictions)#, columns = list(test_target.columns.values))
print(predictions.iloc[0:4,:])
#target_values = test_target.idxmax(axis=1)
#test_prediction = predictions.idxmax(axis=1)

#print(target_values.iloc[0:4])
#print(test_prediction.iloc[0:4])
#cf = confusion_matrix(target_values,test_prediction)
cf = confusion_matrix(test_target_cumulative, predictions)
print(cf)

cnm_plot.plot_confusion_matrix(cf,classes=list(test_target.columns.values))

print(mlp.score(test_data,test_target_cumulative))

print("Accuracy score: ", accuracy_score(test_target_cumulative, predictions) )


#SVM
# We start with a simple linear classifier

#print(train_data.shape)
#print(train_target.shape)

#x = np.asarray(train_data)
#y = np.asarray(train_target)
"""
for i, col in enumerate(y_train_res.columns.tolist(), 1):
    y_train_res.loc[:, col] *= i
    #print(i)

train_target_svm = y_train_res.sum(axis=1)
print(train_target_svm[5:10])
"""
#svc_linear = svm.SVC(kernel='linear', C=20)
svc_linear = svm.SVC(C = 100)
#svc_linear.fit(np.asarray(train_data), np.asarray(train_target))
svc_linear.fit(x_train_res[0:1000], train_target_svm[0:1000])
print("fitted")
predicted= svc_linear.predict(test_data)
print("predicted", predicted[0:10])
#predicted = pd.DataFrame(predicted, columns = list(test_target.columns.values))
#print(predicted.iloc[0:5,:])
#test_predicted = predicted.idxmax(axis=1)
#cnf_matrix = confusion_matrix(target_values, test_predicted)

"""
for i, col in enumerate(test_target.columns.tolist(), 1):
    test_target.loc[:, col] *= i
    #print(i)

test_target_svm = test_target.sum(axis=1)
"""
print("test_target_cumulative[5:10] ", test_target_cumulative[5:10])

cnf_matrix = confusion_matrix(test_target_cumulative, predicted)
print(cnf_matrix)


print("Accuracy score: ", accuracy_score(test_target_cumulative, predicted) )

"""
# OK, not great. Let's try to change the parameter C ...
svc_linear = svm.SVC(kernel='linear', C=100)
svc_linear.fit(X_train, y_train)

predicted= svc_linear.predict(X_test)
cnf_matrix = confusion_matrix(y_test, predicted)
print(cnf_matrix)

#Plot the data to see which type of kernel to use
plt.scatter(data.iloc[:,0], data.iloc[:,1], c = data.iloc[:,2])
plt.show()

# A little better ... Now let's try the radial kernel: The default kernel is 'rbf'. 
# We will use the default parameters at first

svc_radial = svm.SVC()
svc_radial.fit(x_train, y_train)
predicted= svc_radial.predict(x_test)
cnf_matrix = confusion_matrix(y_test, predicted)
print(cnf_matrix)

# Much better. Let's try different values for gamma ...

svc_radial = svm.SVC(gamma = .1)
svc_radial.fit(x_train, y_train)
predicted= svc_radial.predict(x_test)
cnf_matrix = confusion_matrix(y_test, predicted)
print(cnf_matrix)
"""





