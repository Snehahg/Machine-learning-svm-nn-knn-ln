# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:02:55 2018


"""

import sys
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold

data = pd.read_csv(sys.argv[1])
test = pd.read_csv(sys.argv[2])

#print(data.iloc[0:6, :])

#print(data.isnull().any().any())

data1 = data.drop(data.columns[[0, 1, 15, 16]], axis=1, inplace=False)
test1 = test.drop(test.columns[[0, 1, 15, 16]], axis=1, inplace=False)

#print("datat " , data.iloc[0:3, :])

#print("\ndata1 " , data1.iloc[0:3, :])

#train_data = data1.iloc[:,:-1]
train_target = data1.iloc[:,-1]

# Dummy variables
train_data = pd.get_dummies(data.iloc[:,:-1])
#train_target = pd.get_dummies(data.iloc[:,-1])

test_data = pd.get_dummies(test.iloc[:,:-1])
#test_target = pd.get_dummies(test.iloc[:,-1])
test_target = test.iloc[:,-1]



data_dummy_columns = list(train_data.columns.values)
#print(data_dummy_columns[0:3])
test_dummy_columns = list(test_data.columns.values)
#print(test_data.iloc[79:83,79:83])
for i in data_dummy_columns:
    if i not in test_dummy_columns:
        #print(data_dummy_columns.index(i))
       
        test_data.insert(data_dummy_columns.index(i), i, 0)
        

        
        
# here we do a 5-fold CV using the Neural Network - should give a better understanding of the 
# true MSE for this model.
kf = KFold(n_splits=5)
kf.get_n_splits(train_data)
#print("g", g)

# Now we compute the models and average the MSEs:
MSE = 0.0
MSE1 = 0.0
MLPregr = MLPRegressor(hidden_layer_sizes=(10,20,10), max_iter=1000)
    
for train_index, test_index in kf.split(train_data):
    #print("train_index, test_index", train_index, test_index)
    MLPregr.fit(train_data.iloc[train_index],train_target.iloc[train_index])
    predictions = MLPregr.predict(train_data.iloc[test_index])
    #int_pred = []
    #int_pred.append(int(i) for i in predictions)
    #predictions = pd.DataFrame(data = int_pred)
    #predictions_transpose = predictions.transpose()
    MSE += sum((predictions - train_target.iloc[test_index])**2)/len(predictions)
    
print(MSE/5.0)


predicted = MLPregr.predict(test_data)
#test_prediction_int = []
#test_prediction_int.append(int(i) for i in predicted)
    

#test_target_dummy_columns = list(test_target.columns.values)
#print(data_dummy_columns[0:3])
#predicted_dummy_columns = list(test_data.columns.values)
#print(test_data.iloc[79:83,79:83])
"""
for i in data_dummy_columns:
    if i not in test_dummy_columns:
        print(data_dummy_columns.index(i))
       
        test_data.insert(data_dummy_columns.index(i), i, 0)
"""
#print("predicted",predicted.shape)
#print("test_target",test_target.shape)
#test_prediction_int_df = pd.DataFrame(data = test_prediction_int)
#test_prediction_int_df_traponse = test_prediction_int_df.transpose()
#test_target1=pd.DataFrame(data = test_target)

#print(test_target.iloc[0:3])

MSE1 = sum((predicted - test_target)**2)/len(predicted)
"""
for i in range(0,10):
    print(predicted.iloc[i], test_target.iloc[i])
"""
print(MSE1)

#pred = MLPregr.predict(train_data.iloc[0:10,:])
#print(predicted[0:10])
#print("Actual")
#print(test_target[0:10])
    





#Linear regression
# Create linear regression object
regr = linear_model.LinearRegression()

MSE = 0.0
for train_index, test_index in kf.split(train_data):
    #print("train_index, test_index", train_index, test_index)
    regr.fit(train_data.iloc[train_index],train_target.iloc[train_index])
    linear_predictions = regr.predict(train_data.iloc[test_index])
    #int_pred = []
    #int_pred.append(int(i) for i in predictions)
    #predictions = pd.DataFrame(data = int_pred)
    #predictions_transpose = predictions.transpose()
    MSE += sum((linear_predictions - train_target.iloc[test_index])**2)/len(linear_predictions)

print("MSE for Linear regression on train data", MSE/5.0)

MSE1 = 0.0
predicted_linear = regr.predict(test_data)
MSE1 = sum((predicted_linear - test_target)**2)/len(predicted_linear)
print("MSE for Linear regression on test data", MSE1)

#TODO: lasso or ridge




