
# import the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# read the dataset
dataset=pd.read_csv(r'C:\Users\ADMIN\Downloads\23rd- Poly\23rd- Poly\1.POLYNOMIAL REGRESSION\emp_sal.csv')

# x and y variables
x=dataset.iloc[: , 1:2].values
y = dataset.iloc[:,2].values

from sklearn.svm import SVR
regressor=SVR(kernel="poly",degree=4,gamma='auto',C=6,epsilon=1.8)
regressor.fit(x,y)

# predicting a new result with SVR
y_pred_svr =regressor.predict([[6.5]])
y_pred_svr
