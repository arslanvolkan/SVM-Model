import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np


#getting datas from excel
dataset = pd.read_excel(r'C:\Users\VOLKAN\Desktop\deneme.xlsx' )


#defining initial and target values
X = dataset.iloc[0:200000, 0:23].values
Y = dataset.iloc[0:200000,23:24].values.ravel()


#defining training and test data percentages
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.10,random_state=42)
 

#Create and train the model
SVRMODEL = SVR()
SVRMODEL.fit(X_train,Y_train)
Y_pred = SVRMODEL.predict(X_test)


diff = Y_pred - Y_test
percentDiff = (diff / Y_test) * 100
absPercentDiff = np.abs(percentDiff)
mape = absPercentDiff.mean()
print("Avarage absolute error value")
print(mape)
 