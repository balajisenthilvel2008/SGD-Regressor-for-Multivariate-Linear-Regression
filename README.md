# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Load the California housing dataset, extract features (first three columns) and targets (target variable and sixth column), and split the data into training and testing sets.
2. Data Scaling: Standardize the feature and target data using StandardScaler to enhance model performance.
3. Model Training: Create a multi-output regression model with SGDRegressor and fit it to the training data.
4. Prediction and Evaluation: Predict values for the test set using the trained model, calculate the mean squared error, and print the predictions along with the squared error.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the
price of the house and number of occupants in the house with SGD regressor.
Developed by: BALAJI S
RegisterNumber: 212225220015
*/
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data = fetch_california_housing()
X = data.data[:, :3]
Y=np.column_stack((data.target,data.data[:, 6]))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
sgd=SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
print("\nPredictions:\n",Y_pred[:5])
```

## Output:
<img width="587" height="190" alt="image" src="https://github.com/user-attachments/assets/33009df9-9bb3-4a12-96f5-8964e2850a1e" />
<br>

<img width="218" height="112" alt="image" src="https://github.com/user-attachments/assets/1a7c6a40-5b55-4f4a-9633-2fa0e9555041" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
