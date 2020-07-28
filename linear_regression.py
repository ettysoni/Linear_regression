import numpy as np
from sklearn.linear_model import LinearRegression


#Simple linear regression

#Making the array 2d(1 column)
#Reshape(-1, 1) makes it into a 2d array with 1 column and as many rows
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1)) 

y = np.array([5, 20, 14, 32, 22, 38])

#.fit() calculates the optimal intercept and coefficient
model = LinearRegression().fit(x, y)

#Calculates r^2
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)

#Printing the intercept
print('intercept:', model.intercept_)

#Printing the coefficient
print('slope:', model.coef_)

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')








# Multiple linear regression
x = [[0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]]
y = [4, 5, 20, 14, 32, 22, 38, 43]
x, y = np.array(x), np.array(y)

model_1 = LinearRegression().fit(x, y)
r_sq = model_1.score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

y_pred = model_1.predict(x)
print('predicted response:', y_pred, sep='\n')
