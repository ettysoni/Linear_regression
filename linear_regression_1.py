# Multivariate linear reggression

# Take into account multiple input features like fixed acidity, residual sugar, alcohol to predict the quality of wine.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('/Users/ettysoni/Google Drive/Linear_regression/winequality-red.csv')
dataset.shape
dataset.describe()
dataset.isnull().any()

# Remove NA's if any
# dataset = dataset.fillna(method='ffill')

# Dividing the data into attributes and labels
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
Y =  dataset['quality'].values

plt.figure(figsize=(5,5))
plt.tight_layout()
seabornInstance.distplot(dataset['quality'])
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

print(regressor.coef_) 
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})
df1 = df.head(25)

df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='green')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, y_pred)))