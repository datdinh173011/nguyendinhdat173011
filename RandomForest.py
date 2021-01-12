import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics

df1 = pd.read_csv('DataTrain_new.csv')
df2 = pd.read_csv('DataTest_new.csv')

X = df1.drop(['SalePrice'], axis=1)
y = df1['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.4)

from sklearn.ensemble import RandomForestRegressor 
  
 # create regressor object 
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
  
# fit the regressor with x and y data 
regressor.fit(X_train, y_train)   

result = regressor.score(X_test, y_test)
print("Accuracy: %.2f%%" % (result*100.0))
y_predicted = regressor.predict(X_test)

mae = metrics.mean_absolute_error(y_test, y_predicted)
mse = metrics.mean_squared_error(y_test, y_predicted)
r2 = metrics.r2_score(y_test, y_predicted)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))

fig, ax = plt.subplots()
fig.suptitle('Random Forest', fontsize=16)
ax.scatter(y_predicted, y_test, edgecolors=(0, 0, 1))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()