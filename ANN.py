import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import  metrics

df_train = pd.read_csv('DataTrain_new.csv')

X = df_train.drop(['SalePrice'], axis=1)
y = df_train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.4)
X_train = df_train.drop(['SalePrice'],axis=1)
y_train = df_train['SalePrice']

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

adam = keras.optimizers.Adam(lr=0.005)
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units= 50, activation = 'relu', kernel_initializer = 'uniform', input_shape = (174, )))
# Adding the second hidden layer
classifier.add(Dense(units = 25 , kernel_initializer = 'uniform' , activation = 'relu'))
# Adding the third hidden layer
classifier.add(Dense(units = 50 , kernel_initializer = 'uniform' , activation = 'relu'))
# Adding the output layer
classifier.add(Dense(units = 1 , kernel_initializer = 'uniform' ))

classifier.build()
classifier.summary()


from keras import backend as K
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

# Compiling the ANN
classifier.compile(loss=root_mean_squared_error, optimizer= adam, metrics=['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train.values, y_train.values,validation_split=0.20 , batch_size = 10, epochs = 100)

# result = classifier.score(X_test, y_test)
# print("Accuracy: %.2f%%" % (result*100.0))
y_predicted = classifier.predict(X_test)

mae = metrics.mean_absolute_error(y_test, y_predicted)
mse = metrics.mean_squared_error(y_test, y_predicted)
r2 = metrics.r2_score(y_test, y_predicted)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))

fig, ax = plt.subplots()
fig.suptitle('ANN', fontsize=16)
ax.scatter(y_predicted, y_test, edgecolors=(0, 0, 1))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.show()