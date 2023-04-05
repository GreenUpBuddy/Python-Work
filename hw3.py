# ML in Python, homework 3
# name: Martine De Cock
# description: Neural network for predicting personality of Facebook users
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.layers import Dense,Activation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Loading the data
# There are 9500 users (rows)
# There are 81 columns for the LIWC features followed by columns for
# openness, conscientiousness, extraversion, agreeableness, neuroticism
# As the target variable, we select the extraversion column (column 83)
dataset = np.loadtxt("Facebook-User-LIWC-personality-HW3.csv", delimiter=",")
X = dataset[:,0:81]
y = dataset[:,83]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1500)

# Training and testing a linear regression model
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred = linreg.predict(X_test)
print('MSE with linear regression:', metrics.mean_squared_error(y_test, y_pred))

# Training and testing a neural network
model = Sequential()
model.add(Dense(1,input_dim=81,activation = "linear"))
model.add(tf.keras.layers.Dense(4))
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=RMSprop(lr=0.001), loss='logcosh', metrics=['mse'])
model.fit(X_train,y_train,batch_size=64,
    epochs=75,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(X_test, y_test))
y_pred = model.predict(X_test)
print('MSE with neural network:', metrics.mean_squared_error(y_test, y_pred))