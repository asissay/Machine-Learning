#import tensorflow as tf

#from __future__ import absolute_import
#from keras import backend as K
#from keras.utils.generic_utils import get_from_module
import numpy

#import pkg_resources
#pkg_resources.require("keras==2.1.5")
#import keras

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU 
from keras.optimizers import SGD, adam, Nadam

import keras.backend as K
#from  keras.utils.generic_utils import get_from_module

from sklearn.preprocessing import MinMaxScaler

# load the dataset
dataset  = loadtxt('ocs_jj.dat')
dataset1 = loadtxt('ocs_jjj.dat')
# split into input (X) and output (y) variables
X = dataset[:,0:5]
y = dataset[:,5]
#sample_weight = dataset[:,6]
y_norm = y/numpy.amax(y)
X_norm = X/numpy.amax(X)
x_test = dataset1[:,0:5]

y_std = (y - numpy.mean(y))/numpy.std(y)
X_std = (X - numpy.mean(X))/numpy.std(X)

X_j = 2 * (X - numpy.amin(X))/ (numpy.amax(X) - numpy.amin(X)) -1
y_j = 2 * (y - numpy.amin(y))/ (numpy.amax(y) - numpy.amin(y)) -1  

X_jj =  (X - numpy.amin(X))/ (numpy.amax(X) - numpy.amin(X)) 
y_jj =  (y - numpy.amin(y))/ (numpy.amax(y) - numpy.amin(y)) 
x_test_jj = (x_test - numpy.amin(x_test))/ (numpy.amax(x_test) - numpy.amin(x_test)) 

scaler = MinMaxScaler()
#X, y = MinMaxScaler(), MinMaxScaler()
Xj = scaler.fit(X_jj)
yJ = scaler.fit(y_jj.reshape(499,1))
X_jjj = Xj.transform(X_jj)
y_jjj = yJ.transform(y_jj.reshape(499,1))


#define the keras model
# Tanh because values run from neg to pos
model = Sequential()
model.add(Dense(1000, input_dim=5, activation='tanh'))
model.add(Dropout(rate=0.5))
model.add(Dense(500, activation='tanh'))
model.add(Dense(250, activation='tanh'))
model.add(Dropout(rate=0.5))
model.add(Dense(125, activation='tanh'))
model.add(Dense(1, activation='tanh'))
# compile the keras model (lr=0.00000000000001)  # Learning rate actually makes error go down its best to revert to default ? 0.01?
# Accuracy did not work because the data was not normalized and large changes in the data like going from neg to pos will render the accuracy 0
#mse1 = tf.keras.losses.MeanSquaredError(reduction="none", name="msle")
'''Calculates the mean accuracy rate across all predictions for
    multiclass classification problems.
    '''
# New function for calculating error. My values are different.
def custom_accuracy(y_true, y_pred):
    # return K.mean(y_pred, axis=-1)
        return 1-K.abs(y_true-y_pred) #, axis=-1)#, axis=-1)
    #  return K.mean(K.abs(y_pred - y_true))
        #return K.mean(K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1)))




# Regular accuracy command is for binary data ( for 0 or 1) if target data is float values that vary greatly, it will not work.
# How is loss calculated. Do you have to consider loss when calculating error?
model.compile(optimizer='adam', loss='mse', metrics=[custom_accuracy]) # 'categorical_accuracy'] --> gives 100% accuracy

# Check epoch convergence. Might be over-training -> reduce accuracy which will obviously reduce predic. values (bad model)?
# #m.result.numpy()
# # fit the keras model on the dataset
model.fit(X_jjj, y_jjj, epochs=50, batch_size=10, verbose=0)
# evaluate the keras model
_,accuracy = model.evaluate(X_jjj, y_jjj, batch_size=10)
print('Accuracy: %.2f' % (accuracy*100))
# # make predictions with the model
predictions = model.predict(X_jjj)
# #print(predictions)
# summarize the first 5 cases
for i in range(4):
    print('%s => %.5f (expected %.5f)' % (X[i].tolist(), predictions[i], y_jjj[i]))
# #    print('%s => %.3f (expected %.3f)' % (X[i].tolist(), (predictions[i]*(numpy.amax(y_j)-numpy.amin(y_j))+ 2*(numpy.amin(y_j))+1)/2, y[i]))
# #    print('%s => %.5f (expected %.5f)' % (X[i].tolist(), predictions[i]*(numpy.amax(y)-numpy.amin(y)) + numpy.amin(y), y[i]))
#     print('%s => %.5f (expected %.5f)' % (X[i].tolist(), predictions[i] , y_jjj[i]))
