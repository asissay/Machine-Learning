import numpy
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU 
from keras.optimizers import SGD, adam
# load the dataset
dataset = loadtxt('t_0.dat')
dataset1 = loadtxt('t_0.25.dat')
# split into input (X) and output (y) variables
X = dataset[:,0:4]
y = dataset[:,4]
y_norm = y/numpy.amax(y)
X_norm = X/numpy.amax(X)
xx = dataset1[:,0:4]

#define the keras model
model = Sequential()
model.add(Dense(1000, input_dim=4, activation='tanh'))
model.add(Dropout(rate=0.5))
model.add(Dense(500, activation='tanh'))
model.add(Dense(250, activation='tanh'))
model.add(Dropout(rate=0.5))
model.add(Dense(125, activation='tanh'))
model.add(Dense(1, activation='tanh'))
# compile the keras model (lr=0.00000000000001)
# Accuracy did not work because the data was not normalized and large changes in the data like going from neg to pos will render the accuracy 0
model.compile(optimizer='adam',loss='mse', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y_norm, epochs=1000, batch_size=5, verbose=0)
# evaluate the keras model
_, accuracy = model.evaluate(X, y_norm)
print('Accuracy: %.2f' % (accuracy*100))
# make predictions with the model
predictions = model.predict(X)
print(predictions)
# summarize the first 5 cases
for i in range(4):
    print('%s => %.6f (expected %.6f)' % (X[i].tolist(), predictions[i]*numpy.amax(y), y[i]))

