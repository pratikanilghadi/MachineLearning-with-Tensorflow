'''Something'''
import tensorflow as tf
import numpy as np
from tensorflow import keras

#Defines the shape of the model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#Defines what type of calculations we want on the model
model.compile(optimizer='sgd',loss = 'mean_squared_error')

#The data we use to train our model
xs = np.array([-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0],dtype=float)
ys = np.array([-8.0,-5.0,-2.0,1.0,4.0,7.0,10.0,13.0,16.0,19.0,22.0],dtype=float)

#The trainin of our model
model.fit(xs,ys,epochs=500)

#Printing the prediction made by our model after learning from the data we provided and asking for an output for a new input
print(model.predict([10.0]))