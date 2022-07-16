'''The code for identifying the handwritten letters using Convolutions and pooling'''
import tensorflow as tf
import numpy as np
from tensorflow import keras

# YOUR CODE STARTS HERE (Callback Funtion)
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.90):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
# YOUR CODE ENDS HERE

# YOUR CODE STARTS HERE (Refining the Dataset)
mnist = tf.keras.datasets.mnist
(x_images, x_labels), (y_images, y_labels) = mnist.load_data()
x_images = x_images.reshape(-1,28,28,1)
x_images = x_images/255
y_images = y_images.reshape(-1,28,28,1)
y_images = y_images/255
# YOUR CODE ENDS HERE

#YOUR CODE STARTS HERE (Defining the structure of the model)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation = 'relu'),
    tf.keras.layers.Dense(10,activation='softmax'),
])
#YOUR CODE ENDS HERE

# YOUR CODE STARTS HERE (Defining the functionality of the model)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(x_images,x_labels,epochs=10,callbacks=[callbacks])

classifications =  model.predict(y_images)
print(classifications[0])
print(y_labels[0])
# YOUR CODE ENDS HERE
