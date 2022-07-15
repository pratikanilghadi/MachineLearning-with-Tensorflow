'''This program is for the model analysis of a fashion.minst dataset'''
import tensorflow as tf
import numpy as np
from tensorflow import keras

#Importing of the dataset we will be using
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#For visualization of the data we are passing
#import matplotlib.pyplot as plt
#plt.imshow(training_images[0])
#print(training_labels[0])
#print(training_images[0])

#Defines the shape of the model having 3 layers
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(240 , activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#The processing of the model
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Trainging of the model
model.fit(training_images, training_labels, epochs=5)

#Testing our trained model
model.evaluate(test_images, test_labels)

#List for the prediction for the test impage list
classifications = model.predict(test_images)

#Printing the probabiltiy for that specific data location and its label
print(classifications[0])
print(test_labels[0])