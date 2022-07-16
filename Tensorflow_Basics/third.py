'''Something'''

import tensorflow as tf
print(tf.__version__)

#Inheritng the callback function
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.90):
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images.reshape(60_000,28,28,1)
training_images=training_images/255.0
test_images = test_images.reshape(10_000,28,28,1)
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),  #Adds the convolution Layer which extract the necessary components from the required image
  tf.keras.layers.MaxPooling2D(2,2),  #Adds the Pooling layer which Decreaseds the size of the training data's pixels
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(256, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary() #Gives the summary of the model i.e the history of the modification to the model
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

classifications =  model.predict(training_images)
print(classifications[0])
print(training_labels[0])