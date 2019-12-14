# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('musk_csv.csv')
dataset.head()

X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Initialising the ANN
import tensorflow as tf
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(22, input_shape=(166,),
                          activation=tf.nn.tanh),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

# Compiling the ANN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
history = model.fit(X, y, validation_split = 0.2, epochs = 14, verbose = 0)

# Model Accuracy Visualisation
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# Model Loss Visualisation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

# Exporting model to .h5
model.save("ann.h5")
print("Saved model to disk!")

# Get Validation Accuracy, Validation Loss
print(history.history['val_accuracy'][-1])
print(history.history['val_loss'][-1])

# Predicting the Test set results
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
y_pred = model.predict(X_test)
y_pred = y_pred > 0.5
y_pred

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
