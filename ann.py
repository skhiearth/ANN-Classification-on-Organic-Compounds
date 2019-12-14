# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Importing the dataset
dataset = pd.read_csv('musk_csv.csv')
dataset.head()

X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Initialising the ANN
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(40, input_shape=(166,),
                          activation=tf.nn.sigmoid),
    tf.keras.layers.Dense(2, activation=tf.nn.sigmoid)
])

# Compiling the ANN
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
history = model.fit(X, y, validation_split = 0.2, epochs = 100, verbose = 0)

# Model Accuracy Visualisation
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Model Loss Visualisation
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()