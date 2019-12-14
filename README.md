# ANN-Classification-on-Organic-Compounds

The given dataset contains details about organic chemical compounds including their chemical features, isomeric conformation, names and the classes in which they are classified. The compounds are classified as either ‘Musk’ or ‘Non-Musk’ compounds. The task was to classify these compounds accordingly. I used an Artificial Neural Network (Multi-Layer Perceptron) built using Keras to do the classification task. 

The metrics of the model are as follows:
+ Validation Loss: 0.055
+ Validation Accuracy: 97.348%
+ F1-Score: 0.989
+ Precision: 0.991
+ Recall: 0.988

The structure of the neural network was as follows:
+ Input Layer - 166 Features
+ Hidden Layer - 20 Neurons (Activation Function - Tangent Hyperbolic)
+ Output Later - 1 Output Feature - Class Label (Activation Function - Sigmoid)
+ Optimizer - Adam
+ Loss - Binary Crossentropy
+ Epochs - 10
