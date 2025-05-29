# DEEP-LEARNING-PROJECT:

**COMPANY**  : CODTECH IT SOLUTIONS

**NAME**     : POLASANI PURUSHOTHAM REDDY

**INTERN ID**: CT08DN1100

**DOMAIN**   : DATA SCIENCE

**DURATION** : 8 WEEEKS

**MENTOR**   : NEELA SANTOSH

# OUTPUT:


# 🧠 Deep Learning Project: 

# Image Classification with CNN (TensorFlow)

## Description:

This project is an implementation of an image classification model using deep learning techniques. The model is built using a Convolutional Neural Network (CNN) with TensorFlow and trained on the CIFAR-10 dataset. The goal is to classify images into one of ten categories such as airplanes, cars, birds, cats, and more.

This project demonstrates a complete deep learning workflow — from data preprocessing and model training to evaluation and visualization — in a clean, modular, and reproducible way. It is implemented in Python using VS Code as the development environment and TensorFlow/Keras as the deep learning framework.



## 🎯 Project Objectives
Learn to preprocess and normalize image data, build and train a CNN using TensorFlow, evaluate it on test data, visualize performance, and save the model for future use or deployment.

## 📚 Dataset

 **CIFAR-10** dataset  contains 60,000 32x32 color images in 10 different classes, with 6,000 images per class. It is a widely used benchmark dataset for image classification.
#  Classes:
- airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck.
The dataset is automatically downloaded using `tensorflow.keras.datasets`.
## 🛠️ Model Architecture
A simple CNN with 3 convolutional layers, each followed by ReLU and max-pooling, then flattened and passed through a dense layer and a final softmax output layer for classification.
The model was compiled using the **Adam optimizer**, **sparse categorical cross-entropy** as the loss function, and accuracy as the evaluation metric.

## 🧪 Training & Evaluation

The model is trained over **10 epochs** on the training set with real-time validation on the test set. Accuracy and loss values are plotted to visually monitor the training process. After training, the model is evaluated on the test dataset, and the final accuracy is printed.


## 📈 Visualizations

Two visualizations are generated and saved:
- `accuracy_plot.png`: A plot showing how training and validation accuracy progressed over epochs.
- `sample_predictions.png`: A 3x3 grid of test images showing actual vs predicted labels.
These plots are useful for understanding model behavior and spotting signs of overfitting or underfitting.

## 💾 Model Saving:
The trained model is saved in the HDF5 format as `cnn_model.h5`, allowing easy reuse or deployment in applications without retraining.
```python
model.save("cnn_model.h5")

