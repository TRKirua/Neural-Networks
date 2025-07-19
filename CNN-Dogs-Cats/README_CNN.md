# Dog vs Cat Classification with a Convolutional Neural Network (CNN)

This project demonstrates how to classify images of cats and dogs using a simple Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Overview

The dataset consists of labeled images of cats and dogs, divided into training and test sets. Each image is resized to **150x150 pixels**. The goal is to train a CNN to distinguish between the two classes.

## Steps Performed in the Notebook

### 1. Load and Preprocess Data

- The images are stored in separate folders:
  - `training_set/cats`, `training_set/dogs`
  - `test_set/cats`, `test_set/dogs`
- The `ImageDataGenerator` class is used to:
  - Rescale pixel values to the `[0, 1]` range
  - Apply random transformations (shear, zoom, horizontal flip) for data augmentation on training images
- Datasets are loaded using `flow_from_directory()` with:
  - `target_size=(150, 150)`
  - `class_mode='binary'`

### 2. Model Architecture

A simple `Sequential` CNN model with the following layers:

- `Conv2D(64, (3, 3), activation='relu')`
- `MaxPooling2D(pool_size=(2, 2))`
- `Flatten()`
- `Dense(64, activation='relu')`
- `Dense(1, activation='sigmoid')` (for binary classification)

### 3. Compilation

The model is compiled using:

- **Loss**: `binary_crossentropy`
- **Optimizer**: `RMSprop`
- **Metric**: `accuracy`

### 4. Training

- Trained for **20 epochs**
- Batch size: **20**
- Training and validation performed using the respective datasets

### 5. Evaluation and Visualization

- Accuracy and loss are plotted over epochs for both training and validation sets
- Predictions are made on individual test images and visualized using `matplotlib`

## Dependencies

Install the required libraries with:

```bash
pip install tensorflow numpy matplotlib
