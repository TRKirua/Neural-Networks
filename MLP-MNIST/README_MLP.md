# MNIST Classification with a Multilayer Perceptron (MLP)

This project demonstrates how to use a simple Multilayer Perceptron (MLP) model to classify handwritten digits from the MNIST dataset using TensorFlow and Keras.

## Overview

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each image being 28x28 pixels. This project builds, trains, and evaluates an MLP model for digit classification.

### Steps Performed in the Notebook:

1. **Load and Preprocess Data**:  
   The MNIST dataset is loaded using TensorFlow's `keras.datasets.mnist`. The pixel values are normalized to the range [0, 1] by dividing by 255.

2. **Model Architecture**:
   - A `Sequential` model with the following layers:
     - `Flatten`: Flattens the 28x28 input images into a 1D vector.
     - `Dense`: A fully connected layer with 128 units and ReLU activation.
     - `Dropout`: A dropout layer with a rate of 0.2 to prevent overfitting.
     - `Dense`: The output layer with 10 units (one for each digit) with no activation.
  
3. **Compilation**:
   - The model is compiled with the following:
     - **Loss function**: `SparseCategoricalCrossentropy` (since it's a multi-class classification task).
     - **Optimizer**: `Adam`.
     - **Metrics**: Accuracy.

4. **Training**:
   - The model is trained for 5 epochs on the training data (`x_train` and `y_train`).

5. **Evaluation**:
   - The model is evaluated on the test set (`x_test` and `y_test`) to determine accuracy and loss.

## Dependencies

The project requires the following libraries:
- `tensorflow` >= 2.x
- `numpy` (for data manipulation)

You can install these dependencies with:

```bash
pip install tensorflow numpy
