# Assignment 2: Neural Networks, Backpropagation, and Cross-Validation

## Overview
This assignment involves building a custom neural network library from scratch and applying it to two problems: the XOR problem and predicting NYC taxi trip durations.

## Part 1: Neural Network Library Implementation
Implement the following components in a documented and modular fashion:

- [x] **`Layer` (Base Class)**: Define `forward` and `backward` methods.
- [x] **`Linear` Layer**:
    - [x] Forward: $f(\mathbf{x}; \mathbf{w}) = \mathbf{x} \mathbf{w}^T + \mathbf{b}$
    - [x] Backward: Compute gradients for $\mathbf{w}$, $\mathbf{b}$, and input $\mathbf{x}$.
- [x] **`Sigmoid` Layer**: Implement forward/backward for the logistic sigmoid function.
- [x] **`ReLU` Layer**: Implement forward/backward for Rectified Linear Unit.
- [x] **`BinaryCrossEntropyLoss`**: Implement forward/backward for BCE loss.
- [x] **`Sequential` Class**:
    - [x] Maintain a list of layers.
    - [x] Inherit from `Layer`.
    - [x] Implement `forward` and `backward` passes through all layers.
- [x] **Saving and Loading**:
    - [x] Implement functionality to save model weights to a file.
    - [x] Implement functionality to load model weights from a file.

## Part 2: XOR Problem Testing
- [x] Construct a network with 1 hidden layer (2 nodes).
- [x] Solve XOR using `Sigmoid` activations.
- [x] Solve XOR using `Tanh` activations (if implemented/available).
- [x] Compare training ease between `Sigmoid` and `Tanh` in the notebook.
- [x] Save the solved weights as `XOR_solved.w`.

## Part 3: Predicting Trip Duration (NYC Taxi Dataset)
- [x] **Data Loading**: Load `nyc_taxi_data.npy`.
- [x] **Preprocessing**:
    - [x] Experiment with feature selection and transformations.
    - [x] Apply normalization.
    - [x] **Document** features used and transformations in a separate document (with plots).
- [x] **Model Selection**:
    - [x] Experiment with at least 3 hyperparameter configurations.
    - [x] Implement **Early Stopping** (stop after 3 steps of no improvement).
    - [x] Plot training vs. validation loss for each configuration.
    - [x] Evaluate final models on the test set and report accuracy.
- [x] **Benchmarking**: Aim to beat or compare against the benchmark of **0.513 RMSLE**. (Compared: best achieved was 0.9237 RMSLE)

## Submission Requirements
- [x] Zip file containing all relevant code/notebooks.
- [x] Ensure code is reproducible (plots and results).
- [x] Include additional instructions if necessary.
