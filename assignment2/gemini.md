# Assignment 2: Neural Networks, Backpropagation, and Cross-Validation

## Overview
This assignment involves building a custom neural network library from scratch and applying it to two problems: the XOR problem and predicting NYC taxi trip durations.

## Part 1: Neural Network Library Implementation
Implement the following components in a documented and modular fashion:

- [ ] **`Layer` (Base Class)**: Define `forward` and `backward` methods.
- [ ] **`Linear` Layer**:
    - [ ] Forward: $f(\mathbf{x}; \mathbf{w}) = \mathbf{x} \mathbf{w}^T + \mathbf{b}$
    - [ ] Backward: Compute gradients for $\mathbf{w}$, $\mathbf{b}$, and input $\mathbf{x}$.
- [ ] **`Sigmoid` Layer**: Implement forward/backward for the logistic sigmoid function.
- [ ] **`ReLU` Layer**: Implement forward/backward for Rectified Linear Unit.
- [ ] **`BinaryCrossEntropyLoss`**: Implement forward/backward for BCE loss.
- [ ] **`Sequential` Class**:
    - [ ] Maintain a list of layers.
    - [ ] Inherit from `Layer`.
    - [ ] Implement `forward` and `backward` passes through all layers.
- [ ] **Saving and Loading**:
    - [ ] Implement functionality to save model weights to a file.
    - [ ] Implement functionality to load model weights from a file.

## Part 2: XOR Problem Testing
- [ ] Construct a network with 1 hidden layer (2 nodes).
- [ ] Solve XOR using `Sigmoid` activations.
- [ ] Solve XOR using `Tanh` activations (if implemented/available).
- [ ] Compare training ease between `Sigmoid` and `Tanh` in the notebook.
- [ ] Save the solved weights as `XOR_solved.w`.

## Part 3: Predicting Trip Duration (NYC Taxi Dataset)
- [ ] **Data Loading**: Load `nyc_taxi_data.npy`.
- [ ] **Preprocessing**:
    - [ ] Experiment with feature selection and transformations.
    - [ ] Apply normalization.
    - [ ] **Document** features used and transformations in a separate document (with plots).
- [ ] **Model Selection**:
    - [ ] Experiment with at least 3 hyperparameter configurations.
    - [ ] Implement **Early Stopping** (stop after 3 steps of no improvement).
    - [ ] Plot training vs. validation loss for each configuration.
    - [ ] Evaluate final models on the test set and report accuracy.
- [ ] **Benchmarking**: Aim to beat or compare against the benchmark of **0.513 RMSLE**.

## Submission Requirements
- [ ] Zip file containing all relevant code/notebooks.
- [ ] Ensure code is reproducible (plots and results).
- [ ] Include additional instructions if necessary.
