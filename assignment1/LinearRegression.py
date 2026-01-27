import numpy as np
from sklearn.model_selection import train_test_split

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3, 
            learning_rate = 0.01):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        learning_rate: float
            the step size for gradient descent
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate

       
        n_samples, n_features = X.shape
        n_outputs = y.shape[1] if y.ndim > 1 else 1

        self.weights = np.random.randn(n_features, n_outputs) * 0.01
        self.bias = np.zeros((1,n_outputs))

        #split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1)
        
        #init variables for early stopping and training
        best_validation_loss = float('inf')#make inf so future epoch will automatically be lower than first
        patience_counter = 0
        best_weights = self.weights.copy()
        best_bias = self.bias.copy()

        self.losses = []

        #main trainng loop
        for epoch in range(self.max_epochs):
            #shuffle training data at the beginning of each epoch
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            #batch processing loop
            for i in range(0, X_train.shape[0], self.batch_size):
                X_batch = X_train_shuffled[i: i + self.batch_size]#use array-slicing to get the mini-batch
                y_batch = y_train_shuffled[i: i + self.batch_size]
                n_batch = X_batch.shape[0]

                #forward pass
                y_pred = X_batch @ self.weights + self.bias

                #loss calculation
                mse_loss = np.mean((y_pred - y_batch)**2)
                l2_loss = (self.regularization / 2) * np.sum(self.weights ** 2)
                total_loss = mse_loss + l2_loss 
                self.losses.append(total_loss)

                #backward pass (gradient calculation)
                error = y_pred - y_batch
                grad_weights = (2/n_batch) * X_batch.T @ error + self.regularization * self.weights
                grad_bias = (2/n_batch) * np.sum(error, axis = 0)

                #parameter update (weight updates)
                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias

            #early stopping check(at the end of each epoch)
            validation_loss = self.score(X_val, y_val)

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        #restore the best weights found
        self.weights = best_weights
        self.bias = best_bias
        



    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """

        return X @ self.weights + self.bias # y_hat = Xw + b
        

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred)**2) #from MSE formula
        return mse
    
    def save(self, file_path):
        np.savez(file_path, weights = self.weights, bias = self.bias)