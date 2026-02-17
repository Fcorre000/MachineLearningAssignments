import numpy as np
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        self.weights = None
        self.bias = None
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.losses = []

    def _softmax(self, z):
        """
        Subtract the maximum value across each row of 'z' for numerical stability.
        np.max(z, axis=1, keepdims=True) finds the maximum value in each row and
        keeps the dimension, allowing for proper broadcasting during subtraction.
        This step prevents potential overflow errors when computing np.exp() for
        large 'z' values, as subtracting a constant from 'z' does not change
        the final softmax probabilities relative to each other.
        """
        exp_z = np.exp(z - np.max(z, axis = 1, keepdims=True))

        #Calculate the sum of the exponentials for each row.
        #This sum serves as the normalization factor, ensuring that the
        # output probabilities for each sample sum to 1.
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y, learning_rate=0.01):
        self.learning_rate = learning_rate
    
        #Determine the number of samples and features from the input data X
        n_samples, n_features = X.shape

        if y.ndim == 1:
            #assuming y contains integer class labels for multiclass classification
            #get unique classes to determine the number  of output units
            self.classes = np.unique(y)
            n_outputs = len(self.classes)
            
            #one-hot encode y
            y_one_hot = np.zeros((n_samples, n_outputs))
            for i, label in enumerate(y):
                y_one_hot[i, np.where(self.classes == label)[0][0]] = 1
            y_processed = y_one_hot
        else:
            #if y is already 2D (one-hot encoded), use it directly
            n_outputs = y.shape[1]
            y_processed = 1
        #initialize weights with small random values to break symmetry and prevent
        #all features from having the same influence
        #the dimensions are (number of features, number of output classes)
        self.weights = np.random.randn(n_features, n_outputs) * 0.01
        #init biases to zeros
        self.bias = np.zeros((1, n_outputs))
        #split data into training/validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y_processed, test_size=0.1)
        #set to inf so first loss will always be better
        best_validation_loss = float('inf')
        patience_counter = 0
        best_weights = self.weights.copy()
        best_bias = self.bias.copy()
        #init empty lis to store loss values per batch during training 
        self.losses = []

        #----MAIN TRAINING LOOP---: iterates for a maximum number of epochs
        for epoch in range(self.max_epochs):
            #shuffle training data at the beginning of each epoch
            permutation = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            #batch processing loop:iters through shuffled training data in mini batches
            for i in range(0, X_train.shape[0], self.batch_size):
                #extract a mini-batch of features (X_batch) and target labels (y_batch)
                X_batch = X_train_shuffled[i: i+ self.batch_size]
                y_batch = y_train_shuffled[i:i + self.batch_size]

                #get the actual number of samples in the current batch
                #might be less than self.batch_size for the last batch
                n_batch = X_batch.shape[0]

                #FORWARD PASS:
                #1. compute the linear combination of inputs and weights, plus bias (logits)
                #   this is the input to the activation function
                logits = X_batch @ self.weights + self.bias

                #2. apply the softmax function to convert logits into probs
                # these probs represent the models predicted class probs
                y_pred_proba = self._softmax(logits)

                #LOSS CALCULATION:
                #1. compute Cross-Entropy Loss
                #calculates the log-likelihood for each sample. 1e-9 is added to prevent log(0)
                #mean is taken over all samples in the batch
                cross_entropy_loss = -np.mean(np.sum(y_batch * np.log(y_pred_proba + 1e-9), axis=1))

                #2.compute L2 regularization loss. penalizes large weights, prevents overfitting
                l2_loss = (self.regularization / 2) * np.sum(self.weights ** 2)

                #3. combine Cross-Entropy + L2 Regularization loss to get total loss
                total_loss = cross_entropy_loss + l2_loss
                self.losses.append(total_loss)

                #BACKWARD PASS (GRAIDENT CALC):
                #1. calc the error(diff between predicted probs and true labels)
                error = y_pred_proba - y_batch

                #2. calc the gradient of the loss with respect to the weights
                grad_weights = (1 / n_batch) * X_batch.T @ error + self.regularization * self.weights

                #3. calc the gradient of loss w respect to bias
                grad_bias = (1 / n_batch) * np.sum(error, axis=0)

                #param update(weight updates)
                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias

                #Early stopping check(at the end of each epoch):
                validation_loss = self.score(X_val, y_val)

            #check if current validation loss is better than the recorded loss
            if validation_loss < best_validation_loss:
                #if better, update the validation loss
                best_validation_loss = validation_loss

                #store weights/biases as the best so far
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()

                patience_counter = 0
            else:
                patience_counter += 1
            
            #if patience counter reaches 5, stop training early
            if patience_counter > self.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break #exit the main training loop
                
        #restore the best weights and biases found
        self.weights = best_weights
        self.bias = best_bias


    def predict_proba(self, X):
        logits = X @ self.weights + self.bias
           
        # Apply the softmax function to convert the logits into probabilities.
        # This returns an array where each row sums to 1, representing the
        # probability distribution over classes for each sample.
        return self._softmax(logits)


    def predict(self, X):
        # Get the predicted probabilities for each class using predict_proba.
        probabilities = self.predict_proba(X)
             
        # For each sample, find the index of the class with the highest probability.
        # This index corresponds to the predicted class label.
        predicted_classes_indices = np.argmax(probabilities, axis=1)
            
        # Map the indices back to the original class labels if self.classes was set.
        # If self.classes is None (e.g., for binary classification where y was 0 or 1),
        # then the indices themselves are the labels.
        if hasattr(self, 'classes') and self.classes is not None:
            return self.classes[predicted_classes_indices]
        else:
            # This case might happen if y was already one-hot encoded and self.classes wasn't
            #explicitly set.
            # For simplicity, if self.classes isn't available, return the index as the label.
            # In a proper multi-class setup, self.classes should always be populated.
            return predicted_classes_indices

    def score(self, X, y):
        # Ensure y is in the correct format (one-hot encoded) for loss calculation.
        # This mirrors the y_processed logic from the fit method.
        n_samples = X.shape[0]
        if y.ndim == 1:
            # Assuming y contains integer class labels
            if not hasattr(self, 'classes') or self.classes is None:
                raise ValueError("Model not fitted: 'classes' attribute not found. Call fit() first.")
            
            n_outputs = len(self.classes)
            y_one_hot = np.zeros((n_samples, n_outputs))
            for i, label in enumerate(y):
                # Ensure the label is one of the known classes.
                if label in self.classes:
                    y_one_hot[i, np.where(self.classes == label)[0][0]] = 1
                else:
                    # Handle unseen classes in validation if necessary, or raise error.
                    # For now, let's assume all validation labels are in training classes.
                    # A more robust solution might ignore these samples or map them to a default.
                    pass # Or raise an error
            y_processed = y_one_hot
        else:
            y_processed = y

        # Get the predicted probabilities for the given X.
        probabilities = self.predict_proba(X)
        
        # Calculate the Cross-Entropy Loss between true labels (y_processed)
        # and predicted probabilities.
        # Add a small epsilon (1e-9) to probabilities to prevent log(0) errors.
        cross_entropy_loss = -np.mean(np.sum(y_processed * np.log(probabilities + 1e-9), axis=1))
        
        # Return the cross-entropy loss. Lower values are better.
        return cross_entropy_loss

    def save(self, file_path):
        np.savez(file_path, weights=self.weights, bias=self.bias)

    def load(self, file_path):
        data = np.load(file_path)

        self.weights = data['weights']
        self.bias = data['bias']
