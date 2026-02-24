import numpy as np
import pickle

class Layer:
    def __init__(self):
        #store input/output for use during backpropagation
        self.input = None
        self.output = None

    def forward(self, input_data):
        #must be overridden by subclasses
        raise NotImplementedError
    
    def backward(self, output_gradient, learning_rate):
        #must be overriden by subclasses
        raise NotImplementedError
    
class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        #initialize weights with small random numbers
        #np.sqrt(2/input_size) is the initialization, ideal for ReLU
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2./ input_size)
        self.bias = np.zeros((output_size, 1))
    
    def forward(self, input_data):
        self.input = input_data
        #standard matric multiplication: f(x) = Wx + b
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        #dL/dW = (dL/dY) * X.T
        weights_gradient = np.dot(output_gradient, self.input.T)
        #dL/dX = W.T * (dL/dY)
        input_gradient = np.dot(self.weights.T, output_gradient)

        #update weights and bias using Gradient Descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * np.sum(output_gradient, axis=1, keepdims=True)
        
        return input_gradient
    
class Sigmoid(Layer):
    def forward(self, input_data):
        self.output = 1/(1 + np.exp(-input_data))
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        #chain rule: gradient from next layer * derivative of sigmoid
        return output_gradient * (self.output * (1 - self.output))
    
class ReLU(Layer):
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)
    
    def backward(self, output_gradient, learning_rate):
        #gradient is 1 for positive numbers, 0 for negative numbers
        grad = output_gradient.copy()
        grad[self.input <= 0] = 0
        return grad

class BinaryCrossEntropyLoss:
    def forward(self, y_pred, y_true):
        #clip values to prevent log(0) errors
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))
    

class Sequential:
    def __init__(self, layers=None):
        #a list to store our layers in order
        self.layers = layers if layers is not None else []

    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, input_data):
        #assembly line, pass the output of one layer as the input to the next
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, grad, learning_rate):
        #propagate the gradient backwards through all layers
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return grad
    
    def save_weights(self, filepath):
        #collect weights and biases from Linear Layer
        data = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                data.append({'w': layer.weights, 'b': layer.bias})
            else:
                #activation layers don't have weights to save
                data.append(None)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_weights(self, filepath):
        #read the file and put the weights back into the Linear layers
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        for i, layer in enumerate(self.layers):
            if data[i] is not None and isinstance(layer, Linear):
                layer.weights = data[i]['w']
                layer.bias = data[i]['b']

class Tanh(Layer):
    def forward(self, input_data):
        #tanh maps input to a range between -1 and 1
        self.output = np.tanh(input_data)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        #derivative of tanh is 1- output^2
        return output_gradient * (1 - self.output ** 2)
    
class MSE:
    def forward(self, y_pred, y_true):
        #calculate the MSE
        return np.mean(np.power(y_pred - y_true, 2))
    
    def backward(self, y_pred, y_true):
        #the derivative of MSE if 2 * (y_pred - y_true) / n
        #this is the starting gradient for backpropagation
        return 2 * (y_pred - y_true) / y_true.size