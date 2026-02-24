import numpy as np
from NeuralNetwork import Sequential, Linear, Sigmoid, Tanh, BinaryCrossEntropyLoss

#XOR inputs(4, samples, 2 features each)
#We transpose (.T) to make them (2,4) shapes: 2 features, 4 samples
X = np.array([[0,0], [0,1], [1,0], [1,1]]).T

#expected output (1 output per sample)
y = np.array([[0,1,1,0]])


#1. Define the model
model = Sequential()
model.add(Linear(2,2)) #input(2) -> Hidden(2)
model.add(Sigmoid()) #activation
model.add(Linear(2,1)) #hidden(2) -> output(1)
model.add(Sigmoid()) #final activation (correct for 0-1 probability)

#2. define loss and hyperparameters
loss_func = BinaryCrossEntropyLoss()
learning_rate = 0.1
epochs = 10000 #XOR can take a while to converge

#training loop
for i in range(epochs):
    #forward pass
    output = model.forward(X)

    #calculate loss(for monitoring)
    loss = loss_func.forward(output, y)

    #backward pass(learning step)
    grad = loss_func.backward(output, y)
    model.backward(grad, learning_rate)

    if i % 1000 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")

#final test
print("\nFinal Predictions:")
print(model.forward(X))

#save the solved weights
model.save_weights("XOR_solved.w")
print("\nWeights saved to XOR_solved.w")


