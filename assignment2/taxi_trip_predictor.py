import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from NeuralNetwork import Sequential, Linear, ReLU, MSE

#load the dataset
#the .item() is used because the .npy file contains a pickled dictionary
dataset = np.load("nyc_taxi_data-1.npy", allow_pickle=True).item()

#extract the data splits
X_train, y_train = dataset["X_train"], dataset["y_train"]
X_test, y_test = dataset["X_test"], dataset["y_test"]

#---Preprocessing and feature engineering---

#1. feature creation from datetime
for df in [X_train, X_test]:
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.dayofweek #monday=0, Sunday=6
    df['hour'] = df['pickup_datetime'].dt.hour

#2.feature selection
features = [
    'vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
    'dropoff_longitude', 'dropoff_latitude', 'month', 'day', 'hour'
]

X_train = X_train[features]
X_test = X_test[features]

#3.target transformation(log)
y_train = np.log1p(y_train) #np.log1p(x) is = to np.log(x + 1)
y_test = np.log1p(y_test)

#4. create a validatino set from the training data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = 42) #10% for validation

#normalization 
#we calculate the mean and std only from the training set
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

#apply the normalization to all sets
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

#6. convert to numpy arrays for the network
X_train = X_train.values.T
y_train = y_train.values.reshape(1, -1)

X_val = X_val.values.T
y_val = y_val.values.reshape(1, -1)

X_test = X_test.values.T
y_test = y_test.values.reshape(1, -1)

#---Final Data Check---
print("---Processed Data Shapes---")
print("X_train Shape: ", X_train.shape)
print("y_train Shape: ", y_train.shape)
print("X_val Shape: ", X_val.shape)
print("y_val Shape: ", y_val.shape)
print("\n---Sample Processed Data---")
print("Frist trainig Sample (X_train[:, 0]):]\n", X_train[:, 0])
print("Frist training target (y_train[0, 0]):]", y_train[0, 0])






#---NEURAL NETWORK MODEL TRAINING---
#1. define model architecture
#based on assignment(3 layers, ReLU)
input_features = X_train.shape[0] #should be 9
model = Sequential()
model.add(Linear(input_features, 64)) #input layer (9 features) to 64 neurons
model.add(ReLU())
model.add(Linear(64, 32)) #hidden layer to 32 neurons
model.add(ReLU())
model.add(Linear(32, 1)) #output layer (1 neuron for log duration prediction)
#no activation on the final layer for regression, want raw continuous output

#2. hyperparameters
loss_func = MSE()
learning_rate = 0.01
epochs = 1000

#3. early stopping parameters
best_val_loss = float('inf')
patience = 10 #stop after this many epochs without improvement
epochs_no_improve = 0
best_epoch = 0

#store loss history for plotting
train_losses = []
val_losses = []

print("\n---Training Model---")
for epoch in range(epochs):
    #forward pass- trainnig data
    y_pred_train = model.forward(X_train)
    train_loss = loss_func.forward(y_pred_train, y_train)

    #backward pass - update weights
    grad = loss_func.backward(y_pred_train, y_train)
    model.backward(grad, learning_rate)

    #forward pass - validation data(no grad calculation, just evaluate)
    y_pred_val = model.forward(X_val)
    val_loss = loss_func.forward(y_pred_val, y_val)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch + 1} / {epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    #early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_epoch = epoch + 1
        #save the current model weights to a file
        model.save_weights("best_taxi_model.w")
        print(f"--> New best validation loss ({best_val_loss:.6f}). Weights saved.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
            break

print(f"\nloading best model weights from epoch {best_epoch}...")
model.load_weights("best_taxi_model.w")
#Load the best weights before final evaluation
print("\n--- Evaluation on Test Set --- ")
y_pred_test = model.forward(X_test)
test_loss = loss_func.forward(y_pred_test, y_test) #MSE on log_transformed data
print(f"Final Test Loss (MSE on log-transformed data): {test_loss:.6f}")

#calculate RMSLE - which is RMSE on our log-transformed values
rmsle = np.sqrt(test_loss) #RMSE is sqrt of MSE
print(f"Final Test RMSLE: {rmsle:.6f}")

benchmark = 0.513
if rmsle < benchmark:
    print(f" ---> Beat Benchmark of {benchmark} RMSLE!")
else:
    print(f" ---> Did not beat benchmark of {benchmark} RMSLE. Current RMSLE: {rmsle:.6f} ")

print("\n--- Training Complete ---")
print(f"Best validation loss achieved: {best_val_loss:.6f} at epoch {best_epoch}")
print("You can now plot 'train_losses' and 'val_losses' to visualize training progress.")
