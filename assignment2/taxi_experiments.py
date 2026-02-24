import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from NeuralNetwork import Sequential, Linear, ReLU, MSE

# 1. Load Data
dataset = np.load("nyc_taxi_data-1.npy", allow_pickle=True).item()
X_train, y_train = dataset["X_train"], dataset["y_train"]
X_test, y_test = dataset["X_test"], dataset["y_test"]

# 2. Preprocessing
for df in [X_train, X_test]:
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.dayofweek
    df['hour'] = df['pickup_datetime'].dt.hour

features = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
            'dropoff_longitude', 'dropoff_latitude', 'month', 'day', 'hour']

X_train = X_train[features]
X_test = X_test[features]
y_train = np.log1p(y_train)
y_test = np.log1p(y_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

X_train = X_train.values.T
y_train = y_train.values.reshape(1, -1)
X_val = X_val.values.T
y_val = y_val.values.reshape(1, -1)
X_test = X_test.values.T
y_test = y_test.values.reshape(1, -1)

# 3. Experiment Configurations
configs = [
    {"name": "Config_1_64_32_LR0.01", "layers": [64, 32], "lr": 0.01},
    {"name": "Config_2_128_64_LR0.005", "layers": [128, 64], "lr": 0.005},
    {"name": "Config_3_64_64_32_LR0.01", "layers": [64, 64, 32], "lr": 0.01}
]

def run_experiment(config):
    print(f">>> Running: {config['name']}")
    model = Sequential()
    prev_size = X_train.shape[0]
    for size in config['layers']:
        model.add(Linear(prev_size, size))
        model.add(ReLU())
        prev_size = size
    model.add(Linear(prev_size, 1))

    loss_func = MSE()
    lr = config['lr']
    
    best_val_loss = float('inf')
    patience = 3 # As per requirement
    no_improve = 0
    
    train_history = []
    val_history = []

    for epoch in range(100): # Max 100 epochs for experiment
        # Forward/Backward
        y_pred = model.forward(X_train)
        t_loss = loss_func.forward(y_pred, y_train)
        model.backward(loss_func.backward(y_pred, y_train), lr)
        
        # Validation
        y_val_pred = model.forward(X_val)
        v_loss = loss_func.forward(y_val_pred, y_val)
        
        train_history.append(t_loss)
        val_history.append(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            no_improve = 0
            model.save_weights(f"{config['name']}.w")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Plotting
    plt.figure()
    plt.plot(train_history, label='Train Loss')
    plt.plot(val_history, label='Val Loss')
    plt.title(f"Loss for {config['name']}")
    plt.legend()
    plt.savefig(f"{config['name']}_plot.png")
    plt.close()

    # Final Test
    model.load_weights(f"{config['name']}.w")
    test_pred = model.forward(X_test)
    rmsle = np.sqrt(loss_func.forward(test_pred, y_test))
    print(f"Final RMSLE for {config['name']}: {rmsle:.4f}")
    return rmsle

# Run all
results = {}
for c in configs:
    results[c['name']] = run_experiment(c)

print("Summary of Results:")
for name, score in results.items():
    print(f"{name}: {score:.4f}")
