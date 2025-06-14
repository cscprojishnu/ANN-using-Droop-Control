import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Generate synthetic training data
np.random.seed(42)
num_samples = 5000  # Increased dataset size for more complexity
P = np.random.uniform(600, 1400, num_samples)  # Expanded active power range
Q = np.random.uniform(200, 800, num_samples)   # Expanded reactive power range

# Adaptive Droop Coefficients
V_no_ann = 230 - (0.015 + 0.00001 * (P - 1000)) * (P - 1000)  # Dynamic P-f droop
F_no_ann = 50 - (0.005 + 0.00002 * (Q - 500)) * (Q - 500)  # Dynamic Q-V droop

# Introduce nonlinearity to training data
V_train = V_no_ann + np.random.uniform(-2.0, 2.0, num_samples)
F_train = F_no_ann + np.random.uniform(-1.0, 1.0, num_samples)

# Normalize data
scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_Y = MinMaxScaler(feature_range=(-1, 1))

X_train = scaler_X.fit_transform(np.column_stack((P, Q)))
y_train = scaler_Y.fit_transform(np.column_stack((V_train, F_train)))

# Define Complex ANN Model
model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(2,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(256, activation='tanh'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2, activation='linear')
])

model.compile(optimizer='adam', loss='mse')

# Train the ANN Model
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

# Test model with new power variations
P_test = np.array([650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350])
Q_test = np.array([250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950])
X_test = scaler_X.transform(np.column_stack((P_test, Q_test)))

# Predictions using ANN
predictions = scaler_Y.inverse_transform(model.predict(X_test))

# Voltage and Frequency using optimized traditional droop control
V_no_ann_test = 230 - (0.015 + 0.00001 * (P_test - 1000)) * (P_test - 1000)
F_no_ann_test = 50 - (0.005 + 0.00002 * (Q_test - 500)) * (Q_test - 500)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(P_test, V_no_ann_test, 'r--o', label='Voltage without ANN', markersize=8)
plt.plot(P_test, predictions[:, 0], 'b-s', label='Voltage with ANN', markersize=8)
plt.xlabel('Active Power (W)')
plt.ylabel('Voltage (V)')
plt.title('Voltage Response: ANN vs. Optimized Droop')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(Q_test, F_no_ann_test, 'r--o', label='Frequency without ANN', markersize=8)
plt.plot(Q_test, predictions[:, 1], 'b-s', label='Frequency with ANN', markersize=8)
plt.xlabel('Reactive Power (Var)')
plt.ylabel('Frequency (Hz)')
plt.title('Frequency Response: ANN vs. Optimized Droop')
plt.legend()
plt.grid()
plt.show()

# Print results
print("Predicted Voltage and Frequency with ANN:")
for i in range(len(P_test)):
    print(f"P: {P_test[i]}W, Q: {Q_test[i]}Var -> V: {predictions[i][0]:.2f}V, F: {predictions[i][1]:.2f}Hz")