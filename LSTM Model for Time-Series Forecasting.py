from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps, 0])  # Assuming outbreak cases are in column 0
    return np.array(X), np.array(y)

# Prepare LSTM data
n_steps = 12  # Using 12 months of historical data
X, y = create_sequences(scaled_data, n_steps)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM model
lstm_model = Sequential([
    LSTM(100, activation='relu', input_shape=(n_steps, X.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = lstm_model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
lstm_loss, lstm_mae = lstm_model.evaluate(X_test, y_test)
print(f"LSTM Test MAE: {lstm_mae:.4f}")