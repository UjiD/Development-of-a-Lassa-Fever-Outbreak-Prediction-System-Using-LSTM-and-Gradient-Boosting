def predict_future_outbreaks(model, initial_data, steps, n_steps=12):
    predictions = []
    current_batch = initial_data[-n_steps:].reshape((1, n_steps, initial_data.shape[1]))
    
    for i in range(steps):
        current_pred = model.predict(current_batch)[0]
        predictions.append(current_pred)
        
        # Update batch
        current_batch = np.append(current_batch[:,1:,:], [[current_batch[0,-1,:]]], axis=1)
        current_batch[0,-1,0] = current_pred  # Update the outbreak cases prediction
        
    return predictions

# Get last n_steps from training data
last_sequence = scaled_data[-n_steps:]

# Predict next 6 months
future_predictions = predict_future_outbreaks(lstm_model, scaled_data, steps=6)

# Inverse transform predictions
future_predictions = scaler.inverse_transform(
    np.concatenate([
        np.array(future_predictions).reshape(-1, 1),
        np.zeros((len(future_predictions), scaled_data.shape[1]-1))
    ], axis=1)
)[:, 0]

print("\nFuture Outbreak Predictions:")
for i, pred in enumerate(future_predictions, 1):
    print(f"Month {i}: {pred:.0f} predicted cases")