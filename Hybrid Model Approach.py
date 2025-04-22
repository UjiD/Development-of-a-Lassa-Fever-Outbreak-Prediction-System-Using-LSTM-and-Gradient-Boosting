# Use LSTM predictions as feature for XGBoost
lstm_predictions = lstm_model.predict(X_test)

# Create hybrid dataset
X_hybrid = X_test.copy()
X_hybrid['lstm_pred'] = lstm_predictions.flatten()

# Retrain XGBoost with LSTM features
hybrid_model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=10,
    eval_metric=['auc', 'logloss']
)

hybrid_model.fit(
    X_hybrid, y_test,
    eval_set=[(X_hybrid, y_test)],
    verbose=True
)

# Evaluate hybrid model
hybrid_pred = hybrid_model.predict(X_hybrid)
hybrid_proba = hybrid_model.predict_proba(X_hybrid)[:, 1]

print("\nHybrid Model Performance:")
print(f"Accuracy: {accuracy_score(y_test, hybrid_pred):.4f}")
print(f"Precision: {precision_score(y_test, hybrid_pred):.4f}")
print(f"Recall: {recall_score(y_test, hybrid_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, hybrid_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, hybrid_proba):.4f}")