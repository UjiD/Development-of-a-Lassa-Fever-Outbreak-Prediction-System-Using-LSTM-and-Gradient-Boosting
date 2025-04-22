from sklearn.metrics import classification_report

print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred))

print("\nHybrid Model Classification Report:")
print(classification_report(y_test, hybrid_pred))

# Confusion Matrix
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, hybrid_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Hybrid Model Confusion Matrix')

plt.tight_layout()
plt.show()