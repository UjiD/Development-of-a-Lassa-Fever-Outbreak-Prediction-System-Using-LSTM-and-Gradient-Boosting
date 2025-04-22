import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import seaborn as sns

# ROC Curve Comparison
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba)
fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test, hybrid_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, y_proba):.2f})')
plt.plot(fpr_hybrid, tpr_hybrid, label=f'Hybrid (AUC = {roc_auc_score(y_test, hybrid_proba):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()

# Feature Importance
xgb.plot_importance(xgb_model, max_num_features=10)
plt.title('XGBoost Feature Importance')
plt.show()

xgb.plot_importance(hybrid_model, max_num_features=10)
plt.title('Hybrid Model Feature Importance')
plt.show()