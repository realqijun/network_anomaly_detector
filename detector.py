import joblib
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from models import Autoencoder

df = pd.read_csv("working/processed_data.csv")

X = df.drop(columns=['Label'])
y_raw = df['Label']
y_true = (y_raw != 'BENIGN').astype(int).values

scaler = joblib.load("working/min_max_scaler.pkl")
X_scaled = scaler.transform(X.values)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

input_dim = X_tensor.shape[1]
model = Autoencoder(input_dim)
model.load_state_dict(torch.load("working/anomaly_detector_model.pth"))
model.eval()

# Get reconstruction error
with torch.no_grad():
    reconstructed = model(X_tensor)
    mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()

# --- NEW: Calculate optimal threshold on the full dataset ---
if len(np.unique(y_true)) > 1: # Ensure both classes are present for ROC calculation
    fpr, tpr, thresholds_roc_curve = roc_curve(y_true, mse)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold_for_full_data = thresholds_roc_curve[optimal_idx]
    print(f"\nCalculated Optimal Threshold for Full Data (maximizing TPR-FPR): {optimal_threshold_for_full_data:.4f}")
else:
    optimal_threshold_for_full_data = 0.0094
    print("Warning: Only one class found in y_true. Cannot calculate optimal threshold from ROC curve.")

# threshold = np.load("working/optimal_threshold.npy").item()
threshold = optimal_threshold_for_full_data

y_pred = (mse > threshold).astype(int)

print(f"\n--- Evaluation on Full Processed Dataset (using threshold: {threshold:.4f}) ---")
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4, target_names=['Benign', 'Anomaly']))

if len(np.unique(y_true)) > 1:
    roc_auc = roc_auc_score(y_true, mse)
    print("ROC-AUC Score:", roc_auc)
else:
    print("ROC-AUC Score cannot be calculated: Only one class present in y_true.")

plt.figure(figsize=(10, 6))
plt.hist(mse, bins=100, alpha=0.7, label='All Samples MSE')
plt.hist(mse[y_true == 0], bins=100, alpha=0.6, label='Benign MSE', color='skyblue')
plt.hist(mse[y_true == 1], bins=100, alpha=0.6, label='Anomaly MSE', color='salmon')

plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
plt.title("Reconstruction Error Distribution and Threshold (Full Dataset)")
plt.xlabel("Mean Squared Error (MSE)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()