import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import joblib
from models import Autoencoder

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using {device}')
    df = pd.read_csv('working/processed_data.csv')
    benign_df = df[df['Label'] == 'BENIGN'].drop(columns=['Label'])
    anomalous_df = df[df['Label'] != 'BENIGN'].drop(columns=['Label'])

    X_benign_np = benign_df.values.astype('float32')
    X_anomalies_np = anomalous_df.values.astype('float32')

    X_train_benign_subset, X_test_benign_subset = train_test_split(X_benign_np, test_size=0.2, random_state=42)

    scaler = joblib.load("working/min_max_scaler.pkl")
    X_train_benign_scaled = scaler.transform(X_train_benign_subset)
    X_test_benign_scaled = scaler.transform(X_test_benign_subset)
    X_test_anomalies_scaled = scaler.transform(X_anomalies_np)

    X_train_autoencoder_input = X_train_benign_scaled
    X_test_benign_autoencoder_input = X_test_benign_scaled
    X_test_anomalies_autoencoder_input = X_test_anomalies_scaled

    # Prepare combined test set for evaluation (0.2 benign and all anomalous)
    X_test_eval_np = np.concatenate((X_test_benign_autoencoder_input, X_test_anomalies_autoencoder_input), axis=0)

    y_test_benign_labels = np.zeros(len(X_test_benign_autoencoder_input))
    y_test_anomaly_labels = np.ones(len(X_test_anomalies_autoencoder_input))
    y_test_eval = np.concatenate((y_test_benign_labels, y_test_anomaly_labels), axis=0)

    X_train_benign_torch = torch.tensor(X_train_autoencoder_input, dtype=torch.float32).to(device)
    X_test_eval_torch = torch.tensor(X_test_eval_np, dtype=torch.float32).to(device)
    y_test_eval_torch = torch.tensor(y_test_eval, dtype=torch.long).to(device)

    input_dim = X_train_benign_torch.shape[1]
    model = Autoencoder(input_dim)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    EPOCHS = 60
    print(f"\nStarting training")
    for epoch in range(EPOCHS):
        model.train()

        outputs = model(X_train_benign_torch)
        loss = criterion(outputs, X_train_benign_torch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.4f}")

        scheduler.step(loss)

    model.eval()
    with torch.no_grad():
        reconstructed_data = model(X_test_eval_torch)
        anomaly_scores = torch.mean((X_test_eval_torch - reconstructed_data) ** 2, dim=1).cpu().numpy()

    X_test_benign_torch_eval = torch.tensor(X_test_benign_autoencoder_input, dtype=torch.float32).to(device)
    benign_reconstruction_errors = torch.mean(
        (X_test_benign_torch_eval - model(X_test_benign_torch_eval)).detach() ** 2, dim=1).cpu().numpy()

    threshold = np.percentile(benign_reconstruction_errors, 85)
    predicted_labels = (anomaly_scores > threshold).astype(int)

    print(f"\nThreshold for anomaly detection (from {np.percentile(benign_reconstruction_errors, 85):.1f}th percentile of benign errors): {threshold:.4f}")
    print("\n✅ Anomaly Detection Report (0: Benign, 1: Anomaly):")
    print(classification_report(y_test_eval, predicted_labels, target_names=['Benign', 'Anomaly']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_eval, predicted_labels))

    if len(np.unique(y_test_eval)) > 1:
        auc_roc = roc_auc_score(y_test_eval, anomaly_scores)
        print(f"\nAUC-ROC Score: {auc_roc:.4f}")

        fpr, tpr, thresholds_roc_curve = roc_curve(y_test_eval, anomaly_scores)

        # Find optimal threshold using (TPR-FPR)
        optimal_idx = np.argmax(tpr - fpr)
        new_optimal_threshold = thresholds_roc_curve[optimal_idx]

        print(f"\nOptimal threshold (maximizing TPR-FPR): {new_optimal_threshold:.4f}")
        predicted_labels_optimal = (anomaly_scores > new_optimal_threshold).astype(int) # Use a new variable for labels
        print("\n✅ Anomaly Detection Report with Optimized Threshold (0: Benign, 1: Anomaly):")
        print(classification_report(y_test_eval, predicted_labels_optimal, target_names=['Benign', 'Anomaly']))
        print("Confusion Matrix with Optimized Threshold:")
        print(confusion_matrix(y_test_eval, predicted_labels_optimal))

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_roc:.4f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Guessing')
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR - Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.grid(True)

        idx_initial_on_plot = np.argmin(np.abs(thresholds_roc_curve - threshold))
        plt.scatter(fpr[idx_initial_on_plot], tpr[idx_initial_on_plot], marker='x', color='purple', s=100,
                    label=f'Initial Threshold ({threshold:.4f})')

        idx_optimal_on_plot = np.argmin(np.abs(thresholds_roc_curve - new_optimal_threshold))
        plt.scatter(fpr[idx_optimal_on_plot], tpr[idx_optimal_on_plot], marker='o', color='green', s=100,
                    label=f'Optimal Operating Point ({new_optimal_threshold:.4f})')

        plt.legend(loc='lower right')
        plt.savefig("working/roc_curve_optimized.png")
        print(f"✅ Optimized ROC curve saved to: working/roc_curve_optimized.png")
        plt.show()

        np.save("working/optimal_threshold.npy", new_optimal_threshold)
        print(f"✅ Optimal threshold saved to: working/optimal_threshold.npy")

    torch.save(model.state_dict(), "working/anomaly_detector_model.pth")
    print(f"✅ Anomaly detector model saved to: working/anomaly_detector_model.pth")