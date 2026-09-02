import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from models import ConvAutoencoder
from dataset import TemporalDataset

WINDOW_SIZE = 10
BATCH_SIZE = 8192
LR = 0.001
EPOCHS = 50
LATENT_DIM = 24
L1_LAMBDA = 1e-5

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        X_train = np.load("working/X_train.npy")
        X_test_benign = np.load("working/X_test_benign.npy")
        X_test_attack = np.load("working/X_test_attack.npy")
    except FileNotFoundError:
        exit(1)

    train_dataset = TemporalDataset(X_train, window_size=WINDOW_SIZE, stride=5)
    test_benign_dataset = TemporalDataset(
        X_test_benign, window_size=WINDOW_SIZE, stride=1)
    test_attack_dataset = TemporalDataset(
        X_test_attack, window_size=WINDOW_SIZE, stride=1)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    input_dim = X_train.shape[1]
    model = ConvAutoencoder(input_dim)
    model.to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    history = {'loss': []}
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)

            l1_norm = sum(p.abs().sum()
                          for p in model.encoder[-2].parameters())
            total_loss = loss + (L1_LAMBDA * l1_norm)

            total_loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_loss = train_loss / len(train_loader)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "working/best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Huber Loss: {avg_loss:.6f}")

        if epochs_no_improve > 6:  # Faster early stopping
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load("working/best_model.pth"))
    model.eval()

    def evaluate_dataset(dataset):
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        scores = []
        with torch.no_grad():
            for data, _ in loader:
                data = data.to(device)
                recon = model(data)
                error = torch.mean((data - recon) ** 2, dim=[1, 2])
                scores.extend(error.cpu().numpy())
        return scores

    benign_scores = evaluate_dataset(test_benign_dataset)
    attack_scores = evaluate_dataset(test_attack_dataset)

    mse_loss = np.array(benign_scores + attack_scores)
    y_eval = np.array([0]*len(benign_scores) + [1]*len(attack_scores))

    auc_score = roc_auc_score(y_eval, mse_loss)
    print(f"\nAUC-ROC Score: {auc_score:.4f}")

    fpr, tpr, thresholds = roc_curve(y_eval, mse_loss)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    preds = (mse_loss > optimal_threshold).astype(int)
    print(classification_report(y_eval, preds,
          target_names=['Benign', 'Anomaly']))
    print("Confusion Matrix:\n", confusion_matrix(y_eval, preds))

    np.save("working/optimal_threshold.npy", optimal_threshold)

    plt.figure(figsize=(10, 6))
    plt.hist(benign_scores, bins=100, alpha=0.6, label='Benign',
             density=True, log=True, color='blue')
    plt.hist(attack_scores, bins=100, alpha=0.6, label='Attack',
             density=True, log=True, color='red')
    plt.axvline(optimal_threshold, color='k', linestyle='--',
                label=f'Threshold ({optimal_threshold:.4f})')
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Frequency (Log Scale)")
    plt.title(f'Final Performance (AUC={auc_score:.4f})')
    plt.legend()
    plt.savefig("working/error_distribution.png")
