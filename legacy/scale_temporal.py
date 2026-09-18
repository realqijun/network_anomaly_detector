import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import joblib

WINDOW_SIZE = 10  # Look at 10 packets at a time

def scale_temporal_data():
    try:
        df = pd.read_parquet("working/processed_data.parquet")
    except:
        df = pd.read_csv("working/processed_data.csv")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    df = df.loc[:, (df != df.iloc[0]).any()]

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].min() >= 0:
            df[col] = np.log1p(df[col])

    benign_df = df[df['Label'] == 'BENIGN']
    attack_df = df[df['Label'] != 'BENIGN']

    X_benign = benign_df.drop('Label', axis=1).values
    X_attack = attack_df.drop('Label', axis=1).values

    train_size = int(len(X_benign) * 0.8)
    X_train_benign = X_benign[:train_size]
    X_test_benign = X_benign[train_size:]

    scaler_pipeline = Pipeline([
        ('robust', RobustScaler(quantile_range=(5, 95))),
        ('minmax', MinMaxScaler())
    ])

    X_train_scaled = scaler_pipeline.fit_transform(X_train_benign)
    X_test_benign_scaled = scaler_pipeline.transform(X_test_benign)
    X_attack_scaled = scaler_pipeline.transform(X_attack)

    np.save("working/X_train.npy", X_train_scaled.astype(np.float32))
    np.save("working/X_test_benign.npy", X_test_benign_scaled.astype(np.float32))
    np.save("working/X_test_attack.npy", X_attack_scaled.astype(np.float32))
    joblib.dump(scaler_pipeline, "working/scaler.pkl")


if __name__ == "__main__":
    scale_temporal_data()
