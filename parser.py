import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data():
    """
    takes all the .csv files in datasets/ and prunes nas and combines them into 1 big dataframe
    :return: pandas dataframe
    """
    df_list = []

    for file in os.listdir("datasets"):
        if file.endswith(".csv"):
            path = os.path.join("datasets", file)
            df_res = pd.read_csv(path)
            df_res.columns = df_res.columns.str.strip()
            df_res.dropna(how='all', inplace=True)
            df_res['Label'] = df_res['Label'].astype(str).str.strip()
            df_list.append(df_res)

    df_res = pd.concat(df_list, ignore_index=True)
    return df_res

df = load_data()

# remove these columns
non_feature_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Label']
drop_cols = [col for col in non_feature_cols if col in df.columns]
feature_df = df.drop(columns=drop_cols, errors='ignore')

# change values to int
feature_df = feature_df.apply(pd.to_numeric, errors='coerce')
feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
feature_df.dropna(inplace=True)

# Align label DataFrame with cleaned rows
df = df.loc[feature_df.index]

scaler = MinMaxScaler()
scaler.fit(feature_df.values)
feature_df_scaled = pd.DataFrame(scaler.transform(feature_df.values), columns=feature_df.columns)

# Add label column back, dropped from previous changes
feature_df_scaled['Label'] = df['Label'].values

output_path = os.path.join("working", "processed_data.csv")
feature_df_scaled.to_csv(output_path, index=False)
print(f"✅ Processed data saved to: {output_path}")

joblib.dump(scaler, "working/min_max_scaler.pkl")
print(f"✅ MinMaxScaler saved to: working/min_max_scaler.pkl")
