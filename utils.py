import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_iemocap(csv_path="data/iemocap_labels.csv"):
    df = pd.read_csv(csv_path)
    print("Columns found:", df.columns.tolist())
    print("Shape:", df.shape)
    print("Emotion distribution:\n", df['emotion'].value_counts())  # adjust col name if needed

    # Encode labels
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['emotion'])  # change 'emotion' to your actual column name

    # Separate features (all numeric columns except label/emotion cols)
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in ['label']]

    X = df[feature_cols].values
    y = df['label'].values

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, le, df