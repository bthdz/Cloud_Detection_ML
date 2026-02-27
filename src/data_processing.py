# src/data_processing.py

import pandas as pd
from sklearn.model_selection import train_test_split
import config


def load_and_clean_data(file_path):
    print(f"Đang đọc dữ liệu từ: {file_path}...")
    df = pd.read_csv(file_path)

    X = df[config.FEATURES]
    y = df[config.TARGET]

    # Xử lý giá trị rỗng (như code gốc của bạn)
    if X.isnull().values.any():
        print("Phát hiện giá trị null, đang điền bằng 0...")
        X = X.fillna(0)

    return X, y


def split_data(X, y, test_size=0.3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Kích thước Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test
