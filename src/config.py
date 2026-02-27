# src/config.py

DATA_PATH = "../data/S1S2_cloud_training_samples_Ha_Noi_City.csv"
MODEL_DIR = "../models/"  # Đổi thành thư mục để lưu nhiều model
OUTPUT_DIR = "../outputs/"

FEATURES = [
    "B2",
    "B3",
    "B4",
    "B8",
    "B8A",
    "B11",
    "B12",
    "NDVI",
    "NDWI",
    "NDSI",
    "BRIGHT",
    "VV",
    "VH",
    "VV_minus_VH",
]
TARGET = "label"

# Tham số cho Random Forest
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
}

# Tham số cho XGBoost
XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [3, 5, 7, 10],
    "subsample": [0.8, 1.0],
}

# Tham số cho SVM
SVM_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.1, 0.01],
    "kernel": ["rbf", "linear"],
}
