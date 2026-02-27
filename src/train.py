# src/train.py

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import config
from data_processing import load_and_clean_data, split_data


def train_and_evaluate_models():
    # 1. Chuẩn bị dữ liệu
    X, y = load_and_clean_data(config.DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Đảm bảo thư mục lưu model tồn tại
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 2. Định nghĩa danh sách các mô hình
    models_to_train = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": config.RF_PARAM_GRID,
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric="logloss"
            ),
            "params": config.XGB_PARAM_GRID,
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": config.SVM_PARAM_GRID,
        },
    }

    results = []

    # 3. Vòng lặp huấn luyện từng mô hình
    for model_name, mp in models_to_train.items():
        print(f"\n{'='*40}")
        print(f"Đang huấn luyện và Tuning: {model_name}...")
        print(f"{'='*40}")

        # Chạy RandomizedSearchCV
        search = RandomizedSearchCV(
            estimator=mp["model"],
            param_distributions=mp["params"],
            n_iter=10,
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        print(f"Tham số tốt nhất cho {model_name}: {search.best_params_}")

        # Đánh giá trên tập test
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Độ chính xác (Accuracy): {acc:.4f}")
        print(classification_report(y_test, y_pred))

        # Lưu kết quả để so sánh
        results.append({"Model": model_name, "Accuracy": acc})

        # Lưu mô hình ra file .pkl
        model_path = os.path.join(config.MODEL_DIR, f"{model_name}_best_model.pkl")
        joblib.dump(best_model, model_path)
        print(f"Đã lưu mô hình tại: {model_path}")

    # 4. In bảng so sánh kết quả cuối cùng
    print("\n🏆 TỔNG KẾT SO SÁNH CÁC MÔ HÌNH 🏆")
    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
    print(results_df.to_string(index=False))

    # Lưu bảng so sánh ra CSV
    results_df.to_csv(
        os.path.join(config.OUTPUT_DIR, "model_comparison.csv"), index=False
    )


if __name__ == "__main__":
    train_and_evaluate_models()
