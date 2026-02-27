# src/predict.py

import os
import numpy as np
import rasterio
import joblib
import config


def predict_cloud_on_tif(
    input_tif_path, model_name="RandomForest", output_tif_name="cloud_mask_result.tif"
):
    """
    Đọc ảnh vệ tinh (.tif), dùng mô hình đã train dự đoán mây cho từng pixel,
    và xuất ra một ảnh .tif mới chứa nhãn dự đoán.
    """
    print(f"\n{'='*40}")
    print(f"ĐANG TIẾN HÀNH DỰ ĐOÁN MÂY TRÊN ẢNH VỆ TINH")
    print(f"{'='*40}")

    # 1. Tải mô hình
    model_path = os.path.join(config.MODEL_DIR, f"{model_name}_best_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy mô hình tại {model_path}!")

    print(f"Đang tải mô hình {model_name}...")
    model = joblib.load(model_path)

    # 2. Mở và đọc file ảnh GeoTIFF
    print(f"Đang đọc ảnh đầu vào: {input_tif_path}")
    with rasterio.open(input_tif_path) as src:
        meta = src.meta  # Lưu lại siêu dữ liệu (Hệ tọa độ, kích thước, độ phân giải...)
        img_data = src.read()  # Định dạng: (Số kênh màu, Chiều cao, Chiều rộng)

        channels, height, width = img_data.shape
        print(f"Kích thước ảnh: {height}x{width} pixels, Số kênh: {channels}")

        # LƯU Ý QUAN TRỌNG:
        # Số lượng kênh (channels) trong file .tif MỚI này phải đúng bằng số features lúc train.
        # Và thứ tự các kênh trong ảnh phải khớp với thứ tự trong config.FEATURES.
        if channels != len(config.FEATURES):
            print(
                f"Cảnh báo: Ảnh có {channels} kênh, nhưng mô hình cần {len(config.FEATURES)} features!"
            )

        # 3. Chuyển đổi ma trận 3D thành 2D để đưa vào mô hình Scikit-Learn/XGBoost
        # Từ (channels, height, width) -> (height * width, channels)
        img_data_2d = np.transpose(
            img_data, (1, 2, 0)
        )  # Chuyển thành (height, width, channels)
        pixels_features = img_data_2d.reshape(
            -1, channels
        )  # Trải phẳng thành danh sách các pixel

        # Xử lý các giá trị NaN, Null hoặc NoData (thường là các vùng viền đen của ảnh vệ tinh)
        pixels_features = np.nan_to_num(pixels_features, nan=0.0)

        # 4. Chạy dự đoán
        print("Đang phân tích từng pixel (Có thể mất vài phút với ảnh lớn)...")
        predictions_1d = model.predict(pixels_features)

        # 5. Khôi phục lại hình dáng bức ảnh ban đầu
        # Từ mảng 1D (height * width) -> Mảng 2D (height, width)
        cloud_mask = predictions_1d.reshape(height, width)

        # 6. Cập nhật siêu dữ liệu để lưu file mới
        # File xuất ra chỉ có 1 kênh (lớp nhãn 0 và 1) và kiểu dữ liệu là số nguyên (uint8)
        meta.update(
            {
                "count": 1,
                "dtype": rasterio.uint8,
                "nodata": None,  # Hoặc gán bằng một giá trị nếu có mask NoData riêng
            }
        )

        # Tạo thư mục outputs nếu chưa có
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(config.OUTPUT_DIR, output_tif_name)

        # 7. Lưu file ảnh Mask
        print(f"Đang xuất ảnh kết quả ra file TIF...")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(cloud_mask.astype(rasterio.uint8), 1)

    print(f"HOÀN THÀNH! Ảnh Cloud Mask đã được lưu tại: {output_path}")


if __name__ == "__main__":
    TEST_TIF_PATH = "../data/Sentinel1_2_Stacked_Image.tif"

    # Bỏ comment dòng dưới để chạy nếu bạn đã có file TIF chuẩn trong thư mục data/
    predict_cloud_on_tif(
        input_tif_path=TEST_TIF_PATH,
        model_name="RandomForest",
        output_tif_name="Hanoi_Cloud_Mask.tif",
    )
