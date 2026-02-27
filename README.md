# Satellite Cloud Detection with Machine Learning

## Tổng quan (Overview)

Dự án ứng dụng Machine Learning để nhận diện và tạo mặt nạ mây (Cloud Mask) từ dữ liệu vệ tinh đa quang phổ (Sentinel-2) và radar (Sentinel-1). Hỗ trợ phân tích viễn thám chính xác hơn bằng cách tự động loại bỏ nhiễu do mây che phủ trên ảnh.

## Dữ liệu & Hiệu suất (Data & Performance)

- **Khu vực nghiên cứu:** Hà Nội (68.000 mẫu pixels).
- **Đặc trưng (14 features):** \* Quang học: `B2, B3, B4, B8, B8A, B11, B12`
  - Chỉ số: `NDVI, NDWI, NDSI, BRIGHT`
  - Radar: `VV, VH, VV_minus_VH`
- **Hiệu suất (Best Model - Random Forest):** Đạt độ chính xác (Accuracy) **95.21%**, F1-Score **0.95**. Kênh B2 (Blue) và chỉ số NDVI đóng vai trò quan trọng nhất trong quyết định phân loại.

## Cấu trúc dự án (Project Structure)

```text
Satellite-Cloud-Detection/
├── data/                   # Chứa dữ liệu train (.csv) và ảnh cần dự đoán (.tif)
├── models/                 # Chứa các mô hình đã huấn luyện (.pkl)
├── outputs/                # Chứa kết quả (Cloud mask .tif, báo cáo CSV)
├── notebooks/              # Chứa các file Jupyter Notebook phân tích EDA
├── src/
│   ├── config.py           # Thiết lập đường dẫn và siêu tham số
│   ├── data_processing.py  # Tiền xử lý và làm sạch dữ liệu
│   ├── train.py            # Huấn luyện mô hình (RF, XGBoost, SVM)
│   └── predict.py          # Dự đoán lớp phủ mây trên ảnh GeoTIFF
├── requirements.txt        # Danh sách thư viện cần cài đặt
└── README.md               # Tài liệu dự án
```

## Cài đặt & Sử dụng (Installation & Usage)

### 1. Cài đặt môi trường

Đảm bảo bạn đã cài đặt Python 3.8+. Clone repository và cài đặt các thư viện cần thiết:

```bash
git clone https://github.com/your-username/Satellite-Cloud-Detection.git
cd Satellite-Cloud-Detection
pip install -r requirements.txt
```

### 2. Huấn luyện mô hình (Training)

Lệnh dưới đây sẽ tự động huấn luyện các mô hình, tinh chỉnh tham số và lưu mô hình tốt nhất vào thư mục `models/`.

```bash
python src/train.py
```

### 3. Dự đoán mây trên ảnh vệ tinh (.tif)

Để dự đoán mặt nạ mây cho một bức ảnh Sentinel mới:

1. Đặt ảnh `.tif` (đã stack đủ 14 bands) vào thư mục `data/`.
2. Cập nhật tên file trong `src/predict.py`.
3. Chạy lệnh:

```bash
python src/predict.py
```
