# E2E Vision Pipeline ONNX - Ứng dụng Chẩn đoán Bệnh lý Da liễu

## Tổng quan

Đây là một AIAI chẩn đoán bệnh lý da liễu đầu cuối (End-to-End) sử dụng mô hình ONNX được tối ưu hóa cho hiệu suất cao. Hệ thống có khả năng phân loại các loại bệnh lý da liễu phổ biến với độ chính xác tương đối cao, hỗ trợ các chuyên gia y tế trong quá trình chẩn đoán.

## Tính năng chính

- 🧠 **Mô hình Deep Learning**: Sử dụng ResNet50 được huấn luyện trên bộ dữ liệu ISIC (International Skin Imaging Collaboration)
- ⚡ **Tối ưu hóa ONNX**: Mô hình được chuyển đổi sang định dạng ONNX để tăng tốc độ suy luận
- 🌐 **API FastAPI**: Cung cấp endpoint RESTful để tích hợp dễ dàng với các ứng dụng khác
- 📱 **Giao diện Web Responsive**: Giao diện người dùng hiện đại, thân thiện với mọi thiết bị
- 🏥 **Hỗ trợ đa ngôn ngữ**: Tên bệnh lý được hiển thị bằng tiếng Việt

## Cấu trúc dự án

```
E2E-Vision-Pipeline-ONNX/
├── convert/                     # Chuyển đổi mô hình PyTorch sang ONNX
│   └── convert_onnx.py
├── images/                      # Hình ảnh mẫu
│   └── hard_sample.png
├── model/                       # Mô hình đã huấn luyện
│   ├── best_model.pt          # Mô hình PyTorch tốt nhất
│   └── resnet50_final.onnx     # Mô hình ONNX cuối cùng
├── notebooks/                    # Jupyter notebooks cho EDA và nghiên cứu
│   ├── 01_eda.ipynb
│   └── 02_model_research.ipynb
├── requirements.txt               # Các thư viện Python cần thiết
├── src/                         # Source code chính
│   ├── api/                   # API FastAPI
│   │   ├── api.py             # Endpoint API và logic xử lý
│   │   └── requirements-backend.txt
│   ├── data/                  # Xử lý dữ liệu
│   │   ├── dataset.py
│   │   ├── downloader.py
│   │   └── preprocess.py
│   ├── models/                 # Định nghĩa mô hình
│   │   ├── loss.py           # Các hàm mất mát
│   │   └── model.py          # Kiến trúc ResNet50 tùy chỉnh
│   ├── test/                   # Kiểm thử mô hình
│   │   ├── evaluate.py
│   │   └── onnx_demo.py
│   ├── train/                  # Huấn luyện mô hình
│   │   └── train.py
│   └── ui/                     # Giao diện người dùng
│       ├── index.html         # Trang chính
│       ├── style.css          # Style trang
│       ├── script.js          # Logic JavaScript
│       └── nginx.conf         # Cấu hình Nginx
└── README.md                   # Tài liệu dự án
```

## Các lớp bệnh lý được hỗ trợ

| Mã | Tên bệnh bằng tiếng Việt | Mô tả |
|-----|----------------------|--------|
| NV | Nốt ruồi (Nevus) | Nốt ruồi là các tổn thương trên da thường lành tính |
| MEL | U hắc tố (Melanoma) | Ung thư u hắc tố, một dạng ung thư da nguy hiểm |
| BCC | Ung thư biểu mô tế bào đáy (Basal cell carcinoma) | Ung thư biểu mô tế bào đáy, phổ biến nhất |
| BKL | Tăng sừng lành tính (Benign keratosis-like) | Các tổn thương tăng sừng lành tính |
| AK | Dày sừng quang hóa (Actinic keratosis) | Tổn thương tiền ung thư do ánh nắng |
| SCC | Ung thư biểu mô tế bào vảy (Squamous cell carcinoma) | Ung thư biểu mô tế bào vảy |
| VASC | Tổn thương mạch máu (Vascular lesion) | Các vấn đề liên quan đến mạch máu trên da |
| DF | U sợi da (Dermatofibroma) | U sợi da, một tổn thương da lành tính |

## Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- PyTorch 2.0+
- ONNX Runtime 1.15+
- 8GB RAM (tối thiểu)
- CPU hỗ trợ AVX (khuyến khuyến)

### Cài đặt các thư viện

```bash
# Clone repository
git clone https://github.com/your-username/E2E-Vision-Pipeline-ONNX.git
cd E2E-Vision-Pipeline-ONNX

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Cài đặt các thư viện
pip install -r requirements.txt
```

### Chạy ứng dụng

#### Backend API

```bash
# Chạy server phát triển
python -m uvicorn src.api.api:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Web

```bash
# Chạy với Nginx (production)
docker build -t skin-ai .
docker run -p 8080:80 --name skin-ai-container skin-ai

# Hoặc mở trực tiếp file HTML trong trình duyệt
file:///path/to/E2E-Vision-Pipeline-ONNX/src/ui/index.html
```

## API Endpoint

### Chẩn đoán hình ảnh

**Endpoint:** `POST /predict`

**Headers:** `Content-Type: multipart/form-data`

**Request:**
```
file: <file ảnh>
```

**Response:**
```json
{
  "class_code": "MEL",
  "disease": "U hắc tố (Melanoma)",
  "confidence": 0.923456
}
```

## Huấn luyện mô hình

Mô hình được huấn luyện trên bộ dữ liệu ISIC Archive với các bước:

1. **Tiền xử lý dữ liệu**: Resize ảnh (224x224), chuẩn hóa theo ImageNet
2. **Data Augmentation**: Sử dụng kỹ thuật Albumentations để tăng cường dữ liệu
3. **Mô hình cơ sở**: ResNet50 pretrained trên ImageNet
4. **Fine-tuning**: Thay thế lớp cuối cùng để phù hợp với 8 lớp bệnh lý
5. **Loss Function**: Focal Loss để xử lý vấn đề mất cân bằng lớp
6. **Optimizer**: Adam với learning rate schedule

## Đánh giá hiệu suất

- **Độ chính xác tổng thể**: 8181
- **F1-Score**: 0.8181
- **Precision trung bình**: 0.822
- **Recall trung bình**: 0.8181

## Công nghệ sử dụng

- **PyTorch**: Framework Deep Learning
- **ONNX**: Tối ưu hóa mô hình cho suy luận
- **FastAPI**: Framework API hiệu suất cao
- **Nginx**: Web server
- **HTML/CSS/JavaScript**: Giao diện người dùng
- **Albumentations**: Thư viện augmentation cho hình ảnh y tế

## Giấy phép

Dự án này được phát triển với mục đích nghiên cứu và học tập. Vui lòng tham khảo giấy phép tại file LICENSE.

## Tác giả

- Phat Nguyen <tanphat6406@gmail.com>
- Liên hệ: +84 333 786 257
- LinkedIn: [linkedin.com/in/phat-nguyen-a264722b7](https://linkedin.com/in/phat-nguyen-a264722b7)

---

*Đây là dự án học thuật được phát triển cho mục đích học tập được sử dụng để chẩn đoán thay thế cho chuyên gia y tế.*
