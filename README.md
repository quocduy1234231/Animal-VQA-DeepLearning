# Visual Question Answering (VQA) trên Tập dữ liệu Động vật

Đồ án cuối kỳ môn **Học Sâu (Deep Learning)** – Trường Đại học Tôn Đức Thắng.

Dự án xây dựng và so sánh các mô hình Visual Question Answering (VQA) để trả lời các câu hỏi tiếng Anh liên quan đến hình ảnh động vật (ví dụ: “What animal is this?”, “How many…”).  
Dự án so sánh hiệu suất giữa mô hình tự xây dựng CNN (Train from Scratch) và mô hình Pre-trained ResNet-50, đồng thời đánh giá tác động của cơ chế Attention.

---

## 👥 Thành viên thực hiện

| STT | Họ và tên | MSSV |
|:---:|:---|:---|
| 1 | Nguyễn Quốc Duy | 52200196 |
| 2 | Nguyễn Hoàng Ân | 52200183 |
| 3 | Nguyễn Nhật Trường | 52200192 |

**Giảng viên hướng dẫn:** PGS.TS. Lê Anh Cường

---

## 📂 Cấu trúc Dự án

- `dataset.ipynb`: Xử lý dữ liệu (tải ảnh COCO, lọc câu hỏi từ VQA, tiền xử lý văn bản).
- `model_training.ipynb`: Huấn luyện mô hình **Train From Scratch** (Custom CNN).
- `model_pre-trained.ipynb`: Huấn luyện mô hình **Pre-trained ResNet-50**.
- `midterm_report.pdf`: Báo cáo chi tiết phương pháp và kết quả.

---

## 📊 Dữ liệu (Dataset)

Dự án sử dụng 2 bộ dữ liệu lớn, được lọc theo category **Animal**:

1. **Hình ảnh:** COCO Train 2014 — chỉ giữ ảnh chứa động vật.  
2. **Câu hỏi/Trả lời:** VQA v2.0 — lọc các câu hỏi "what animal..." và "how many...".

### Quy trình xử lý:

- **Ảnh:** Resize về `224x224`, chuyển Tensor, Normalize theo ImageNet.  
- **Văn bản:** Tokenization, tạo vocab, chuyển chuỗi thành vector và padding.

---

## 🏗️ Kiến trúc Mô hình

Dự án thử nghiệm 4 cấu hình dựa trên kết hợp giữa:

### 1️⃣ Trích xuất đặc trưng ảnh
- **Custom CNN (Train from Scratch):** 3 lớp Conv2d + BatchNorm + MaxPool.  
- **ResNet-50 (Pre-trained):** Lấy đặc trưng từ tầng FC trước khi phân loại (2048-d).

### 2️⃣ Xử lý câu hỏi (Question Processing)
- **LSTM** để xử lý chuỗi văn bản.  
- **Word Embeddings (GloVe)** để biểu diễn từ.

### 3️⃣ Cơ chế Attention
- Trọng số hóa thông tin để mô hình tập trung vào vùng quan trọng của câu hỏi và hình ảnh.

---

## 📈 Kết quả Thực nghiệm (50 epochs)

| Mô hình | Attention | Train Acc | Val Acc | Nhận xét |
|:---|:---:|:---:|:---:|:---|
| Train from Scratch | ❌ | ~78% | ~33% | Overfitting nặng |
| Train from Scratch | ✔️ | ~80% | ~32% | Attention giúp học nhanh nhưng không giảm overfitting |
| Pre-trained ResNet | ❌ | ~64% | ~37% | Tổng quát hóa tốt hơn |
| **Pre-trained ResNet** | **✔️** | **~73%** | **~41%** | **Kết quả tốt nhất** |

---

## 🚀 Hướng dẫn Cài đặt & Chạy

### Yêu cầu hệ thống
- Python 3.10+
- Các thư viện:  
  `torch`, `torchvision`, `pandas`, `numpy`, `matplotlib`,  
  `nltk`, `tqdm`, `Pillow`

---

### Các bước chạy dự án

#### 1️⃣ Clone repository
```bash
git clone https://github.com/quocduy1234231/Animal-VQA-DeepLearning.git
