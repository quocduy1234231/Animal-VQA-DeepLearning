# 🦊 Visual Question Answering (VQA) for Animal Images  
Một dự án cá nhân xây dựng hệ thống **Visual Question Answering** có khả năng trả lời câu hỏi tiếng Anh dựa trên hình ảnh động vật.  
Ví dụ:  
- “What animal is this?”  
- “How many animals are there?”  
- “What color is the animal?”  

Dự án tập trung thử nghiệm các mô hình **CNN**, **ResNet-50**, **LSTM**, và **Attention** để đánh giá khả năng hiểu hình ảnh + ngôn ngữ tự nhiên.

---

## 🚀 Mục tiêu dự án

- Xây dựng pipeline đầy đủ cho VQA: xử lý dữ liệu → trích xuất đặc trưng → mô hình → đánh giá.  
- So sánh hiệu suất giữa:
  - **CNN tự xây dựng từ đầu (Train from Scratch)**  
  - **ResNet-50 Pre-trained**  
- Khám phá tác động của **Attention** trong việc kết hợp thông tin ảnh và câu hỏi.  

---

## 📊 Dataset

Dự án sử dụng 2 nguồn dữ liệu lớn:

### **1. COCO Train 2014**
Dùng để lấy ảnh có chứa động vật.  
Chỉ giữ lại những ảnh thuộc các category: dog, cat, bear, zebra, giraffe, sheep, cow, horse…

### **2. VQA v2.0 (2017)**
Lọc các câu hỏi liên quan đến:
- Nhận dạng (“what animal is…”)  
- Đếm số lượng (“how many…”)  
- Mô tả đặc điểm (“what color…”)  

### **Tiền xử lý**
- Ảnh: resize `224x224`, normalize, chuyển Tensor  
- Văn bản: tokenize, tạo từ điển (vocab), padding  

---

## 🏗️ Kiến trúc mô hình

Dự án thử nghiệm **4 cấu hình**:

### 🔹 1. Image Feature Extraction
- **Custom CNN:** 3 lớp Conv2D + BatchNorm + MaxPool  
- **ResNet-50 Pretrained:** trích xuất feature 2048 chiều

### 🔹 2. Question Encoder
- **Word Embedding (GloVe)**  
- **LSTM** để mã hóa câu hỏi

### 🔹 3. Fusion
- Có hoặc không sử dụng **Attention Mechanism**  
- Kết hợp đặc trưng ảnh + đặc trưng câu hỏi

### 🔹 4. Classifier
- Multi-layer perceptron  
- Dự đoán câu trả lời dạng phân loại (classification)

---

## 📈 Kết quả thực nghiệm

Sau 50 epoch huấn luyện:

| Mô hình | Attention | Train Acc | Val Acc | Nhận xét |
|--------|:---------:|:---------:|:--------:|----------|
| CNN (Scratch) | ❌ | ~78% | ~33% | Overfitting mạnh |
| CNN (Scratch) | ✔️ | ~80% | ~32% | Attention không giúp tổng quát hóa |
| ResNet-50 | ❌ | ~64% | ~37% | Tốt hơn mô hình scratch |
| **ResNet-50** | **✔️** | **~73%** | **~41%** | **Hiệu suất tốt nhất** |

👉 **Kết luận:**  
ResNet-50 + Attention = mô hình mạnh nhất, cân bằng giữa học tốt và tổng quát hóa.

---

## 🛠️ Hướng dẫn chạy

### 1️⃣ Clone project
```bash
git clone https://github.com/ngduy-dev/Animal-VQA-DeepLearning.git
