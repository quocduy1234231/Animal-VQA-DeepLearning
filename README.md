# Visual Question Answering (VQA) trên Tập dữ liệu Động vật

[cite_start]Đồ án cuối kỳ môn **Học Sâu (Deep Learning)** - Trường Đại học Tôn Đức Thắng[cite: 1, 3].

Dự án này xây dựng và so sánh các mô hình Visual Question Answering (VQA) để trả lời các câu hỏi tiếng Anh liên quan đến hình ảnh động vật (ví dụ: "What animal is this?", "How many..."). [cite_start]Dự án so sánh hiệu suất giữa việc tự xây dựng mô hình CNN (Train from scratch) và sử dụng mô hình đã huấn luyện trước (Pre-trained ResNet-50), đồng thời đánh giá tác động của cơ chế Attention[cite: 911, 1022].

## 👥 Thành viên thực hiện
| STT | Họ và tên | MSSV |
|:---:|:---|:---|
| 1 | Nguyễn Quốc Duy | 52200196 |
| 2 | Nguyễn Hoàng Ân | 52200183 |
| 3 | Nguyễn Nhật Trường | 52200192 |

**Giảng viên hướng dẫn:** PGS.TS. [cite_start]Lê Anh Cường[cite: 6].

## 📂 Cấu trúc Dự án
* [cite_start]`dataset.ipynb`: Notebook xử lý dữ liệu (tải ảnh từ COCO, lọc câu hỏi từ VQA v2.0, tiền xử lý văn bản)[cite: 826].
* [cite_start]`model_training.ipynb`: Huấn luyện mô hình **Train from Scratch** (Tự xây dựng CNN)[cite: 965].
* [cite_start]`model_pre-trained.ipynb`: Huấn luyện mô hình **Pre-trained** (Sử dụng ResNet-50)[cite: 958].
* `midterm_report.pdf`: Báo cáo chi tiết về phương pháp và kết quả thực nghiệm.

## 📊 Dữ liệu (Dataset)
Dự án sử dụng kết hợp hai bộ dữ liệu lớn, được lọc riêng cho category **Animal**:
1.  **Hình ảnh:** Tập **COCO train 2014**. [cite_start]Được lọc để chỉ lấy các ảnh chứa động vật để tránh dữ liệu thưa[cite: 704].
2.  **Câu hỏi/Trả lời:** Tập **VQA v2.0 (2017)**. [cite_start]Lọc các câu hỏi liên quan đến nhận dạng ("what animal is") và đếm số lượng ("how many")[cite: 712, 760].

**Quy trình xử lý:**
* [cite_start]**Ảnh:** Resize về kích thước `224x224`, chuyển thành Tensor và chuẩn hóa (Normalize) theo ImageNet[cite: 879, 881].
* [cite_start]**Văn bản:** Tokenization, tạo từ điển (Vocab), chuyển thành vector và padding độ dài cố định[cite: 893, 907].

## 🏗️ Kiến trúc Mô hình
Dự án thực nghiệm 4 cấu hình mô hình khác nhau dựa trên sự kết hợp của các thành phần sau:

### 1. Trích xuất đặc trưng ảnh (Image Feature Extraction)
* [cite_start]**Custom CNN (Train from Scratch):** Mạng tích chập tự xây dựng gồm 3 lớp Conv2d, BatchNorm và MaxPool[cite: 965, 969].
* [cite_start]**ResNet-50 (Pre-trained):** Sử dụng mạng ResNet-50 đã huấn luyện trên ImageNet, đóng băng trọng số và lấy đặc trưng tại lớp trước Fully Connected (kích thước 2048)[cite: 960].

### 2. Xử lý ngôn ngữ (Question Processing)
* [cite_start]Sử dụng mạng **LSTM** (Long Short-Term Memory) để xử lý chuỗi từ[cite: 987].
* [cite_start]Sử dụng **Word Embeddings** (GloVe)[cite: 657].

### 3. Cơ chế Attention
* [cite_start]Sử dụng cơ chế Attention đơn giản để trọng số hóa thông tin từ LSTM, giúp mô hình tập trung vào các phần quan trọng của câu hỏi và hình ảnh[cite: 991, 1124].

## 📈 Kết quả Thực nghiệm

Dưới đây là tóm tắt kết quả huấn luyện sau 50 epochs:

| Mô hình | Cơ chế | Train Accuracy | Val Accuracy | Nhận xét |
|:---|:---:|:---:|:---:|:---|
| **Train from Scratch** | Non-Attention | ~78% | ~33% | [cite_start]Overfitting nặng, Validation Loss tăng dần sau epoch 25[cite: 1153, 1154]. |
| **Train from Scratch** | **Attention** | ~80% | ~32% | [cite_start]Attention giúp học nhanh hơn nhưng vẫn bị overfitting nặng[cite: 1193, 1194]. |
| **Pre-trained (ResNet)**| Non-Attention | ~64% | ~37% | [cite_start]Ổn định hơn, tổng quát hóa tốt hơn mô hình tự xây[cite: 1040, 1041]. |
| **Pre-trained (ResNet)**| **Attention** | **~73%** | **~41%** | **Kết quả tốt nhất**. [cite_start]Attention giúp cải thiện độ chính xác trên tập Validation[cite: 1250, 1288]. |

## 🚀 Hướng dẫn Cài đặt & Chạy
### Yêu cầu hệ thống
* Python 3.10+
* Thư viện: `torch`, `torchvision`, `pandas`, `numpy`, `matplotlib`, `nltk`, `tqdm`, `Pillow`.

### Các bước thực hiện
1.  **Clone repository:**
    ```bash
    git clone https://github.com/quocduy1234231/Animal-VQA-DeepLearning.git
    ```

2.  **Tải dữ liệu:**
    Chạy file `dataset.ipynb` để tải ảnh từ COCO API và file json từ VQA, sau đó tiền xử lý tạo ra file `dataset_vqa.csv`.

3.  **Huấn luyện:**
    * Để train mô hình tự xây dựng: Chạy `model_training.ipynb`.
    * Để train mô hình ResNet-50: Chạy `model_pre-trained.ipynb`.
    * *Lưu ý: Có thể bật/tắt biến `use_attention = True/False` trong code để thử nghiệm các cấu hình khác nhau.*

## 🔗 Tham khảo
* [cite_start][VQA Dataset Website](https://visualqa.org/) [cite: 1417]
* [cite_start][MS COCO Dataset](https://cocodataset.org/) [cite: 1419]
* [Pytorch Documentation](https://pytorch.org/docs/stable/index.html)