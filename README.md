# 🏦 Consumer Credit Risk Classification & Scoring System
### Phân Loại & Chấm Điểm Rủi Ro Tín Dụng Khách Hàng Vay Tiêu Dùng (Lending Club Dataset)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

---

## 📌 Tổng Quan Dự Án (Project Overview)

Dự án xây dựng quy trình **Machine Learning End-to-End** trên tập dữ liệu thực tế quy mô lớn của **Lending Club** (>2.26 triệu hồ sơ vay, 145 đặc trưng) nhằm dự báo khả năng vỡ nợ (Default / Charged Off) của khách hàng vay tiêu dùng. 

Dự án giải quyết hai bài toán cốt lõi trong lĩnh vực Tài chính - Ngân hàng:
1. **Rò rỉ dữ liệu (Data Leakage)** và **Ngoại lai cực đoan (Extreme Outliers)** trong hồ sơ tài chính thực tế.
2. **Mất cân bằng nhãn nghiêm trọng (Class Imbalance ~78:22)** giữa nhóm khách hàng trả nợ tốt và nhóm nợ xấu.

---

## 🎯 Câu Hỏi Nghiên Cứu (Research Questions)

1. **RQ1:** Những yếu tố tài chính và đặc trưng tín dụng nào ảnh hưởng mạnh nhất đến xác suất vỡ nợ của khách hàng?
2. **RQ2:** Phương pháp xử lý mất cân bằng dữ liệu (`class_weight`, `SMOTE`, `Undersampling`) và thuật toán học máy nào mang lại hiệu quả phân loại tối ưu nhất?
3. **RQ3:** Kết quả từ mô hình học máy có thể chuyển hóa thành các chính sách phê duyệt tự động và định giá lãi suất theo rủi ro (Risk-Based Pricing) như thế nào?

---

## 🛠️ Công Nghệ & Thư Viện (Tech Stack)

- **Ngôn ngữ:** Python
- **Xử lý & Phân tích dữ liệu:** `Pandas`, `NumPy`
- **Trực quan hóa dữ liệu (EDA & Evaluation):** `Matplotlib`, `Seaborn`
- **Học máy & Tiền xử lý:** `Scikit-learn` (`OneHotEncoder`, `StandardScaler`, `PCA`, `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `LinearDiscriminantAnalysis`)
- **Boosting & Xử lý lệch nhãn:** `XGBoost`, `Imbalanced-learn` (`SMOTE`, `RandomUnderSampler`)

---

## ⚙️ Quy Trình Thực Hiện (Methodology Pipeline)

### 1. Khám phá dữ liệu (Exploratory Data Analysis - EDA)
- Phân tích hơn **2,260,668 bản ghi** và **145 thuộc tính** gốc.
- Trực quan hóa 18 biểu đồ phân tích phân phối đơn biến, đa biến và ma trận tương quan (Correlation Matrix).
- Đánh giá tỷ lệ nợ xấu theo Hạng tín dụng (`grade`), Mục đích vay (`purpose`), Kỳ hạn vay (`term`), Tình trạng nhà ở (`home_ownership`) và Thu nhập (`annual_inc`).

### 2. Tiền xử lý & Chống rò rỉ dữ liệu (Data Preprocessing & Leakage Prevention)
- **Xác định nhãn mục tiêu:** Lọc các khoản vay đã kết thúc (`1,338,224` hồ sơ). Gán nhãn `0` (*Fully Paid* - 78%) và nhãn `1` (*Charged Off / Default / Late* - 22%).
- **Chống Data Leakage:** Loại bỏ triệt để các biến phát sinh sau thời điểm giải ngân (`recoveries`, `total_rec_prncp`, `last_pymnt_amnt`,...).
- **Xử lý Missing Values & Outliers:** Loại bỏ các cột khuyết thiếu >50%, điền khuyết bằng `Median`/`Mode` và áp dụng kỹ thuật **Winsorization (1% – 99%)** cho các biến tài chính (`annual_inc`, `dti`, `loan_amnt`).
- **Chuẩn hóa chống rò rỉ:** Chia tập Train/Test (80/20 - Stratify) **trước** khi `fit_transform` với `OneHotEncoder` và `StandardScaler` (tạo ra 51 đặc trưng đầu vào).

### 3. Huấn luyện & Đối sánh 8 Phương án Học máy (Model Benchmarking)
1. **Logistic Regression (Baseline)** với `class_weight='balanced'`
2. **Decision Tree Classifier** (`max_depth=6`, `class_weight='balanced'`)
3. **Random Forest Classifier** (Bagging Ensemble)
4. **XGBoost Classifier** (`tree_method='hist'`, `scale_pos_weight`)
5. **LDA (Linear Discriminant Analysis)**
6. **PCA (10 Components) + Logistic Regression**
7. **SMOTE (Oversampling) + Logistic Regression**
8. **Random Undersampling + Logistic Regression**

---

## 📊 Kết Quả Thực Nghiệm (Model Performance)

Đánh giá trên tập kiểm thử độc lập (**267,645 hồ sơ vay**):

| STT | Phương án / Mô hình | Accuracy | Precision (Nhãn 1) | Recall (Nhãn 1) | F1-Score (Nhãn 1) | ROC-AUC |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Logistic Regression (Baseline)** | 63.71% | 33.82% | 66.81% | 44.91% | 0.704 |
| 2 | **Decision Tree** | 60.58% | 32.25% | **70.89%** | 44.33% | 0.697 |
| 3 | **Random Forest** | 62.20% | 33.05% | 68.97% | 44.69% | 0.702 |
| 4 | **XGBoost Classifier** ⭐ | 64.79% | 34.76% | 67.33% | **45.85%** | **0.715** |
| 5 | **LDA (Linear Discriminant Analysis)** ⚠️ | **78.10%** | **52.16%** | 12.92% | 20.71% | 0.703 |
| 6 | **PCA + Logistic Regression** | 65.48% | 34.61% | 62.91% | 44.66% | 0.700 |
| 7 | **SMOTE + Logistic Regression** | 63.73% | 33.79% | 66.56% | 44.82% | 0.703 |
| 8 | **Undersampling + Logistic Regression** | 63.72% | 33.83% | 66.77% | 44.90% | 0.704 |

### 💡 Điểm nhấn phân tích:
- **Mô hình tối ưu nhất:** **XGBoost Classifier** đạt sự cân bằng tốt nhất với **ROC-AUC = 0.715** và **F1-Score = 45.85%**, nhận diện thành công hơn 67% tổng số ca nợ xấu thực tế.
- **Cảnh báo "Bẫy Accuracy":** Mô hình **LDA** đạt Accuracy cao nhất (78.10%) nhưng **Recall chỉ đạt 12.92%** (bỏ lọt >87% khách hàng vỡ nợ), minh chứng việc sử dụng Accuracy đơn thuần trên dữ liệu lệch nhãn sẽ gây rủi ro mất vốn nghiêm trọng.
- **Tối ưu chi phí tính toán:** Kỹ thuật `class_weight='balanced'` mang lại hiệu năng tương đương `SMOTE` nhưng tiết kiệm hơn 95% thời gian huấn luyện.

---

## 💼 Đề Xuất Chính Sách Kinh Doanh (Business Recommendations)

Từ phân tích **Feature Importance** của XGBoost (trong đó Hạng tín dụng `grade_A`, Lãi suất `int_rate`, Kỳ hạn `term`, Tình trạng nhà ở `home_ownership` và `dti` là các yếu tố quyết định):

- **Luồng xanh phê duyệt tự động (Green Channel):** Tự động duyệt nhanh cho hồ sơ thuộc `Grade A, B` nhằm giảm chi phí vận hành và tăng trải nghiệm khách hàng.
- **Định giá dựa trên rủi ro (Risk-Based Pricing):** Điều chỉnh biên độ lãi suất và giới hạn kỳ hạn vay 36 tháng (thay vì 60 tháng) đối với phân khúc rủi ro trung bình - cao (`Grade C, D, E`) và nhóm mục đích vay `small_business`.
- **Chiến lược Cut-off 3 tầng:** Phân luồng hồ sơ dựa trên xác suất vỡ nợ dự báo từ XGBoost thành 3 nhóm: *Tự động phê duyệt* – *Chuyển thẩm định viên (Manual Review)* – *Tự động từ chối*.

---

## 📂 Cấu Trúc Thư Mục (Repository Structure)

```text
├── PhanTichDataset.ipynb   # Notebook chính chứa toàn bộ Source Code, Biểu đồ & Phân tích (Phần A -> D)
├── README.md                    # Tài liệu giới thiệu và hướng dẫn dự án
```

---

## 🚀 Hướng Dẫn Chạy Dự Án (How to Run)

1. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn jupyter
   ```
2. **Tải tập dữ liệu:**
   - Tải bộ dữ liệu Lending Club (`loan.csv`) từ Kaggle và đặt vào thư mục gốc của dự án (hoặc kết nối Google Drive nếu chạy trên Google Colab).
3. **Khởi chạy Jupyter Notebook:**
   ```bash
   jupyter notebook PhanTichDataset.ipynb
   ```
