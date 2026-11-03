# 🧠 KNN From Scratch — C++ + WebAssembly

> **Đồ án Machine Learning**: Cài đặt thuật toán K-Nearest Neighbors bằng C++17, biên dịch sang WebAssembly, chạy trực tiếp trong trình duyệt.

🌐 **[Xem Demo Trực Tuyến →](https://YOUR_USERNAME.github.io/KNN)**

---

## 📋 Giới Thiệu Thuật Toán KNN

**K-Nearest Neighbors (KNN)** là thuật toán học máy có giám sát dạng **lazy learner** — không xây dựng mô hình khi huấn luyện mà chỉ ghi nhớ dữ liệu.

### Nguyên lý hoạt động

1. **Lưu** toàn bộ tập huấn luyện `{(x₁,y₁), ..., (xₙ,yₙ)}`
2. Khi cần dự đoán điểm mới `q`:
   - Tính **khoảng cách Euclidean** từ `q` đến mọi điểm huấn luyện
   - Lấy **k điểm gần nhất**
   - Dùng **majority voting** → nhãn nào nhiều phiếu nhất là kết quả

### Công thức khoảng cách Euclidean

$$d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2}$$

### Độ phức tạp

| Phase | Thời gian | Không gian |
|-------|-----------|------------|
| Huấn luyện (`fit`) | `O(1)` | `O(n·d)` |
| Dự đoán (`predict`) | `O(n·d)` | `O(1)` |

> `n` = số mẫu, `d` = số features, `k` = số láng giềng

---

## 🗂️ Cấu Trúc Dự Án

```
KNN/
├── src/
│   ├── knn.hpp          # Interface: class KNNClassifier
│   ├── knn.cpp          # Thuật toán KNN (C++17)
│   └── bindings.cpp     # Emscripten bindings (C++ ↔ JS)
├── web/
│   ├── index.html       # Giao diện web app
│   ├── style.css        # Dark theme UI
│   ├── app.js           # Logic frontend
│   └── iris_data.js     # Dataset Iris (150 mẫu)
├── docs/                # GitHub Pages (auto-generated)
├── .github/workflows/
│   └── build.yml        # CI/CD: build WASM + deploy Pages
├── Makefile
└── README.md
```

---

## 🚀 Chạy Dự Án

### Cách 1 — GitHub Pages (không cần cài gì)
Truy cập: `https://YOUR_USERNAME.github.io/KNN`

### Cách 2 — Build local (cần Emscripten)

```bash
# 1. Cài Emscripten
git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install latest && ./emsdk activate latest
source ./emsdk_env.sh   # Linux/Mac
# emsdk_env.bat          # Windows

# 2. Clone & Build
git clone https://github.com/YOUR_USERNAME/KNN.git
cd KNN
make build    # Compile C++ → WASM
make docs     # Copy sang docs/
make serve    # Mở http://localhost:8080
```

### Cách 3 — Chỉ xem web (không cần WASM)
> File `docs/knn.js` và `docs/knn.wasm` đã được commit sẵn. Chỉ cần mở `docs/index.html` bằng Live Server trong VS Code.

---

## 📊 Dataset: Iris

| Đặc trưng | Mô tả | Đơn vị |
|-----------|-------|--------|
| Sepal Length | Chiều dài đài hoa | cm |
| Sepal Width | Chiều rộng đài hoa | cm |
| Petal Length | Chiều dài cánh hoa | cm |
| Petal Width | Chiều rộng cánh hoa | cm |

**3 lớp:** Iris Setosa (0), Iris Versicolor (1), Iris Virginica (2)  
**150 mẫu** · Chia **70% train / 30% test**

### Kết quả thực nghiệm

| k | Accuracy (test set) |
|---|---------------------|
| 1 | ~93.3% |
| 3 | ~95.6% |
| **5** | **~97.8%** |
| 7 | ~97.8% |
| 9 | ~95.6% |

---

## 🌐 Tính Năng Web App

- **Scatter Plot tương tác** — chọn 2 trong 4 features để visualize
- **Thanh kéo k** — điều chỉnh k từ 1–15, accuracy cập nhật ngay
- **Nhập điểm mới** — nhập 4 feature → dự đoán tức thì bằng C++/WASM
- **Bảng k láng giềng** — hiển thị khoảng cách + lớp của từng láng giềng
- **Xác suất từng lớp** — tỉ lệ phiếu bầu trong k láng giềng

---

## 🔧 Stack Công Nghệ

| Thành phần | Công nghệ |
|---|---|
| Thuật toán | **C++17** (không dùng thư viện ngoài) |
| Biên dịch | **Emscripten** (C++ → `.wasm`) |
| Frontend | **HTML5 + CSS3 + Vanilla JS** |
| Chart | **Chart.js 4** |
| CI/CD | **GitHub Actions** |
| Hosting | **GitHub Pages** |

---

## 📁 API C++

```cpp
KNNClassifier clf;

// Huấn luyện
clf.fit(X_train, y_train);

// Dự đoán
int label = clf.predict(x_query, k);

// Xác suất từng lớp
auto proba = clf.predictProba(x_query, k);

// k láng giềng gần nhất
auto neighbors = clf.kNeighbors(x_query, k);

// Độ chính xác trên tập test
double acc = clf.score(X_test, y_test, k);
```

---

## 👤 Tác Giả

Đồ án môn **Machine Learning** — Cài đặt KNN from scratch bằng C++17 + WebAssembly

---

*Được xây dựng với ❤️ bằng C++ và WebAssembly*
