#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <map>

// =====================================================================
//  KNNClassifier — K‑Nearest Neighbors từ đầu (from scratch)
//  Tác giả  : Đồ án Machine Learning
//  Thuật toán: Euclidean distance + Majority voting
// =====================================================================

struct Neighbor {
    double distance;
    int    label;
};

class KNNClassifier {
public:
    // ---- Huấn luyện ------------------------------------------------
    // X_train : ma trận (n_samples × n_features)
    // y_train : nhãn lớp (0, 1, 2, ...)
    void fit(const std::vector<std::vector<double>>& X_train,
             const std::vector<int>&                 y_train);

    // ---- Dự đoán một điểm -----------------------------------------
    // Trả về nhãn dự đoán
    int predict(const std::vector<double>& x, int k) const;

    // ---- Dự đoán + trả về k láng giềng ----------------------------
    std::vector<Neighbor> kNeighbors(const std::vector<double>& x, int k) const;

    // ---- Độ chính xác trên tập test --------------------------------
    // Trả về % accuracy (0–100)
    double score(const std::vector<std::vector<double>>& X_test,
                 const std::vector<int>&                 y_test,
                 int k) const;

    // ---- Phân phối xác suất các lớp --------------------------------
    // Trả về map<label, tỉ lệ> dựa trên k láng giềng
    std::map<int,double> predictProba(const std::vector<double>& x, int k) const;

    // ---- Kích thước tập huấn luyện ---------------------------------
    int trainingSize() const { return (int)X_.size(); }
    int numFeatures()  const { return X_.empty() ? 0 : (int)X_[0].size(); }

    // ---- Tính khoảng cách Euclidean (public để test) ---------------
    static double euclidean(const std::vector<double>& a,
                            const std::vector<double>& b);

private:
    std::vector<std::vector<double>> X_;  // Dữ liệu huấn luyện
    std::vector<int>                 y_;  // Nhãn huấn luyện
};
