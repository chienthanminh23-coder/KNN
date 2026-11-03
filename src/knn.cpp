#include "knn.hpp"
#include <numeric>
#include <stdexcept>

// =====================================================================
//  euclidean — Khoảng cách Euclidean giữa hai vector
//  d(a,b) = sqrt( sum( (a_i - b_i)^2 ) )
// =====================================================================
double KNNClassifier::euclidean(const std::vector<double>& a,
                                const std::vector<double>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("Vectors must have the same dimension");

    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// =====================================================================
//  fit — Ghi nhớ toàn bộ tập huấn luyện
//  KNN là lazy learner: không học gì ở bước này, chỉ lưu dữ liệu.
// =====================================================================
void KNNClassifier::fit(const std::vector<std::vector<double>>& X_train,
                        const std::vector<int>&                 y_train) {
    if (X_train.size() != y_train.size())
        throw std::invalid_argument("X and y must have the same number of samples");
    if (X_train.empty())
        throw std::invalid_argument("Training data must not be empty");

    X_ = X_train;
    y_ = y_train;
}

// =====================================================================
//  kNeighbors — Tìm k điểm gần nhất với điểm truy vấn x
//  1. Tính khoảng cách x đến mọi điểm trong tập huấn luyện
//  2. Sắp xếp tăng dần theo khoảng cách
//  3. Lấy k phần tử đầu tiên
// =====================================================================
std::vector<Neighbor> KNNClassifier::kNeighbors(const std::vector<double>& x,
                                                  int k) const {
    if (X_.empty())
        throw std::runtime_error("Model has not been fitted yet");
    if (k <= 0 || k > (int)X_.size())
        throw std::invalid_argument("k must be between 1 and training set size");

    // Tính tất cả khoảng cách
    std::vector<Neighbor> neighbors;
    neighbors.reserve(X_.size());
    for (size_t i = 0; i < X_.size(); ++i) {
        neighbors.push_back({ euclidean(x, X_[i]), y_[i] });
    }

    // Sắp xếp tăng dần theo khoảng cách (partial_sort: chỉ sắp k phần tử đầu)
    std::partial_sort(neighbors.begin(),
                      neighbors.begin() + k,
                      neighbors.end(),
                      [](const Neighbor& a, const Neighbor& b){
                          return a.distance < b.distance;
                      });

    neighbors.resize(k);
    return neighbors;
}

// =====================================================================
//  predict — Trả về nhãn dự đoán bằng Majority Voting
//  Đếm số phiếu của mỗi lớp trong k láng giềng, chọn lớp nhiều nhất.
// =====================================================================
int KNNClassifier::predict(const std::vector<double>& x, int k) const {
    auto neighbors = kNeighbors(x, k);

    // Đếm phiếu
    std::map<int,int> votes;
    for (auto& nb : neighbors)
        votes[nb.label]++;

    // Tìm nhãn có nhiều phiếu nhất
    return std::max_element(votes.begin(), votes.end(),
                            [](const std::pair<int,int>& a,
                               const std::pair<int,int>& b){
                                return a.second < b.second;
                            })->first;
}

// =====================================================================
//  predictProba — Xác suất từng lớp (tỉ lệ phiếu trong k láng giềng)
// =====================================================================
std::map<int,double> KNNClassifier::predictProba(const std::vector<double>& x,
                                                   int k) const {
    auto neighbors = kNeighbors(x, k);

    std::map<int,double> proba;
    for (auto& nb : neighbors)
        proba[nb.label] += 1.0;

    for (auto& p : proba)
        p.second /= k;

    return proba;
}

// =====================================================================
//  score — Độ chính xác (%) trên tập test
// =====================================================================
double KNNClassifier::score(const std::vector<std::vector<double>>& X_test,
                             const std::vector<int>&                 y_test,
                             int k) const {
    if (X_test.size() != y_test.size())
        throw std::invalid_argument("X_test and y_test size mismatch");
    if (X_test.empty()) return 0.0;

    int correct = 0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        if (predict(X_test[i], k) == y_test[i])
            ++correct;
    }
    return 100.0 * correct / X_test.size();
}
