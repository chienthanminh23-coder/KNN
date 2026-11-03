// =====================================================================
//  bindings.cpp — Emscripten Bindings
//  Expose C++ KNN functions to JavaScript via WebAssembly
// =====================================================================
#include <emscripten/bind.h>
#include "knn.hpp"
#include <sstream>

using namespace emscripten;

// ---- Helper: chuyển JS array → C++ vector -------------------------
static std::vector<double> jsToVec(const val& jsArr) {
    std::vector<double> v;
    int len = jsArr["length"].as<int>();
    v.reserve(len);
    for (int i = 0; i < len; ++i)
        v.push_back(jsArr[i].as<double>());
    return v;
}

// ---- Wrapper class để dùng từ JavaScript --------------------------
class KNNWrapper {
public:
    KNNWrapper() = default;

    // fit(flatX, nSamples, nFeatures, labels)
    // flatX: mảng 1D [f0, f1, ..., fn, f0, f1, ..., fn, ...]
    void fit(const val& flatX, int nSamples, int nFeatures, const val& labels) {
        std::vector<double> flat = jsToVec(flatX);
        std::vector<int> y;
        for (int i = 0; i < nSamples; ++i)
            y.push_back(labels[i].as<int>());

        std::vector<std::vector<double>> X(nSamples, std::vector<double>(nFeatures));
        for (int i = 0; i < nSamples; ++i)
            for (int j = 0; j < nFeatures; ++j)
                X[i][j] = flat[i * nFeatures + j];

        clf_.fit(X, y);
    }

    // predict(features, k) → nhãn dự đoán
    int predict(const val& features, int k) {
        return clf_.predict(jsToVec(features), k);
    }

    // predictProba(features, k) → JSON string "{0: 0.4, 1: 0.6, ...}"
    std::string predictProba(const val& features, int k) {
        auto proba = clf_.predictProba(jsToVec(features), k);
        std::ostringstream oss;
        oss << "{";
        bool first = true;
        for (auto& p : proba) {
            if (!first) oss << ",";
            oss << "\"" << p.first << "\":" << p.second;
            first = false;
        }
        oss << "}";
        return oss.str();
    }

    // kNeighbors(features, k) → JSON string với distances + labels
    std::string kNeighbors(const val& features, int k) {
        auto neighbors = clf_.kNeighbors(jsToVec(features), k);
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < neighbors.size(); ++i) {
            if (i > 0) oss << ",";
            oss << "{\"distance\":" << neighbors[i].distance
                << ",\"label\":"    << neighbors[i].label << "}";
        }
        oss << "]";
        return oss.str();
    }

    // score(flatX, nSamples, nFeatures, labels, k) → accuracy %
    double score(const val& flatX, int nSamples, int nFeatures,
                 const val& labels, int k) {
        std::vector<double> flat = jsToVec(flatX);
        std::vector<int> y;
        for (int i = 0; i < nSamples; ++i)
            y.push_back(labels[i].as<int>());

        std::vector<std::vector<double>> X(nSamples, std::vector<double>(nFeatures));
        for (int i = 0; i < nSamples; ++i)
            for (int j = 0; j < nFeatures; ++j)
                X[i][j] = flat[i * nFeatures + j];

        return clf_.score(X, y, k);
    }

    int trainingSize() { return clf_.trainingSize(); }
    int numFeatures()  { return clf_.numFeatures(); }

private:
    KNNClassifier clf_;
};

// ---- Đăng ký với Emscripten ---------------------------------------
EMSCRIPTEN_BINDINGS(knn_module) {
    class_<KNNWrapper>("KNNClassifier")
        .constructor<>()
        .function("fit",          &KNNWrapper::fit)
        .function("predict",      &KNNWrapper::predict)
        .function("predictProba", &KNNWrapper::predictProba)
        .function("kNeighbors",   &KNNWrapper::kNeighbors)
        .function("score",        &KNNWrapper::score)
        .function("trainingSize", &KNNWrapper::trainingSize)
        .function("numFeatures",  &KNNWrapper::numFeatures);
}
