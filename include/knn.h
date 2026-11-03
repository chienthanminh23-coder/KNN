// knn.h - Simple K-Nearest Neighbors classifier
// -------------------------------------------------
// Provides a lightweight, header‑only interface for a KNN model.
// Designed for numeric CSV data where the last column is the class label.
//
// Usage example (see README):
//   KNN knn(k);
//   knn.fit(train_features, train_labels);
//   auto preds = knn.predict(test_features);
//
// The implementation lives in knn.cpp; this header only declares the class.

#ifndef KNN_H
#define KNN_H

#include <vector>
#include <string>

class KNN {
public:
    // Construct with number of neighbors (k). k must be >=1.
    explicit KNN(std::size_t k = 3);

    // Fit the model with training data.
    //   features: NxM matrix (N samples, M features)
    //   labels:   length N vector of class names (strings)
    void fit(const std::vector<std::vector<double>>& features,
             const std::vector<std::string>& labels);

    // Predict the class label for each sample in test_features.
    // Returns a vector of predicted labels (same order as input).
    std::vector<std::string> predict(const std::vector<std::vector<double>>& test_features) const;

    // Accessors
    std::size_t k() const { return k_; }
    const std::vector<std::vector<double>>& training_features() const { return train_features_; }
    const std::vector<std::string>& training_labels() const { return train_labels_; }

private:
    std::size_t k_; // number of neighbors
    std::vector<std::vector<double>> train_features_;
    std::vector<std::string> train_labels_;

    // Helper: Euclidean distance between two feature vectors.
    static double euclidean(const std::vector<double>& a, const std::vector<double>& b);
};

#endif // KNN_H
