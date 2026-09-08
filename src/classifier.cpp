#include "gesture_engine/classifier.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>

GestureClassifier::GestureClassifier(const std::string& model_path, const std::string& labels_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "GestureClassifier") {
    
    session_options_.SetIntraOpNumThreads(2);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

    load_labels(labels_path);
}

void GestureClassifier::load_labels(const std::string& labels_path) {
    std::ifstream file(labels_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open labels file: " + labels_path);
    }
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            labels_.push_back(line);
        }
    }
}

std::vector<float> GestureClassifier::preprocess(const cv::Mat& frame) {
    cv::Mat resized, rgb, normalized;
    
    // 1. Ресайз до 128x128
    cv::resize(frame, resized, cv::Size(input_width_, input_height_));
    
    // 2. BGR -> RGB
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    
    // 3. Преобразование в float32 и масштабирование в [0.0, 1.0]
    rgb.convertTo(normalized, CV_32FC3, 1.0f / 255.0f);

    // 4. Копирование в буфер NHWC [1, 128, 128, 3]
    std::vector<float> input_tensor_values(1 * input_height_ * input_width_ * input_channels_);
    std::memcpy(input_tensor_values.data(), normalized.data, input_tensor_values.size() * sizeof(float));

    return input_tensor_values;
}

DetectionResult GestureClassifier::predict(const cv::Mat& frame) {
    std::vector<float> input_tensor_values = preprocess(frame);

    std::vector<int64_t> input_shape = {1, input_height_, input_width_, input_channels_};
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_tensor_values.data(),
        input_tensor_values.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Запуск вычислений в ONNX Runtime
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_node_names_.data(),
        &input_tensor,
        1,
        output_node_names_.data(),
        output_node_names_.size()
    );

    float* raw_output = output_tensors.front().GetTensorMutableData<float>();
    size_t num_classes = labels_.size();

    int best_class_id = 0;
    float max_score = raw_output[0];
    for (size_t i = 1; i < num_classes; ++i) {
        if (raw_output[i] > max_score) {
            max_score = raw_output[i];
            best_class_id = static_cast<int>(i);
        }
    }

    DetectionResult result;
    result.class_id = best_class_id;
    result.confidence = max_score;
    result.label = (best_class_id < static_cast<int>(labels_.size())) ? labels_[best_class_id] : "Unknown";

    return result;
}
