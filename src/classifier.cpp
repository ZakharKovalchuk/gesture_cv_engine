#include "gesture_engine/classifier.hpp"
#include <opencv2/dnn.hpp>
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
    cv::Mat blob;
    
    // cv::dnn::blobFromImage выполняет:
    // 1. Ресайз до (input_width_, input_height_) = 128x128
    // 2. Масштабирование: деление пикселей на 255.0f
    // 3. swapRB = true: преобразование BGR -> RGB
    // 4. Переупаковку памяти из HWC в NCHW [1, 3, 128, 128]
    cv::dnn::blobFromImage(
        frame,
        blob,
        1.0f / 255.0f,
        cv::Size(input_width_, input_height_),
        cv::Scalar(0, 0, 0),
        true,   // swapRB
        false   // crop
    );

    // Копируем непрерывный буфер float32 значений в вектор
    return std::vector<float>(blob.ptr<float>(), blob.ptr<float>() + blob.total());
}

DetectionResult GestureClassifier::predict(const cv::Mat& frame) {
    std::vector<float> input_tensor_values = preprocess(frame);

    // Формат PyTorch: NCHW [Batch, Channels, Height, Width] -> [1, 3, 128, 128]
    std::vector<int64_t> input_shape = {1, input_channels_, input_height_, input_width_};
    
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