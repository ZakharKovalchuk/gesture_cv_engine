#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct DetectionResult {
    int class_id;
    std::string label;
    float confidence;
};

class GestureClassifier {
public:
    GestureClassifier(const std::string& model_path, const std::string& labels_path);
    ~GestureClassifier() = default;

    DetectionResult predict(const cv::Mat& frame);

private:
    void load_labels(const std::string& labels_path);
    std::vector<float> preprocess(const cv::Mat& frame);

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    std::unique_ptr<Ort::Session> session_;

    std::vector<std::string> labels_;
    
    // Спецификации Keras/ONNX модели
    const int input_width_ = 128;
    const int input_height_ = 128;
    const int input_channels_ = 3;

    std::vector<const char*> input_node_names_ = {"input_tensor"};
    std::vector<const char*> output_node_names_ = {"Identity:0"};
};
