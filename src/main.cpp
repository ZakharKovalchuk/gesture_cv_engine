#include <iostream>
#include "gesture_engine/classifier.hpp"

int main() {
    try {
        std::cout << "[INFO] Initializing Gesture Classifier..." << std::endl;
        GestureClassifier classifier("../models/gesture_model.onnx", "../models/labels.txt");

        // Синтетический тестовый кадр 640x480 (черное изображение)
        cv::Mat dummy_frame = cv::Mat::zeros(480, 640, CV_8UC3);

        std::cout << "[INFO] Running test inference on dummy frame..." << std::endl;
        DetectionResult res = classifier.predict(dummy_frame);

        std::cout << "[SUCCESS] Class: " << res.label 
                  << " (ID: " << res.class_id << ")"
                  << " | Confidence: " << res.confidence << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
