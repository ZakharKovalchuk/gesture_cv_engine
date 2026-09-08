#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "[INFO] Gesture CV Engine starting..." << std::endl;
    std::cout << "[INFO] OpenCV version: " << CV_VERSION << std::endl;

    // Инициализация среды ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "GestureTest");
    std::cout << "[INFO] ONNX Runtime environment initialized successfully." << std::endl;

    return 0;
}
