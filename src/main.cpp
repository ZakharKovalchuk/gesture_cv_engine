#include <iostream>
#include <chrono>
#include <vector>
#include <deque>
#include <unordered_map>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "gesture_engine/classifier.hpp"

namespace fs = std::filesystem;

// Класс для временного сглаживания потока предсказаний (мажоритарное голосование)
class TemporalSmoother {
public:
    explicit TemporalSmoother(size_t window_size = 6) : max_size_(window_size) {}

    DetectionResult update(const DetectionResult& raw) {
        history_.push_back(raw);
        if (history_.size() > max_size_) {
            history_.pop_front();
        }

        std::unordered_map<std::string, int> votes;
        std::unordered_map<std::string, float> conf_sum;

        for (const auto& item : history_) {
            votes[item.label]++;
            conf_sum[item.label] += item.confidence;
        }

        std::string best_label = raw.label;
        int max_votes = 0;

        for (const auto& [label, count] : votes) {
            if (count > max_votes) {
                max_votes = count;
                best_label = label;
            }
        }

        float avg_confidence = conf_sum[best_label] / votes[best_label];

        DetectionResult result = raw;
        result.label = best_label;
        result.confidence = avg_confidence;
        return result;
    }

private:
    size_t max_size_;
    std::deque<DetectionResult> history_;
};

int main() {
    try {
        std::cout << "[INFO] Initializing Gesture CV Engine..." << std::endl;

        // Автоматический выбор пути: запуск из корня или из папки build/
        std::string model_path = fs::exists("models/gesture_model.onnx") ? "models/gesture_model.onnx" : "../models/gesture_model.onnx";
        std::string labels_path = fs::exists("models/labels.txt") ? "models/labels.txt" : "../models/labels.txt";

        GestureClassifier classifier(model_path, labels_path);
        TemporalSmoother smoother(6); // Окно сглаживания

        // Порог уверенности для отсечения шума и пустого фона
        constexpr float CONFIDENCE_THRESHOLD = 0.70f;

        // Захват видео через V4L2
        cv::VideoCapture cap(0, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);

        if (!cap.isOpened()) {
            std::cerr << "[WARN] Video capture device unavailable. Running fallback loop." << std::endl;
        }

        const std::string window_name = "Gesture CV Engine [C++ / ONNX Runtime]";
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

        cv::Mat frame;
        cv::Rect roi(200, 100, 240, 240); // Область интереса под кисть

        auto prev_time = std::chrono::steady_clock::now();
        double fps = 0.0;

        while (true) {
            if (cap.isOpened()) {
                cap >> frame;
                if (frame.empty()) break;
                cv::flip(frame, frame, 1); // Зеркальное отображение для комфортного взаимодействия
            } else {
                frame = cv::Mat(480, 640, CV_8UC3, cv::Scalar(30, 30, 30));
            }

            // Вырезаем ROI для инференса
            cv::Mat hand_roi = frame(roi);

            // Инференс через ONNX Runtime
            DetectionResult raw_result = classifier.predict(hand_roi);

            // Сглаживание шумов во времени
            DetectionResult stable_result = smoother.update(raw_result);

            // Замер FPS
            auto current_time = std::chrono::steady_clock::now();
            double duration = std::chrono::duration<double>(current_time - prev_time).count();
            prev_time = current_time;
            if (duration > 0.0) {
                fps = 0.9 * fps + 0.1 * (1.0 / duration);
            }

            // Логика визуализации на основе порога уверенности
            cv::Scalar box_color;
            std::string label_text;

            if (stable_result.confidence >= CONFIDENCE_THRESHOLD) {
                // Жест распознан уверенно
                box_color = cv::Scalar(0, 255, 0); // Зеленый
                label_text = stable_result.label + " [" + cv::format("%.1f%%", stable_result.confidence * 100.0f) + "]";
            } else {
                // Руки нет, фон или переходное состояние
                box_color = cv::Scalar(100, 100, 100); // Нейтральный серый
                label_text = "Searching for gesture...";
            }

            // Отрисовка рамки ROI
            cv::rectangle(frame, roi, box_color, 2);

            // Отрисовка текста метки над ROI
            cv::putText(frame, label_text, cv::Point(roi.x, roi.y - 12),
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, box_color, 2);

            // Телеметрия: FPS и Latency
            std::string telemetry = cv::format("FPS: %.1f | LATENCY: ~%.1fms", fps, (fps > 0 ? 1000.0 / fps : 0.0));
            cv::putText(frame, telemetry, cv::Point(15, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(240, 240, 240), 2);

            cv::putText(frame, "ESC or 'q' to exit", cv::Point(15, frame.rows - 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 150), 1);

            cv::imshow(window_name, frame);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "[FATAL ERROR] " << e.what() << std::endl;
        return -1;
    }

    return 0;
}