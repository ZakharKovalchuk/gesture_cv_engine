#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "gesture_engine/classifier.hpp"

int main() {
    try {
        std::cout << "[INFO] Loading model and labels..." << std::endl;
        GestureClassifier classifier("../models/gesture_model.onnx", "../models/labels.txt");

        // Открываем RGB-ноду /dev/video0 через бэкенд V4L2
        cv::VideoCapture cap(0, cv::CAP_V4L2);

        // Принудительно выставляем сжатый кодек MJPG ДО разрешения,
        // чтобы избежать таймаута шины USB/IP
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);

        if (!cap.isOpened()) {
            std::cerr << "[WARN] Camera with index 0 could not be opened." << std::endl;
            std::cerr << "[INFO] Falling back to synthetic test loop (press 'q' or ESC in window to exit)..." << std::endl;
        }

        const std::string window_name = "Gesture CV Engine (C++)";
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

        cv::Mat frame;
        cv::Rect roi(200, 100, 240, 240); // Квадратная зона под руку

        auto prev_time = std::chrono::steady_clock::now();
        double fps = 0.0;

        while (true) {
            if (cap.isOpened()) {
                cap >> frame;
                if (frame.empty()) break;
                // Зеркалим по горизонтали для привычного отображения
                cv::flip(frame, frame, 1);
            } else {
                // Фоллбэк на темный холст, если дескриптор недоступен
                frame = cv::Mat(480, 640, CV_8UC3, cv::Scalar(40, 40, 40));
            }

            // Вырезаем область ROI для инференса
            cv::Mat hand_roi = frame(roi);

            // Инференс через ONNX Runtime
            DetectionResult result = classifier.predict(hand_roi);

            // Расчет сглаженного FPS
            auto current_time = std::chrono::steady_clock::now();
            double duration = std::chrono::duration<double>(current_time - prev_time).count();
            prev_time = current_time;
            if (duration > 0.0) {
                fps = 0.9 * fps + 0.1 * (1.0 / duration);
            }

            // Рамка ROI (зеленый при уверенности > 60%, иначе оранжевый)
            cv::Scalar box_color = (result.confidence > 0.60f) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 165, 255);
            cv::rectangle(frame, roi, box_color, 2);

            // Метка класса и вероятность
            std::string label_text = result.label + " (" + cv::format("%.1f%%", result.confidence * 100.0f) + ")";
            cv::putText(frame, label_text, cv::Point(roi.x, roi.y - 12),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2);

            // Счетчик FPS
            std::string fps_text = cv::format("FPS: %.1f", fps);
            cv::putText(frame, fps_text, cv::Point(15, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

            // Подсказка выхода
            cv::putText(frame, "Press 'q' or ESC to exit", cv::Point(15, frame.rows - 15),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(180, 180, 180), 1);

            // Отрисовка кадра
            cv::imshow(window_name, frame);

            int key = cv::waitKey(1);
            if (key == 27 || key == 'q' || key == 'Q') {
                break;
            }
        }

        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << std::endl;
        return -1;
    }

    return 0;
}