#include <iostream>
#include <chrono>
#include <vector>
#include <deque>
#include <unordered_map>
#include <filesystem>
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>
#include "gesture_engine/classifier.hpp"

namespace fs = std::filesystem;

// UI Color Palette (BGR)
const cv::Scalar COLOR_CYAN     = cv::Scalar(255, 230, 0);   // Target locked state
const cv::Scalar COLOR_AMBER    = cv::Scalar(0, 140, 255);   // Standby / scanning state
const cv::Scalar COLOR_BG_DARK  = cv::Scalar(18, 18, 18);    // Graphite dark panel base
const cv::Scalar COLOR_WHITE    = cv::Scalar(240, 240, 240); // Primary HUD typography
const cv::Scalar COLOR_MUTED    = cv::Scalar(135, 135, 135); // Secondary indicators

// Temporal smoother for label stabilization via majority voting
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

// Check standard deviation to filter out flat, featureless backgrounds (e.g. blank walls)
bool isHandPresent(const cv::Mat& roi, double stddev_thresh = 18.0) {
    cv::Mat gray;
    if (roi.channels() == 3) {
        cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = roi;
    }

    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);

    return stddev[0] > stddev_thresh;
}

// Render reticle brackets and crosshair indicator
void drawTargetReticle(cv::Mat& frame, const cv::Rect& roi, const cv::Scalar& color, int len = 25, int thickness = 2) {
    int x = roi.x, y = roi.y, w = roi.width, h = roi.height;

    // Top-left
    cv::line(frame, {x, y}, {x + len, y}, color, thickness, cv::LINE_AA);
    cv::line(frame, {x, y}, {x, y + len}, color, thickness, cv::LINE_AA);

    // Top-right
    cv::line(frame, {x + w, y}, {x + w - len, y}, color, thickness, cv::LINE_AA);
    cv::line(frame, {x + w, y}, {x + w, y + len}, color, thickness, cv::LINE_AA);

    // Bottom-left
    cv::line(frame, {x, y + h}, {x + len, y + h}, color, thickness, cv::LINE_AA);
    cv::line(frame, {x, y + h}, {x, y + h - len}, color, thickness, cv::LINE_AA);

    // Bottom-right
    cv::line(frame, {x + w, y + h}, {x + w - len, y + h}, color, thickness, cv::LINE_AA);
    cv::line(frame, {x + w, y + h}, {x + w, y + h - len}, color, thickness, cv::LINE_AA);

    // Center crosshair
    cv::Point center(x + w / 2, y + h / 2);
    int cs = 6;
    cv::line(frame, {center.x - cs, center.y}, {center.x + cs, center.y}, color, 1, cv::LINE_AA);
    cv::line(frame, {center.x, center.y - cs}, {center.x, center.y + cs}, color, 1, cv::LINE_AA);
}

// Render dynamic confidence progress bar
void drawConfidenceBar(cv::Mat& frame, const cv::Rect& roi, float conf, const cv::Scalar& color) {
    int bar_y = roi.y + roi.height + 12;
    int bar_h = 4;
    int total_w = roi.width;

    cv::rectangle(frame, cv::Rect(roi.x, bar_y, total_w, bar_h), cv::Scalar(40, 40, 40), -1);

    int fill_w = static_cast<int>(total_w * std::clamp(conf, 0.0f, 1.0f));
    if (fill_w > 0) {
        cv::rectangle(frame, cv::Rect(roi.x, bar_y, fill_w, bar_h), color, -1);
    }
}

// Render HUD elements with zero heap allocations for overlays
void renderHUD(cv::Mat& canvas, const cv::Rect& roi, const std::string& label, float confidence, double fps, double latency_ms, bool is_detected) {
    // Top banner overlay
    cv::Rect top_rect(0, 0, canvas.cols, 38);
    cv::Mat top_roi = canvas(top_rect);
    cv::Mat top_bg(top_roi.size(), top_roi.type(), COLOR_BG_DARK);
    cv::addWeighted(top_bg, 0.70, top_roi, 0.30, 0, top_roi);

    // Bottom banner overlay
    cv::Rect bottom_rect(0, canvas.rows - 26, canvas.cols, 26);
    cv::Mat bottom_roi = canvas(bottom_rect);
    cv::Mat bottom_bg(bottom_roi.size(), bottom_roi.type(), COLOR_BG_DARK);
    cv::addWeighted(bottom_bg, 0.70, bottom_roi, 0.30, 0, bottom_roi);

    // Separation accent line
    cv::line(canvas, cv::Point(0, 38), cv::Point(canvas.cols, 38), cv::Scalar(50, 50, 50), 1);

    // Telemetry stats
    std::string stats = cv::format("SYS_FPS: %.1f  |  LATENCY: ~%.1fms", fps, latency_ms);
    cv::putText(canvas, stats, cv::Point(18, 24), cv::FONT_HERSHEY_DUPLEX, 0.50, COLOR_WHITE, 1, cv::LINE_AA);

    // Target lock indicator
    cv::Scalar theme_color = is_detected ? COLOR_CYAN : COLOR_AMBER;
    cv::Point led_dot(canvas.cols - 145, 21);
    cv::circle(canvas, led_dot, 4, theme_color, -1, cv::LINE_AA);

    std::string status_text = is_detected ? "LOCKED" : "SCANNING";
    cv::putText(canvas, status_text, cv::Point(canvas.cols - 130, 25), cv::FONT_HERSHEY_DUPLEX, 0.50, theme_color, 1, cv::LINE_AA);

    // Reticle & Confidence gauge
    drawTargetReticle(canvas, roi, theme_color);
    drawConfidenceBar(canvas, roi, confidence, theme_color);

    // Classification readout
    if (is_detected) {
        std::string clean_label = label;
        auto pos = clean_label.find('_');
        if (pos != std::string::npos) {
            clean_label = clean_label.substr(pos + 1);
        }
        std::transform(clean_label.begin(), clean_label.end(), clean_label.begin(), ::toupper);

        std::string badge = "GESTURE: " + clean_label;
        std::string pct = cv::format("%.0f%%", confidence * 100.0f);

        cv::putText(canvas, badge, cv::Point(roi.x, roi.y - 12), cv::FONT_HERSHEY_DUPLEX, 0.60, COLOR_WHITE, 1, cv::LINE_AA);
        cv::putText(canvas, pct, cv::Point(roi.x + roi.width - 55, roi.y - 12), cv::FONT_HERSHEY_DUPLEX, 0.55, theme_color, 1, cv::LINE_AA);
    } else {
        cv::putText(canvas, "ALIGN HAND TO SENSOR", cv::Point(roi.x + 12, roi.y - 12), cv::FONT_HERSHEY_DUPLEX, 0.45, COLOR_MUTED, 1, cv::LINE_AA);
    }

    cv::putText(canvas, "ROI: 240x240 NCHW", cv::Point(roi.x, roi.y + roi.height + 26), cv::FONT_HERSHEY_DUPLEX, 0.35, COLOR_MUTED, 1, cv::LINE_AA);
    cv::putText(canvas, "SYS_HALT: ESC / 'q'", cv::Point(18, canvas.rows - 8), cv::FONT_HERSHEY_DUPLEX, 0.38, COLOR_MUTED, 1, cv::LINE_AA);
}

int main(int argc, char** argv) {
    try {
        std::cout << "[INFO] Initializing Gesture CV Engine..." << std::endl;

        // Dynamic model/labels path resolution
        std::string model_path = fs::exists("models/gesture_model.onnx") ? "models/gesture_model.onnx" : "../models/gesture_model.onnx";
        std::string labels_path = fs::exists("models/labels.txt") ? "models/labels.txt" : "../models/labels.txt";

        GestureClassifier classifier(model_path, labels_path);
        TemporalSmoother smoother(6);

        constexpr float CONFIDENCE_THRESHOLD = 0.80f;

        // Camera device index configurable via CLI argument (default: 0)
        int camera_index = 0;
        if (argc > 1) {
            try {
                camera_index = std::stoi(argv[1]);
            } catch (...) {
                std::cerr << "[WARN] Invalid camera index provided, falling back to 0." << std::endl;
                camera_index = 0;
            }
        }

        cv::VideoCapture cap(camera_index, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);

        if (!cap.isOpened()) {
            std::cerr << "[WARN] Video capture device " << camera_index << " unavailable. Running simulated loop." << std::endl;
        }

        const std::string window_name = "Gesture CV Engine [C++ / ONNX Runtime]";
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);

        constexpr int CANVAS_WIDTH = 854;
        constexpr int CANVAS_HEIGHT = 480;
        constexpr int OFFSET_X = (CANVAS_WIDTH - 640) / 2;

        cv::Rect raw_roi(200, 100, 240, 240);
        cv::Rect canvas_roi(raw_roi.x + OFFSET_X, raw_roi.y, raw_roi.width, raw_roi.height);

        cv::Mat frame;
        auto prev_time = std::chrono::steady_clock::now();
        double fps = 0.0;
        int empty_frame_count = 0;

        while (true) {
            if (cap.isOpened()) {
                cap >> frame;
                if (frame.empty()) {
                    empty_frame_count++;
                    if (empty_frame_count > 30) {
                        std::cerr << "[ERROR] Camera disconnected or stopped streaming frames." << std::endl;
                        break;
                    }
                    continue;
                }
                empty_frame_count = 0;
                cv::flip(frame, frame, 1);
            } else {
                frame = cv::Mat(480, 640, CV_8UC3, cv::Scalar(25, 25, 25));
            }

            cv::Mat hand_roi = frame(raw_roi);
            DetectionResult raw_result;

            // Presence gating: check texture/variance to prevent false positives on flat walls
            if (isHandPresent(hand_roi, 18.0)) {
                raw_result = classifier.predict(hand_roi);
            } else {
                raw_result.label = "none";
                raw_result.confidence = 0.0f;
            }

            DetectionResult stable_result = smoother.update(raw_result);

            auto current_time = std::chrono::steady_clock::now();
            double duration = std::chrono::duration<double>(current_time - prev_time).count();
            prev_time = current_time;
            if (duration > 0.0) {
                fps = 0.9 * fps + 0.1 * (1.0 / duration);
            }

            double latency_ms = (fps > 0.0) ? (1000.0 / fps) : 0.0;
            bool is_detected = (stable_result.label != "none" && stable_result.confidence >= CONFIDENCE_THRESHOLD);

            cv::Mat canvas(CANVAS_HEIGHT, CANVAS_WIDTH, CV_8UC3, COLOR_BG_DARK);
            frame.copyTo(canvas(cv::Rect(OFFSET_X, 0, frame.cols, frame.rows)));

            renderHUD(canvas, canvas_roi, stable_result.label, stable_result.confidence, fps, latency_ms, is_detected);

            cv::imshow(window_name, canvas);

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