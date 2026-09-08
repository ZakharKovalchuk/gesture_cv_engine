cmake_minimum_required(VERSION 3.20)
project(gesture_cv_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Поиск OpenCV
find_package(OpenCV REQUIRED)

# 2. Подключение ONNX Runtime из third_party
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.18.0")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIBS "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

# 3. Подключаем заголовочные файлы
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# 4. Исполняемый файл
add_executable(gesture_engine
    src/main.cpp
)

# 5. Линковка библиотек
target_link_libraries(gesture_engine
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBS}
)

# Копируем путь к .so библиотеке при запуске
set_target_properties(gesture_engine PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)
