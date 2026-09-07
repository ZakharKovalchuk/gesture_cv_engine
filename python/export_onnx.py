import os
import sys
from pathlib import Path

# Отключаем логи CUDA/GPU, используем чистый CPU для конвертации
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import tf2onnx
import onnxruntime as ort

MODEL_H5_PATH = "models/gesture_model.h5"
OUTPUT_ONNX_PATH = "models/gesture_model.onnx"

def main():
    h5_path = Path(MODEL_H5_PATH)
    if not h5_path.exists():
        print(f"[ERROR] Source model not found: {MODEL_H5_PATH}")
        sys.exit(1)

    print(f"[INFO] Loading Keras model from {h5_path}...")
    model = tf.keras.models.load_model(h5_path)

    # Параметры из summary: 128x128x3, batch=1
    target_dims = (1, 128, 128, 3)
    input_signature = [tf.TensorSpec(target_dims, tf.float32, name="input_tensor")]

    @tf.function(input_signature=input_signature)
    def model_func(input_tensor):
        return model(input_tensor, training=False)

    print(f"[INFO] Converting graph via tf2onnx (Opset 13)...")
    # Передаем саму model_func, tf2onnx сам извлечет concrete function
    tf2onnx.convert.from_function(
        model_func,
        input_signature=input_signature,
        opset=13,
        output_path=OUTPUT_ONNX_PATH
    )
    print(f"[SUCCESS] ONNX model saved to {OUTPUT_ONNX_PATH}")

    # Валидация модели через onnxruntime
    print("[INFO] Verifying ONNX graph via onnxruntime...")
    session = ort.InferenceSession(OUTPUT_ONNX_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dummy_input = np.random.randn(*target_dims).astype(np.float32)
    result = session.run([output_name], {input_name: dummy_input})

    print(f"[SUCCESS] Input tensor name: {input_name}")
    print(f"[SUCCESS] Output tensor name: {output_name}")
    print(f"[SUCCESS] Test run passed! Output shape: {result[0].shape}")

if __name__ == "__main__":
    main()