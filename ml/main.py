import tensorflow as tf
from model import build_deepfake_detector

def save_quantized_model(keras_model_path):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    with open('model_quantized.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model berhasil dikonversi ke INT8 TFLite!")

if __name__ == "__main__":
    print("Menjalankan pipeline konversi ke Edge...")
    # save_quantized_model('model.h5')