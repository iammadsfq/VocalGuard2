import tensorflow as tf
import os

def convert_to_quantized_tflite(model_path, output_name):
    if not os.path.exists(model_path):
        print(f"File {model_path} tidak ditemukan, lewati konversi.")
        return

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    with open(output_name, 'wb') as f:
        f.write(tflite_model)
    print(f"Berhasil: {model_path} -> {output_name} (INT8)")

if __name__ == "__main__":
    print("Menjalankan pipeline konversi Dual-Model...")
    
    convert_to_quantized_tflite('model_la.h5', 'model_la_quantized.tflite')
    convert_to_quantized_tflite('model_pa.h5', 'model_pa_quantized.tflite')