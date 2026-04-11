# Vocal-Guard
Findit

---

# The Complete LFCC Pipeline

1. **Pre-emphasis (Crucial Optimization):** Before framing, you pass the raw audio through a simple first-order high-pass filter. The formula is $y[t] = x[t] - \alpha x[t-1]$, where $\alpha$ is typically 0.97. This balances the frequency spectrum by heavily amplifying the high frequencies, which forces the AI to pay closer attention to the exact region where deepfake artifacts hide.
2. **Framing & Windowing:** Slicing the audio into overlapping 25ms frames and applying a Hann window to prevent edge noise.
3. **Fast Fourier Transform (FFT):** Converting the time-domain frames into the frequency domain.
4. **Linear Filterbank:** Applying a set of linearly spaced triangular filters across the FFT magnitudes. Unlike Mel, this keeps the high-frequency bins separated and distinct.
5. **Logarithmic Compression:** Taking the log of the filterbank energies to mimic human loudness perception and compress the dynamic range.
6. **Discrete Cosine Transform (DCT):** This mathematical step decorrelates the overlapping filterbank energies, yielding the final Linear Frequency Cepstral Coefficients (LFCCs).

# Pre-Processing Optimizations for MobileNetV2

MobileNetV2 expects an image with three color channels (RGB) shaped as `(Batch, 3, 224, 224)`. If you only feed it the 1D grayscale LFCC matrix, you are leaving the model's spatial reasoning capabilities underutilized. 

* **The 3-Channel Trick (Deltas):** Instead of duplicating the grayscale LFCCs three times, calculate the **Delta** (first derivative, or how the audio changes over time) and **Delta-Delta** (second derivative, or the acceleration of the audio changes) of your LFCCs. 
    * Channel 1 (Red): Static LFCCs
    * Channel 2 (Green): Delta LFCCs
    * Channel 3 (Blue): Delta-Delta LFCCs
    This effectively hands MobileNetV2 a map of the audio's temporal dynamics, making it incredibly sensitive to unnatural AI voice transitions.
* **Utterance-Level Normalization:** Standardize your matrices so they have a mean of zero and a variance of one before feeding them to the model. This makes the AI robust against variations in microphone volume.
* **INT8 Quantized:** Train the model in INT8 Quantized form so that we have the absolute model.

# Interoperating Python with High-Performance Rust

You absolutely can, and for a real-time edge deployment, you *must* separate the training logic from the mathematical execution. Python is excellent for orchestrating PyTorch or TensorFlow, but it is far too slow and resource-heavy for live audio buffering on an edge device. 

Rust is uniquely positioned to solve this due to its memory safety, zero-cost abstractions, and excellent interoperability tooling. You can write your expensive LFCC pipeline once in Rust and use it everywhere.

**Phase 1: Model Training (Python + PyO3)**
During training, you want your data pipeline to be as fast as possible to keep your GPU fed. You can write the framing, FFT, and LFCC logic in Rust. By using the **`PyO3`** crate, you can compile that Rust code directly into a native Python module. You `import` it into Python just like any other library, but it executes at bare-metal speeds, drastically accelerating your dataset generation.

**Phase 2: Edge Deployment (Android/Linux Edge)**
When deploying the system, Python is removed entirely. 
* Your trained MobileNetV2 is exported as a quantized **TensorFlow Lite (.tflite)** or **ONNX Runtime Edge** model.
* Your exact same Rust LFCC code is compiled for the target architecture (e.g., using `aarch64-linux-android`).
* If deploying on Android, you use **JNI (Java Native Interface)** via the Android NDK. The native Android audio APIs capture the mic buffer, hand the raw bytes to your Rust library, Rust instantly computes the 3-channel LFCC tensor, and then feeds it directly into the TFLite C++ API for inference. 

This architecture guarantees that the mathematical features the model sees in Python during training are bit-for-bit identical to what it sees on the edge device during production.
