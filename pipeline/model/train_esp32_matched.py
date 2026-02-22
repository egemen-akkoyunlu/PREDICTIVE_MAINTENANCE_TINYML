"""
ESP32-Matched Training Pipeline
================================

This script trains the autoencoder using the EXACT SAME spectrogram format
as the ESP32 inference code (stft_processor.hpp).

Critical Parameters (must match ESP32):
- FFT Size: 512
- Hop Size: 128  
- Num Frames: 32
- Freq Bins: 32 (first 32 bins of 256)
- Normalization: Min-Max to [0, 1]

Author: Predictive Maintenance Project
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from pathlib import Path
import os
import urllib.request
import zipfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =============================================================================
# ESP32-MATCHED DSP CONFIGURATION
# =============================================================================

class ESP32Config:
    """
    Configuration matching stft_processor.hpp on ESP32.
    DO NOT MODIFY unless you also change ESP32 code!
    """
    # Audio
    SAMPLE_RATE = 16000
    
    # STFT (must match ESP32)
    FFT_SIZE = 512          # 32ms window
    HOP_SIZE = 128          # 8ms hop, 75% overlap
    NUM_FRAMES = 32         # Time frames
    NUM_FREQ_BINS = 256     # FFT_SIZE / 2
    
    # NN Input (cropped from full STFT)
    NN_FREQ_BINS = 32       # First 32 bins (0-1000 Hz)
    NN_TIME_FRAMES = 32     # 32 time frames
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    THRESHOLD_PERCENTILE = 95
    
    # Latent dimension
    LATENT_DIM = 16


# =============================================================================
# ESP32-MATCHED STFT COMPUTATION
# =============================================================================

def compute_esp32_stft(audio: np.ndarray, config: ESP32Config = ESP32Config) -> np.ndarray:
    """
    Compute STFT exactly as ESP32 does it.
    
    This replicates stft_processor.hpp behavior:
    1. Apply Hann window
    2. Compute FFT
    3. Take magnitude of first FFT_SIZE/2 bins
    4. NO log/dB conversion (ESP32 uses linear magnitude)
    
    Returns:
        spectrogram: Shape (NUM_FRAMES, NUM_FREQ_BINS)
    """
    required_samples = config.FFT_SIZE + (config.NUM_FRAMES - 1) * config.HOP_SIZE
    
    if len(audio) < required_samples:
        # Pad with zeros if needed
        audio = np.pad(audio, (0, required_samples - len(audio)))
    
    # Generate Hann window (same as ESP32)
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(config.FFT_SIZE) / (config.FFT_SIZE - 1)))
    
    spectrogram = np.zeros((config.NUM_FRAMES, config.NUM_FREQ_BINS), dtype=np.float32)
    
    for frame in range(config.NUM_FRAMES):
        start_idx = frame * config.HOP_SIZE
        frame_data = audio[start_idx:start_idx + config.FFT_SIZE]
        
        # Apply window
        windowed = frame_data * window
        
        # Compute FFT
        fft_result = np.fft.fft(windowed)
        
        # Take magnitude of positive frequencies (first half)
        magnitude = np.abs(fft_result[:config.NUM_FREQ_BINS])
        
        spectrogram[frame, :] = magnitude
    
    return spectrogram


def crop_for_nn(spectrogram: np.ndarray, config: ESP32Config = ESP32Config) -> np.ndarray:
    """
    Crop spectrogram to match NN input size.
    Takes first 32 frequency bins (0-1000 Hz at 16kHz sample rate).
    
    This matches resizeSpectrogramForNN() in anomaly_detector.cpp
    """
    return spectrogram[:config.NN_TIME_FRAMES, :config.NN_FREQ_BINS]


def normalize_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """
    Min-max normalize to [0, 1].
    Matches STFTProcessor::normalizeSpectrogram() on ESP32.
    """
    min_val = spectrogram.min()
    max_val = spectrogram.max()
    
    if max_val - min_val > 0:
        return (spectrogram - min_val) / (max_val - min_val)
    else:
        return np.zeros_like(spectrogram)


def audio_to_nn_input(audio: np.ndarray, config: ESP32Config = ESP32Config) -> np.ndarray:
    """
    Complete pipeline: audio -> ESP32-matched spectrogram -> NN input
    
    Returns:
        Normalized spectrogram ready for NN, shape (32, 32, 1)
    """
    # Compute STFT (same as ESP32)
    spec = compute_esp32_stft(audio, config)
    
    # Crop to NN size
    spec = crop_for_nn(spec, config)
    
    # Normalize
    spec = normalize_spectrogram(spec)
    
    # Add channel dimension for CNN
    return spec.reshape(config.NN_TIME_FRAMES, config.NN_FREQ_BINS, 1)


# =============================================================================
# SYNTHETIC DATA GENERATION (For Testing)
# =============================================================================

def generate_synthetic_motor_audio(duration_sec: float, 
                                    sample_rate: int,
                                    fundamental_hz: float = 50.0,
                                    add_fault: bool = False,
                                    fault_type: str = "bearing") -> np.ndarray:
    """
    Generate synthetic motor audio that creates STFT patterns
    similar to real rotating machinery.
    """
    num_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    
    # Healthy motor: fundamental + harmonics
    audio = (
        0.5 * np.sin(2 * np.pi * fundamental_hz * t) +
        0.25 * np.sin(2 * np.pi * 2 * fundamental_hz * t) +
        0.1 * np.sin(2 * np.pi * 3 * fundamental_hz * t) +
        0.02 * np.random.randn(num_samples)  # Noise floor
    )
    
    if add_fault:
        if fault_type == "bearing":
            # Bearing fault: modulated high frequency
            bearing_freq = 180
            fault = 0.3 * np.sin(2 * np.pi * bearing_freq * t)
            fault *= (1 + 0.5 * np.sin(2 * np.pi * fundamental_hz * t))
            audio += fault
            
        elif fault_type == "unbalance":
            # Unbalance: excessive 1x
            audio += 0.8 * np.sin(2 * np.pi * fundamental_hz * t)
            
        elif fault_type == "looseness":
            # Looseness: sub-harmonics + noise
            audio += 0.3 * np.sin(2 * np.pi * 0.5 * fundamental_hz * t)
            audio += 0.2 * np.random.randn(num_samples)
    
    return audio.astype(np.float32)


def create_synthetic_dataset(num_normal: int = 500,
                             num_anomaly: int = 100,
                             config: ESP32Config = ESP32Config) -> tuple:
    """
    Create synthetic dataset with ESP32-matched spectrograms.
    
    Returns:
        train_data: Normal samples for training (N, 32, 32, 1)
        test_data: Mixed samples for testing (M, 32, 32, 1)
        test_labels: 0=normal, 1=anomaly
    """
    # Calculate required audio duration for one spectrogram
    required_samples = config.FFT_SIZE + (config.NUM_FRAMES - 1) * config.HOP_SIZE
    duration = required_samples / config.SAMPLE_RATE + 0.1  # Add margin
    
    print(f"Generating {num_normal} normal + {num_anomaly} anomaly samples...")
    
    # Generate normal samples (training data)
    normal_specs = []
    for i in range(num_normal):
        # Vary fundamental frequency slightly for diversity
        fundamental = 45 + np.random.rand() * 15  # 45-60 Hz
        audio = generate_synthetic_motor_audio(duration, config.SAMPLE_RATE, fundamental)
        spec = audio_to_nn_input(audio, config)
        normal_specs.append(spec)
    
    train_data = np.array(normal_specs, dtype=np.float32)
    
    # Generate test set (normal + anomaly)
    test_specs = []
    test_labels = []
    
    # Normal test samples
    for i in range(num_normal // 5):
        fundamental = 45 + np.random.rand() * 15
        audio = generate_synthetic_motor_audio(duration, config.SAMPLE_RATE, fundamental)
        spec = audio_to_nn_input(audio, config)
        test_specs.append(spec)
        test_labels.append(0)
    
    # Anomaly test samples
    fault_types = ["bearing", "unbalance", "looseness"]
    for i in range(num_anomaly):
        fundamental = 45 + np.random.rand() * 15
        fault = fault_types[i % len(fault_types)]
        audio = generate_synthetic_motor_audio(duration, config.SAMPLE_RATE, fundamental,
                                               add_fault=True, fault_type=fault)
        spec = audio_to_nn_input(audio, config)
        test_specs.append(spec)
        test_labels.append(1)
    
    test_data = np.array(test_specs, dtype=np.float32)
    test_labels = np.array(test_labels)
    
    # Shuffle test data
    indices = np.random.permutation(len(test_data))
    test_data = test_data[indices]
    test_labels = test_labels[indices]
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Test anomalies: {test_labels.sum()}")
    
    return train_data, test_data, test_labels


# =============================================================================
# REAL AUDIO DATA LOADING
# =============================================================================

def load_wav_file(filepath: str, target_sr: int = 16000) -> np.ndarray:
    """Load WAV file and resample if needed."""
    import librosa
    audio, sr = librosa.load(filepath, sr=target_sr, mono=True)
    return audio.astype(np.float32)


def load_audio_dataset(audio_dir: str, 
                       config: ESP32Config = ESP32Config,
                       max_samples: int = None) -> np.ndarray:
    """
    Load audio files from directory and convert to ESP32-matched spectrograms.
    
    Args:
        audio_dir: Directory containing WAV files
        config: ESP32 configuration
        max_samples: Maximum number of samples to load
        
    Returns:
        Array of spectrograms (N, 32, 32, 1)
    """
    audio_dir = Path(audio_dir)
    wav_files = list(audio_dir.glob("*.wav"))
    
    if max_samples:
        wav_files = wav_files[:max_samples]
    
    print(f"Found {len(wav_files)} WAV files in {audio_dir}")
    
    required_samples = config.FFT_SIZE + (config.NUM_FRAMES - 1) * config.HOP_SIZE
    spectrograms = []
    
    for i, wav_path in enumerate(wav_files):
        try:
            audio = load_wav_file(str(wav_path), config.SAMPLE_RATE)
            
            # Split long audio into multiple spectrograms
            num_specs = len(audio) // required_samples
            for j in range(num_specs):
                segment = audio[j * required_samples:(j + 1) * required_samples]
                spec = audio_to_nn_input(segment, config)
                spectrograms.append(spec)
                
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(wav_files)} files...")
    
    print(f"Generated {len(spectrograms)} spectrograms")
    return np.array(spectrograms, dtype=np.float32)


# =============================================================================
# AUTOENCODER MODEL (Same architecture, different input)
# =============================================================================

def build_autoencoder(config: ESP32Config = ESP32Config) -> Model:
    """
    Build Convolutional Autoencoder for 32x32 spectrograms.
    Architecture matches the original but input is now linear STFT.
    """
    input_shape = (config.NN_TIME_FRAMES, config.NN_FREQ_BINS, 1)
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Encoder
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)  # 16x16
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 8x8
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 4x4
    
    # Bottleneck
    x = layers.Flatten()(x)
    x = layers.Dense(config.LATENT_DIM, activation='relu', name='latent')(x)
    
    # Decoder
    x = layers.Dense(4 * 4 * 32, activation='relu')(x)
    x = layers.Reshape((4, 4, 32))(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)  # 8x8
    x = layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu')(x)  # 16x16
    x = layers.Conv2DTranspose(1, (3, 3), strides=2, padding='same', activation='sigmoid')(x)  # 32x32
    
    model = Model(inputs, x, name='esp32_matched_autoencoder')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    return model


# =============================================================================
# TRAINING
# =============================================================================

def train_autoencoder(model: Model, 
                      train_data: np.ndarray,
                      config: ESP32Config = ESP32Config) -> keras.callbacks.History:
    """Train autoencoder on normal data only."""
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_data, train_data,  # Reconstruction task
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def calculate_threshold(model: Model, 
                        normal_data: np.ndarray,
                        percentile: float = 95) -> float:
    """Calculate anomaly threshold from normal data reconstruction errors."""
    reconstructions = model.predict(normal_data, verbose=0)
    mse = np.mean(np.square(normal_data - reconstructions), axis=(1, 2, 3))
    threshold = np.percentile(mse, percentile)
    
    print(f"\n[THRESHOLD]")
    print(f"   Mean MSE on normal data: {mse.mean():.6f}")
    print(f"   Std MSE: {mse.std():.6f}")
    print(f"   {percentile}th percentile threshold: {threshold:.6f}")
    
    return threshold


# =============================================================================
# MODEL EXPORT FOR ESP32
# =============================================================================

def convert_to_tflite_int8(model: Model,
                           representative_data: np.ndarray,
                           output_path: str = "model_int8.tflite") -> str:
    """Convert to INT8 quantized TFLite for ESP32."""
    
    def representative_dataset():
        for i in range(min(100, len(representative_data))):
            sample = representative_data[i:i+1].astype(np.float32)
            yield [sample]
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    print("\n[CONVERT] Converting to TFLite INT8...")
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"   [OK] Saved: {output_path}")
    print(f"   [SIZE] {len(tflite_model) / 1024:.2f} KB")
    
    return output_path


def generate_c_header(tflite_path: str, output_path: str = "model_data.h") -> str:
    """Generate C header for ESP32 embedding."""
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    c_code = f"""// Auto-generated TFLite model - ESP32 Matched STFT
// Generated with linear STFT (not Mel spectrogram)
// Model size: {len(model_data)} bytes

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

const unsigned int model_data_len = {len(model_data)};

alignas(8) const uint8_t model_data[] = {{
"""
    
    for i in range(0, len(model_data), 12):
        chunk = model_data[i:i+12]
        hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
        c_code += f"    {hex_values},\n"
    
    c_code += """};

#endif // MODEL_DATA_H
"""
    
    with open(output_path, 'w') as f:
        f.write(c_code)
    
    print(f"   [OK] C header: {output_path}")
    return output_path


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_history(history):
    """Plot training curves."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train')
    plt.plot(history.history['val_mae'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()


def plot_reconstructions(model, test_data, num_samples=5):
    """Visualize original vs reconstructed spectrograms."""
    reconstructions = model.predict(test_data[:num_samples], verbose=0)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    
    for i in range(num_samples):
        axes[0, i].imshow(test_data[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        mse = np.mean(np.square(test_data[i] - reconstructions[i]))
        axes[1, i].imshow(reconstructions[i, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
        axes[1, i].set_title(f'MSE: {mse:.4f}')
        axes[1, i].axis('off')
    
    plt.suptitle('ESP32-Matched Linear STFT Spectrograms', fontsize=14)
    plt.tight_layout()
    plt.savefig('reconstructions.png', dpi=150)
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  ESP32-Matched Autoencoder Training")
    print("  Using Linear STFT (NOT Mel Spectrogram)")
    print("=" * 70)
    
    config = ESP32Config()
    
    print(f"\n[CONFIG] ESP32-Matched Configuration:")
    print(f"   FFT Size: {config.FFT_SIZE}")
    print(f"   Hop Size: {config.HOP_SIZE}")
    print(f"   Num Frames: {config.NUM_FRAMES}")
    print(f"   Freq Bins (full): {config.NUM_FREQ_BINS}")
    print(f"   NN Input: {config.NN_TIME_FRAMES}x{config.NN_FREQ_BINS}")
    
    # Generate synthetic data
    print("\n[DATA] Creating synthetic dataset...")
    train_data, test_data, test_labels = create_synthetic_dataset(
        num_normal=800,
        num_anomaly=100,
        config=config
    )
    
    # Build model
    print("\n[BUILD] Building autoencoder...")
    model = build_autoencoder(config)
    model.summary()
    
    # Train
    print("\n[TRAIN] Training on normal data only...")
    history = train_autoencoder(model, train_data, config)
    
    # Plot training
    plot_training_history(history)
    
    # Calculate threshold
    threshold = calculate_threshold(model, train_data, config.THRESHOLD_PERCENTILE)
    
    # Evaluate on test set
    print("\n[EVAL] Evaluating on test set...")
    reconstructions = model.predict(test_data, verbose=0)
    mse = np.mean(np.square(test_data - reconstructions), axis=(1, 2, 3))
    predictions = (mse > threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (test_labels == 1))
    fp = np.sum((predictions == 1) & (test_labels == 0))
    tn = np.sum((predictions == 0) & (test_labels == 0))
    fn = np.sum((predictions == 0) & (test_labels == 1))
    
    accuracy = (tp + tn) / len(test_labels)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    print(f"\n[RESULTS]")
    print(f"   Accuracy:  {accuracy:.2%}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall:    {recall:.2%}")
    print(f"   Threshold: {threshold:.6f}")
    
    # Visualize
    plot_reconstructions(model, test_data)
    
    # Save models
    print("\n[SAVE] Saving models...")
    model.save('autoencoder_esp32_matched.keras')
    
    # Convert to TFLite
    convert_to_tflite_int8(model, train_data, 'model_int8.tflite')
    generate_c_header('model_int8.tflite', 'model_data.h')
    
    print("\n" + "=" * 70)
    print("  [COMPLETE] TRAINING COMPLETE")
    print("=" * 70)
    print(f"""
    Files Generated:
    ────────────────
    • autoencoder_esp32_matched.keras - Keras model
    • model_int8.tflite - Quantized TFLite (for ESP32)
    • model_data.h - C header (embed in firmware)
    
    IMPORTANT: Update anomaly_detector.cpp threshold:
    ──────────────────────────────────────────────────
    constexpr float ANOMALY_THRESHOLD = {threshold:.4f}f;
    
    Copy model_data.h to your ESP32 project 'main/' directory
    and rebuild the firmware!
    """)


if __name__ == "__main__":
    main()
