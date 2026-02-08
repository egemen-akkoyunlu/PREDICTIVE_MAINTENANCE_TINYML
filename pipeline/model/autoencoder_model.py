"""
Convolutional Autoencoder for Industrial Anomaly Detection
===========================================================

This script implements a lightweight Convolutional Autoencoder (CAE) designed
to run on ESP32 microcontrollers for unsupervised anomaly detection.

Key Design Principles:
1. LIGHTWEIGHT - Fits in ESP32's ~520KB SRAM
2. QUANTIZABLE - Optimized for INT8 quantization
3. UNSUPERVISED - Learns "normal" patterns, detects anomalies via reconstruction error

Anomaly Detection Logic:
- Train on HEALTHY data only
- High reconstruction error = ANOMALY (unknown pattern)
- No labeled fault data required!

Author: AI-Powered Predictive Maintenance Project
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# =============================================================================
# CONFIGURATION - Optimized for ESP32
# =============================================================================

class ModelConfig:
    """
    Model configuration optimized for TinyML on ESP32.
    
    ESP32 Constraints:
    - SRAM: ~520 KB (model + activations must fit)
    - Flash: ~4 MB (model weights stored here)
    - No hardware FPU preference for INT8
    
    Target: Model < 50KB quantized, < 100KB RAM during inference
    """
    
    # Input spectrogram dimensions
    INPUT_HEIGHT = 32         # Frequency bins (mel bands or STFT bins)
    INPUT_WIDTH = 32          # Time frames
    INPUT_CHANNELS = 1        # Grayscale spectrogram
    
    # Encoder architecture (progressively compress)
    ENCODER_FILTERS = [8, 16, 32]     # Lightweight filter counts
    KERNEL_SIZE = (3, 3)
    POOL_SIZE = (2, 2)
    
    # Latent space dimension
    LATENT_DIM = 16           # Compressed representation size
    
    # Training parameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    
    # Anomaly detection threshold (percentile of training reconstruction error)
    THRESHOLD_PERCENTILE = 95


# =============================================================================
# CONVOLUTIONAL AUTOENCODER MODEL
# =============================================================================

def build_encoder(input_shape: tuple, config: ModelConfig = ModelConfig) -> Model:
    """
    Build the Encoder part of the CAE.
    
    Architecture:
    Input (32x32x1) → Conv2D → MaxPool → Conv2D → MaxPool → Conv2D → Flatten → Dense(latent)
    
    Progressively reduces spatial dimensions while increasing feature depth.
    """
    inputs = keras.Input(shape=input_shape, name='encoder_input')
    x = inputs
    
    # Convolutional blocks
    for i, filters in enumerate(config.ENCODER_FILTERS):
        x = layers.Conv2D(
            filters=filters,
            kernel_size=config.KERNEL_SIZE,
            strides=(1, 1),
            padding='same',
            activation='relu',
            name=f'encoder_conv_{i+1}'
        )(x)
        x = layers.MaxPooling2D(
            pool_size=config.POOL_SIZE,
            name=f'encoder_pool_{i+1}'
        )(x)
    
    # Flatten and compress to latent space
    x = layers.Flatten(name='encoder_flatten')(x)
    latent = layers.Dense(
        config.LATENT_DIM,
        activation='relu',
        name='latent_space'
    )(x)
    
    encoder = Model(inputs, latent, name='encoder')
    return encoder


def build_decoder(latent_dim: int, output_shape: tuple, config: ModelConfig = ModelConfig) -> Model:
    """
    Build the Decoder part of the CAE.
    
    Architecture:
    Dense(latent) → Reshape → Conv2DTranspose → UpSample → Conv2DTranspose → Output
    
    Mirrors the encoder to reconstruct the input.
    """
    # Calculate the shape after encoder pooling
    # 32x32 → 16x16 → 8x8 → 4x4 (after 3 pooling layers)
    num_pools = len(config.ENCODER_FILTERS)
    reduced_size = output_shape[0] // (2 ** num_pools)  # 32 / 8 = 4
    
    inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
    
    # Expand from latent space
    x = layers.Dense(
        reduced_size * reduced_size * config.ENCODER_FILTERS[-1],
        activation='relu',
        name='decoder_dense'
    )(inputs)
    x = layers.Reshape(
        (reduced_size, reduced_size, config.ENCODER_FILTERS[-1]),
        name='decoder_reshape'
    )(x)
    
    # Deconvolutional blocks (reverse order of encoder filters)
    decoder_filters = list(reversed(config.ENCODER_FILTERS[:-1])) + [output_shape[-1]]
    
    for i, filters in enumerate(decoder_filters):
        x = layers.Conv2DTranspose(
            filters=config.ENCODER_FILTERS[-(i+1)] if i < len(config.ENCODER_FILTERS)-1 else filters,
            kernel_size=config.KERNEL_SIZE,
            strides=(2, 2),  # Upsampling via strided transpose conv
            padding='same',
            activation='relu' if i < len(decoder_filters) - 1 else 'sigmoid',
            name=f'decoder_deconv_{i+1}'
        )(x)
    
    decoder = Model(inputs, x, name='decoder')
    return decoder


def build_autoencoder(config: ModelConfig = ModelConfig) -> tuple:
    """
    Build the complete Convolutional Autoencoder.
    
    Returns:
    --------
    autoencoder : Model
        Full autoencoder (input → encoder → decoder → reconstruction)
    encoder : Model  
        Encoder only (for extracting latent features)
    decoder : Model
        Decoder only (for generation/visualization)
    """
    input_shape = (config.INPUT_HEIGHT, config.INPUT_WIDTH, config.INPUT_CHANNELS)
    
    # Build encoder
    encoder = build_encoder(input_shape, config)
    
    # Build decoder
    decoder = build_decoder(config.LATENT_DIM, input_shape, config)
    
    # Connect encoder and decoder
    inputs = keras.Input(shape=input_shape, name='autoencoder_input')
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    
    autoencoder = Model(inputs, decoded, name='convolutional_autoencoder')
    
    return autoencoder, encoder, decoder


def compile_model(autoencoder: Model, config: ModelConfig = ModelConfig) -> Model:
    """
    Compile the autoencoder with MSE loss.
    
    MSE (Mean Squared Error) is ideal for reconstruction tasks:
    - Measures pixel-wise difference between input and output
    - Anomalies produce high MSE (model can't reconstruct unknown patterns)
    """
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='mse',  # Mean Squared Error for reconstruction
        metrics=['mae']  # Mean Absolute Error for monitoring
    )
    return autoencoder


# =============================================================================
# ALTERNATIVE: EVEN LIGHTER MODEL FOR VERY CONSTRAINED DEVICES
# =============================================================================

def build_tiny_autoencoder(config: ModelConfig = ModelConfig) -> Model:
    """
    Ultra-lightweight autoencoder for extremely memory-constrained devices.
    
    Uses depthwise separable convolutions to reduce parameters.
    Target: < 10KB quantized model size.
    """
    input_shape = (config.INPUT_HEIGHT, config.INPUT_WIDTH, config.INPUT_CHANNELS)
    inputs = keras.Input(shape=input_shape, name='input')
    
    # Encoder (using separable convolutions)
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)  # 16x16
    x = layers.SeparableConv2D(16, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 8x8
    x = layers.SeparableConv2D(8, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)  # 4x4
    
    # Bottleneck
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu', name='latent')(x)
    
    # Decoder
    x = layers.Dense(4 * 4 * 8, activation='relu')(x)
    x = layers.Reshape((4, 4, 8))(x)
    x = layers.Conv2DTranspose(8, (3, 3), strides=2, padding='same', activation='relu')(x)  # 8x8
    x = layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu')(x) # 16x16
    x = layers.Conv2DTranspose(1, (3, 3), strides=2, padding='same', activation='sigmoid')(x) # 32x32
    
    model = Model(inputs, x, name='tiny_autoencoder')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def create_synthetic_dataset(
    num_samples: int = 1000,
    config: ModelConfig = ModelConfig,
    add_anomalies: bool = False,
    anomaly_ratio: float = 0.1
) -> tuple:
    """
    Create synthetic spectrogram dataset for testing.
    
    Normal patterns: Horizontal lines (motor harmonics)
    Anomalies: Additional random patterns (faults)
    """
    np.random.seed(42)
    
    h, w = config.INPUT_HEIGHT, config.INPUT_WIDTH
    
    # Generate normal spectrograms (horizontal lines = motor harmonics)
    normal_data = np.zeros((num_samples, h, w, 1), dtype=np.float32)
    
    for i in range(num_samples):
        # Add fundamental frequency (random position)
        fundamental = np.random.randint(3, 8)
        normal_data[i, fundamental, :, 0] = 0.8 + 0.2 * np.random.rand(w)
        
        # Add harmonics
        for harmonic in range(2, 5):
            if fundamental * harmonic < h:
                intensity = 0.5 / harmonic
                normal_data[i, fundamental * harmonic, :, 0] = intensity + 0.1 * np.random.rand(w)
        
        # Add noise
        normal_data[i] += 0.05 * np.random.rand(h, w, 1)
    
    labels = np.zeros(num_samples)  # 0 = normal
    
    if add_anomalies:
        num_anomalies = int(num_samples * anomaly_ratio)
        anomaly_data = np.zeros((num_anomalies, h, w, 1), dtype=np.float32)
        
        for i in range(num_anomalies):
            # Anomaly: random blobs + extra frequencies
            anomaly_data[i] = normal_data[i % num_samples].copy()
            
            # Add random fault signature
            fault_freq = np.random.randint(10, 25)
            anomaly_data[i, fault_freq, :, 0] += 0.6 * np.random.rand(w)
            
            # Add broadband noise (e.g., cavitation)
            if np.random.rand() > 0.5:
                anomaly_data[i, 15:25, :, 0] += 0.3 * np.random.rand(10, w)
        
        all_data = np.concatenate([normal_data, anomaly_data], axis=0)
        labels = np.concatenate([np.zeros(num_samples), np.ones(num_anomalies)])
        
        # Shuffle
        indices = np.random.permutation(len(all_data))
        return all_data[indices], labels[indices]
    
    return normal_data, labels


def train_autoencoder(
    autoencoder: Model,
    train_data: np.ndarray,
    config: ModelConfig = ModelConfig
) -> keras.callbacks.History:
    """
    Train the autoencoder on NORMAL data only.
    
    The key insight: We only train on healthy/normal patterns.
    The model learns to reconstruct normal data well.
    Anomalies (unseen patterns) will have HIGH reconstruction error.
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = autoencoder.fit(
        train_data, train_data,  # Input = Output (reconstruction)
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_split=config.VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


# =============================================================================
# ANOMALY DETECTION
# =============================================================================

def calculate_reconstruction_error(autoencoder: Model, data: np.ndarray) -> np.ndarray:
    """
    Calculate per-sample reconstruction error (MSE).
    
    Higher error = more anomalous
    """
    reconstructions = autoencoder.predict(data, verbose=0)
    mse = np.mean(np.square(data - reconstructions), axis=(1, 2, 3))
    return mse


def determine_threshold(
    autoencoder: Model, 
    normal_data: np.ndarray,
    percentile: float = 95
) -> float:
    """
    Determine anomaly threshold from normal training data.
    
    Uses percentile of reconstruction error on normal data.
    Samples with error above this threshold are classified as anomalies.
    """
    errors = calculate_reconstruction_error(autoencoder, normal_data)
    threshold = np.percentile(errors, percentile)
    
    print(f"\n📊 Threshold Calculation:")
    print(f"   Mean reconstruction error: {errors.mean():.6f}")
    print(f"   Std reconstruction error:  {errors.std():.6f}")
    print(f"   {percentile}th percentile threshold: {threshold:.6f}")
    
    return threshold


def detect_anomalies(
    autoencoder: Model,
    data: np.ndarray,
    threshold: float
) -> tuple:
    """
    Detect anomalies based on reconstruction error.
    
    Returns:
    --------
    predictions : np.ndarray
        Binary predictions (0 = normal, 1 = anomaly)
    errors : np.ndarray
        Reconstruction error for each sample
    """
    errors = calculate_reconstruction_error(autoencoder, data)
    predictions = (errors > threshold).astype(int)
    return predictions, errors


# =============================================================================
# MODEL EXPORT FOR ESP32 (TFLite INT8 Quantization)
# =============================================================================

def convert_to_tflite_int8(
    model: Model,
    representative_data: np.ndarray,
    output_path: str = "model_int8.tflite"
) -> str:
    """
    Convert Keras model to TensorFlow Lite with INT8 quantization.
    
    INT8 quantization benefits:
    - 4x smaller model size
    - Faster inference on ESP32 (no FPU needed)
    - Lower power consumption
    
    Parameters:
    -----------
    representative_data : np.ndarray
        Sample of training data for calibration (100-500 samples recommended)
    """
    
    def representative_dataset():
        """Generator for calibration data."""
        for i in range(min(100, len(representative_data))):
            sample = representative_data[i:i+1].astype(np.float32)
            yield [sample]
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Convert
    print("\n🔧 Converting to TFLite INT8...")
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    # Report size
    size_kb = len(tflite_model) / 1024
    print(f"   ✅ Saved to: {output_path}")
    print(f"   📦 Model size: {size_kb:.2f} KB")
    
    return output_path


def convert_to_tflite_float32(
    model: Model,
    output_path: str = "model_float32.tflite"
) -> str:
    """
    Convert to TFLite without quantization (for comparison/debugging).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = len(tflite_model) / 1024
    print(f"\n📦 Float32 model size: {size_kb:.2f} KB")
    
    return output_path


def generate_c_header(tflite_path: str, output_path: str = "model_data.h") -> str:
    """
    Convert TFLite model to C header file for ESP32 embedding.
    
    This creates a byte array that can be included directly in ESP32 firmware.
    """
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    # Generate C array
    c_code = f"""// Auto-generated TFLite model for ESP32
// Model size: {len(model_data)} bytes

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

const unsigned int model_data_len = {len(model_data)};

alignas(8) const uint8_t model_data[] = {{
"""
    
    # Add bytes in rows of 12
    for i in range(0, len(model_data), 12):
        chunk = model_data[i:i+12]
        hex_values = ', '.join(f'0x{b:02x}' for b in chunk)
        c_code += f"    {hex_values},\n"
    
    c_code += """};

#endif // MODEL_DATA_H
"""
    
    with open(output_path, 'w') as f:
        f.write(c_code)
    
    print(f"   ✅ C header saved to: {output_path}")
    return output_path


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_history(history: keras.callbacks.History):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Autoencoder Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_reconstructions(autoencoder: Model, test_data: np.ndarray, num_samples: int = 5):
    """Visualize original vs reconstructed spectrograms."""
    reconstructions = autoencoder.predict(test_data[:num_samples], verbose=0)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    fig.suptitle('Original vs Reconstructed Spectrograms', fontsize=14)
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(test_data[i, :, :, 0], aspect='auto', origin='lower', cmap='magma')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(reconstructions[i, :, :, 0], aspect='auto', origin='lower', cmap='magma')
        mse = np.mean(np.square(test_data[i] - reconstructions[i]))
        axes[1, i].set_title(f'Recon (MSE: {mse:.4f})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_anomaly_detection(errors: np.ndarray, labels: np.ndarray, threshold: float):
    """Visualize anomaly detection results."""
    plt.figure(figsize=(12, 5))
    
    # Scatter plot of reconstruction errors
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    plt.subplot(1, 2, 1)
    plt.scatter(np.where(normal_mask)[0], errors[normal_mask], 
                c='green', alpha=0.5, label='Normal', s=20)
    plt.scatter(np.where(anomaly_mask)[0], errors[anomaly_mask],
                c='red', alpha=0.5, label='Anomaly', s=20)
    plt.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.4f})')
    plt.xlabel('Sample Index')
    plt.ylabel('Reconstruction Error (MSE)')
    plt.title('Anomaly Detection Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Histogram
    plt.subplot(1, 2, 2)
    plt.hist(errors[normal_mask], bins=30, alpha=0.7, label='Normal', color='green')
    plt.hist(errors[anomaly_mask], bins=30, alpha=0.7, label='Anomaly', color='red')
    plt.axvline(x=threshold, color='orange', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN - Complete Pipeline Demonstration
# =============================================================================

def main():
    """
    Complete demonstration of the CAE anomaly detection pipeline.
    """
    print("=" * 70)
    print("  Convolutional Autoencoder for Industrial Anomaly Detection")
    print("  TinyML Pipeline for ESP32")
    print("=" * 70)
    
    # Configuration
    config = ModelConfig()
    
    print(f"\n📐 Model Configuration:")
    print(f"   Input shape: {config.INPUT_HEIGHT}x{config.INPUT_WIDTH}x{config.INPUT_CHANNELS}")
    print(f"   Encoder filters: {config.ENCODER_FILTERS}")
    print(f"   Latent dimension: {config.LATENT_DIM}")
    
    # Build model
    print("\n🏗️  Building Convolutional Autoencoder...")
    autoencoder, encoder, decoder = build_autoencoder(config)
    autoencoder = compile_model(autoencoder, config)
    
    # Model summary
    print("\n📊 Model Architecture:")
    autoencoder.summary()
    
    # Count parameters
    total_params = autoencoder.count_params()
    print(f"\n   Total parameters: {total_params:,}")
    print(f"   Estimated size (float32): {total_params * 4 / 1024:.2f} KB")
    print(f"   Estimated size (int8): {total_params / 1024:.2f} KB")
    
    # Generate synthetic dataset
    print("\n📊 Generating synthetic dataset...")
    train_data, _ = create_synthetic_dataset(num_samples=1000, config=config, add_anomalies=False)
    test_data, test_labels = create_synthetic_dataset(num_samples=200, config=config, 
                                                       add_anomalies=True, anomaly_ratio=0.2)
    
    print(f"   Training samples (normal only): {len(train_data)}")
    print(f"   Test samples (with anomalies): {len(test_data)}")
    print(f"   Test anomalies: {int(test_labels.sum())}")
    
    # Train
    print("\n🎯 Training autoencoder on NORMAL data only...")
    history = train_autoencoder(autoencoder, train_data, config)
    
    # Plot training
    plot_training_history(history)
    
    # Visualize reconstructions
    print("\n🖼️  Visualizing reconstructions...")
    plot_reconstructions(autoencoder, test_data)
    
    # Determine threshold
    threshold = determine_threshold(autoencoder, train_data, config.THRESHOLD_PERCENTILE)
    
    # Detect anomalies
    print("\n🔍 Detecting anomalies in test set...")
    predictions, errors = detect_anomalies(autoencoder, test_data, threshold)
    
    # Calculate metrics
    true_positives = np.sum((predictions == 1) & (test_labels == 1))
    false_positives = np.sum((predictions == 1) & (test_labels == 0))
    true_negatives = np.sum((predictions == 0) & (test_labels == 0))
    false_negatives = np.sum((predictions == 0) & (test_labels == 1))
    
    accuracy = (true_positives + true_negatives) / len(test_labels)
    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    print(f"\n📈 Detection Results:")
    print(f"   Accuracy:  {accuracy:.2%}")
    print(f"   Precision: {precision:.2%}")
    print(f"   Recall:    {recall:.2%}")
    print(f"   F1 Score:  {f1:.2%}")
    
    # Plot results
    plot_anomaly_detection(errors, test_labels, threshold)
    
    # Save and convert model
    print("\n💾 Saving and converting model...")
    
    # Save Keras model
    autoencoder.save('autoencoder_model.keras')
    print("   ✅ Keras model saved: autoencoder_model.keras")
    
    # Convert to TFLite (float32)
    convert_to_tflite_float32(autoencoder, 'model_float32.tflite')
    
    # Convert to TFLite (INT8 quantized)
    convert_to_tflite_int8(autoencoder, train_data, 'model_int8.tflite')
    
    # Generate C header for ESP32
    generate_c_header('model_int8.tflite', 'model_data.h')
    
    # Summary
    print("\n" + "=" * 70)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 70)
    print("""
    Files Generated:
    ────────────────
    • autoencoder_model.keras  - Full Keras model (for Python)
    • model_float32.tflite     - TFLite model (for comparison)
    • model_int8.tflite        - Quantized TFLite (for ESP32)
    • model_data.h             - C header (embed in firmware)
    
    Next Steps for ESP32 Integration:
    ──────────────────────────────────
    1. Copy model_data.h to your ESP32 project
    2. Include TensorFlow Lite Micro library
    3. Load model and run inference on spectrograms
    4. Compare reconstruction error to threshold
    5. If error > threshold → ANOMALY DETECTED!
    
    Deployment Notes:
    ─────────────────
    • Model runs entirely on-device (no cloud needed)
    • Inference time: ~10-50ms on ESP32
    • Memory usage: ~50-100KB RAM
    • Power: Can run on battery with deep sleep
    """)


# Also provide a quick build function for external use
def quick_build_and_train(spectrograms: np.ndarray, epochs: int = 50) -> tuple:
    """
    Quick function to build and train autoencoder on your data.
    
    Parameters:
    -----------
    spectrograms : np.ndarray
        Array of shape (num_samples, height, width, 1)
        Should contain ONLY normal/healthy samples
    
    Returns:
    --------
    autoencoder : Model
        Trained autoencoder
    threshold : float
        Anomaly detection threshold
    """
    # Adjust config to match input shape
    config = ModelConfig()
    config.INPUT_HEIGHT = spectrograms.shape[1]
    config.INPUT_WIDTH = spectrograms.shape[2]
    config.EPOCHS = epochs
    
    # Build and train
    autoencoder, _, _ = build_autoencoder(config)
    autoencoder = compile_model(autoencoder, config)
    train_autoencoder(autoencoder, spectrograms, config)
    
    # Determine threshold
    threshold = determine_threshold(autoencoder, spectrograms)
    
    return autoencoder, threshold


if __name__ == "__main__":
    main()
