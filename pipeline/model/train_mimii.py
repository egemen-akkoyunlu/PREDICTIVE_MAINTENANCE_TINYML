"""
DCASE 2020 / MIMII Dataset Training Pipeline
=============================================

Train the autoencoder on REAL industrial machine audio from the
DCASE 2020 Task 2 development dataset (MIMII-based).

Dataset: https://zenodo.org/records/3678171
Download: dev_data_fan.zip (~1.3 GB)

Usage:
    python train_mimii.py --data_dir path/to/dev_data/fan
    python train_mimii.py --data_dir path/to/dev_data/fan --machine_id id_00
    python train_mimii.py --data_dir path/to/dev_data/fan --max_train 500

Directory structure expected:
    dev_data/fan/
    ├── train/
    │   ├── normal_id_00_00000000.wav
    │   ├── normal_id_02_00000000.wav
    │   └── ...
    └── test/
        ├── normal_id_00_00000000.wav
        ├── anomaly_id_00_00000000.wav
        └── ...
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from glob import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import shared functions from existing training pipeline
from train_esp32_matched import (
    ESP32Config,
    audio_to_nn_input,
    load_wav_file,
    build_autoencoder,
    train_autoencoder,
    calculate_threshold,
    convert_to_tflite_int8,
    generate_c_header,
    plot_training_history,
    plot_reconstructions,
)


# =============================================================================
# DCASE / MIMII DATA LOADING
# =============================================================================

def load_dcase_split(data_dir: str,
                     split: str,
                     machine_id: str = None,
                     condition: str = None,
                     config: ESP32Config = ESP32Config(),
                     max_files: int = None) -> tuple:
    """
    Load DCASE 2020 Task 2 audio files and convert to spectrograms.
    
    Args:
        data_dir: Path to machine type dir (e.g., dev_data/fan)
        split: 'train' or 'test'
        machine_id: Filter by machine ID (e.g., 'id_00'). None = all IDs.
        condition: Filter by 'normal' or 'anomaly'. None = all.
        config: ESP32 configuration
        max_files: Limit number of files loaded
        
    Returns:
        spectrograms: np.ndarray of shape (N, 32, 32, 1)
        labels: np.ndarray of shape (N,) — 0=normal, 1=anomaly
    """
    split_dir = Path(data_dir) / split
    
    if not split_dir.exists():
        print(f"ERROR: Directory not found: {split_dir}")
        sys.exit(1)
    
    # Build file filter pattern
    wav_files = sorted(split_dir.glob("*.wav"))
    
    # Filter by machine ID if specified
    if machine_id:
        wav_files = [f for f in wav_files if machine_id in f.name]
    
    # Filter by condition if specified
    if condition:
        wav_files = [f for f in wav_files if f.name.startswith(condition)]
    
    if max_files:
        wav_files = wav_files[:max_files]
    
    print(f"  Loading {len(wav_files)} files from {split_dir.name}/"
          f"{' ('+machine_id+')' if machine_id else ''}"
          f"{' ('+condition+')' if condition else ''}")
    
    required_samples = config.FFT_SIZE + (config.NUM_FRAMES - 1) * config.HOP_SIZE
    spectrograms = []
    labels = []
    
    for i, wav_path in enumerate(wav_files):
        try:
            audio = load_wav_file(str(wav_path), config.SAMPLE_RATE)
            is_anomaly = 1 if wav_path.name.startswith("anomaly") else 0
            
            # Each WAV is ~10 seconds. Split into multiple spectrogram windows.
            num_specs = len(audio) // required_samples
            for j in range(num_specs):
                segment = audio[j * required_samples:(j + 1) * required_samples]
                spec = audio_to_nn_input(segment, config)
                spectrograms.append(spec)
                labels.append(is_anomaly)
                
        except Exception as e:
            print(f"  Error loading {wav_path.name}: {e}")
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(wav_files)} files "
                  f"({len(spectrograms)} spectrograms)...")
    
    spectrograms = np.array(spectrograms, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    n_normal = np.sum(labels == 0)
    n_anomaly = np.sum(labels == 1)
    print(f"  Result: {len(spectrograms)} spectrograms "
          f"({n_normal} normal, {n_anomaly} anomaly)")
    
    return spectrograms, labels


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def evaluate_model(model, test_data, test_labels, threshold):
    """Compute detection metrics on test set."""
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
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    print(f"\n{'='*50}")
    print(f"  DETECTION RESULTS (DCASE/MIMII)")
    print(f"{'='*50}")
    print(f"  Threshold:  {threshold:.6f}")
    print(f"  Accuracy:   {accuracy:.2%}")
    print(f"  Precision:  {precision:.2%}")
    print(f"  Recall:     {recall:.2%}")
    print(f"  F1 Score:   {f1:.2%}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"{'='*50}")
    
    # MSE distribution
    normal_mse = mse[test_labels == 0]
    anomaly_mse = mse[test_labels == 1]
    print(f"\n  Normal MSE:  mean={np.mean(normal_mse):.6f}  "
          f"std={np.std(normal_mse):.6f}  max={np.max(normal_mse):.6f}")
    if len(anomaly_mse) > 0:
        print(f"  Anomaly MSE: mean={np.mean(anomaly_mse):.6f}  "
              f"std={np.std(anomaly_mse):.6f}  min={np.min(anomaly_mse):.6f}")
    
    return {
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1,
        'threshold': threshold, 'mse': mse
    }


def plot_mse_distribution(mse, labels, threshold, save_path='mse_distribution.png'):
    """Plot MSE histogram for normal vs anomaly samples."""
    import matplotlib.pyplot as plt
    
    normal_mse = mse[labels == 0]
    anomaly_mse = mse[labels == 1]
    
    plt.figure(figsize=(10, 5))
    plt.hist(normal_mse, bins=50, alpha=0.7, label=f'Normal (n={len(normal_mse)})', 
             color='green')
    if len(anomaly_mse) > 0:
        plt.hist(anomaly_mse, bins=50, alpha=0.7, label=f'Anomaly (n={len(anomaly_mse)})', 
                 color='red')
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                label=f'Threshold={threshold:.4f}')
    plt.xlabel('Reconstruction MSE')
    plt.ylabel('Count')
    plt.title('DCASE/MIMII — Normal vs Anomaly MSE Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train autoencoder on DCASE 2020 / MIMII dataset')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to machine dir (e.g., dev_data/fan)')
    parser.add_argument('--machine_id', type=str, default=None,
                        help='Filter by machine ID (e.g., id_00). Default: all IDs')
    parser.add_argument('--max_train', type=int, default=None,
                        help='Max training files to load. Default: all')
    parser.add_argument('--max_test', type=int, default=None,
                        help='Max test files to load. Default: all')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for model files')
    args = parser.parse_args()
    
    config = ESP32Config()
    
    print("=" * 60)
    print("  DCASE/MIMII Autoencoder Training")
    print("  Real Industrial Machine Audio")
    print("=" * 60)
    print(f"\n  Data dir:    {args.data_dir}")
    print(f"  Machine ID:  {args.machine_id or 'all'}")
    print(f"  NN Input:    {config.NN_TIME_FRAMES}x{config.NN_FREQ_BINS}")
    print(f"  Sample Rate: {config.SAMPLE_RATE} Hz")
    
    # -------------------------------------------------------------------------
    # 1. Load training data (normal only)
    # -------------------------------------------------------------------------
    print(f"\n[TRAIN DATA] Loading normal training samples...")
    train_data, train_labels = load_dcase_split(
        args.data_dir, 'train',
        machine_id=args.machine_id,
        condition='normal',
        config=config,
        max_files=args.max_train
    )
    
    if len(train_data) == 0:
        print("ERROR: No training data loaded!")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # 2. Load test data (normal + anomaly)
    # -------------------------------------------------------------------------
    print(f"\n[TEST DATA] Loading test samples (normal + anomaly)...")
    test_data, test_labels = load_dcase_split(
        args.data_dir, 'test',
        machine_id=args.machine_id,
        config=config,
        max_files=args.max_test
    )
    
    # -------------------------------------------------------------------------
    # 3. Build and train
    # -------------------------------------------------------------------------
    print(f"\n[BUILD] Building autoencoder...")
    model = build_autoencoder(config)
    model.summary()
    
    print(f"\n[TRAIN] Training on {len(train_data)} normal spectrograms...")
    history = train_autoencoder(model, train_data, config)
    
    # -------------------------------------------------------------------------
    # 4. Plot training
    # -------------------------------------------------------------------------
    plot_training_history(history)
    
    # -------------------------------------------------------------------------
    # 5. Calculate threshold & evaluate
    # -------------------------------------------------------------------------
    threshold = calculate_threshold(model, train_data, config.THRESHOLD_PERCENTILE)
    
    if len(test_data) > 0:
        results = evaluate_model(model, test_data, test_labels, threshold)
        plot_mse_distribution(results['mse'], test_labels, threshold)
        plot_reconstructions(model, test_data)
    
    # -------------------------------------------------------------------------
    # 6. Export for ESP32
    # -------------------------------------------------------------------------
    print(f"\n[EXPORT] Saving models...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    keras_path = str(output_dir / 'autoencoder_mimii.keras')
    tflite_path = str(output_dir / 'model_int8.tflite')
    header_path = str(output_dir / 'model_data.h')
    
    model.save(keras_path)
    convert_to_tflite_int8(model, train_data, tflite_path)
    generate_c_header(tflite_path, header_path)
    
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"""
    Files Generated:
    ────────────────
    • {keras_path} — Keras model
    • {tflite_path} — Quantized TFLite (for ESP32)
    • {header_path} — C header (embed in firmware)
    • training_history.png — Loss curves
    • mse_distribution.png — Normal vs Anomaly MSE histogram
    • reconstructions.png — Sample reconstructions
    
    ESP32 Threshold:
    ────────────────
    constexpr float ANOMALY_THRESHOLD = {threshold:.4f}f;
    
    Next Steps:
    ───────────
    1. Copy model_data.h to main/model/
    2. Update threshold in anomaly_detector.cpp
    3. Rebuild: idf.py build flash monitor
    """)


if __name__ == "__main__":
    main()
