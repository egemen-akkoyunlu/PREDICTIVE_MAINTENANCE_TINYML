"""
Signal Processing Pipeline for Rotating Machinery Fault Detection
==================================================================

This script demonstrates the conversion of raw time-series sensor data
(vibration/audio) into 2D spectrograms suitable for neural network input.

Spectrogram Types:
- STFT Spectrogram (scipy) - Linear frequency scale, good for vibration
- Mel Spectrogram (librosa) - Perceptual scale, excellent for audio/acoustic

Target Application: Industrial rotating machinery (pumps, motors, fans)
Target Frequencies: 50Hz - 1000Hz (motor harmonics, bearing faults)

Author: AI-Powered Predictive Maintenance Project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import os

# Librosa for audio-focused analysis
import librosa
import librosa.display

# =============================================================================
# CONFIGURATION - Optimized for Motor/Bearing Fault Detection
# =============================================================================

class DSPConfig:
    """
    DSP Parameters optimized for rotating machinery analysis.
    
    Key Considerations:
    - Motor fundamental frequency: typically 25-60 Hz (1500-3600 RPM)
    - Bearing fault frequencies: 50-500 Hz
    - Gear mesh frequencies: 100-2000 Hz
    - Cavitation in pumps: 1-10 kHz broadband
    """
    
    # Sampling Configuration
    SAMPLE_RATE = 16000  # Hz - Captures up to 8kHz (Nyquist)
    
    # STFT Parameters (explained in detail below)
    WINDOW_SIZE = 512     # samples (~32ms at 16kHz)
    HOP_LENGTH = 128      # samples (~8ms) - 75% overlap
    N_FFT = 512           # FFT size = window size
    
    # Frequency Range of Interest
    FREQ_MIN = 50         # Hz - Below motor fundamental
    FREQ_MAX = 2000       # Hz - Captures most fault signatures
    
    # Mel Spectrogram Parameters (librosa)
    N_MELS = 64           # Number of Mel bands
    FMIN = 50             # Minimum frequency for Mel filterbank
    FMAX = 4000           # Maximum frequency for Mel filterbank
    
    # Spectrogram Output Size (for NN input)
    TARGET_TIME_FRAMES = 64   # Temporal resolution
    TARGET_FREQ_BINS = 64     # Frequency resolution


# =============================================================================
# SIGNAL SIMULATION - Synthetic Rotating Machinery Signals
# =============================================================================

def generate_healthy_motor_signal(duration_sec: float, sample_rate: int) -> np.ndarray:
    """
    Simulate a healthy rotating motor signal.
    
    Components:
    - Fundamental frequency (50Hz for a 3000 RPM motor)
    - 2nd and 3rd harmonics (normal for any motor)
    - Low-level broadband noise
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    
    # Motor fundamental (50Hz = 3000 RPM / 60)
    fundamental = 50  # Hz
    
    # Healthy motor: strong fundamental, weak harmonics
    signal_healthy = (
        1.0 * np.sin(2 * np.pi * fundamental * t) +           # Fundamental
        0.3 * np.sin(2 * np.pi * 2 * fundamental * t) +       # 2nd harmonic
        0.1 * np.sin(2 * np.pi * 3 * fundamental * t) +       # 3rd harmonic
        0.05 * np.random.randn(len(t))                         # Noise floor
    )
    
    return signal_healthy.astype(np.float32)


def generate_faulty_motor_signal(duration_sec: float, sample_rate: int, 
                                  fault_type: str = "bearing") -> np.ndarray:
    """
    Simulate faulty motor signals with different fault signatures.
    
    Fault Types:
    - bearing: High-frequency components (150-400 Hz) + modulation
    - unbalance: Strong 1x fundamental
    - misalignment: Strong 2x fundamental
    - looseness: Many harmonics + sub-harmonics
    """
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    fundamental = 50  # Hz
    
    # Base healthy signal
    base_signal = (
        1.0 * np.sin(2 * np.pi * fundamental * t) +
        0.3 * np.sin(2 * np.pi * 2 * fundamental * t) +
        0.1 * np.sin(2 * np.pi * 3 * fundamental * t)
    )
    
    if fault_type == "bearing":
        # Bearing defect: characteristic frequency + sidebands
        # BPFO (Ball Pass Frequency Outer) ≈ 3-4x RPM
        bearing_freq = 180  # Hz
        fault_component = (
            0.5 * np.sin(2 * np.pi * bearing_freq * t) *
            (1 + 0.5 * np.sin(2 * np.pi * fundamental * t))  # Amplitude modulation
        )
        # Add high-frequency noise burst (impact)
        fault_component += 0.3 * np.random.randn(len(t)) * np.abs(np.sin(2 * np.pi * bearing_freq * t))
        
    elif fault_type == "unbalance":
        # Unbalance: Excessive 1x vibration
        fault_component = 1.5 * np.sin(2 * np.pi * fundamental * t)
        
    elif fault_type == "misalignment":
        # Misalignment: Strong 2x component
        fault_component = 1.2 * np.sin(2 * np.pi * 2 * fundamental * t)
        
    elif fault_type == "looseness":
        # Mechanical looseness: Multiple harmonics + sub-harmonics
        fault_component = (
            0.4 * np.sin(2 * np.pi * 0.5 * fundamental * t) +  # Sub-harmonic
            0.5 * np.sin(2 * np.pi * 4 * fundamental * t) +
            0.4 * np.sin(2 * np.pi * 5 * fundamental * t) +
            0.3 * np.sin(2 * np.pi * 6 * fundamental * t)
        )
    else:
        fault_component = np.zeros_like(t)
    
    # Add noise
    noise = 0.1 * np.random.randn(len(t))
    
    return (base_signal + fault_component + noise).astype(np.float32)


# =============================================================================
# STFT SPECTROGRAM COMPUTATION
# =============================================================================

def compute_stft_spectrogram(
    signal_data: np.ndarray,
    sample_rate: int,
    window_size: int,
    hop_length: int,
    n_fft: int
) -> tuple:
    """
    Compute Short-Time Fourier Transform (STFT) spectrogram.
    
    Parameters:
    -----------
    signal_data : np.ndarray
        1D time-series signal
    sample_rate : int
        Sampling rate in Hz
    window_size : int
        Number of samples per window (determines frequency resolution)
    hop_length : int
        Number of samples between successive frames (determines time resolution)
    n_fft : int
        FFT size (usually = window_size, can be larger for zero-padding)
    
    Returns:
    --------
    frequencies : np.ndarray
        Frequency bins in Hz
    times : np.ndarray
        Time bins in seconds
    spectrogram : np.ndarray
        2D magnitude spectrogram (frequency x time)
    
    Why These Parameters Matter:
    ----------------------------
    WINDOW SIZE (512 samples @ 16kHz = 32ms):
        - Frequency resolution = sample_rate / window_size = 16000/512 = 31.25 Hz
        - This allows us to distinguish between motor harmonics (50, 100, 150 Hz)
        - Larger window = better frequency resolution but worse time resolution
        - For 50Hz motor, we need at least 1 complete cycle (20ms), so 32ms is good
    
    HOP LENGTH (128 samples @ 16kHz = 8ms):
        - Time resolution = hop_length / sample_rate = 8ms
        - Overlap = 1 - (hop_length/window_size) = 75%
        - Higher overlap = smoother spectrogram, better for transient detection
        - For bearing impacts (brief events), we need good time resolution
    
    For Motor Frequencies (50-1000 Hz):
        - We need frequency bins at least every 25 Hz to separate harmonics
        - Window of 32ms gives us 31.25 Hz resolution ✓
        - Hop of 8ms gives us 125 frames/second for transient detection ✓
    """
    
    # Compute STFT using scipy
    frequencies, times, Zxx = signal.stft(
        signal_data,
        fs=sample_rate,
        window='hann',           # Hann window reduces spectral leakage
        nperseg=window_size,     # Samples per segment
        noverlap=window_size - hop_length,  # Overlap
        nfft=n_fft,
        return_onesided=True     # Only positive frequencies
    )
    
    # Convert to magnitude (dB scale for better visualization)
    magnitude = np.abs(Zxx)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)  # Add epsilon to avoid log(0)
    
    return frequencies, times, magnitude_db


# =============================================================================
# LIBROSA-BASED SPECTROGRAM COMPUTATION (Recommended for Audio)
# =============================================================================

def compute_mel_spectrogram(
    signal_data: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 128,
    n_mels: int = 64,
    fmin: float = 50,
    fmax: float = 4000
) -> tuple:
    """
    Compute Mel-frequency spectrogram using librosa.
    
    Why Mel Spectrogram for Audio?
    ------------------------------
    - Mel scale is perceptually motivated (based on human hearing)
    - Compresses high frequencies, expands low frequencies
    - Reduces dimensionality (e.g., 64 mel bands vs 257 linear bins)
    - Standard input for audio ML (CNN, autoencoders)
    
    For Industrial Acoustic Monitoring:
    - Cavitation noise: broadband high-frequency (1-10 kHz)
    - Bearing rattle: impulsive mid-frequency (200-2000 Hz)
    - Motor hum: low-frequency harmonics (50-500 Hz)
    
    Parameters:
    -----------
    n_mels : int
        Number of Mel bands (typically 64-128 for industrial audio)
    fmin : float
        Minimum frequency (Hz) - set to motor fundamental
    fmax : float
        Maximum frequency (Hz) - set to capture fault signatures
    
    Returns:
    --------
    mel_spectrogram : np.ndarray
        2D Mel spectrogram in dB scale (n_mels x time_frames)
    """
    # Ensure float32 for librosa
    signal_data = signal_data.astype(np.float32)
    
    # Compute Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=signal_data,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0  # Power spectrogram
    )
    
    # Convert to dB scale (log scale)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db


def compute_librosa_stft(
    signal_data: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int = 128
) -> tuple:
    """
    Compute STFT using librosa (alternative to scipy).
    
    Librosa advantages:
    - Consistent API with mel spectrograms
    - Built-in amplitude/power to dB conversion
    - Easy MFCC extraction for audio features
    """
    signal_data = signal_data.astype(np.float32)
    
    # Compute STFT
    stft_matrix = librosa.stft(
        signal_data,
        n_fft=n_fft,
        hop_length=hop_length,
        window='hann'
    )
    
    # Convert to dB
    stft_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)
    
    # Compute frequency and time axes
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    times = librosa.times_like(stft_matrix, sr=sample_rate, hop_length=hop_length)
    
    return frequencies, times, stft_db


def plot_mel_spectrogram(
    mel_spec_db: np.ndarray,
    sample_rate: int,
    hop_length: int,
    title: str = "Mel Spectrogram",
    save_path: str = None
):
    """
    Visualize Mel spectrogram using librosa display.
    """
    plt.figure(figsize=(12, 5))
    
    librosa.display.specshow(
        mel_spec_db,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        cmap='magma'
    )
    
    plt.colorbar(label='dB')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Mel Frequency')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Mel spectrogram to: {save_path}")
    
    plt.show()


def compare_stft_vs_mel(
    signal_data: np.ndarray,
    sample_rate: int,
    title: str = "STFT vs Mel Spectrogram"
):
    """
    Side-by-side comparison of STFT and Mel spectrogram.
    Shows why Mel is often preferred for audio classification.
    """
    # Compute both
    freq, times, stft_db = compute_librosa_stft(
        signal_data, sample_rate,
        DSPConfig.N_FFT, DSPConfig.HOP_LENGTH
    )
    
    mel_db = compute_mel_spectrogram(
        signal_data, sample_rate,
        DSPConfig.N_FFT, DSPConfig.HOP_LENGTH,
        DSPConfig.N_MELS, DSPConfig.FMIN, DSPConfig.FMAX
    )
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # STFT
    img1 = librosa.display.specshow(
        stft_db, sr=sample_rate, hop_length=DSPConfig.HOP_LENGTH,
        x_axis='time', y_axis='hz', ax=axes[0], cmap='viridis'
    )
    axes[0].set_title('STFT Spectrogram (Linear Frequency)')
    axes[0].set_ylim([0, DSPConfig.FREQ_MAX])
    fig.colorbar(img1, ax=axes[0], label='dB')
    
    # Mel
    img2 = librosa.display.specshow(
        mel_db, sr=sample_rate, hop_length=DSPConfig.HOP_LENGTH,
        x_axis='time', y_axis='mel', ax=axes[1], cmap='magma'
    )
    axes[1].set_title(f'Mel Spectrogram ({DSPConfig.N_MELS} bands)')
    fig.colorbar(img2, ax=axes[1], label='dB')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Comparison:")
    print(f"   STFT shape: {stft_db.shape} (freq_bins x time)")
    print(f"   Mel shape:  {mel_db.shape} (mel_bands x time)")
    print(f"   → Mel reduces dimensionality by {stft_db.shape[0]/mel_db.shape[0]:.1f}x")


def extract_frequency_band(
    spectrogram: np.ndarray,
    frequencies: np.ndarray,
    freq_min: float,
    freq_max: float
) -> tuple:
    """
    Extract a specific frequency band from the spectrogram.
    
    For rotating machinery, we typically focus on 50-2000 Hz
    where most fault signatures appear.
    """
    freq_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
    filtered_spectrogram = spectrogram[freq_mask, :]
    filtered_frequencies = frequencies[freq_mask]
    
    return filtered_frequencies, filtered_spectrogram


def normalize_for_nn(spectrogram: np.ndarray) -> np.ndarray:
    """
    Normalize spectrogram for neural network input.
    
    Options:
    1. Min-Max normalization to [0, 1]
    2. Z-score normalization (mean=0, std=1)
    3. Per-frequency normalization
    
    We use min-max for image-like representation.
    """
    min_val = spectrogram.min()
    max_val = spectrogram.max()
    
    if max_val - min_val > 0:
        normalized = (spectrogram - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(spectrogram)
    
    return normalized


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_signal_and_spectrogram(
    signal_data: np.ndarray,
    sample_rate: int,
    spectrogram: np.ndarray,
    frequencies: np.ndarray,
    times: np.ndarray,
    title: str = "Signal Analysis",
    save_path: str = None
):
    """
    Create comprehensive visualization of time-domain signal and spectrogram.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Time-domain signal
    time_axis = np.linspace(0, len(signal_data) / sample_rate, len(signal_data))
    axes[0].plot(time_axis, signal_data, 'b-', linewidth=0.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Time-Domain Signal')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, time_axis[-1]])
    
    # Spectrogram (full frequency range)
    im1 = axes[1].pcolormesh(times, frequencies, spectrogram, 
                              shading='gouraud', cmap='viridis')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title(f'STFT Spectrogram (Full Range: 0-{frequencies[-1]:.0f} Hz)')
    plt.colorbar(im1, ax=axes[1], label='Magnitude (dB)')
    
    # Zoomed spectrogram (motor frequency range)
    freq_mask = frequencies <= DSPConfig.FREQ_MAX
    im2 = axes[2].pcolormesh(times, frequencies[freq_mask], spectrogram[freq_mask, :],
                              shading='gouraud', cmap='magma')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_title(f'Motor Frequency Range ({DSPConfig.FREQ_MIN}-{DSPConfig.FREQ_MAX} Hz)')
    plt.colorbar(im2, ax=axes[2], label='Magnitude (dB)')
    
    # Add horizontal lines for typical fault frequencies
    fault_freqs = [50, 100, 150, 180, 200, 300]  # Motor harmonics + bearing freq
    for freq in fault_freqs:
        if freq <= DSPConfig.FREQ_MAX:
            axes[2].axhline(y=freq, color='white', linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    
    plt.show()


def plot_comparison(healthy_spec, faulty_spec, frequencies, times, fault_type, save_path=None):
    """
    Side-by-side comparison of healthy vs faulty spectrograms.
    This is the "image" that the neural network will learn to distinguish.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Healthy vs {fault_type.title()} Fault - Spectrogram Comparison', 
                 fontsize=14, fontweight='bold')
    
    # Limit to motor frequency range
    freq_mask = frequencies <= DSPConfig.FREQ_MAX
    
    # Healthy
    im1 = axes[0].pcolormesh(times, frequencies[freq_mask], healthy_spec[freq_mask, :],
                              shading='gouraud', cmap='magma', vmin=-60, vmax=0)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('HEALTHY Motor')
    plt.colorbar(im1, ax=axes[0], label='dB')
    
    # Faulty
    im2 = axes[1].pcolormesh(times, frequencies[freq_mask], faulty_spec[freq_mask, :],
                              shading='gouraud', cmap='magma', vmin=-60, vmax=0)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title(f'FAULTY Motor ({fault_type.upper()})')
    plt.colorbar(im2, ax=axes[1], label='dB')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to: {save_path}")
    
    plt.show()


def plot_nn_input(spectrogram: np.ndarray, title: str = "Neural Network Input"):
    """
    Show the final normalized spectrogram that will be fed to the NN.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Normalized Magnitude')
    plt.xlabel('Time Frames')
    plt.ylabel('Frequency Bins')
    plt.title(f'{title}\nShape: {spectrogram.shape} (ready for CNN input)')
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN - Demonstration
# =============================================================================

def main():
    """
    Main demonstration of the signal processing pipeline.
    """
    print("=" * 60)
    print("  Signal Processing Pipeline for Fault Detection")
    print("  STFT Spectrogram Generation")
    print("=" * 60)
    
    # Configuration summary
    print("\n📊 DSP Configuration:")
    print(f"   Sample Rate:     {DSPConfig.SAMPLE_RATE} Hz")
    print(f"   Window Size:     {DSPConfig.WINDOW_SIZE} samples ({1000*DSPConfig.WINDOW_SIZE/DSPConfig.SAMPLE_RATE:.1f} ms)")
    print(f"   Hop Length:      {DSPConfig.HOP_LENGTH} samples ({1000*DSPConfig.HOP_LENGTH/DSPConfig.SAMPLE_RATE:.1f} ms)")
    print(f"   Overlap:         {100*(1-DSPConfig.HOP_LENGTH/DSPConfig.WINDOW_SIZE):.0f}%")
    print(f"   Freq Resolution: {DSPConfig.SAMPLE_RATE/DSPConfig.WINDOW_SIZE:.2f} Hz")
    print(f"   Time Resolution: {1000*DSPConfig.HOP_LENGTH/DSPConfig.SAMPLE_RATE:.1f} ms")
    print(f"   Frequency Range: {DSPConfig.FREQ_MIN}-{DSPConfig.FREQ_MAX} Hz")
    
    # Generate signals
    duration = 2.0  # seconds
    print(f"\n🔊 Generating {duration}s of simulated sensor data...")
    
    healthy_signal = generate_healthy_motor_signal(duration, DSPConfig.SAMPLE_RATE)
    bearing_fault_signal = generate_faulty_motor_signal(duration, DSPConfig.SAMPLE_RATE, "bearing")
    unbalance_signal = generate_faulty_motor_signal(duration, DSPConfig.SAMPLE_RATE, "unbalance")
    
    print(f"   Signal length: {len(healthy_signal)} samples")
    
    # Compute spectrograms
    print("\n📈 Computing STFT spectrograms...")
    
    freq_healthy, time_healthy, spec_healthy = compute_stft_spectrogram(
        healthy_signal, DSPConfig.SAMPLE_RATE,
        DSPConfig.WINDOW_SIZE, DSPConfig.HOP_LENGTH, DSPConfig.N_FFT
    )
    
    freq_bearing, time_bearing, spec_bearing = compute_stft_spectrogram(
        bearing_fault_signal, DSPConfig.SAMPLE_RATE,
        DSPConfig.WINDOW_SIZE, DSPConfig.HOP_LENGTH, DSPConfig.N_FFT
    )
    
    print(f"   Spectrogram shape: {spec_healthy.shape} (freq_bins x time_frames)")
    print(f"   Frequency bins: {len(freq_healthy)} ({freq_healthy[0]:.1f} to {freq_healthy[-1]:.1f} Hz)")
    print(f"   Time frames: {len(time_healthy)} ({time_healthy[0]:.3f} to {time_healthy[-1]:.3f} s)")
    
    # Visualizations
    print("\n🖼️  Generating visualizations...")
    
    # 1. Full analysis of healthy signal
    plot_signal_and_spectrogram(
        healthy_signal, DSPConfig.SAMPLE_RATE,
        spec_healthy, freq_healthy, time_healthy,
        title="Healthy Motor Signal Analysis"
    )
    
    # 2. Comparison: Healthy vs Bearing Fault
    plot_comparison(
        spec_healthy, spec_bearing,
        freq_healthy, time_healthy,
        fault_type="bearing"
    )
    
    # 3. Librosa Mel Spectrogram (for audio)
    print("\n🎵 Computing Mel Spectrograms with librosa...")
    
    mel_healthy = compute_mel_spectrogram(
        healthy_signal, DSPConfig.SAMPLE_RATE,
        DSPConfig.N_FFT, DSPConfig.HOP_LENGTH,
        DSPConfig.N_MELS, DSPConfig.FMIN, DSPConfig.FMAX
    )
    
    print(f"   Mel spectrogram shape: {mel_healthy.shape} (mel_bands x time)")
    
    # Compare STFT vs Mel
    compare_stft_vs_mel(healthy_signal, DSPConfig.SAMPLE_RATE, 
                        title="Healthy Motor: STFT vs Mel Spectrogram")
    
    # Plot Mel spectrogram for bearing fault
    mel_bearing = compute_mel_spectrogram(
        bearing_fault_signal, DSPConfig.SAMPLE_RATE,
        DSPConfig.N_FFT, DSPConfig.HOP_LENGTH,
        DSPConfig.N_MELS, DSPConfig.FMIN, DSPConfig.FMAX
    )
    
    plot_mel_spectrogram(mel_bearing, DSPConfig.SAMPLE_RATE, DSPConfig.HOP_LENGTH,
                         title="Bearing Fault - Mel Spectrogram (NN Input)")
    
    # 4. Prepare for Neural Network
    print("\n🧠 Preparing data for Neural Network input...")
    
    # Extract frequency band of interest
    freq_filtered, spec_filtered = extract_frequency_band(
        spec_healthy, freq_healthy,
        DSPConfig.FREQ_MIN, DSPConfig.FREQ_MAX
    )
    
    # Normalize
    spec_normalized = normalize_for_nn(spec_filtered)
    mel_normalized = normalize_for_nn(mel_healthy)
    
    print(f"   STFT filtered shape: {spec_filtered.shape}")
    print(f"   Mel spectrogram shape: {mel_healthy.shape}")
    print(f"   → Mel is more compact and preferred for audio NN input!")
    
    plot_nn_input(mel_normalized, "Normalized Mel Spectrogram for CNN")
    
    # Summary
    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print("=" * 60)
    print("""
    Window Size & Hop Length Explanation:
    =====================================
    
    For Motor Fault Detection (50-1000 Hz range):
    
    WINDOW SIZE = 512 samples (32ms @ 16kHz)
    ─────────────────────────────────────────
    • Frequency resolution = 16000/512 = 31.25 Hz
    • This lets us separate:
      - 50 Hz fundamental from 100 Hz 2nd harmonic ✓
      - Bearing fault frequencies (~150-400 Hz) ✓
    • Captures at least 1.5 cycles of 50 Hz signal
    • Trade-off: Larger window = better freq resolution 
                 but worse time resolution
    
    HOP LENGTH = 128 samples (8ms @ 16kHz)  
    ─────────────────────────────────────────
    • 75% overlap between windows
    • Time resolution = 8ms (125 frames/second)
    • Good for detecting:
      - Transient impacts from bearing defects
      - Intermittent faults
    • Trade-off: Smaller hop = more frames, smoother 
                 spectrogram, but more computation
    
    The resulting 2D spectrogram is an "image" where:
    • X-axis = Time (captures how faults evolve)
    • Y-axis = Frequency (captures fault signatures)
    • Color = Energy (intensity of each frequency)
    
    This image can be fed into a CNN for classification
    or an Autoencoder for anomaly detection!
    """)


if __name__ == "__main__":
    main()
