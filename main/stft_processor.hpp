/**
 * @file stft_processor.hpp
 * @brief Real-time STFT Spectrogram Generator using ESP-DSP
 * 
 * This module implements Short-Time Fourier Transform (STFT) for
 * converting time-domain audio samples into frequency-domain spectrograms.
 * 
 * Uses esp-dsp library's optimized FFT functions for ESP32.
 * 
 * Memory Optimization:
 * - Uses sliding window to minimize memory usage
 * - Pre-allocates all buffers during initialization
 * - Avoids dynamic allocation during processing
 */

#pragma once

#include "freertos/FreeRTOS.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_dsp.h"
#include <cstdint>
#include <cmath>
#include <cstring>

namespace pdm {

/**
 * @brief STFT configuration
 */
struct STFTConfig {
    size_t fft_size;        // FFT size (must be power of 2)
    size_t hop_size;        // Hop length between frames
    size_t num_frames;      // Number of time frames in output
    size_t sample_rate;     // Sample rate for frequency calculation
};

/**
 * @brief Default STFT configuration for anomaly detection
 * 
 * FFT Size: 512 samples (32ms at 16kHz) → 256 frequency bins
 * Hop Size: 128 samples (8ms) → 75% overlap
 * Frames: 32 time frames for 32x32 spectrogram output
 */
constexpr STFTConfig DEFAULT_STFT_CONFIG = {
    .fft_size = 512,
    .hop_size = 128,
    .num_frames = 32,
    .sample_rate = 16000
};

/**
 * @class STFTProcessor
 * @brief Sliding Window STFT using esp-dsp
 * 
 * Generates a 2D spectrogram matrix from audio samples.
 * Output shape: (num_freq_bins x num_time_frames)
 * 
 * For anomaly detection, we use magnitude spectrogram normalized to [0, 1].
 */
class STFTProcessor {
public:
    /**
     * @brief Construct STFT processor
     * @param config STFT configuration
     */
    explicit STFTProcessor(const STFTConfig& config = DEFAULT_STFT_CONFIG);
    
    /**
     * @brief Destructor - frees allocated buffers
     */
    ~STFTProcessor();
    
    // Prevent copying
    STFTProcessor(const STFTProcessor&) = delete;
    STFTProcessor& operator=(const STFTProcessor&) = delete;
    
    /**
     * @brief Initialize processor and allocate buffers
     * @return ESP_OK on success
     */
    esp_err_t init();
    
    /**
     * @brief Process audio samples and generate spectrogram
     * 
     * @param audio_samples Input audio buffer (float, normalized [-1, 1])
     * @param num_samples Number of samples in input buffer
     * @param spectrogram Output spectrogram buffer (must be pre-allocated)
     *                    Size: getNumFreqBins() * getNumFrames()
     * @return ESP_OK on success
     * 
     * The spectrogram is stored in row-major order:
     * spectrogram[frame * num_freq_bins + freq_bin]
     */
    esp_err_t process(const float* audio_samples, size_t num_samples, float* spectrogram);
    
    /**
     * @brief Process a single FFT frame
     * 
     * @param audio_frame Input frame (fft_size samples)
     * @param magnitude_output Output magnitude spectrum (fft_size/2 + 1 bins)
     * @return ESP_OK on success
     */
    esp_err_t processFrame(const float* audio_frame, float* magnitude_output);
    
    /**
     * @brief Normalize spectrogram to [0, 1] range
     */
    void normalizeSpectrogram(float* spectrogram, size_t size);
    
    /**
     * @brief Convert to dB scale (optional)
     */
    void convertToDb(float* spectrogram, size_t size, float min_db = -80.0f);
    
    /**
     * @brief Get number of frequency bins in output
     */
    size_t getNumFreqBins() const { return config_.fft_size / 2; }
    
    /**
     * @brief Get number of time frames in output
     */
    size_t getNumFrames() const { return config_.num_frames; }
    
    /**
     * @brief Get required input samples for full spectrogram
     */
    size_t getRequiredSamples() const {
        return config_.fft_size + (config_.num_frames - 1) * config_.hop_size;
    }
    
    /**
     * @brief Get output spectrogram size (total floats)
     */
    size_t getSpectrogramSize() const {
        return getNumFreqBins() * getNumFrames();
    }
    
    /**
     * @brief Get frequency resolution (Hz per bin)
     */
    float getFrequencyResolution() const {
        return static_cast<float>(config_.sample_rate) / config_.fft_size;
    }
    
private:
    static constexpr const char* TAG = "STFT";
    
    STFTConfig config_;
    
    // Pre-allocated buffers (allocated on heap to avoid stack overflow)
    float* fft_input_;          // FFT input buffer (fft_size * 2 for complex)
    float* window_;             // Hann window coefficients
    float* frame_buffer_;       // Windowed frame buffer
    
    bool is_initialized_;
    
    /**
     * @brief Generate Hann window coefficients
     */
    void generateHannWindow();
    
    /**
     * @brief Apply window function to frame
     */
    void applyWindow(const float* input, float* output);
};

// ============================================================================
// Implementation
// ============================================================================

inline STFTProcessor::STFTProcessor(const STFTConfig& config)
    : config_(config)
    , fft_input_(nullptr)
    , window_(nullptr)
    , frame_buffer_(nullptr)
    , is_initialized_(false)
{
}

inline STFTProcessor::~STFTProcessor() {
    if (fft_input_) free(fft_input_);
    if (window_) free(window_);
    if (frame_buffer_) free(frame_buffer_);
}

inline esp_err_t STFTProcessor::init() {
    if (is_initialized_) {
        return ESP_OK;
    }
    
    ESP_LOGI(TAG, "Initializing STFT processor");
    ESP_LOGI(TAG, "  FFT size: %d, Hop size: %d, Frames: %d",
             config_.fft_size, config_.hop_size, config_.num_frames);
    ESP_LOGI(TAG, "  Output shape: %d x %d", getNumFreqBins(), getNumFrames());
    ESP_LOGI(TAG, "  Required input samples: %d", getRequiredSamples());
    
    // Initialize esp-dsp FFT
    esp_err_t ret = dsps_fft2r_init_fc32(nullptr, config_.fft_size);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize FFT: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Allocate buffers on heap (IMPORTANT: avoid stack overflow)
    // FFT input: complex array, size = fft_size * 2
    fft_input_ = static_cast<float*>(heap_caps_malloc(
        config_.fft_size * 2 * sizeof(float), MALLOC_CAP_8BIT));
    if (!fft_input_) {
        ESP_LOGE(TAG, "Failed to allocate FFT input buffer");
        return ESP_ERR_NO_MEM;
    }
    
    // Window coefficients
    window_ = static_cast<float*>(heap_caps_malloc(
        config_.fft_size * sizeof(float), MALLOC_CAP_8BIT));
    if (!window_) {
        ESP_LOGE(TAG, "Failed to allocate window buffer");
        return ESP_ERR_NO_MEM;
    }
    
    // Frame buffer for windowed samples
    frame_buffer_ = static_cast<float*>(heap_caps_malloc(
        config_.fft_size * sizeof(float), MALLOC_CAP_8BIT));
    if (!frame_buffer_) {
        ESP_LOGE(TAG, "Failed to allocate frame buffer");
        return ESP_ERR_NO_MEM;
    }
    
    // Generate Hann window
    generateHannWindow();
    
    is_initialized_ = true;
    ESP_LOGI(TAG, "STFT processor initialized successfully");
    
    return ESP_OK;
}

inline void STFTProcessor::generateHannWindow() {
    // Hann window: w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
    for (size_t n = 0; n < config_.fft_size; n++) {
        window_[n] = 0.5f * (1.0f - cosf(2.0f * M_PI * n / (config_.fft_size - 1)));
    }
}

inline void STFTProcessor::applyWindow(const float* input, float* output) {
    for (size_t i = 0; i < config_.fft_size; i++) {
        output[i] = input[i] * window_[i];
    }
}

inline esp_err_t STFTProcessor::processFrame(const float* audio_frame, float* magnitude_output) {
    if (!is_initialized_) {
        return ESP_ERR_INVALID_STATE;
    }
    
    // Apply Hann window
    applyWindow(audio_frame, frame_buffer_);
    
    // Prepare complex FFT input (real, imag interleaved)
    for (size_t i = 0; i < config_.fft_size; i++) {
        fft_input_[i * 2] = frame_buffer_[i];      // Real part
        fft_input_[i * 2 + 1] = 0.0f;               // Imaginary part
    }
    
    // Perform FFT
    dsps_fft2r_fc32(fft_input_, config_.fft_size);
    
    // Bit-reverse the output
    dsps_bit_rev_fc32(fft_input_, config_.fft_size);
    
    // Calculate magnitude for positive frequencies only (fft_size/2 bins)
    for (size_t i = 0; i < config_.fft_size / 2; i++) {
        float real = fft_input_[i * 2];
        float imag = fft_input_[i * 2 + 1];
        magnitude_output[i] = sqrtf(real * real + imag * imag);
    }
    
    return ESP_OK;
}

inline esp_err_t STFTProcessor::process(const float* audio_samples, size_t num_samples, float* spectrogram) {
    if (!is_initialized_) {
        ESP_LOGE(TAG, "Not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    size_t required = getRequiredSamples();
    if (num_samples < required) {
        ESP_LOGE(TAG, "Insufficient samples: got %d, need %d", num_samples, required);
        return ESP_ERR_INVALID_ARG;
    }
    
    size_t num_freq_bins = getNumFreqBins();
    
    // Process each frame with sliding window
    for (size_t frame = 0; frame < config_.num_frames; frame++) {
        size_t start_idx = frame * config_.hop_size;
        
        // Get pointer to frame start in audio buffer
        const float* frame_start = audio_samples + start_idx;
        
        // Get pointer to output row in spectrogram
        float* output_row = spectrogram + frame * num_freq_bins;
        
        // Process this frame
        esp_err_t ret = processFrame(frame_start, output_row);
        if (ret != ESP_OK) {
            return ret;
        }
    }
    
    return ESP_OK;
}

inline void STFTProcessor::normalizeSpectrogram(float* spectrogram, size_t size) {
    // Find min and max
    float min_val = spectrogram[0];
    float max_val = spectrogram[0];
    
    for (size_t i = 1; i < size; i++) {
        if (spectrogram[i] < min_val) min_val = spectrogram[i];
        if (spectrogram[i] > max_val) max_val = spectrogram[i];
    }
    
    // Normalize to [0, 1]
    float range = max_val - min_val;
    if (range > 0) {
        for (size_t i = 0; i < size; i++) {
            spectrogram[i] = (spectrogram[i] - min_val) / range;
        }
    } else {
        // All values are the same
        memset(spectrogram, 0, size * sizeof(float));
    }
}

inline void STFTProcessor::convertToDb(float* spectrogram, size_t size, float min_db) {
    float reference = 1.0f;  // Reference for dB calculation
    
    for (size_t i = 0; i < size; i++) {
        float db = 20.0f * log10f(spectrogram[i] / reference + 1e-10f);
        
        // Clamp to minimum dB
        if (db < min_db) {
            db = min_db;
        }
        
        spectrogram[i] = db;
    }
}

} // namespace pdm
