/**
 * @file i2s_microphone.hpp
 * @brief I2S Microphone Driver for INMP441
 * 
 * This driver captures audio from an INMP441 digital MEMS microphone
 * using the ESP32's I2S peripheral with DMA for efficient buffering.
 * 
 * Hardware Connection:
 * - INMP441 VDD  → 3.3V
 * - INMP441 GND  → GND
 * - INMP441 SD   → GPIO32 (Data In)
 * - INMP441 SCK  → GPIO33 (Bit Clock)
 * - INMP441 WS   → GPIO25 (Word Select / LRCLK)
 * - INMP441 L/R  → GND (Left channel) or VDD (Right channel)
 */

#pragma once

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2s_std.h"
#include "esp_log.h"
#include "esp_err.h"
#include <cstdint>
#include <cstring>

namespace pdm {

/**
 * @brief I2S Microphone configuration
 */
struct I2SMicConfig {
    gpio_num_t bck_pin;         // Bit Clock pin
    gpio_num_t ws_pin;          // Word Select (LRCLK) pin
    gpio_num_t data_pin;        // Data (SD) pin
    uint32_t sample_rate;       // Sample rate in Hz
    size_t dma_buffer_count;    // Number of DMA buffers
    size_t dma_buffer_size;     // Size of each DMA buffer in samples
    i2s_port_t i2s_port;        // I2S port number
};

/**
 * @brief Default configuration for INMP441
 */
constexpr I2SMicConfig DEFAULT_MIC_CONFIG = {
    .bck_pin = GPIO_NUM_33,
    .ws_pin = GPIO_NUM_25,
    .data_pin = GPIO_NUM_32,
    .sample_rate = 16000,
    .dma_buffer_count = 8,
    .dma_buffer_size = 1024,
    .i2s_port = I2S_NUM_0
};

/**
 * @class I2SMicrophone
 * @brief Driver for INMP441 I2S MEMS Microphone
 * 
 * Features:
 * - Configurable sample rate (8000-48000 Hz)
 * - DMA-based double buffering for continuous capture
 * - 24-bit audio data from INMP441
 * - Conversion to normalized float samples
 */
class I2SMicrophone {
public:
    /**
     * @brief Construct I2S Microphone driver
     * @param config Configuration parameters
     */
    explicit I2SMicrophone(const I2SMicConfig& config = DEFAULT_MIC_CONFIG);
    
    /**
     * @brief Destructor - releases I2S resources
     */
    ~I2SMicrophone();
    
    // Prevent copying
    I2SMicrophone(const I2SMicrophone&) = delete;
    I2SMicrophone& operator=(const I2SMicrophone&) = delete;
    
    /**
     * @brief Initialize I2S peripheral
     * @return ESP_OK on success
     */
    esp_err_t init();
    
    /**
     * @brief Start audio capture
     * @return ESP_OK on success
     */
    esp_err_t start();
    
    /**
     * @brief Stop audio capture
     * @return ESP_OK on success
     */
    esp_err_t stop();
    
    /**
     * @brief Read raw audio samples (blocking)
     * @param buffer Destination buffer for int32_t samples
     * @param num_samples Number of samples to read
     * @param timeout_ms Timeout in milliseconds
     * @return Number of samples actually read
     */
    size_t readRaw(int32_t* buffer, size_t num_samples, uint32_t timeout_ms = 1000);
    
    /**
     * @brief Read normalized float samples [-1.0, 1.0]
     * @param buffer Destination buffer for float samples
     * @param num_samples Number of samples to read
     * @param timeout_ms Timeout in milliseconds
     * @return Number of samples actually read
     */
    size_t readFloat(float* buffer, size_t num_samples, uint32_t timeout_ms = 1000);
    
    /**
     * @brief Get current sample rate
     */
    uint32_t getSampleRate() const { return config_.sample_rate; }
    
    /**
     * @brief Check if driver is initialized and running
     */
    bool isRunning() const { return is_running_; }
    
private:
    static constexpr const char* TAG = "I2S_MIC";
    
    // INMP441 outputs 24-bit data in 32-bit frame
    static constexpr int BITS_PER_SAMPLE = 32;
    static constexpr int INMP441_DATA_BITS = 24;
    
    // Normalization factor for 24-bit signed audio
    static constexpr float NORMALIZE_FACTOR = 1.0f / (1 << (INMP441_DATA_BITS - 1));
    
    I2SMicConfig config_;
    i2s_chan_handle_t rx_channel_;
    bool is_initialized_;
    bool is_running_;
    
    // Pre-allocated buffer for readFloat conversion (avoids per-call malloc)
    int32_t* raw_buffer_;
    size_t raw_buffer_size_;
    
    /**
     * @brief Configure I2S channel
     */
    esp_err_t configureChannel();
};

// ============================================================================
// Implementation
// ============================================================================

inline I2SMicrophone::I2SMicrophone(const I2SMicConfig& config)
    : config_(config)
    , rx_channel_(nullptr)
    , is_initialized_(false)
    , is_running_(false)
    , raw_buffer_(nullptr)
    , raw_buffer_size_(0)
{
}

inline I2SMicrophone::~I2SMicrophone() {
    if (is_running_) {
        stop();
    }
    if (is_initialized_ && rx_channel_ != nullptr) {
        i2s_del_channel(rx_channel_);
    }
    if (raw_buffer_) {
        free(raw_buffer_);
    }
}

inline esp_err_t I2SMicrophone::init() {
    if (is_initialized_) {
        ESP_LOGW(TAG, "Already initialized");
        return ESP_OK;
    }
    
    ESP_LOGI(TAG, "Initializing I2S microphone on port %d", config_.i2s_port);
    ESP_LOGI(TAG, "  BCK: GPIO%d, WS: GPIO%d, DATA: GPIO%d",
             config_.bck_pin, config_.ws_pin, config_.data_pin);
    ESP_LOGI(TAG, "  Sample rate: %lu Hz", config_.sample_rate);
    
    // Channel configuration
    i2s_chan_config_t chan_cfg = {
        .id = config_.i2s_port,
        .role = I2S_ROLE_MASTER,
        .dma_desc_num = static_cast<uint32_t>(config_.dma_buffer_count),
        .dma_frame_num = static_cast<uint32_t>(config_.dma_buffer_size),
        .auto_clear = true,
    };
    
    esp_err_t ret = i2s_new_channel(&chan_cfg, nullptr, &rx_channel_);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create I2S channel: %s", esp_err_to_name(ret));
        return ret;
    }
    
    // Configure as standard I2S
    ret = configureChannel();
    if (ret != ESP_OK) {
        i2s_del_channel(rx_channel_);
        rx_channel_ = nullptr;
        return ret;
    }
    
    is_initialized_ = true;
    ESP_LOGI(TAG, "I2S microphone initialized successfully");
    
    return ESP_OK;
}

inline esp_err_t I2SMicrophone::configureChannel() {
    // Standard I2S configuration for INMP441
    // Use macros for ESP-IDF v5.5+ compatibility
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(config_.sample_rate),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = config_.bck_pin,
            .ws = config_.ws_pin,
            .dout = I2S_GPIO_UNUSED,
            .din = config_.data_pin,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv = false,
            },
        },
    };
    
    // Configure for left channel only (INMP441 L/R pin to GND)
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
    
    esp_err_t ret = i2s_channel_init_std_mode(rx_channel_, &std_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure I2S channel: %s", esp_err_to_name(ret));
    }
    
    return ret;
}

inline esp_err_t I2SMicrophone::start() {
    if (!is_initialized_) {
        ESP_LOGE(TAG, "Not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    if (is_running_) {
        return ESP_OK;
    }
    
    esp_err_t ret = i2s_channel_enable(rx_channel_);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to enable I2S channel: %s", esp_err_to_name(ret));
        return ret;
    }
    
    is_running_ = true;
    ESP_LOGI(TAG, "I2S capture started");
    
    return ESP_OK;
}

inline esp_err_t I2SMicrophone::stop() {
    if (!is_running_) {
        return ESP_OK;
    }
    
    esp_err_t ret = i2s_channel_disable(rx_channel_);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to disable I2S channel: %s", esp_err_to_name(ret));
        return ret;
    }
    
    is_running_ = false;
    ESP_LOGI(TAG, "I2S capture stopped");
    
    return ESP_OK;
}

inline size_t I2SMicrophone::readRaw(int32_t* buffer, size_t num_samples, uint32_t timeout_ms) {
    if (!is_running_) {
        ESP_LOGW(TAG, "Not running, cannot read");
        return 0;
    }
    
    size_t bytes_to_read = num_samples * sizeof(int32_t);
    size_t bytes_read = 0;
    
    esp_err_t ret = i2s_channel_read(rx_channel_, buffer, bytes_to_read, 
                                      &bytes_read, pdMS_TO_TICKS(timeout_ms));
    
    if (ret != ESP_OK && ret != ESP_ERR_TIMEOUT) {
        ESP_LOGE(TAG, "I2S read error: %s", esp_err_to_name(ret));
        return 0;
    }
    
    // INMP441 data is in upper 24 bits of 32-bit word
    // Shift right to get proper signed value
    size_t samples_read = bytes_read / sizeof(int32_t);
    for (size_t i = 0; i < samples_read; i++) {
        buffer[i] = buffer[i] >> 8;  // Shift to get 24-bit signed value
    }
    
    return samples_read;
}

inline size_t I2SMicrophone::readFloat(float* buffer, size_t num_samples, uint32_t timeout_ms) {
    // Ensure pre-allocated buffer is large enough
    if (raw_buffer_ == nullptr || num_samples > raw_buffer_size_) {
        if (raw_buffer_) free(raw_buffer_);
        raw_buffer_ = static_cast<int32_t*>(malloc(num_samples * sizeof(int32_t)));
        if (raw_buffer_ == nullptr) {
            ESP_LOGE(TAG, "Failed to allocate raw buffer");
            raw_buffer_size_ = 0;
            return 0;
        }
        raw_buffer_size_ = num_samples;
    }
    
    size_t samples_read = readRaw(raw_buffer_, num_samples, timeout_ms);
    
    // Convert to normalized float
    for (size_t i = 0; i < samples_read; i++) {
        buffer[i] = static_cast<float>(raw_buffer_[i]) * NORMALIZE_FACTOR;
    }
    
    return samples_read;
}

} // namespace pdm
