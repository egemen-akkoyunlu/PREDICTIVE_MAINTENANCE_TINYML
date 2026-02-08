/**
 * @file anomaly_detector.cpp
 * @brief Self-Calibrating Anomaly Detection System
 * 
 * This system automatically calibrates to any machine:
 * 1. CALIBRATION PHASE: First N samples learn the baseline
 * 2. MONITORING PHASE: Detects anomalies using learned threshold
 * 
 * Features:
 * - Auto-calibration on first boot or button press
 * - Saves calibration to NVS flash (survives reboot)  
 * - Adaptive threshold: baseline_mean + sensitivity * baseline_std
 * - Works with any rotating machinery without retraining model
 * 
 * Hardware:
 * - ESP32 DevKit
 * - INMP441 I2S Microphone (BCK=33, WS=25, SD=32)
 * - MPU6050 Accelerometer (SDA=21, SCL=22) - optional
 * - LED on GPIO2 for status indication
 */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "driver/gpio.h"
#include "nvs_flash.h"
#include "nvs.h"

// Project headers
#include "drivers/i2s_microphone.hpp"
#include "dsp/stft_processor.hpp"
#include "model/tflite_inference.hpp"
#include "drivers/mpu6050_watchdog.hpp"
#include "drivers/sd_card_logger.hpp"

// Include the embedded TFLite model
#include "model/model_data.h"

static const char* TAG = "AnomalyDetector";

// =============================================================================
// CONFIGURATION
// =============================================================================

namespace config {

    // GPIO Configuration
    constexpr gpio_num_t LED_ANOMALY_PIN = GPIO_NUM_2;       // Built-in LED
    constexpr gpio_num_t CALIBRATE_BUTTON_PIN = GPIO_NUM_0;  // BOOT button for recalibration
    
    // I2S Microphone Pins (INMP441)
    constexpr gpio_num_t I2S_BCK_PIN = GPIO_NUM_33;
    constexpr gpio_num_t I2S_WS_PIN = GPIO_NUM_25;
    constexpr gpio_num_t I2S_DATA_PIN = GPIO_NUM_32;
    
    // MPU6050 I2C Pins
    constexpr gpio_num_t MPU_SDA_PIN = GPIO_NUM_21;
    constexpr gpio_num_t MPU_SCL_PIN = GPIO_NUM_22;
    constexpr i2c_port_t MPU_I2C_PORT = I2C_NUM_0;
    
    // Audio Configuration 
    constexpr uint32_t SAMPLE_RATE = 16000;
    constexpr size_t AUDIO_BUFFER_SIZE = 8192;
    
    // STFT Configuration
    constexpr size_t FFT_SIZE = 512;
    constexpr size_t HOP_SIZE = 128;
    constexpr size_t NUM_FRAMES = 32;
    constexpr size_t NUM_FREQ_BINS = FFT_SIZE / 2;
    
    // Spectrogram dimensions for NN
    constexpr size_t SPEC_HEIGHT = 32;
    constexpr size_t SPEC_WIDTH = 32;
    
    // TFLite Configuration  
    constexpr size_t TENSOR_ARENA_SIZE = 80 * 1024;
    
    // =========================================================================
    // CALIBRATION SETTINGS
    // =========================================================================
    
    // Number of samples to collect during calibration
    constexpr size_t CALIBRATION_SAMPLES = 60;           // ~90 seconds @ 1.5s interval
    
    // Sensitivity multiplier for threshold
    // threshold = mean + (SENSITIVITY * std_dev)
    // Lower = more sensitive, Higher = fewer false positives
    constexpr float SENSITIVITY = 2.0f;                  // 2 sigma rule
    
    // Minimum threshold (prevents too-sensitive detection)
    constexpr float MIN_THRESHOLD_OFFSET = 0.01f;        // At least 0.01 above mean
    
    // NVS storage namespace
    constexpr const char* NVS_NAMESPACE = "anomaly_cal";
    
    // =========================================================================
    // MPU6050 Settings (optional)
    // =========================================================================
    
    constexpr bool ENABLE_MPU6050 = true;               // MPU6050 is now working!
    constexpr float ACCEL_RMS_THRESHOLD = 0.5f;
    constexpr float ACCEL_PEAK_THRESHOLD = 1.5f;
    constexpr size_t ACCEL_SAMPLES_PER_DETECTION = 50;
    constexpr uint32_t ACCEL_SAMPLE_DELAY_MS = 5;
    
    // Detection Loop
    constexpr uint32_t DETECTION_INTERVAL_MS = 1500;
    constexpr uint32_t ANOMALY_LED_DURATION_MS = 2000;
    constexpr size_t MAIN_TASK_STACK_SIZE = 10240;

} // namespace config

// =============================================================================
// CALIBRATION DATA STRUCTURE
// =============================================================================

struct CalibrationData {
    float baseline_mean;      // Average MSE during normal operation
    float baseline_std;       // Standard deviation of MSE
    float threshold;          // Calculated audio anomaly threshold
    float vib_baseline_rms;   // Baseline RMS from accelerometer
    float vib_threshold;      // Vibration RMS threshold (baseline + offset)
    bool is_valid;            // True if calibration is complete
    uint32_t sample_count;    // Number of samples used for calibration
};

// =============================================================================
// VIBRATION FEATURES
// =============================================================================

struct VibrationFeatures {
    float rms;
    float peak;
    float crest_factor;
    bool is_fault;
};

// =============================================================================
// GLOBAL STATE
// =============================================================================

static float* g_audio_buffer = nullptr;
static float* g_spectrogram = nullptr;
static float* g_nn_input = nullptr;

static CalibrationData g_calibration = {0};
static bool g_is_calibrating = false;
static float* g_calibration_samples = nullptr;      // Audio MSE samples
static float* g_vib_calibration_samples = nullptr;  // Vibration RMS samples
static size_t g_calibration_index = 0;

static bool g_anomaly_detected = false;
static TickType_t g_anomaly_time = 0;
static uint32_t g_detection_count = 0;
static uint32_t g_anomaly_count = 0;

// =============================================================================
// NVS STORAGE FUNCTIONS
// =============================================================================

/**
 * @brief Save calibration data to NVS flash
 */
static esp_err_t saveCalibrationToNVS(const CalibrationData& cal) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open(config::NVS_NAMESPACE, NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS: %s", esp_err_to_name(err));
        return err;
    }
    
    // Save as blob
    err = nvs_set_blob(handle, "cal_data", &cal, sizeof(CalibrationData));
    if (err == ESP_OK) {
        err = nvs_commit(handle);
    }
    
    nvs_close(handle);
    
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Calibration saved to flash");
    }
    return err;
}

/**
 * @brief Load calibration data from NVS flash
 */
static esp_err_t loadCalibrationFromNVS(CalibrationData& cal) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open(config::NVS_NAMESPACE, NVS_READONLY, &handle);
    if (err != ESP_OK) {
        return err;  // Not found is normal on first boot
    }
    
    size_t size = sizeof(CalibrationData);
    err = nvs_get_blob(handle, "cal_data", &cal, &size);
    
    nvs_close(handle);
    return err;
}

/**
 * @brief Clear calibration from NVS (for recalibration)
 */
static esp_err_t clearCalibrationNVS() {
    nvs_handle_t handle;
    esp_err_t err = nvs_open(config::NVS_NAMESPACE, NVS_READWRITE, &handle);
    if (err != ESP_OK) return err;
    
    nvs_erase_key(handle, "cal_data");
    nvs_commit(handle);
    nvs_close(handle);
    
    ESP_LOGI(TAG, "Calibration cleared from flash");
    return ESP_OK;
}

// =============================================================================
// CALIBRATION FUNCTIONS
// =============================================================================

/**
 * @brief Start calibration mode
 */
static void startCalibration() {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  STARTING DUAL-SENSOR CALIBRATION");
    ESP_LOGI(TAG, "  Collecting %d samples (~%d seconds)", 
             config::CALIBRATION_SAMPLES,
             (config::CALIBRATION_SAMPLES * config::DETECTION_INTERVAL_MS) / 1000);
    ESP_LOGI(TAG, "  Keep machine running NORMALLY");
    ESP_LOGI(TAG, "  Calibrating: Audio + Vibration");
    ESP_LOGI(TAG, "========================================");
    
    g_is_calibrating = true;
    g_calibration_index = 0;
    g_calibration.is_valid = false;
    
    // Allocate calibration buffers
    if (g_calibration_samples == nullptr) {
        g_calibration_samples = static_cast<float*>(
            heap_caps_malloc(config::CALIBRATION_SAMPLES * sizeof(float), MALLOC_CAP_8BIT));
    }
    if (g_vib_calibration_samples == nullptr) {
        g_vib_calibration_samples = static_cast<float*>(
            heap_caps_malloc(config::CALIBRATION_SAMPLES * sizeof(float), MALLOC_CAP_8BIT));
    }
    
    // Fast blink LED to indicate calibration mode
    for (int i = 0; i < 10; i++) {
        gpio_set_level(config::LED_ANOMALY_PIN, 1);
        vTaskDelay(pdMS_TO_TICKS(50));
        gpio_set_level(config::LED_ANOMALY_PIN, 0);
        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

/**
 * @brief Add sample to calibration buffers (audio + vibration)
 */
static void addCalibrationSample(float mse, float vib_rms) {
    if (g_calibration_index < config::CALIBRATION_SAMPLES) {
        g_calibration_samples[g_calibration_index] = mse;
        g_vib_calibration_samples[g_calibration_index] = vib_rms;
        g_calibration_index++;
        
        // Progress update every 10 samples
        if (g_calibration_index % 10 == 0) {
            ESP_LOGI(TAG, "Calibration progress: %d/%d (MSE=%.4f, RMS=%.3fg)", 
                     g_calibration_index, config::CALIBRATION_SAMPLES, mse, vib_rms);
            
            // Blink LED to show progress
            gpio_set_level(config::LED_ANOMALY_PIN, 1);
            vTaskDelay(pdMS_TO_TICKS(100));
            gpio_set_level(config::LED_ANOMALY_PIN, 0);
        }
    }
}

/**
 * @brief Complete calibration and calculate thresholds for audio and vibration
 */
static void completeCalibration() {
    if (g_calibration_index < 10) {
        ESP_LOGE(TAG, "Not enough samples for calibration");
        return;
    }
    
    // ========== AUDIO CALIBRATION ==========
    float sum = 0;
    for (size_t i = 0; i < g_calibration_index; i++) {
        sum += g_calibration_samples[i];
    }
    float mean = sum / g_calibration_index;
    
    float var_sum = 0;
    for (size_t i = 0; i < g_calibration_index; i++) {
        float diff = g_calibration_samples[i] - mean;
        var_sum += diff * diff;
    }
    float std_dev = sqrtf(var_sum / g_calibration_index);
    
    float threshold_offset = config::SENSITIVITY * std_dev;
    if (threshold_offset < config::MIN_THRESHOLD_OFFSET) {
        threshold_offset = config::MIN_THRESHOLD_OFFSET;
    }
    float audio_threshold = mean + threshold_offset;
    
    // ========== VIBRATION CALIBRATION ==========
    float vib_sum = 0;
    for (size_t i = 0; i < g_calibration_index; i++) {
        vib_sum += g_vib_calibration_samples[i];
    }
    float vib_baseline = vib_sum / g_calibration_index;
    
    // Vibration threshold: baseline + fixed offset (0.3g above baseline)
    float vib_threshold = vib_baseline + 0.3f; // Think here for better approximation
                                                // Why did I choose 0.3f?
    
    // Store calibration
    g_calibration.baseline_mean = mean;
    g_calibration.baseline_std = std_dev;
    g_calibration.threshold = audio_threshold;
    g_calibration.vib_baseline_rms = vib_baseline;
    g_calibration.vib_threshold = vib_threshold;
    g_calibration.sample_count = g_calibration_index;
    g_calibration.is_valid = true;
    
    // Save to flash
    saveCalibrationToNVS(g_calibration);
    
    // End calibration mode
    g_is_calibrating = false;
    
    // Free calibration buffers
    if (g_calibration_samples != nullptr) {
        heap_caps_free(g_calibration_samples);
        g_calibration_samples = nullptr;
    }
    if (g_vib_calibration_samples != nullptr) {
        heap_caps_free(g_vib_calibration_samples);
        g_vib_calibration_samples = nullptr;
    }
    
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  DUAL-SENSOR CALIBRATION COMPLETE!");
    ESP_LOGI(TAG, "  Audio: baseline=%.4f, threshold=%.4f", mean, audio_threshold);
    ESP_LOGI(TAG, "  Vibration: baseline=%.3fg, threshold=%.3fg", vib_baseline, vib_threshold);
    ESP_LOGI(TAG, "  Samples used: %d", g_calibration_index);
    ESP_LOGI(TAG, "========================================");
    
    // Solid LED for 2 seconds to confirm
    gpio_set_level(config::LED_ANOMALY_PIN, 1);
    vTaskDelay(pdMS_TO_TICKS(2000));
    gpio_set_level(config::LED_ANOMALY_PIN, 0);
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

static esp_err_t initGPIO() {
    gpio_config_t io_conf = {};
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pin_bit_mask = (1ULL << config::LED_ANOMALY_PIN);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    
    esp_err_t ret = gpio_config(&io_conf);
    if (ret == ESP_OK) {
        gpio_set_level(config::LED_ANOMALY_PIN, 0);
    }
    
    // Configure BOOT button as input with pull-up for recalibration
    io_conf.mode = GPIO_MODE_INPUT;
    io_conf.pin_bit_mask = (1ULL << config::CALIBRATE_BUTTON_PIN);
    io_conf.pull_up_en = GPIO_PULLUP_ENABLE;
    gpio_config(&io_conf);
    
    return ret;
}

static esp_err_t allocateBuffers() {
    size_t audio_bytes = config::AUDIO_BUFFER_SIZE * sizeof(float);
    size_t spec_bytes = config::NUM_FREQ_BINS * config::NUM_FRAMES * sizeof(float);
    size_t nn_bytes = config::SPEC_HEIGHT * config::SPEC_WIDTH * sizeof(float);
    
    g_audio_buffer = static_cast<float*>(
        heap_caps_malloc(audio_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!g_audio_buffer) {
        g_audio_buffer = static_cast<float*>(
            heap_caps_malloc(audio_bytes, MALLOC_CAP_8BIT));
    }
    
    g_spectrogram = static_cast<float*>(
        heap_caps_malloc(spec_bytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!g_spectrogram) {
        g_spectrogram = static_cast<float*>(
            heap_caps_malloc(spec_bytes, MALLOC_CAP_8BIT));
    }
    
    g_nn_input = static_cast<float*>(
        heap_caps_malloc(nn_bytes, MALLOC_CAP_8BIT));
    
    if (!g_audio_buffer || !g_spectrogram || !g_nn_input) {
        return ESP_ERR_NO_MEM;
    }
    
    return ESP_OK;
}

static void resizeSpectrogramForNN(const float* full_spec, float* nn_spec) {
    for (size_t frame = 0; frame < config::SPEC_WIDTH; frame++) {
        for (size_t freq = 0; freq < config::SPEC_HEIGHT; freq++) {
            size_t src_idx = frame * config::NUM_FREQ_BINS + freq;
            size_t dst_idx = frame * config::SPEC_HEIGHT + freq;
            nn_spec[dst_idx] = full_spec[src_idx];
        }
    }
}

static VibrationFeatures computeVibrationFeatures(pdm::MPU6050Watchdog& mpu) {
    VibrationFeatures features = {0};
    features.is_fault = false;
    
    float sum_sq = 0.0f;
    float max_val = 0.0f;
    int valid_samples = 0;
    
    for (size_t i = 0; i < config::ACCEL_SAMPLES_PER_DETECTION; i++) {
        pdm::AccelData data;
        if (mpu.readRawData(data) == ESP_OK) {
            float mag_g = data.magnitudeG(pdm::AccelRange::RANGE_8G);
            float deviation = fabsf(mag_g - 1.0f);
            sum_sq += deviation * deviation;
            if (deviation > max_val) max_val = deviation;
            valid_samples++;
        }
        vTaskDelay(pdMS_TO_TICKS(config::ACCEL_SAMPLE_DELAY_MS));
    }
    
    constexpr int MIN_VALID_SAMPLES = 10;
    if (valid_samples >= MIN_VALID_SAMPLES) {
        features.rms = sqrtf(sum_sq / valid_samples);
        features.peak = max_val;
        features.crest_factor = (features.rms > 0.01f) ? (features.peak / features.rms) : 1.0f;
        features.is_fault = (features.rms > config::ACCEL_RMS_THRESHOLD) ||
                           (features.peak > config::ACCEL_PEAK_THRESHOLD);
    }
    
    return features;
}

// =============================================================================
// MAIN DETECTION TASK
// =============================================================================

static void anomalyDetectionTask(void* pvParameters) {
    ESP_LOGI(TAG, "Starting self-calibrating anomaly detection");
    
    // Allow power to stabilize (fixes some SD card init issues)
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Initialize SD Card Logger FIRST (Black Box)
    pdm::SDCardConfig sd_config;  // Uses default pins
    pdm::SDCardLogger sdLogger(sd_config);
    esp_err_t sd_ret = sdLogger.init();
    bool sd_available = (sd_ret == ESP_OK);
    if (!sd_available) {
        ESP_LOGW(TAG, "SD card logging disabled (Error: %s) - continuing without black box", esp_err_to_name(sd_ret));
    } else {
        ESP_LOGI(TAG, "SD card logging ENABLED and READY");
    }

    // Initialize MPU6050 (optional)
    bool mpu_available = false;
    pdm::MPU6050Watchdog mpu(config::MPU_I2C_PORT, config::MPU_SDA_PIN, config::MPU_SCL_PIN);
    
    if (config::ENABLE_MPU6050) {
        if (mpu.init() == ESP_OK) {
            mpu_available = true;
            ESP_LOGI(TAG, "MPU6050 initialized");
        } else {
            ESP_LOGW(TAG, "MPU6050 not found");
        }
    }
    
    // Initialize I2S Microphone
    pdm::I2SMicConfig mic_config = {
        .bck_pin = config::I2S_BCK_PIN,
        .ws_pin = config::I2S_WS_PIN,
        .data_pin = config::I2S_DATA_PIN,
        .sample_rate = config::SAMPLE_RATE,
        .dma_buffer_count = 8,
        .dma_buffer_size = 1024,
        .i2s_port = I2S_NUM_0
    };
    
    pdm::I2SMicrophone microphone(mic_config);
    if (microphone.init() != ESP_OK) {
        ESP_LOGE(TAG, "Microphone init failed");
        vTaskDelete(nullptr);
        return;
    }
    
    // Initialize STFT
    pdm::STFTConfig stft_config = {
        .fft_size = config::FFT_SIZE,
        .hop_size = config::HOP_SIZE,
        .num_frames = config::NUM_FRAMES,
        .sample_rate = config::SAMPLE_RATE
    };
    
    pdm::STFTProcessor stft(stft_config);
    if (stft.init() != ESP_OK) {
        ESP_LOGE(TAG, "STFT init failed");
        vTaskDelete(nullptr);
        return;
    }
    
    // Initialize TFLite (use a placeholder threshold, we use calibrated one)
    pdm::InferenceConfig inference_config = {
        .tensor_arena_size = config::TENSOR_ARENA_SIZE,
        .anomaly_threshold = 0.1f,  // Placeholder, we use calibrated threshold
        .input_height = config::SPEC_HEIGHT,
        .input_width = config::SPEC_WIDTH
    };
    
    pdm::TFLiteInference inference(model_data, model_data_len, inference_config);
    if (inference.init() != ESP_OK) {
        ESP_LOGE(TAG, "TFLite init failed");
        vTaskDelete(nullptr);
        return;
    }
    
    microphone.start();
    
    // Load calibration from NVS or start calibration
    if (loadCalibrationFromNVS(g_calibration) == ESP_OK && g_calibration.is_valid) {
        ESP_LOGI(TAG, "========================================");
        ESP_LOGI(TAG, "  LOADED CALIBRATION FROM FLASH");
        ESP_LOGI(TAG, "  Audio: threshold=%.4f", g_calibration.threshold);
        ESP_LOGI(TAG, "  Vibration: baseline=%.3fg, threshold=%.3fg",
                 g_calibration.vib_baseline_rms, g_calibration.vib_threshold);
        ESP_LOGI(TAG, "  Hold BOOT button 3s to recalibrate");
        ESP_LOGI(TAG, "========================================");
    } else {
        ESP_LOGI(TAG, "No calibration found - starting calibration");
        startCalibration();
    }
    
    size_t required_samples = stft.getRequiredSamples();
    TickType_t button_press_start = 0;
    bool button_held = false;
    
    // Main loop
    while (true) {
        uint32_t loop_start = xTaskGetTickCount();
        
        // Check for recalibration button (BOOT button, hold 3 seconds)
        if (gpio_get_level(config::CALIBRATE_BUTTON_PIN) == 0) {
            if (!button_held) {
                button_press_start = xTaskGetTickCount();
                button_held = true;
            } else {
                uint32_t held_ms = (xTaskGetTickCount() - button_press_start) * portTICK_PERIOD_MS;
                if (held_ms > 3000 && !g_is_calibrating) {
                    ESP_LOGW(TAG, "Recalibration requested via button");
                    clearCalibrationNVS();
                    startCalibration();
                }
            }
        } else {
            button_held = false;
        }
        
        // Capture audio and run inference
        float audio_mse = 0.0f;
        bool audio_fault = false;
        
        size_t samples_read = microphone.readFloat(g_audio_buffer, required_samples, 2000);
        
        // Collect vibration data (needed for both calibration and detection)
        VibrationFeatures vib = {0};
        float temp_c = 0.0f;
        if (mpu_available) {
            vib = computeVibrationFeatures(mpu);
            mpu.readTemperature(temp_c);  // Temperature from MPU6050
        }
        
        if (samples_read >= required_samples) {
            esp_err_t ret = stft.process(g_audio_buffer, samples_read, g_spectrogram);
            if (ret == ESP_OK) {
                stft.normalizeSpectrogram(g_spectrogram, stft.getSpectrogramSize());
                resizeSpectrogramForNN(g_spectrogram, g_nn_input);
                
                pdm::InferenceResult result;
                ret = inference.runInference(g_nn_input, result);
                
                if (ret == ESP_OK) {
                    audio_mse = result.reconstruction_error;
                    
                    // During calibration, collect both audio and vibration samples
                    if (g_is_calibrating) {
                        addCalibrationSample(audio_mse, vib.rms);
                        
                        if (g_calibration_index >= config::CALIBRATION_SAMPLES) {
                            completeCalibration();
                        }
                    } else if (g_calibration.is_valid) {
                        // Use calibrated thresholds
                        audio_fault = audio_mse > g_calibration.threshold;
                    }
                }
            }
        }
        
        // Vibration fault check using CALIBRATED threshold
        bool accel_fault = false;
        if (mpu_available && g_calibration.is_valid && !g_is_calibrating) {
            accel_fault = vib.rms > g_calibration.vib_threshold;
        }
        
        // Output
        if (g_is_calibrating) {
            ESP_LOGI(TAG, "[CALIBRATING %d/%d] MSE=%.4f, RMS=%.3fg", 
                     g_calibration_index, config::CALIBRATION_SAMPLES, audio_mse, vib.rms);
        } else if (g_calibration.is_valid) {
            g_detection_count++;
            bool combined_fault = audio_fault || accel_fault;
            
            if (combined_fault) {
                g_anomaly_count++;
                g_anomaly_detected = true;
                g_anomaly_time = xTaskGetTickCount();
                
                const char* source = audio_fault && accel_fault ? "BOTH" : 
                                    audio_fault ? "AUDIO" : "VIBRATION";
                
                ESP_LOGW(TAG, "ANOMALY [%s] MSE=%.4f (th=%.4f) | RMS=%.3fg (th=%.3fg) | Temp=%.1fC", 
                         source, audio_mse, g_calibration.threshold,
                         vib.rms, g_calibration.vib_threshold, temp_c);
                
                // Log to SD card (Black Box)
                if (sd_available) {
                    float threshold = audio_fault ? g_calibration.threshold : g_calibration.vib_threshold;
                    float value = audio_fault ? audio_mse : vib.rms;
                    sdLogger.logAnomaly(source, value, threshold);
                }
                
                gpio_set_level(config::LED_ANOMALY_PIN, 1);
            } else {
                if (g_anomaly_detected) {
                    uint32_t elapsed = (xTaskGetTickCount() - g_anomaly_time) * portTICK_PERIOD_MS;
                    if (elapsed > config::ANOMALY_LED_DURATION_MS) {
                        gpio_set_level(config::LED_ANOMALY_PIN, 0);
                        g_anomaly_detected = false;
                    }
                }
                
                ESP_LOGI(TAG, "Normal: MSE=%.4f (th=%.4f) | RMS=%.3fg (th=%.3fg) | Temp=%.1fC", 
                         audio_mse, g_calibration.threshold, 
                         vib.rms, g_calibration.vib_threshold, temp_c);
            }
            
            // Stats every 20 detections
            if (g_detection_count % 20 == 0) {
                float rate = 100.0f * g_anomaly_count / g_detection_count;
                ESP_LOGI(TAG, "Stats: %d detections, %d anomalies (%.1f%%)",
                         g_detection_count, g_anomaly_count, rate);
            }
        } else {
            ESP_LOGW(TAG, "No calibration - press BOOT button for 3s or restart");
        }
        
        // Maintain interval
        uint32_t elapsed_ms = (xTaskGetTickCount() - loop_start) * portTICK_PERIOD_MS;
        if (elapsed_ms < config::DETECTION_INTERVAL_MS) {
            vTaskDelay(pdMS_TO_TICKS(config::DETECTION_INTERVAL_MS - elapsed_ms));
        }
    }
}

// =============================================================================
// APP MAIN
// =============================================================================

extern "C" void app_main(void) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Self-Calibrating Anomaly Detector");
    ESP_LOGI(TAG, "  Works on ANY machine!");
    ESP_LOGI(TAG, "========================================");
    
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    
    if (initGPIO() != ESP_OK) {
        ESP_LOGE(TAG, "GPIO init failed");
        return;
    }
    
    if (allocateBuffers() != ESP_OK) {
        ESP_LOGE(TAG, "Buffer allocation failed");
        return;
    }
    
    ESP_LOGI(TAG, "Free heap: %d bytes", esp_get_free_heap_size());
    
    xTaskCreatePinnedToCore(
        anomalyDetectionTask,
        "anomaly_detect",
        config::MAIN_TASK_STACK_SIZE,
        nullptr,
        5,
        nullptr,
        1
    );
    
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(10000));
        ESP_LOGI(TAG, "System running... Free heap: %d bytes", esp_get_free_heap_size());
    }
}
