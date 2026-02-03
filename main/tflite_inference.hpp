/**
 * @file tflite_inference.hpp
 * @brief TensorFlow Lite Micro Inference Engine for Anomaly Detection
 * 
 * This module wraps TFLite Micro for running autoencoder inference
 * on ESP32 to detect anomalies via reconstruction error.
 * 
 * Memory Usage:
 * - Tensor Arena: ~40-100KB depending on model
 * - Allocated from PSRAM if available, otherwise internal RAM
 */

#pragma once

#include "freertos/FreeRTOS.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"

// TensorFlow Lite Micro headers
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <cstdint>
#include <cmath>

namespace pdm {

/**
 * @brief Inference configuration
 */
struct InferenceConfig {
    size_t tensor_arena_size;       // Size of tensor arena in bytes
    float anomaly_threshold;         // Reconstruction error threshold
    size_t input_height;            // Spectrogram height (freq bins)
    size_t input_width;             // Spectrogram width (time frames)
};

/**
 * @brief Default inference configuration
 */
constexpr InferenceConfig DEFAULT_INFERENCE_CONFIG = {
    .tensor_arena_size = 80 * 1024,  // 80KB
    .anomaly_threshold = 0.035f,      // Tune based on training
    .input_height = 32,
    .input_width = 32
};

/**
 * @brief Inference result
 */
struct InferenceResult {
    float reconstruction_error;      // MSE between input and output
    bool is_anomaly;                  // True if error > threshold
    float max_error;                  // Maximum single-value error
    uint32_t inference_time_ms;      // Time taken for inference
};

/**
 * @class TFLiteInference
 * @brief TensorFlow Lite Micro inference engine for autoencoder anomaly detection
 * 
 * Usage:
 * 1. Initialize with model data
 * 2. Call runInference() with spectrogram
 * 3. Check result.is_anomaly
 */
class TFLiteInference {
public:
    /**
     * @brief Construct inference engine
     * @param model_data Pointer to TFLite model data (from C header)
     * @param model_size Size of model data in bytes
     * @param config Inference configuration
     */
    TFLiteInference(const uint8_t* model_data, size_t model_size,
                    const InferenceConfig& config = DEFAULT_INFERENCE_CONFIG);
    
    /**
     * @brief Destructor - frees resources
     */
    ~TFLiteInference();
    
    // Prevent copying
    TFLiteInference(const TFLiteInference&) = delete;
    TFLiteInference& operator=(const TFLiteInference&) = delete;
    
    /**
     * @brief Initialize TFLite interpreter
     * @return ESP_OK on success
     */
    esp_err_t init();
    
    /**
     * @brief Run inference on spectrogram
     * 
     * @param spectrogram Input spectrogram (normalized to [0, 1])
     *                    Size: input_height * input_width
     * @param result Output inference result
     * @return ESP_OK on success
     */
    esp_err_t runInference(const float* spectrogram, InferenceResult& result);
    
    /**
     * @brief Run inference with INT8 quantized input
     * 
     * @param spectrogram_int8 Input spectrogram as int8_t [-128, 127]
     * @param result Output inference result
     * @return ESP_OK on success
     */
    esp_err_t runInferenceInt8(const int8_t* spectrogram_int8, InferenceResult& result);
    
    /**
     * @brief Set anomaly detection threshold
     */
    void setThreshold(float threshold) { config_.anomaly_threshold = threshold; }
    
    /**
     * @brief Get current threshold
     */
    float getThreshold() const { return config_.anomaly_threshold; }
    
    /**
     * @brief Get input tensor size
     */
    size_t getInputSize() const { return config_.input_height * config_.input_width; }
    
    /**
     * @brief Check if initialized
     */
    bool isInitialized() const { return is_initialized_; }
    
    /**
     * @brief Get model version info
     */
    const char* getModelVersion() const;
    
private:
    static constexpr const char* TAG = "TFLITE";
    
    const uint8_t* model_data_;
    size_t model_size_;
    InferenceConfig config_;
    
    // TFLite components
    const tflite::Model* model_;
    tflite::MicroInterpreter* interpreter_;
    uint8_t* tensor_arena_;
    
    // Input/Output tensors
    TfLiteTensor* input_tensor_;
    TfLiteTensor* output_tensor_;
    
    bool is_initialized_;
    
    /**
     * @brief Calculate Mean Squared Error between input and output
     */
    float calculateMSE(const float* input, const float* output, size_t size);
    
    /**
     * @brief Register required operations
     */
    static tflite::MicroMutableOpResolver<30>& getOpResolver();
};

// ============================================================================
// Implementation
// ============================================================================

inline TFLiteInference::TFLiteInference(const uint8_t* model_data, size_t model_size,
                                         const InferenceConfig& config)
    : model_data_(model_data)
    , model_size_(model_size)
    , config_(config)
    , model_(nullptr)
    , interpreter_(nullptr)
    , tensor_arena_(nullptr)
    , input_tensor_(nullptr)
    , output_tensor_(nullptr)
    , is_initialized_(false)
{
}

inline TFLiteInference::~TFLiteInference() {
    if (interpreter_) {
        delete interpreter_;
    }
    if (tensor_arena_) {
        heap_caps_free(tensor_arena_);
    }
}

inline tflite::MicroMutableOpResolver<30>& TFLiteInference::getOpResolver() {
    // Static resolver with operations needed for autoencoder
    static tflite::MicroMutableOpResolver<30> resolver;
    static bool initialized = false;
    
    if (!initialized) {
        // Core operations for autoencoder
        resolver.AddConv2D();
        resolver.AddMaxPool2D();
        resolver.AddReshape();
        resolver.AddFullyConnected();
        resolver.AddRelu();
        resolver.AddLogistic();  // Sigmoid
        resolver.AddTransposeConv();
        
        // Quantization ops
        resolver.AddQuantize();
        resolver.AddDequantize();
        
        // Shape/tensor manipulation ops
        resolver.AddShape();
        resolver.AddPack();
        resolver.AddUnpack();
        resolver.AddPad();
        resolver.AddGather();
        resolver.AddConcatenation();
        resolver.AddStridedSlice();
        
        // Math ops
        resolver.AddMul();
        resolver.AddAdd();
        resolver.AddSoftmax();
        resolver.AddMean();
        
        initialized = true;
    }
    
    return resolver;
}

inline esp_err_t TFLiteInference::init() {
    if (is_initialized_) {
        return ESP_OK;
    }
    
    ESP_LOGI(TAG, "Initializing TFLite Micro inference engine");
    ESP_LOGI(TAG, "  Model size: %d bytes", model_size_);
    ESP_LOGI(TAG, "  Tensor arena: %d KB", config_.tensor_arena_size / 1024);
    ESP_LOGI(TAG, "  Input shape: %dx%d", config_.input_height, config_.input_width);
    ESP_LOGI(TAG, "  Anomaly threshold: %.4f", config_.anomaly_threshold);
    
    // Load model
    model_ = tflite::GetModel(model_data_);
    if (model_->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch: got %d, expected %d",
                 model_->version(), TFLITE_SCHEMA_VERSION);
        return ESP_ERR_INVALID_VERSION;
    }
    
    // Allocate tensor arena
    // Try PSRAM first if available, fall back to internal RAM
    tensor_arena_ = static_cast<uint8_t*>(
        heap_caps_malloc(config_.tensor_arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    
    if (!tensor_arena_) {
        ESP_LOGW(TAG, "PSRAM not available, using internal RAM");
        tensor_arena_ = static_cast<uint8_t*>(
            heap_caps_malloc(config_.tensor_arena_size, MALLOC_CAP_8BIT));
    }
    
    if (!tensor_arena_) {
        ESP_LOGE(TAG, "Failed to allocate tensor arena");
        return ESP_ERR_NO_MEM;
    }
    
    // Create interpreter
    interpreter_ = new tflite::MicroInterpreter(
        model_,
        getOpResolver(),
        tensor_arena_,
        config_.tensor_arena_size);
    
    // Allocate tensors
    TfLiteStatus status = interpreter_->AllocateTensors();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Failed to allocate tensors");
        return ESP_FAIL;
    }
    
    // Get input and output tensors
    input_tensor_ = interpreter_->input(0);
    output_tensor_ = interpreter_->output(0);
    
    ESP_LOGI(TAG, "Input tensor: type=%d, dims=[%d, %d, %d, %d]",
             input_tensor_->type,
             input_tensor_->dims->data[0],
             input_tensor_->dims->data[1],
             input_tensor_->dims->data[2],
             input_tensor_->dims->data[3]);
    
    ESP_LOGI(TAG, "Output tensor: type=%d, dims=[%d, %d, %d, %d]",
             output_tensor_->type,
             output_tensor_->dims->data[0],
             output_tensor_->dims->data[1],
             output_tensor_->dims->data[2],
             output_tensor_->dims->data[3]);
    
    // Report memory usage
    size_t used_bytes = interpreter_->arena_used_bytes();
    ESP_LOGI(TAG, "Tensor arena used: %d bytes (%.1f%%)",
             used_bytes, 100.0f * used_bytes / config_.tensor_arena_size);
    
    is_initialized_ = true;
    ESP_LOGI(TAG, "TFLite inference engine initialized successfully");
    
    return ESP_OK;
}

inline esp_err_t TFLiteInference::runInference(const float* spectrogram, InferenceResult& result) {
    if (!is_initialized_) {
        ESP_LOGE(TAG, "Not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    uint32_t start_time = xTaskGetTickCount();
    
    // Copy input data
    size_t input_size = getInputSize();
    
    if (input_tensor_->type == kTfLiteFloat32) {
        // Float model
        float* input_data = input_tensor_->data.f;
        memcpy(input_data, spectrogram, input_size * sizeof(float));
    } else if (input_tensor_->type == kTfLiteInt8) {
        // Quantized model - convert float to int8
        int8_t* input_data = input_tensor_->data.int8;
        float scale = input_tensor_->params.scale;
        int32_t zero_point = input_tensor_->params.zero_point;
        
        for (size_t i = 0; i < input_size; i++) {
            int32_t quantized = static_cast<int32_t>(roundf(spectrogram[i] / scale)) + zero_point;
            // Clamp to int8 range
            if (quantized < -128) quantized = -128;
            if (quantized > 127) quantized = 127;
            input_data[i] = static_cast<int8_t>(quantized);
        }
    } else {
        ESP_LOGE(TAG, "Unsupported input tensor type: %d", input_tensor_->type);
        return ESP_ERR_NOT_SUPPORTED;
    }
    
    // Run inference
    TfLiteStatus status = interpreter_->Invoke();
    if (status != kTfLiteOk) {
        ESP_LOGE(TAG, "Inference failed");
        return ESP_FAIL;
    }
    
    // Get output and calculate reconstruction error
    float* output_data;
    float* output_float_buffer = nullptr;
    
    if (output_tensor_->type == kTfLiteFloat32) {
        output_data = output_tensor_->data.f;
    } else if (output_tensor_->type == kTfLiteInt8) {
        // Dequantize output
        output_float_buffer = static_cast<float*>(malloc(input_size * sizeof(float)));
        if (!output_float_buffer) {
            return ESP_ERR_NO_MEM;
        }
        
        int8_t* output_int8 = output_tensor_->data.int8;
        float scale = output_tensor_->params.scale;
        int32_t zero_point = output_tensor_->params.zero_point;
        
        for (size_t i = 0; i < input_size; i++) {
            output_float_buffer[i] = (static_cast<float>(output_int8[i]) - zero_point) * scale;
        }
        output_data = output_float_buffer;
    } else {
        ESP_LOGE(TAG, "Unsupported output tensor type: %d", output_tensor_->type);
        return ESP_ERR_NOT_SUPPORTED;
    }
    
    // Calculate reconstruction error (MSE)
    result.reconstruction_error = calculateMSE(spectrogram, output_data, input_size);
    result.is_anomaly = (result.reconstruction_error > config_.anomaly_threshold);
    
    // Find max error
    result.max_error = 0;
    for (size_t i = 0; i < input_size; i++) {
        float error = fabsf(spectrogram[i] - output_data[i]);
        if (error > result.max_error) {
            result.max_error = error;
        }
    }
    
    if (output_float_buffer) {
        free(output_float_buffer);
    }
    
    result.inference_time_ms = (xTaskGetTickCount() - start_time) * portTICK_PERIOD_MS;
    
    return ESP_OK;
}

inline esp_err_t TFLiteInference::runInferenceInt8(const int8_t* spectrogram_int8, InferenceResult& result) {
    if (!is_initialized_) {
        return ESP_ERR_INVALID_STATE;
    }
    
    if (input_tensor_->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Model does not use INT8 input");
        return ESP_ERR_NOT_SUPPORTED;
    }
    
    uint32_t start_time = xTaskGetTickCount();
    
    // Copy INT8 input directly
    size_t input_size = getInputSize();
    memcpy(input_tensor_->data.int8, spectrogram_int8, input_size);
    
    // Run inference
    TfLiteStatus status = interpreter_->Invoke();
    if (status != kTfLiteOk) {
        return ESP_FAIL;
    }
    
    // Calculate error in quantized domain
    int8_t* output_data = output_tensor_->data.int8;
    float mse_sum = 0;
    result.max_error = 0;
    
    for (size_t i = 0; i < input_size; i++) {
        float diff = static_cast<float>(spectrogram_int8[i] - output_data[i]);
        mse_sum += diff * diff;
        if (fabsf(diff) > result.max_error) {
            result.max_error = fabsf(diff);
        }
    }
    
    // Scale error to float range
    float scale = input_tensor_->params.scale;
    result.reconstruction_error = (mse_sum / input_size) * scale * scale;
    result.is_anomaly = (result.reconstruction_error > config_.anomaly_threshold);
    result.max_error *= scale;
    
    result.inference_time_ms = (xTaskGetTickCount() - start_time) * portTICK_PERIOD_MS;
    
    return ESP_OK;
}

inline float TFLiteInference::calculateMSE(const float* input, const float* output, size_t size) {
    float sum = 0;
    for (size_t i = 0; i < size; i++) {
        float diff = input[i] - output[i];
        sum += diff * diff;
    }
    return sum / size;
}

inline const char* TFLiteInference::getModelVersion() const {
    if (model_) {
        return model_->description()->c_str();
    }
    return "Unknown";
}

} // namespace pdm
