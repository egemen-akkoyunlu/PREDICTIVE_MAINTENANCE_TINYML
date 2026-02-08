/**
 * @file app_main.cpp
 * @brief AI-Powered Multi-Modal Sensor Fusion Node - Main Application
 * 
 * This is the entry point for the predictive maintenance firmware.
 * Implements "Sleep-until-Vibration" logic for energy-efficient
 * monitoring of rotating machinery.
 * 
 * Hardware Configuration:
 *   - MPU6050 SDA: GPIO21
 *   - MPU6050 SCL: GPIO22
 *   - MPU6050 INT: GPIO4 (RTC GPIO for deep sleep wakeup)
 */

#include <cstdio>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_system.h"

#include "drivers/mpu6050_watchdog.hpp"
#include "drivers/deep_sleep_manager.hpp"

static const char* TAG = "PdM-Main";

// ============================================================================
// Hardware Pin Configuration
// ============================================================================
namespace config {
    // I2C Configuration
    constexpr i2c_port_t I2C_PORT     = I2C_NUM_0;
    constexpr gpio_num_t I2C_SDA_PIN  = GPIO_NUM_21;
    constexpr gpio_num_t I2C_SCL_PIN  = GPIO_NUM_22;
    
    // MPU6050 Interrupt Pin (must be RTC GPIO for deep sleep wakeup)
    constexpr gpio_num_t MPU_INT_PIN  = GPIO_NUM_4;
    
    // Motion Detection Configuration
    constexpr float MOTION_THRESHOLD_G = 0.1f;   // g-force (tune for your machinery)
    constexpr uint8_t MOTION_DURATION_MS = 1;    // Minimum motion duration
    
    // Optional: Periodic wakeup for health checks (0 = disabled)
    constexpr uint64_t TIMER_WAKEUP_SECONDS = 3600;  // Wake every hour even if no motion
    
    // Active monitoring duration before returning to sleep
    constexpr uint32_t ACTIVE_MONITORING_MS = 5000;  // 5 seconds of monitoring
    constexpr uint32_t SAMPLE_INTERVAL_MS = 100;     // Sample rate during active monitoring
}

// ============================================================================
// Function Prototypes
// ============================================================================
void handleColdBoot(pdm::MPU6050Watchdog& sensor, pdm::DeepSleepManager& sleepMgr);
void handleMotionWakeup(pdm::MPU6050Watchdog& sensor, pdm::DeepSleepManager& sleepMgr);
void handleTimerWakeup(pdm::MPU6050Watchdog& sensor, pdm::DeepSleepManager& sleepMgr);
void performActiveMonitoring(pdm::MPU6050Watchdog& sensor);
void enterLowPowerMode(pdm::MPU6050Watchdog& sensor, pdm::DeepSleepManager& sleepMgr);

// ============================================================================
// Main Application Entry Point
// ============================================================================
extern "C" void app_main(void) {
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  Predictive Maintenance Sensor Node");
    ESP_LOGI(TAG, "  Sleep-until-Vibration Firmware v1.0");
    ESP_LOGI(TAG, "========================================");

    // Get deep sleep manager instance
    pdm::DeepSleepManager& sleepMgr = pdm::DeepSleepManager::getInstance(config::MPU_INT_PIN);
    
    // Log boot information
    ESP_LOGI(TAG, "Boot count: %lu", sleepMgr.getBootCount());
    ESP_LOGI(TAG, "Wakeup cause: %s", 
             pdm::DeepSleepManager::wakeupCauseToString(sleepMgr.getWakeupCause()));

    // Create MPU6050 sensor instance (constructor initializes I2C)
    pdm::MPU6050Watchdog sensor(
        config::I2C_PORT,
        config::I2C_SDA_PIN,
        config::I2C_SCL_PIN
    );

    // Initialize sensor
    esp_err_t ret = sensor.init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize MPU6050! Error: %s", esp_err_to_name(ret));
        ESP_LOGE(TAG, "Check wiring: SDA=GPIO%d, SCL=GPIO%d", 
                 config::I2C_SDA_PIN, config::I2C_SCL_PIN);
        
        // Wait and restart
        vTaskDelay(pdMS_TO_TICKS(5000));
        esp_restart();
    }

    // Handle wakeup based on cause
    switch (sleepMgr.getWakeupCause()) {
        case pdm::WakeupCause::POWER_ON:
            handleColdBoot(sensor, sleepMgr);
            break;
            
        case pdm::WakeupCause::EXT0_MOTION:
            handleMotionWakeup(sensor, sleepMgr);
            break;
            
        case pdm::WakeupCause::TIMER:
            handleTimerWakeup(sensor, sleepMgr);
            break;
            
        default:
            ESP_LOGW(TAG, "Unexpected wakeup cause, treating as cold boot");
            handleColdBoot(sensor, sleepMgr);
            break;
    }

    // Should never reach here
    ESP_LOGE(TAG, "Unexpected exit from main loop!");
    esp_restart();
}

// ============================================================================
// Boot Handlers
// ============================================================================

/**
 * @brief Handle initial power-on or reset
 */
void handleColdBoot(pdm::MPU6050Watchdog& sensor, pdm::DeepSleepManager& sleepMgr) {
    ESP_LOGI(TAG, "Cold boot detected - initializing system");
    
    // Print system information
    ESP_LOGI(TAG, "Free heap: %lu bytes", esp_get_free_heap_size());

    // Read initial sensor data
    pdm::AccelData accel;
    if (sensor.readRawData(accel) == ESP_OK) {
        ESP_LOGI(TAG, "Initial accel: X=%d, Y=%d, Z=%d (mag=%.2f)", 
                 accel.x, accel.y, accel.z, 
                 accel.magnitudeG(pdm::AccelRange::RANGE_8G));
    }

    // Enter low power mode
    enterLowPowerMode(sensor, sleepMgr);
}

/**
 * @brief Handle wakeup due to motion detection
 */
void handleMotionWakeup(pdm::MPU6050Watchdog& sensor, pdm::DeepSleepManager& sleepMgr) {
    ESP_LOGI(TAG, "==> MOTION DETECTED! Machine may be starting...");
    
    // Clear the interrupt flag
    sensor.clearInterrupt();
    
    // Re-initialize sensor to exit Cycle Mode for proper data reading
    // (Cycle Mode doesn't update accelerometer data registers)
    sensor.init();
    
    // Perform active monitoring
    performActiveMonitoring(sensor);
    
    // Return to low power mode
    enterLowPowerMode(sensor, sleepMgr);
}

/**
 * @brief Handle wakeup due to timer (periodic health check)
 */
void handleTimerWakeup(pdm::MPU6050Watchdog& sensor, pdm::DeepSleepManager& sleepMgr) {
    ESP_LOGI(TAG, "Timer wakeup - performing health check");
    
    // Quick sensor check
    pdm::AccelData accel;
    if (sensor.readRawData(accel) == ESP_OK) {
        float mag = accel.magnitudeG(pdm::AccelRange::RANGE_8G);
        ESP_LOGI(TAG, "Health check: magnitude=%.3fg (expect ~1.0g at rest)", mag);
        
        // Sanity check - if magnitude is way off, sensor might be faulty
        if (mag < 0.5f || mag > 1.5f) {
            ESP_LOGW(TAG, "Unusual reading - sensor may need recalibration");
        }
    }
    
    // Return to low power mode
    enterLowPowerMode(sensor, sleepMgr);
}

// ============================================================================
// Core Functions
// ============================================================================

/**
 * @brief Perform active monitoring while machine is running
 */
void performActiveMonitoring(pdm::MPU6050Watchdog& sensor) {
    ESP_LOGI(TAG, "Starting active monitoring for %lu ms...", config::ACTIVE_MONITORING_MS);
    
    uint32_t samples = 0;
    uint32_t start_time = xTaskGetTickCount() * portTICK_PERIOD_MS;
    
    float max_magnitude = 0.0f;
    float sum_magnitude = 0.0f;
    
    while ((xTaskGetTickCount() * portTICK_PERIOD_MS - start_time) < config::ACTIVE_MONITORING_MS) {
        pdm::AccelData accel;
        
        if (sensor.readRawData(accel) == ESP_OK) {
            float mag = accel.magnitudeG(pdm::AccelRange::RANGE_8G);
            samples++;
            sum_magnitude += mag;
            
            if (mag > max_magnitude) {
                max_magnitude = mag;
            }
            
            // Log every 10th sample to avoid flooding
            if (samples % 10 == 0) {
                ESP_LOGI(TAG, "Sample %lu: X=%+6d Y=%+6d Z=%+6d | Mag=%.3fg",
                         samples, accel.x, accel.y, accel.z, mag);
            }
        }
        
        vTaskDelay(pdMS_TO_TICKS(config::SAMPLE_INTERVAL_MS));
    }
    
    // Report statistics
    if (samples > 0) {
        float avg_magnitude = sum_magnitude / samples;
        ESP_LOGI(TAG, "Monitoring complete: %lu samples", samples);
        ESP_LOGI(TAG, "  Average magnitude: %.3fg", avg_magnitude);
        ESP_LOGI(TAG, "  Maximum magnitude: %.3fg", max_magnitude);
        
        // TODO: Here you would add anomaly detection logic
        // For now, just log a placeholder
        if (max_magnitude > 2.0f) {
            ESP_LOGW(TAG, ">>> HIGH VIBRATION DETECTED - Possible anomaly! <<<");
        }
    }
}

/**
 * @brief Configure sensor for wake-on-motion and enter deep sleep
 */
void enterLowPowerMode(pdm::MPU6050Watchdog& sensor, pdm::DeepSleepManager& sleepMgr) {
    ESP_LOGI(TAG, "Configuring low-power mode...");
    
    // Configure motion detection
    pdm::MotionConfig motionConfig;
    motionConfig.threshold_g = config::MOTION_THRESHOLD_G;
    motionConfig.duration_ms = config::MOTION_DURATION_MS;
    motionConfig.accel_range = pdm::AccelRange::RANGE_8G;
    motionConfig.wake_freq = pdm::WakeFrequency::FREQ_20_HZ;
    
    esp_err_t ret = sensor.enableWakeOnMotion(motionConfig);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure wake-on-motion!");
        vTaskDelay(pdMS_TO_TICKS(5000));
        esp_restart();
    }
    
    // Configure ext0 wakeup (MPU6050 INT is active HIGH)
    ret = sleepMgr.configureExt0Wakeup(config::MPU_INT_PIN, pdm::WakeupLevel::HIGH);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to configure ext0 wakeup!");
        vTaskDelay(pdMS_TO_TICKS(5000));
        esp_restart();
    }
    
    // Configure optional timer wakeup for periodic health checks
    if (config::TIMER_WAKEUP_SECONDS > 0) {
        sleepMgr.configureTimerWakeup(config::TIMER_WAKEUP_SECONDS);
    }
    
    // Isolate GPIOs for minimal leakage
    sleepMgr.isolateGPIOs();
    
    ESP_LOGI(TAG, "Entering deep sleep - waiting for motion...");
    ESP_LOGI(TAG, "  Motion threshold: %.2fg", config::MOTION_THRESHOLD_G);
    ESP_LOGI(TAG, "  Wake GPIO: %d", config::MPU_INT_PIN);
    
    // Enter deep sleep (this never returns)
    sleepMgr.enterDeepSleep();
}
