/**
 * @file deep_sleep_manager.cpp
 * @brief Implementation of DeepSleepManager for ESP32
 */

// FreeRTOS must be included first
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "deep_sleep_manager.hpp"
#include "esp_log.h"
#include "driver/rtc_io.h"

static const char* TAG = "DeepSleep";

namespace pdm {

// RTC memory variable - survives deep sleep
RTC_DATA_ATTR uint32_t DeepSleepManager::boot_count_ = 0;

// ============================================================================
// Singleton Implementation
// ============================================================================

DeepSleepManager& DeepSleepManager::getInstance(gpio_num_t wakeup_pin) {
    static DeepSleepManager instance(wakeup_pin);
    return instance;
}

DeepSleepManager::DeepSleepManager(gpio_num_t wakeup_pin)
    : wakeup_pin_(wakeup_pin)
    , ext0_configured_(false)
    , timer_configured_(false)
{
    // Increment boot counter
    boot_count_++;
    
    ESP_LOGI(TAG, "DeepSleepManager initialized (boot #%lu)", boot_count_);
}

// ============================================================================
// Wakeup Cause Detection
// ============================================================================

WakeupCause DeepSleepManager::getWakeupCause() const {
    esp_sleep_wakeup_cause_t cause = esp_sleep_get_wakeup_cause();

    switch (cause) {
        case ESP_SLEEP_WAKEUP_UNDEFINED:
            return WakeupCause::POWER_ON;
        case ESP_SLEEP_WAKEUP_EXT0:
            return WakeupCause::EXT0_MOTION;
        case ESP_SLEEP_WAKEUP_EXT1:
            return WakeupCause::EXT1;
        case ESP_SLEEP_WAKEUP_TIMER:
            return WakeupCause::TIMER;
        case ESP_SLEEP_WAKEUP_TOUCHPAD:
            return WakeupCause::TOUCHPAD;
        case ESP_SLEEP_WAKEUP_ULP:
            return WakeupCause::ULP;
        default:
            return WakeupCause::UNKNOWN;
    }
}

const char* DeepSleepManager::wakeupCauseToString(WakeupCause cause) {
    switch (cause) {
        case WakeupCause::POWER_ON:     return "Power-On/Reset";
        case WakeupCause::EXT0_MOTION:  return "EXT0 (Motion Detected)";
        case WakeupCause::EXT1:         return "EXT1 (Multiple GPIO)";
        case WakeupCause::TIMER:        return "Timer";
        case WakeupCause::TOUCHPAD:     return "Touchpad";
        case WakeupCause::ULP:          return "ULP Coprocessor";
        case WakeupCause::UNKNOWN:      
        default:                        return "Unknown";
    }
}

// ============================================================================
// Wakeup Configuration
// ============================================================================

esp_err_t DeepSleepManager::configureExt0Wakeup(gpio_num_t pin, WakeupLevel level) {
    wakeup_pin_ = pin;

    // Validate GPIO is RTC capable
    if (!rtc_gpio_is_valid_gpio(pin)) {
        ESP_LOGE(TAG, "GPIO%d is not a valid RTC GPIO for deep sleep wakeup", pin);
        return ESP_ERR_INVALID_ARG;
    }

    // Initialize RTC GPIO
    esp_err_t ret = rtc_gpio_init(pin);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to init RTC GPIO%d: %s", pin, esp_err_to_name(ret));
        return ret;
    }

    // Set as input
    ret = rtc_gpio_set_direction(pin, RTC_GPIO_MODE_INPUT_ONLY);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to set GPIO%d direction: %s", pin, esp_err_to_name(ret));
        return ret;
    }

    // Enable pulldown if waking on HIGH (MPU6050 INT is active HIGH)
    if (level == WakeupLevel::HIGH) {
        rtc_gpio_pulldown_en(pin);
        rtc_gpio_pullup_dis(pin);
    } else {
        rtc_gpio_pullup_en(pin);
        rtc_gpio_pulldown_dis(pin);
    }

    // Configure ext0 wakeup
    ret = esp_sleep_enable_ext0_wakeup(pin, static_cast<int>(level));
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to enable ext0 wakeup: %s", esp_err_to_name(ret));
        return ret;
    }

    ext0_configured_ = true;
    ESP_LOGI(TAG, "ext0 wakeup configured on GPIO%d (level=%s)",
             pin, level == WakeupLevel::HIGH ? "HIGH" : "LOW");

    return ESP_OK;
}

esp_err_t DeepSleepManager::configureTimerWakeup(uint64_t seconds) {
    uint64_t time_us = seconds * 1000000ULL;
    
    esp_err_t ret = esp_sleep_enable_timer_wakeup(time_us);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to enable timer wakeup: %s", esp_err_to_name(ret));
        return ret;
    }

    timer_configured_ = true;
    ESP_LOGI(TAG, "Timer wakeup configured for %llu seconds", seconds);

    return ESP_OK;
}

// ============================================================================
// Deep Sleep Entry
// ============================================================================

void DeepSleepManager::isolateGPIOs() {
    // Isolate GPIO12 to reduce leakage current (common source of leakage)
    rtc_gpio_isolate(GPIO_NUM_12);
    
    // Note: Don't isolate the wakeup pin!
    ESP_LOGI(TAG, "GPIOs isolated for low-power sleep");
}

[[noreturn]] void DeepSleepManager::enterDeepSleep() {
    if (!ext0_configured_ && !timer_configured_) {
        ESP_LOGW(TAG, "No wakeup source configured! Device may not wake up.");
    }

    ESP_LOGI(TAG, "Entering deep sleep...");
    
    // Flush logs before sleeping
    esp_log_level_set("*", ESP_LOG_NONE);
    fflush(stdout);
    
    // Short delay to ensure all operations complete
    vTaskDelay(pdMS_TO_TICKS(10));

    // Enter deep sleep (this never returns)
    esp_deep_sleep_start();

    // Compiler hint that this never returns
    while (true) {}
}

// ============================================================================
// Utility Methods
// ============================================================================

uint32_t DeepSleepManager::getBootCount() const {
    return boot_count_;
}

bool DeepSleepManager::isColdBoot() const {
    return (getWakeupCause() == WakeupCause::POWER_ON);
}

}  // namespace pdm
