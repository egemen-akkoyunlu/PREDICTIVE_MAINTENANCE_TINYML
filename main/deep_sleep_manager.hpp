/**
 * @file deep_sleep_manager.hpp
 * @brief ESP32 Deep Sleep Manager with ext0 Wakeup Support
 * 
 * Provides a high-level interface for configuring ESP32 deep sleep
 * with external GPIO wakeup (ext0) for motion-triggered applications.
 */

#pragma once

#include "driver/gpio.h"
#include "esp_sleep.h"
#include "esp_err.h"

namespace pdm {

/**
 * @brief Wakeup cause enumeration
 */
enum class WakeupCause {
    POWER_ON,       ///< Initial power-on or reset
    EXT0_MOTION,    ///< ext0 wakeup (motion detected)
    EXT1,           ///< ext1 wakeup (multiple GPIOs)
    TIMER,          ///< Timer wakeup
    TOUCHPAD,       ///< Touchpad wakeup
    ULP,            ///< ULP coprocessor wakeup
    UNKNOWN         ///< Unknown cause
};

/**
 * @brief GPIO wakeup level
 */
enum class WakeupLevel {
    LOW = 0,   ///< Wake on LOW level
    HIGH = 1   ///< Wake on HIGH level
};

/**
 * @class DeepSleepManager
 * @brief Manages ESP32 deep sleep and wakeup configuration
 * 
 * This class provides a clean interface for configuring deep sleep
 * with ext0 wakeup on a specific GPIO pin. Designed for use with
 * motion detection sensors like the MPU6050.
 * 
 * @example
 * @code
 * pdm::DeepSleepManager& sleepMgr = pdm::DeepSleepManager::getInstance();
 * 
 * // Check why we woke up
 * auto cause = sleepMgr.getWakeupCause();
 * if (cause == pdm::WakeupCause::EXT0_MOTION) {
 *     // Handle motion event
 * }
 * 
 * // Configure and enter deep sleep
 * sleepMgr.configureExt0Wakeup(GPIO_NUM_4, pdm::WakeupLevel::HIGH);
 * sleepMgr.enterDeepSleep();  // Never returns
 * @endcode
 */
class DeepSleepManager {
public:
    /**
     * @brief Get singleton instance
     * @param wakeup_pin GPIO pin for ext0 wakeup (only used on first call)
     * @return Reference to the singleton instance
     */
    static DeepSleepManager& getInstance(gpio_num_t wakeup_pin = GPIO_NUM_4);

    // Delete copy/move operations (singleton)
    DeepSleepManager(const DeepSleepManager&) = delete;
    DeepSleepManager& operator=(const DeepSleepManager&) = delete;
    DeepSleepManager(DeepSleepManager&&) = delete;
    DeepSleepManager& operator=(DeepSleepManager&&) = delete;

    /**
     * @brief Get the cause of the most recent wakeup
     * @return WakeupCause enum indicating why the ESP32 woke up
     */
    WakeupCause getWakeupCause() const;

    /**
     * @brief Get human-readable string for wakeup cause
     * @param cause The wakeup cause to describe
     * @return String description
     */
    static const char* wakeupCauseToString(WakeupCause cause);

    /**
     * @brief Configure ext0 wakeup on the specified GPIO
     * 
     * The GPIO must be an RTC GPIO for deep sleep wakeup to work.
     * Valid RTC GPIOs: 0, 2, 4, 12-15, 25-27, 32-39
     * 
     * @param pin GPIO pin number (must be RTC GPIO)
     * @param level Wakeup level (HIGH or LOW)
     * @return ESP_OK on success, error code otherwise
     */
    esp_err_t configureExt0Wakeup(gpio_num_t pin, WakeupLevel level);

    /**
     * @brief Configure optional timer wakeup as backup
     * 
     * Useful for periodic health checks even if no motion detected.
     * 
     * @param seconds Time in seconds before timer wakeup
     * @return ESP_OK on success, error code otherwise
     */
    esp_err_t configureTimerWakeup(uint64_t seconds);

    /**
     * @brief Enter deep sleep mode
     * 
     * This function does not return. The ESP32 will reset when
     * it wakes up, and app_main() will be called again.
     * 
     * @note Call configureExt0Wakeup() before this to set wakeup source
     */
    [[noreturn]] void enterDeepSleep();

    /**
     * @brief Prepare GPIOs for minimal power consumption
     * 
     * Isolates GPIO pins to prevent current leakage during deep sleep.
     * Call this before enterDeepSleep() for optimal power consumption.
     */
    void isolateGPIOs();

    /**
     * @brief Get the current boot count (survives deep sleep)
     * @return Number of times the device has booted/woken up
     */
    uint32_t getBootCount() const;

    /**
     * @brief Check if this is a fresh power-on boot
     * @return true if this is the first boot after power-on
     */
    bool isColdBoot() const;

    /**
     * @brief Get the configured wakeup pin
     * @return GPIO number configured for wakeup
     */
    gpio_num_t getWakeupPin() const { return wakeup_pin_; }

private:
    /**
     * @brief Private constructor (singleton)
     * @param wakeup_pin Default GPIO for ext0 wakeup
     */
    explicit DeepSleepManager(gpio_num_t wakeup_pin);

    gpio_num_t wakeup_pin_;
    bool ext0_configured_;
    bool timer_configured_;

    // Boot counter stored in RTC memory (survives deep sleep)
    static RTC_DATA_ATTR uint32_t boot_count_;
};

}  // namespace pdm
