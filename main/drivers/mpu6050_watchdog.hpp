/**
 * @file mpu6050_watchdog.hpp
 * @brief MPU6050 Low-Power Motion Detection Driver for ESP-IDF
 * 
 * This class provides a high-level interface for configuring the MPU6050
 * in low-power "Cycle Mode" with motion detection interrupt capability.
 * Designed for battery-powered Industrial IoT applications.
 * 
 * @author AI-Powered Predictive Maintenance Project
 * @date 2026
 */

#pragma once

#include <cstdint>
#include <cmath>
#include "driver/i2c.h"
#include "esp_err.h"

namespace pdm {  // Predictive Maintenance namespace

/**
 * @brief Accelerometer full-scale range options
 */
enum class AccelRange : uint8_t {
    RANGE_2G  = 0x00,  ///< ±2g  (16384 LSB/g)
    RANGE_4G  = 0x08,  ///< ±4g  (8192 LSB/g)
    RANGE_8G  = 0x10,  ///< ±8g  (4096 LSB/g) - Recommended for industrial
    RANGE_16G = 0x18   ///< ±16g (2048 LSB/g)
};

/**
 * @brief Low-power wake-up frequency options (Cycle Mode)
 */
enum class WakeFrequency : uint8_t {
    FREQ_1_25_HZ = 0x00,  ///< 1.25 Hz - Lowest power (~10µA)
    FREQ_5_HZ    = 0x40,  ///< 5 Hz
    FREQ_20_HZ   = 0x80,  ///< 20 Hz
    FREQ_40_HZ   = 0xC0   ///< 40 Hz - Most responsive (~40µA)
};

/**
 * @brief Motion detection configuration parameters
 */
struct MotionConfig {
    float threshold_g;              ///< Motion threshold in g-force (0.0 - 1.0 typical)
    uint8_t duration_ms;            ///< Motion duration before interrupt (1-255 ms)
    AccelRange accel_range;         ///< Accelerometer range
    WakeFrequency wake_freq;        ///< Low-power sampling frequency

    /**
     * @brief Default configuration for industrial machinery monitoring
     */
    static MotionConfig defaultConfig() {
        return MotionConfig{
            .threshold_g = 0.5f,                    // 0.5g threshold
            .duration_ms = 1,                        // 1ms duration
            .accel_range = AccelRange::RANGE_8G,    // ±8g range
            .wake_freq = WakeFrequency::FREQ_20_HZ  // 20Hz sampling
        };
    }
};

/**
 * @brief Raw accelerometer data from MPU6050
 */
struct AccelData {
    int16_t x;  ///< X-axis raw value
    int16_t y;  ///< Y-axis raw value
    int16_t z;  ///< Z-axis raw value

    /**
     * @brief Calculate magnitude of acceleration vector
     * @return Magnitude in raw LSB units
     */
    float magnitude() const {
        return sqrtf(static_cast<float>(x * x + y * y + z * z));
    }

    /**
     * @brief Convert to g-force based on range setting
     * @param range The accelerometer range used during measurement
     * @return Magnitude in g-force units
     */
    float magnitudeG(AccelRange range) const {
        float scale;
        switch (range) {
            case AccelRange::RANGE_2G:  scale = 16384.0f; break;
            case AccelRange::RANGE_4G:  scale = 8192.0f;  break;
            case AccelRange::RANGE_8G:  scale = 4096.0f;  break;
            case AccelRange::RANGE_16G: scale = 2048.0f;  break;
            default: scale = 4096.0f;
        }
        return magnitude() / scale;
    }
};

/**
 * @class MPU6050Watchdog
 * @brief High-level driver for MPU6050 motion-triggered wakeup
 * 
 * This class encapsulates all low-level register manipulation for
 * configuring the MPU6050 as a low-power motion watchdog. The device
 * operates in Cycle Mode, periodically waking to check for motion
 * and asserting an interrupt when motion exceeds the threshold.
 * 
 * @example
 * @code
 * pdm::MPU6050Watchdog sensor(I2C_NUM_0, GPIO_NUM_21, GPIO_NUM_22);
 * 
 * if (sensor.init() == ESP_OK) {
 *     auto config = pdm::MotionConfig::defaultConfig();
 *     config.threshold_g = 0.3f;  // More sensitive
 *     
 *     sensor.enableWakeOnMotion(config);
 *     // MPU6050 will now assert INT pin when motion detected
 * }
 * @endcode
 */
class MPU6050Watchdog {
public:
    /**
     * @brief Construct MPU6050Watchdog and initialize I2C
     * 
     * @param i2c_port I2C port number (I2C_NUM_0 or I2C_NUM_1)
     * @param sda_pin GPIO pin for I2C SDA
     * @param scl_pin GPIO pin for I2C SCL
     * @param device_addr I2C address (0x68 default, 0x69 if AD0 high)
     * @param i2c_freq_hz I2C clock frequency (default 400kHz)
     */
    MPU6050Watchdog(i2c_port_t i2c_port, 
                    gpio_num_t sda_pin, 
                    gpio_num_t scl_pin,
                    uint8_t device_addr = MPU6050_DEFAULT_ADDR,
                    uint32_t i2c_freq_hz = 400000);

    /**
     * @brief Destructor - releases I2C driver
     */
    ~MPU6050Watchdog();

    // Disable copy (I2C resource management)
    MPU6050Watchdog(const MPU6050Watchdog&) = delete;
    MPU6050Watchdog& operator=(const MPU6050Watchdog&) = delete;

    /**
     * @brief Initialize the MPU6050 sensor
     * 
     * Wakes the device from sleep, verifies WHO_AM_I register,
     * and performs initial configuration.
     * 
     * @return ESP_OK on success, error code otherwise
     */
    esp_err_t init();

    /**
     * @brief Configure and enable Wake-on-Motion detection
     * 
     * This is the primary high-level method. It configures:
     * - Motion detection threshold and duration
     * - Low-power Cycle Mode
     * - Interrupt assertion on motion detection
     * 
     * After calling this, the INT pin will go HIGH when motion
     * exceeding the threshold is detected.
     * 
     * @param config Motion detection configuration
     * @return ESP_OK on success, error code otherwise
     */
    esp_err_t enableWakeOnMotion(const MotionConfig& config);

    /**
     * @brief Read raw acceleration data from sensor
     * 
     * Note: In Cycle Mode, this reads the most recent sample.
     * The sensor samples at the configured wake frequency.
     * 
     * @param data Output structure for acceleration data
     * @return ESP_OK on success, error code otherwise
     */
    esp_err_t readRawData(AccelData& data);

    /**
     * @brief Clear the motion interrupt flag
     * 
     * Must be called after handling a motion interrupt to
     * re-arm the interrupt for the next motion event.
     * 
     * @return ESP_OK on success, error code otherwise
     */
    esp_err_t clearInterrupt();

    /**
     * @brief Put sensor into full sleep mode
     * 
     * Lowest power consumption (~5µA) but no motion detection.
     * Call init() to wake the sensor again.
     * 
     * @return ESP_OK on success, error code otherwise
     */
    esp_err_t sleep();

    /**
     * @brief Check if the sensor is responding
     * @return true if WHO_AM_I register returns expected value
     */
    bool isConnected();

    /**
     * @brief Get the current motion configuration
     * @return Current MotionConfig settings
     */
    const MotionConfig& getConfig() const { return config_; }

    /**
     * @brief Read temperature from MPU6050 internal sensor
     * 
     * The MPU6050 has a built-in temperature sensor.
     * Accuracy: ±1°C, useful for thermal monitoring.
     * 
     * @param temp_c Output: Temperature in Celsius
     * @return ESP_OK on success, error code otherwise
     */
    esp_err_t readTemperature(float& temp_c);

private:
    // ============== Private Constants ==============
    static constexpr uint8_t MPU6050_DEFAULT_ADDR = 0x68;
    static constexpr uint8_t WHO_AM_I_EXPECTED    = 0x68;

    // Register Addresses
    static constexpr uint8_t REG_WHO_AM_I      = 0x75;
    static constexpr uint8_t REG_PWR_MGMT_1    = 0x6B;
    static constexpr uint8_t REG_PWR_MGMT_2    = 0x6C;
    static constexpr uint8_t REG_ACCEL_CONFIG  = 0x1C;
    static constexpr uint8_t REG_INT_PIN_CFG   = 0x37;
    static constexpr uint8_t REG_INT_ENABLE    = 0x38;
    static constexpr uint8_t REG_INT_STATUS    = 0x3A;
    static constexpr uint8_t REG_MOT_THR       = 0x1F;
    static constexpr uint8_t REG_MOT_DUR       = 0x20;
    static constexpr uint8_t REG_ACCEL_XOUT_H  = 0x3B;
    static constexpr uint8_t REG_TEMP_OUT_H    = 0x41;  // Temperature sensor
    static constexpr uint8_t REG_MOT_DETECT_CTRL = 0x69;

    // Bit masks for PWR_MGMT_1
    static constexpr uint8_t BIT_DEVICE_RESET = 0x80;
    static constexpr uint8_t BIT_SLEEP        = 0x40;
    static constexpr uint8_t BIT_CYCLE        = 0x20;
    static constexpr uint8_t BIT_TEMP_DIS     = 0x08;
    static constexpr uint8_t BIT_CLKSEL_PLL   = 0x01;

    // Bit masks for PWR_MGMT_2
    static constexpr uint8_t BIT_STBY_XG = 0x04;
    static constexpr uint8_t BIT_STBY_YG = 0x02;
    static constexpr uint8_t BIT_STBY_ZG = 0x01;
    static constexpr uint8_t BITS_STBY_ALL_GYRO = BIT_STBY_XG | BIT_STBY_YG | BIT_STBY_ZG;

    // Bit masks for INT_ENABLE
    static constexpr uint8_t BIT_MOT_EN = 0x40;

    // Bit masks for INT_PIN_CFG
    static constexpr uint8_t BIT_LATCH_INT_EN = 0x20;
    static constexpr uint8_t BIT_INT_RD_CLEAR = 0x10;

    // ============== Private Members ==============
    i2c_port_t i2c_port_;
    gpio_num_t sda_pin_;
    gpio_num_t scl_pin_;
    uint8_t device_addr_;
    uint32_t i2c_freq_hz_;
    bool i2c_initialized_;
    bool sensor_initialized_;
    MotionConfig config_;

    // ============== Private Methods ==============
    
    /**
     * @brief Initialize I2C master driver
     * @return ESP_OK on success
     */
    esp_err_t initI2C();

    /**
     * @brief Write a single byte to a register
     * @param reg Register address
     * @param value Value to write
     * @return ESP_OK on success
     */
    esp_err_t writeRegister(uint8_t reg, uint8_t value);

    /**
     * @brief Read a single byte from a register
     * @param reg Register address
     * @param value Output buffer for read value
     * @return ESP_OK on success
     */
    esp_err_t readRegister(uint8_t reg, uint8_t* value);

    /**
     * @brief Read multiple bytes starting from a register
     * @param reg Starting register address
     * @param buffer Output buffer
     * @param length Number of bytes to read
     * @return ESP_OK on success
     */
    esp_err_t readRegisters(uint8_t reg, uint8_t* buffer, size_t length);

    /**
     * @brief Configure motion detection threshold register
     * @param threshold_g Threshold in g-force
     * @return ESP_OK on success
     */
    esp_err_t configureMotionThreshold(float threshold_g);

    /**
     * @brief Configure motion detection duration register
     * @param duration_ms Duration in milliseconds
     * @return ESP_OK on success
     */
    esp_err_t configureMotionDuration(uint8_t duration_ms);

    /**
     * @brief Set accelerometer full-scale range
     * @param range Accel range enum value
     * @return ESP_OK on success
     */
    esp_err_t setAccelRange(AccelRange range);

    /**
     * @brief Enable low-power Cycle Mode with gyro disabled
     * @param wake_freq Wake frequency for cycle mode
     * @return ESP_OK on success
     */
    esp_err_t enableCycleMode(WakeFrequency wake_freq);

    /**
     * @brief Configure interrupt pin behavior
     * @return ESP_OK on success
     */
    esp_err_t configureInterruptPin();

    /**
     * @brief Enable motion detection interrupt
     * @return ESP_OK on success
     */
    esp_err_t enableMotionInterrupt();

    /**
     * @brief Convert g-force threshold to register value
     * @param threshold_g Threshold in g-force
     * @return Register value (0-255)
     */
    uint8_t gForceToThreshold(float threshold_g) const;
};

}  // namespace pdm
