/**
 * @file mpu6050_watchdog.cpp
 * @brief Implementation of MPU6050Watchdog class for ESP-IDF
 * 
 * Low-level I2C communication and register manipulation for
 * motion-triggered wakeup functionality.
 */

// FreeRTOS must be included first
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "mpu6050_watchdog.hpp"
#include "esp_log.h"
#include <cstring>
#include <cmath>

static const char* TAG = "MPU6050";

namespace pdm {

// ============================================================================
// Constructor / Destructor
// ============================================================================

MPU6050Watchdog::MPU6050Watchdog(i2c_port_t i2c_port,
                                 gpio_num_t sda_pin,
                                 gpio_num_t scl_pin,
                                 uint8_t device_addr,
                                 uint32_t i2c_freq_hz)
    : i2c_port_(i2c_port)
    , sda_pin_(sda_pin)
    , scl_pin_(scl_pin)
    , device_addr_(device_addr)
    , i2c_freq_hz_(i2c_freq_hz)
    , i2c_initialized_(false)
    , sensor_initialized_(false)
    , config_(MotionConfig::defaultConfig())
{
    ESP_LOGI(TAG, "MPU6050Watchdog created on I2C%d (SDA=%d, SCL=%d, addr=0x%02X)",
             i2c_port_, sda_pin_, scl_pin_, device_addr_);
}

MPU6050Watchdog::~MPU6050Watchdog() {
    if (i2c_initialized_) {
        i2c_driver_delete(i2c_port_);
        ESP_LOGI(TAG, "I2C driver released");
    }
}

// ============================================================================
// Public Methods - High-Level API
// ============================================================================

esp_err_t MPU6050Watchdog::init() {
    esp_err_t ret;

    // Initialize I2C if not already done
    if (!i2c_initialized_) {
        ret = initI2C();
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "Failed to initialize I2C: %s", esp_err_to_name(ret));
            return ret;
        }
    }

    // Verify sensor presence
    if (!isConnected()) {
        ESP_LOGE(TAG, "MPU6050 not detected on I2C bus");
        return ESP_ERR_NOT_FOUND;
    }

    // Reset device to known state
    ret = writeRegister(REG_PWR_MGMT_1, BIT_DEVICE_RESET);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to reset device");
        return ret;
    }
    vTaskDelay(pdMS_TO_TICKS(100));  // Wait for reset

    // Wake up device (clear SLEEP bit, use PLL with X-axis gyro reference)
    ret = writeRegister(REG_PWR_MGMT_1, BIT_CLKSEL_PLL);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to wake device");
        return ret;
    }
    vTaskDelay(pdMS_TO_TICKS(10));

    sensor_initialized_ = true;
    ESP_LOGI(TAG, "MPU6050 initialized successfully");
    return ESP_OK;
}

esp_err_t MPU6050Watchdog::enableWakeOnMotion(const MotionConfig& config) {
    if (!sensor_initialized_) {
        ESP_LOGE(TAG, "Sensor not initialized. Call init() first.");
        return ESP_ERR_INVALID_STATE;
    }

    esp_err_t ret;
    config_ = config;

    ESP_LOGI(TAG, "Configuring Wake-on-Motion: threshold=%.2fg, duration=%dms",
             config.threshold_g, config.duration_ms);

    // Step 1: Set accelerometer range
    ret = setAccelRange(config.accel_range);
    if (ret != ESP_OK) return ret;

    // Step 2: Configure motion detection threshold
    ret = configureMotionThreshold(config.threshold_g);
    if (ret != ESP_OK) return ret;

    // Step 3: Configure motion detection duration
    ret = configureMotionDuration(config.duration_ms);
    if (ret != ESP_OK) return ret;

    // Step 4: Configure interrupt pin behavior
    ret = configureInterruptPin();
    if (ret != ESP_OK) return ret;

    // Step 5: Enable motion detection interrupt
    ret = enableMotionInterrupt();
    if (ret != ESP_OK) return ret;

    // Step 6: Enable low-power Cycle Mode
    ret = enableCycleMode(config.wake_freq);
    if (ret != ESP_OK) return ret;

    ESP_LOGI(TAG, "Wake-on-Motion enabled successfully");
    return ESP_OK;
}

esp_err_t MPU6050Watchdog::readRawData(AccelData& data) {
    uint8_t buffer[6];
    
    esp_err_t ret = readRegisters(REG_ACCEL_XOUT_H, buffer, 6);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read acceleration data");
        return ret;
    }

    // MPU6050 data is big-endian (MSB first)
    data.x = static_cast<int16_t>((buffer[0] << 8) | buffer[1]);
    data.y = static_cast<int16_t>((buffer[2] << 8) | buffer[3]);
    data.z = static_cast<int16_t>((buffer[4] << 8) | buffer[5]);

    return ESP_OK;
}

esp_err_t MPU6050Watchdog::clearInterrupt() {
    uint8_t status;
    // Reading INT_STATUS register clears the interrupt
    return readRegister(REG_INT_STATUS, &status);
}

esp_err_t MPU6050Watchdog::sleep() {
    esp_err_t ret = writeRegister(REG_PWR_MGMT_1, BIT_SLEEP);
    if (ret == ESP_OK) {
        sensor_initialized_ = false;
        ESP_LOGI(TAG, "Sensor put to sleep");
    }
    return ret;
}

bool MPU6050Watchdog::isConnected() {
    uint8_t who_am_i = 0;
    esp_err_t ret = readRegister(REG_WHO_AM_I, &who_am_i);
    
    if (ret != ESP_OK) {
        ESP_LOGW(TAG, "I2C read failed during connection check");
        return false;
    }

    bool connected = (who_am_i == WHO_AM_I_EXPECTED);
    ESP_LOGD(TAG, "WHO_AM_I: 0x%02X (expected 0x%02X) - %s",
             who_am_i, WHO_AM_I_EXPECTED, connected ? "OK" : "MISMATCH");
    
    return connected;
}

// ============================================================================
// Private Methods - I2C Communication
// ============================================================================

esp_err_t MPU6050Watchdog::initI2C() {
    i2c_config_t conf = {};
    conf.mode = I2C_MODE_MASTER;
    conf.sda_io_num = sda_pin_;
    conf.scl_io_num = scl_pin_;
    conf.sda_pullup_en = GPIO_PULLUP_ENABLE;
    conf.scl_pullup_en = GPIO_PULLUP_ENABLE;
    conf.master.clk_speed = i2c_freq_hz_;

    esp_err_t ret = i2c_param_config(i2c_port_, &conf);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C param config failed: %s", esp_err_to_name(ret));
        return ret;
    }

    ret = i2c_driver_install(i2c_port_, conf.mode, 0, 0, 0);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C driver install failed: %s", esp_err_to_name(ret));
        return ret;
    }

    i2c_initialized_ = true;
    ESP_LOGI(TAG, "I2C initialized at %lu Hz", i2c_freq_hz_);
    return ESP_OK;
}

esp_err_t MPU6050Watchdog::writeRegister(uint8_t reg, uint8_t value) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (device_addr_ << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);
    i2c_master_write_byte(cmd, value, true);
    i2c_master_stop(cmd);

    esp_err_t ret = i2c_master_cmd_begin(i2c_port_, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C write to reg 0x%02X failed: %s", reg, esp_err_to_name(ret));
    }
    return ret;
}

esp_err_t MPU6050Watchdog::readRegister(uint8_t reg, uint8_t* value) {
    return readRegisters(reg, value, 1);
}

esp_err_t MPU6050Watchdog::readRegisters(uint8_t reg, uint8_t* buffer, size_t length) {
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();

    // Write register address
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (device_addr_ << 1) | I2C_MASTER_WRITE, true);
    i2c_master_write_byte(cmd, reg, true);

    // Repeated start and read
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (device_addr_ << 1) | I2C_MASTER_READ, true);
    
    if (length > 1) {
        i2c_master_read(cmd, buffer, length - 1, I2C_MASTER_ACK);
    }
    i2c_master_read_byte(cmd, buffer + length - 1, I2C_MASTER_NACK);
    i2c_master_stop(cmd);

    esp_err_t ret = i2c_master_cmd_begin(i2c_port_, cmd, pdMS_TO_TICKS(100));
    i2c_cmd_link_delete(cmd);

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C read from reg 0x%02X failed: %s", reg, esp_err_to_name(ret));
    }
    return ret;
}

// ============================================================================
// Private Methods - Register Configuration
// ============================================================================

esp_err_t MPU6050Watchdog::configureMotionThreshold(float threshold_g) {
    uint8_t threshold_reg = gForceToThreshold(threshold_g);
    ESP_LOGD(TAG, "Motion threshold: %.3fg -> reg value %d", threshold_g, threshold_reg);
    return writeRegister(REG_MOT_THR, threshold_reg);
}

esp_err_t MPU6050Watchdog::configureMotionDuration(uint8_t duration_ms) {
    // MOT_DUR register: 1 LSB = 1ms
    ESP_LOGD(TAG, "Motion duration: %d ms", duration_ms);
    return writeRegister(REG_MOT_DUR, duration_ms);
}

esp_err_t MPU6050Watchdog::setAccelRange(AccelRange range) {
    ESP_LOGD(TAG, "Setting accel range: 0x%02X", static_cast<uint8_t>(range));
    return writeRegister(REG_ACCEL_CONFIG, static_cast<uint8_t>(range));
}

esp_err_t MPU6050Watchdog::enableCycleMode(WakeFrequency wake_freq) {
    esp_err_t ret;

    // PWR_MGMT_2: Set LP_WAKE_CTRL bits and disable all gyroscopes
    uint8_t pwr_mgmt_2 = static_cast<uint8_t>(wake_freq) | BITS_STBY_ALL_GYRO;
    ret = writeRegister(REG_PWR_MGMT_2, pwr_mgmt_2);
    if (ret != ESP_OK) return ret;

    // PWR_MGMT_1: Enable CYCLE mode, disable SLEEP, disable temp sensor
    // Use internal 8MHz oscillator for cycle mode
    uint8_t pwr_mgmt_1 = BIT_CYCLE | BIT_TEMP_DIS;
    ret = writeRegister(REG_PWR_MGMT_1, pwr_mgmt_1);
    if (ret != ESP_OK) return ret;

    ESP_LOGI(TAG, "Cycle mode enabled with wake freq: 0x%02X", static_cast<uint8_t>(wake_freq));
    return ESP_OK;
}

esp_err_t MPU6050Watchdog::configureInterruptPin() {
    // INT_PIN_CFG register:
    // - LATCH_INT_EN: Interrupt pulse held until cleared
    // - INT_RD_CLEAR: Interrupt cleared on any read
    // - Active HIGH (default)
    // - Push-Pull (default)
    uint8_t int_cfg = BIT_LATCH_INT_EN | BIT_INT_RD_CLEAR;
    return writeRegister(REG_INT_PIN_CFG, int_cfg);
}

esp_err_t MPU6050Watchdog::enableMotionInterrupt() {
    // Enable Motion Detection interrupt
    esp_err_t ret = writeRegister(REG_INT_ENABLE, BIT_MOT_EN);
    if (ret != ESP_OK) return ret;

    // Configure motion detection control for better sensitivity
    // MOT_DETECT_CTRL: Reduce motion detection delay
    ret = writeRegister(REG_MOT_DETECT_CTRL, 0x15);  // Motion detect with delay counter decrement
    
    return ret;
}

uint8_t MPU6050Watchdog::gForceToThreshold(float threshold_g) const {
    // Motion threshold register sensitivity depends on accel range
    // In general: 1 LSB = (range / 128) g for the 8-bit register
    // For ±8g range: 1 LSB ≈ 0.0625g = 62.5mg
    // For ±2g range: 1 LSB ≈ 0.0156g = 15.6mg
    
    float lsb_per_g;
    switch (config_.accel_range) {
        case AccelRange::RANGE_2G:  lsb_per_g = 128.0f / 2.0f;  break;  // 64 LSB/g
        case AccelRange::RANGE_4G:  lsb_per_g = 128.0f / 4.0f;  break;  // 32 LSB/g
        case AccelRange::RANGE_8G:  lsb_per_g = 128.0f / 8.0f;  break;  // 16 LSB/g
        case AccelRange::RANGE_16G: lsb_per_g = 128.0f / 16.0f; break;  // 8 LSB/g
        default: lsb_per_g = 16.0f;
    }

    float reg_value = threshold_g * lsb_per_g;
    
    // Clamp to 8-bit range
    if (reg_value < 1.0f) reg_value = 1.0f;
    if (reg_value > 255.0f) reg_value = 255.0f;

    return static_cast<uint8_t>(reg_value);
}

}  // namespace pdm
