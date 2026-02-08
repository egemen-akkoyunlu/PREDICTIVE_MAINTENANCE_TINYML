/**
 * @file sd_card_logger.hpp
 * @brief Black Box Logger for Predictive Maintenance Device
 * 
 * Provides robust SD card logging via SPI interface for anomaly events.
 * Implements power-loss safe logging by opening/writing/closing for each entry.
 * 
 * Hardware: MicroSD Card Module (SPI Interface)
 * - MOSI: GPIO23
 * - MISO: GPIO19
 * - CLK:  GPIO18
 * - CS:   GPIO5
 * 
 * @note Requires menuconfig: Component config → FAT Filesystem → Long filename support
 */

#ifndef SD_CARD_LOGGER_HPP
#define SD_CARD_LOGGER_HPP

#include "esp_err.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include <cstdio>

namespace pdm {

/**
 * @brief Configuration for SD Card Logger
 */
struct SDCardConfig {
    gpio_num_t mosi_pin = GPIO_NUM_23;
    gpio_num_t miso_pin = GPIO_NUM_19;
    gpio_num_t clk_pin  = GPIO_NUM_18;
    gpio_num_t cs_pin   = GPIO_NUM_5;
    
    const char* mount_point = "/sdcard";
    const char* log_filename = "/sdcard/log.csv";  // Short 8.3 name for compatibility
    
    // SPI configuration
    spi_host_device_t spi_host = SPI3_HOST; // Use VSPI (Native pins 18,19,23,5)
    int max_freq_khz = 4000;  // Standard speed
};

/**
 * @brief SD Card Logger Class
 * 
 * Provides power-loss safe logging to SD card. Each log entry opens,
 * writes, and closes the file to prevent corruption on sudden power loss.
 */
class SDCardLogger {
public:
    /**
     * @brief Constructor
     * @param config SD card configuration
     */
    explicit SDCardLogger(const SDCardConfig& config = SDCardConfig())
        : config_(config), mounted_(false), card_(nullptr) {}
    
    /**
     * @brief Destructor - unmounts SD card
     */
    ~SDCardLogger() {
        deinit();
    }
    
    /**
     * @brief Initialize SD card and mount filesystem
     * @return ESP_OK on success
     */
    esp_err_t init();
    
    /**
     * @brief Deinitialize and unmount SD card
     */
    void deinit();
    
    /**
     * @brief Log an anomaly event (power-loss safe)
     * 
     * Opens file, appends CSV line, closes file immediately.
     * Format: [SystemUptime_ms],[Fault_Type],[MSE_Value],[Threshold]
     * 
     * @param type Fault type string (e.g., "AUDIO", "VIBRATION", "BOTH")
     * @param mseValue Current MSE value
     * @param threshold Current threshold value
     */
    void logAnomaly(const char* type, float mseValue, float threshold);
    
    /**
     * @brief Log a generic event with custom message
     * 
     * @param eventType Event type string
     * @param message Custom message
     */
    void logEvent(const char* eventType, const char* message);
    
    /**
     * @brief Check if SD card is mounted
     * @return true if mounted and ready
     */
    bool isMounted() const { return mounted_; }
    
    /**
     * @brief Get total and used space on SD card
     * @param total_kb Output: total space in KB
     * @param used_kb Output: used space in KB
     * @return ESP_OK on success
     */
    esp_err_t getSpaceInfo(uint64_t& total_kb, uint64_t& used_kb);

private:
    SDCardConfig config_;
    bool mounted_;
    sdmmc_card_t* card_;
    
    /**
     * @brief Write header row if log file is new/empty
     */
    void ensureHeader();
    
    // Prevent copying
    SDCardLogger(const SDCardLogger&) = delete;
    SDCardLogger& operator=(const SDCardLogger&) = delete;
};

} // namespace pdm

#endif // SD_CARD_LOGGER_HPP
