/**
 * @file sd_card_logger.cpp
 * @brief Implementation of SD Card Logger for Black Box functionality
 */

// FreeRTOS must be included first
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "sd_card_logger.hpp"
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include <cstring>
#include <cstdio>
#include <sys/stat.h>

static const char* TAG = "SDCardLogger";

namespace pdm {

esp_err_t SDCardLogger::init() {
    if (mounted_) {
        ESP_LOGW(TAG, "SD card already mounted");
        return ESP_OK;
    }

    ESP_LOGI(TAG, "Initializing SD card (SPI mode)");
    ESP_LOGI(TAG, "  MOSI: GPIO%d, MISO: GPIO%d, CLK: GPIO%d, CS: GPIO%d",
             config_.mosi_pin, config_.miso_pin, config_.clk_pin, config_.cs_pin);

    // SPI bus configuration
    spi_bus_config_t bus_cfg = {};
    bus_cfg.mosi_io_num = config_.mosi_pin;
    bus_cfg.miso_io_num = config_.miso_pin;
    bus_cfg.sclk_io_num = config_.clk_pin;
    bus_cfg.quadwp_io_num = -1;
    bus_cfg.quadhd_io_num = -1;
    bus_cfg.max_transfer_sz = 4096; // Must be at least sector size (512) or cluster size

    // Use SPI_DMA_CH_AUTO to enable DMA (Required for transfer > 64 bytes)
    esp_err_t ret = spi_bus_initialize(config_.spi_host, &bus_cfg, SPI_DMA_CH_AUTO);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize SPI bus: %s", esp_err_to_name(ret));
        return ret;
    }

    // CRITICAL: Internal pull-up for MISO is required for many SD modules
    // without their own strong pull-ups.
    gpio_set_pull_mode(config_.miso_pin, GPIO_PULLUP_ONLY);
    gpio_set_pull_mode(config_.cs_pin, GPIO_PULLUP_ONLY);

    // SD card mount configuration
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {};
    mount_config.format_if_mount_failed = false;
    mount_config.max_files = 5;
    mount_config.allocation_unit_size = 16 * 1024;

    // SD card SPI device configuration
    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = config_.cs_pin;
    slot_config.host_id = config_.spi_host;

    // SPI host configuration
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    host.slot = config_.spi_host;
    host.max_freq_khz = 400;      // Force 400kHz "Safe Mode" for timeout issues
    host.command_timeout_ms = 1000; // Even longer timeout for slow operations

    // Mount filesystem
    ret = esp_vfs_fat_sdspi_mount(
        config_.mount_point,
        &host,
        &slot_config,
        &mount_config,
        &card_
    );

    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount filesystem. Check SD card format (FAT32).");
        } else {
            ESP_LOGE(TAG, "Failed to initialize SD card: %s", esp_err_to_name(ret));
            if (ret == 0x108) { // ESP_ERR_INVALID_RESPONSE
                ESP_LOGE(TAG, "-> Possible causes: Wiring too long, shared SPI bus, or insufficient power (SD card needs ~100mA spikes)");
                ESP_LOGE(TAG, "-> Current frequency: %d kHz", config_.max_freq_khz);
            }
            ESP_LOGE(TAG, "Check pins: MOSI=%d, MISO=%d, CLK=%d, CS=%d",
                     config_.mosi_pin, config_.miso_pin, config_.clk_pin, config_.cs_pin);
        }
        spi_bus_free(config_.spi_host);
        return ret;
    }

    mounted_ = true;

    // Print card info
    ESP_LOGI(TAG, "SD card mounted successfully!");
    sdmmc_card_print_info(stdout, card_);

    // Create header if log file doesn't exist
    ensureHeader();

    // Log startup event
    logEvent("SYSTEM", "Device started - logging initialized");

    return ESP_OK;
}

void SDCardLogger::deinit() {
    if (mounted_) {
        logEvent("SYSTEM", "Device shutdown - logging stopped");
        
        esp_vfs_fat_sdcard_unmount(config_.mount_point, card_);
        spi_bus_free(config_.spi_host);
        
        mounted_ = false;
        card_ = nullptr;
        
        ESP_LOGI(TAG, "SD card unmounted");
    }
}

void SDCardLogger::ensureHeader() {
    if (!mounted_) return;

    // Check if file exists and has content
    struct stat st;
    bool needs_header = true;
    
    if (stat(config_.log_filename, &st) == 0 && st.st_size > 0) {
        needs_header = false;
    }

    if (needs_header) {
        FILE* f = fopen(config_.log_filename, "w");
        if (f) {
            fprintf(f, "Uptime_ms,Event_Type,MSE_Value,Threshold,Message\n");
            fclose(f);
            ESP_LOGI(TAG, "Created log file with header: %s", config_.log_filename);
        } else {
            ESP_LOGE(TAG, "Failed to create log file: %s", config_.log_filename);
        }
    }
}

void SDCardLogger::logAnomaly(const char* type, float mseValue, float threshold) {
    if (!mounted_) {
        ESP_LOGW(TAG, "SD card not mounted - anomaly not logged");
        return;
    }

    // Get system uptime in milliseconds
    uint32_t uptime_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;

    // Open file in append mode
    FILE* f = fopen(config_.log_filename, "a");
    if (f == nullptr) {
        ESP_LOGE(TAG, "Failed to open log file for writing");
        return;
    }

    // Write CSV line: Uptime_ms,Event_Type,MSE_Value,Threshold,Message
    int written = fprintf(f, "%lu,ANOMALY_%s,%.6f,%.6f,Anomaly detected\n",
                          (unsigned long)uptime_ms, type, mseValue, threshold);
    
    // Force flush to disk
    fflush(f);
    
    // Close immediately for power-loss safety
    fclose(f);

    if (written > 0) {
        ESP_LOGI(TAG, "Logged: ANOMALY_%s MSE=%.4f (uptime=%lu ms)", 
                 type, mseValue, (unsigned long)uptime_ms);
    } else {
        ESP_LOGE(TAG, "Failed to write log entry");
    }
}

void SDCardLogger::logEvent(const char* eventType, const char* message) {
    if (!mounted_) {
        return;
    }

    uint32_t uptime_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;

    FILE* f = fopen(config_.log_filename, "a");
    if (f == nullptr) {
        ESP_LOGE(TAG, "Failed to open log file for event logging");
        return;
    }

    // Write: Uptime_ms,Event_Type,0,0,Message
    fprintf(f, "%lu,%s,0,0,%s\n", (unsigned long)uptime_ms, eventType, message);
    fflush(f);
    fclose(f);

    ESP_LOGD(TAG, "Logged event: %s - %s", eventType, message);
}

esp_err_t SDCardLogger::getSpaceInfo(uint64_t& total_kb, uint64_t& used_kb) {
    if (!mounted_) {
        return ESP_ERR_INVALID_STATE;
    }

    FATFS* fs;
    DWORD fre_clust;
    
    if (f_getfree("0:", &fre_clust, &fs) != FR_OK) {
        return ESP_FAIL;
    }

    uint64_t total_sectors = (fs->n_fatent - 2) * fs->csize;
    uint64_t free_sectors = fre_clust * fs->csize;
    
    // Sector size is typically 512 bytes
    total_kb = total_sectors * 512 / 1024;
    used_kb = (total_sectors - free_sectors) * 512 / 1024;

    return ESP_OK;
}

} // namespace pdm
