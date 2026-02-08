# Main component CMakeLists.txt for Predictive Maintenance

# Option to build only the deep sleep firmware (Task 1)
# or full anomaly detection with TFLite (Task 4)
option(BUILD_ANOMALY_DETECTOR "Build full anomaly detector with TFLite" ON)

if(BUILD_ANOMALY_DETECTOR)
    # Full anomaly detection build with dual sensors + SD card logging
    set(SRCS
        "anomaly_detector.cpp"
        "drivers/mpu6050_watchdog.cpp"
        "drivers/sd_card_logger.cpp"
    )
    
    idf_component_register(
        SRCS 
            ${SRCS}
        INCLUDE_DIRS 
            "."
            "drivers"
            "dsp"
            "model"
        REQUIRES
            driver
            freertos
        PRIV_REQUIRES
            esp_timer
            esp_system
            nvs_flash
            fatfs
    )
    
else()
    # Deep sleep wake-on-vibration build (Task 1)
    set(SRCS
        "app_main.cpp"
        "drivers/mpu6050_watchdog.cpp"
        "drivers/deep_sleep_manager.cpp"
    )
    
    idf_component_register(
        SRCS 
            ${SRCS}
        INCLUDE_DIRS 
            "."
        REQUIRES
            driver
            freertos
    )
endif()

# Compiler options for C++
target_compile_options(${COMPONENT_LIB} PRIVATE
    -Wno-missing-field-initializers
)
