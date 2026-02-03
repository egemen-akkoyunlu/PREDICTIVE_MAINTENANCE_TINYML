# Main component CMakeLists.txt for Predictive Maintenance

# Option to build only the deep sleep firmware (Task 1)
# or full anomaly detection with TFLite (Task 4)
option(BUILD_ANOMALY_DETECTOR "Build full anomaly detector with TFLite" OFF)

if(BUILD_ANOMALY_DETECTOR)
    # Full anomaly detection build
    set(SRCS
        "anomaly_detector.cpp"
    )
    
    idf_component_register(
        SRCS 
            ${SRCS}
        INCLUDE_DIRS 
            "."
        REQUIRES
            driver
            freertos
        PRIV_REQUIRES
            esp_timer
            esp_system
    )
    
else()
    # Deep sleep wake-on-vibration build (Task 1)
    set(SRCS
        "app_main.cpp"
        "mpu6050_watchdog.cpp"
        "deep_sleep_manager.cpp"
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
