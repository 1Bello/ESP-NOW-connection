#include <Arduino.h>
#include "esp_camera.h"
#include "camera_handler.h"
#include "tensorflow/lite/micro/micro_log.h"

#define CAMERA_MODEL_AI_THINKER
#include "camera_pins.h"

bool InitCamera() {
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.grab_mode = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        MicroPrintf("Camera init failed with error 0x%x\n", err);
        return false;
    }

    sensor_t* sensor = esp_camera_sensor_get();
    if (sensor) {
        sensor->set_brightness(sensor, 1);
        sensor->set_contrast(sensor, 1);
        sensor->set_saturation(sensor, 1);
        sensor->set_special_effect(sensor, 0);
    }

    MicroPrintf("Camera initialized successfully");
    return true;
}

bool CaptureImage(float* image_data, int target_width, int target_height) {
    // Validate input parameters
    if (!image_data || target_width <= 0 || target_height <= 0) {
        MicroPrintf("Invalid image data or dimensions");
        return false;
    }

    // Capture frame buffer
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
        MicroPrintf("Camera capture failed");
        return false;
    }

    // Temporary buffer for RGB conversion
    uint8_t* rgb_buffer = nullptr;
    bool conversion_success = false;

    try {
        // Allocate RGB buffer if JPEG
        if (fb->format == PIXFORMAT_JPEG) {
            size_t rgb_len = fb->width * fb->height * 2; // RGB565 uses 2 bytes per pixel
            rgb_buffer = (uint8_t*)malloc(rgb_len);
            
            if (!rgb_buffer) {
                MicroPrintf("Memory allocation failed for RGB buffer");
                esp_camera_fb_return(fb);
                return false;
            }

            // Convert JPEG to RGB565
            conversion_success = jpg2rgb565(fb->buf, fb->len, rgb_buffer, JPG_SCALE_NONE);
            if (!conversion_success) {
                MicroPrintf("JPEG conversion failed");
                free(rgb_buffer);
                esp_camera_fb_return(fb);
                return false;
            }
        }

        // Pointer to source image data
        const uint8_t* source = (fb->format == PIXFORMAT_JPEG) ? rgb_buffer : fb->buf;
        const int src_width = fb->width;
        const int src_height = fb->height;
        
        // Scale and convert to grayscale
        for (int y = 0; y < target_height; y++) {
            for (int x = 0; x < target_width; x++) {
                // Calculate source coordinates with proper scaling
                int src_x = x * src_width / target_width;
                int src_y = y * src_height / target_height;
                
                uint8_t r, g, b;
                if (fb->format == PIXFORMAT_JPEG) {
                    // For RGB565 from JPEG
                    int src_idx = (src_y * src_width + src_x) * 2;
                    uint16_t pixel = (source[src_idx + 1] << 8) | source[src_idx];
                    r = ((pixel >> 11) & 0x1F) << 3;
                    g = ((pixel >> 5) & 0x3F) << 2;
                    b = (pixel & 0x1F) << 3;
                } else {
                    // For raw RGB/grayscale format
                    int src_idx = (src_y * src_width + src_x) * 3; // Assuming RGB888
                    r = source[src_idx];
                    g = source[src_idx + 1];
                    b = source[src_idx + 2];
                }
                
                // Calculate grayscale value and normalize to 0-1
                float gray = (0.299f * r + 0.587f * g + 0.114f * b) / 255.0f;
                image_data[y * target_width + x] = gray;
            }
        }
    }
    catch (...) {
        MicroPrintf("Unexpected error during image processing");
        conversion_success = false;
    }

    // Clean up resources
    if (rgb_buffer) {
        free(rgb_buffer);
    }
    esp_camera_fb_return(fb);

    return conversion_success;
}