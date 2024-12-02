#ifndef CAMERA_HANDLER_H_
#define CAMERA_HANDLER_H_

#include "tensorflow/lite/c/common.h"


struct ProcessedImage {
    uint8_t* jpg_buf;
    size_t jpg_len;
    bool is_new;
    SemaphoreHandle_t mutex;  // Add mutex for thread-safe access
};

bool InitCamera();

void CaptureImageLoop();

// Function to capture an image and return it as a float array
bool CaptureImage(float* image_data, int width, int height);

// Function to convert RGB565 to grayscale
bool fmt2grayscale(uint8_t* input, size_t input_len, int width, int height, float* output);

#endif  // CAMERA_HANDLER_H_
