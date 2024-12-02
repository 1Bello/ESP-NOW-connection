#include "esp_task_wdt.h"
#include "camera_handler.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "model.h"
#include "esp_heap_caps.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include <WiFi.h>
#include <esp_now.h>

#define INF_POWER 0.13
#define ESP_CHANNEL 1

//MAC addres propia 3c:71:bf:ef:67:d0
uint8_t peer_mac[] = {0x0c, 0xdc, 0x7e, 0x3a, 0x34, 0x3c};

int loop_counter = 0;
bool start_loop = false;

// Function to handle received data
void onDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int data_len) {
    // Print the MAC address of the sender
    Serial.printf("Data received from: %02X:%02X:%02X:%02X:%02X:%02X\n",
                  info->src_addr[0], info->src_addr[1], info->src_addr[2],
                  info->src_addr[3], info->src_addr[4], info->src_addr[5]);

    // Print the received data
    Serial.printf("Data: %s\n", (char*)data);

    loop_counter = 0;
    start_loop = true;
}

void onDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
    if (status != ESP_NOW_SEND_SUCCESS) {
        Serial.println("ESP-NOW Send Fail");
    }
}

// Initialize Wi-Fi
void initWiFi() {
    WiFi.mode(WIFI_STA); // Set ESP32 as a station
    WiFi.disconnect();   // Disconnect from any previously connected network
    Serial.println("WiFi initialized");
}

// Initialize ESP-NOW
void initEspNow() {
    if (esp_now_init() == ESP_OK) {
        Serial.println("ESP-NOW initialized successfully");
    } else {
        Serial.println("ESP-NOW initialization failed");
        ESP.restart();
    }
    esp_now_register_recv_cb(onDataRecv);
    esp_now_register_send_cb(onDataSent);
}

// Register a peer
void registerPeer(uint8_t *peer_addr) {
    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, peer_addr, 6);
    peerInfo.channel = ESP_CHANNEL;
    peerInfo.encrypt = false;

    if (esp_now_add_peer(&peerInfo) == ESP_OK) {
        Serial.println("Peer registered successfully");
    } else {
        Serial.println("Failed to register peer");
    }
}

// Send data
void sendData(const uint8_t *peer_addr, const char *data) {
    esp_err_t result = esp_now_send(peer_addr, (uint8_t *)data, strlen(data));
    if (result == ESP_OK) {
        Serial.println("Data sent successfully");
    } else {
        Serial.println("Error sending data");
    }
}

// Increase tensor arena size based on the error message
constexpr int kTensorArenaSize = 100000;  // Increased from 50000
uint8_t* tensor_arena = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

void setup() {
    Serial.begin(115200);
    while(!Serial) delay(100);
    Serial.println("Starting setup...");

    initWiFi();
    initEspNow();
    registerPeer(peer_mac);

    // Initialize PSRAM
    if(!psramInit()) {
        Serial.println("PSRAM initialization failed");
        while(1) delay(1000);
    }

    // Allocate tensor arena from PSRAM with specific capabilities
    tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (tensor_arena == nullptr) {
        Serial.println("Failed to allocate tensor arena in PSRAM");
        // Try allocating from regular memory as fallback
        tensor_arena = (uint8_t*)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_8BIT);
        if (tensor_arena == nullptr) {
            Serial.println("Failed to allocate tensor arena in regular memory");
            while(1) delay(1000);
        }
        Serial.println("Allocated tensor arena in regular memory");
    } else {
        Serial.println("Allocated tensor arena in PSRAM");
    }

    // Initialize the camera
    if (!InitCamera()) {
        Serial.println("Camera initialization failed");
        while(1) delay(1000);
    }

    Serial.println("Loading TFLite model...");
    model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model schema version mismatch!");
        while(1) delay(1000);
    }

    static tflite::MicroMutableOpResolver<7> resolver;
    resolver.AddQuantize();
    resolver.AddConv2D();
    resolver.AddMaxPool2D();
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddDequantize();

    // Create interpreter without error reporter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        while(1) delay(1000);
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("Setup completed successfully");
}

void loop() {
    if (start_loop) {
        const int num_images = 3; // Number of images to capture and process
        const int num_classes = output->dims->data[1];
        float class_scores_sum[6] = {0}; // Adjust size if number of classes is different
        int image_width = input->dims->data[1];
        int image_height = input->dims->data[2];

        for (int img_idx = 0; img_idx < num_images; ++img_idx) {
            float* image_data = (float*)heap_caps_malloc(
                image_width * image_height * sizeof(float),
                MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT
            );

            if (!image_data) {
                Serial.println("Failed to allocate image buffer");
                delay(1000);
                return;
            }

            if (!CaptureImage(image_data, image_width, image_height)) {
                Serial.println("Image capture failed");
                heap_caps_free(image_data);
                delay(1000);
                return;
            }

            // Copy image data to input tensor
            memcpy(input->data.f, image_data, image_width * image_height * sizeof(float));
            heap_caps_free(image_data);

            Serial.println("Running inference...");
            TfLiteStatus invoke_status = interpreter->Invoke();
            if (invoke_status != kTfLiteOk) {
                Serial.println("Invoke failed");
                delay(1000);
                return;
            }

            // Add the output scores to the sum
            for (int i = 0; i < num_classes; ++i) {
                float value = output->type == kTfLiteInt8 
                    ? (output->data.int8[i] - output->params.zero_point) * output->params.scale 
                    : output->data.f[i];
                class_scores_sum[i] += value;
            }
            delay(500); // Slight delay between image captures
        }

        // Average the results
        float class_scores_avg[6] = {0};
        for (int i = 0; i < num_classes; ++i) {
            class_scores_avg[i] = class_scores_sum[i] / num_images;
            Serial.printf("Class %d Avg Score: %.4f\n", i, class_scores_avg[i]);
        }

        // Determine the class with the highest average score
        int predicted_class = 0;
        float max_score = class_scores_avg[0];
        for (int i = 1; i < num_classes; ++i) {
            if (class_scores_avg[i] > max_score) {
                max_score = class_scores_avg[i];
                predicted_class = i;
            }
        }

        Serial.printf("Predicted Class: %d with Score: %.4f\n", predicted_class, max_score);

        char message[2] = {0};
        switch (predicted_class) {
          case 0: message[0] = 'A'; break;
          case 1: message[0] = 'B'; break;
          case 2: message[0] = 'C'; break;
          case 3: message[0] = 'D'; break;
          case 4: message[0] = 'E'; break;
          case 5: message[0] = 'F'; break;
          default: break;
        }
        sendData(peer_mac, message);

        delay(1000); // Prevent rapid looping
        loop_counter++;

        if (loop_counter >= 4){
          start_loop = false;
        }
    }
}