/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "output_handler.h"
#include "tensorflow/lite/micro/micro_log.h"

const char* class_labels[] = {
    "Mano abierta", "Cerrada", "Pulgar", "Rock", "Tres dedos", "Un dedo"
};

void HandleOutput(float* probabilities, int num_classes) {
  if (num_classes != 6) {
    MicroPrintf("Unexpected number of classes: %d\n", num_classes);
    return;
  }

  // Log all probabilities
  MicroPrintf("Class Probabilities:\n");
  for (int i = 0; i < num_classes; ++i) {
    MicroPrintf("Class %d (%s): %.2f%%\n", i, class_labels[i], probabilities[i] * 100);
  }

  // Find the best prediction
  int best_class_index = 0;
  float best_probability = probabilities[0];
  for (int i = 1; i < num_classes; ++i) {
    if (probabilities[i] > best_probability) {
      best_class_index = i;
      best_probability = probabilities[i];
    }
  }

  // Log the best prediction
  MicroPrintf("Best Prediction: Class %d (%s) with %.2f%% confidence\n", 
              best_class_index, class_labels[best_class_index], best_probability * 100);
}
