#pragma once
#include <Arduino.h>
#include <avr/pgmspace.h>
// Placeholder. Replaced by train_export.py after training on collected data.
#define MODEL_TRAINED 0
const int8_t W1[8][3] PROGMEM = {{0}};
const int8_t B1[8] PROGMEM = {0};
const int8_t W2[3][8] PROGMEM = {{0}};
const int8_t B2[3] PROGMEM = {0};
