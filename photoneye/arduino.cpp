#include <Arduino.h>

int lightSensorPin = A0;

void setup() {
    Serial.begin(115200);
}

void loop() {
    int lightLevel = analogRead(lightSensorPin);
    Serial.println(lightLevel);
    delay(1);
}