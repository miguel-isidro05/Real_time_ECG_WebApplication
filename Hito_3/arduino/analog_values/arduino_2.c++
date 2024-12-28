#include <Arduino.h>

void setup() {
  Serial.begin(115200);  // Configura la comunicación serial a 115200 baudios
}

void loop() {
  if (Serial.available() > 0) {
    String receivedData = Serial.readStringUntil('\n');  // Leer hasta salto de línea
    if (receivedData.length() > 0) {
      Serial.println(receivedData);
      Serial.println(",")
    }
  }
}
