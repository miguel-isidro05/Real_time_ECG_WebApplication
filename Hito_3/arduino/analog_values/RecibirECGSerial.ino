#include <Arduino.h>
#include "BluetoothSerial.h"

// Declaramos la variable cadena para almacenar los datos que lleguen
BluetoothSerial BT;

// Variable para almacenar el tiempo de la última lectura
unsigned long lastTime = 0;

// Intervalo de tiempo entre lecturas (en milisegundos)
unsigned long interval = 3.33;

void setup() {
  // Le damos nombre al dispositivo Bluetooth "Doctor Bocanegra"
  BT.begin("CardioSportX");
  Serial.begin(115200);  // Para monitoreo por serial si es necesario
  Serial.println("Bluetooth iniciado, enviando datos...");
}

void loop() {
  if (BT.hasClient()) { // Verifica si hay algún dispositivo conectado
    unsigned long currentTime = millis(); // Obtiene el tiempo actual en milisegundos

    // Solo lee el valor del potenciómetro si ha pasado el intervalo
    if (currentTime - lastTime >= interval) {
      if(Serial.available() > 0){
        // Lee el valor del potenciómetro (de 0 a 1023)
        String potValue = Serial.readStringUntil('\n');
        float potValueInt = potValue.toFloat();

        // Muestra el valor del potenciómetro y el tiempo transcurrido
        BT.print(potValueInt); // Potenciómetro
        BT.print(",");      // Separador de coma
        BT.println(currentTime); // Tiempo en milisegundos

        // Actualiza el tiempo de la última lectura
        lastTime = currentTime;
      }
    }
  }
}

