void setup() {
  pinMode(35,INPUT);
  pinMode(34,INPUT);
  Serial.begin(115200);
}

void loop() {
  int A = digitalRead(2);
  int B = digitalRead(4);

  if (A == 0 || B == 0) {
    double ecg = analogReadMilliVolts(36);
    Serial.println(ecg);
  }
  else {
    Serial.println("Desconectado");
  }
  delay(10);
}