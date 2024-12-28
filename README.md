# Real_time_ECG_WebApplication
Aplicación web para la detección de anomalías en señales de ECG en tiempo real, específicamente diseñada para deportistas en movimiento. Esta aplicación tiene como objetivo proporcionar un monitoreo continuo de la actividad cardíaca de los deportistas, permitiendo la detección temprana de anomalías cardíacas y mejorando la prevención en deportistas de alto rendimiento.

# Introduccion:

Las enfermedades cardiovasculares (ECV) son una de las principales causas de muerte a nivel mundial, y el ejercicio intenso de resistencia puede aumentar el riesgo de anomalías cardíacas, incluso en atletas altamente entrenados. Este proyecto propone un sistema portátil de ECG integrado con machine learning para detectar anomalías cardíacas en tiempo real durante actividades de alta resistencia. Enfocado en deportes como ciclismo, running y baloncesto, busca superar las limitaciones actuales, ofreciendo una solución cómoda, fiable y precisa para mejorar la salud cardiovascular de los deportistas.

# Explicacion:

### Generación y procesamiento de la señal:
El proceso de obtención de la señal comienza en el momento que el ESP32 transmite datos a través de la conexión bluetooth a la aplicación web, de tal forma que se obtendrá los valores a tiempo real y guardará temporalmente en la aplicación web.

Antes del análisis de datos, se carga y extrae la señal desde un archivo .mat, puesto que la base de datos usada (Sport DB 2.0 [8]) posee los archivos en formato .mat, por lo cual es necesario trabajar con dicho formato para simplificar el proceso de extracción  y segmentación de la señal [12]. En ese sentido, se usará una frecuencia de muestreo de 300 Hz, pues de todos los productos comerciales usados en la base de datos el más fiable es  Kadia mobile 6L [8], ya que ha sido aprobada por la FDA para uso en diagnóstico médico [9], por lo cual se usará sólo en los deportes de resistencia (basketball, volleyball y soccer) correspondiente a ese dispositivo para una mejor correlación entre datos obtenidos y reales.

### Detección de picos PQRST:
Las ondas PQRST conforman puntos clave dentro del ECG, aquellos son la consecuencia de los potenciales de acción producidos tras los latidos cardíacos del corazón. En ese sentido, la capacidad que poseen dichas señales para transmitir información acerca del funcionamiento del corazón de un paciente lo convierte en una gran herramienta para monitorear y prevenir enfermedades cardiovasculares sobre todo en deportistas, ya que su corazón puede tener una morfología distinta respecto a una persona "normal".

Para esta detección es vital primero  detectar el pico R, el cual representa la despolarización ventricular, pues al tener la curva más "empinada" de todos los picos, lo convierte en un buen punto de partida para obtener los demás picos.
<img width="276" alt="image" src="https://github.com/user-attachments/assets/c0cffacb-ee31-4a84-a566-beaa5a50d439" />

### Parámetros de la señal:
Después de obtener los picos de la señal, se obtiene el promedio de los parámetros respectivos, ya que se realizará un modelo de ML por cada uno para que se especializan en detectar anomalías en cualquiera de los parámetros, pues cualquier variación anormal a largo plazo podría significar verdaderos problemas de salud.

## Hito 3: 
Carpeta donde se encuentra la aplicacion web creada en el proyecto "Sistema de ECG inalámbrico con aplicación web para monitoreo y deteccion de anomalías cardíacas en deportistas de alto rendimiento" desarrollado como parte del "Proyecto de Biodiseño 1".
#
<p style="text-align: center;">
  <img width="512" alt="image" src="https://github.com/user-attachments/assets/e55a9cfd-bdf7-4158-a226-8abd22423f9d" />
</p>

## main_proccesing: 
Carpeta donde se encuentra el analisis y procesado de la base de datos de SportDB2.0 para el modelo de Isolation Forests.
#

<p style="text-align: center;">
  <img width="795" alt="image" src="https://github.com/user-attachments/assets/c85afe43-5ba3-44ca-bcd9-0ad069996b5a" />
</p>

Enlace del video sobre el funcionamiento de la aplicacion web y el emulador de ECG en conjunto: https://drive.google.com/file/d/1lkdfKj6ReUG6c930V4KiMxVPHxJV2eVt/view?usp=sharing
