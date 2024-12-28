import serial
import time
import datetime
import Funciones

#Lugar donde se generara la señal de 30s
#No anomalo, es decir es normal
file_path=f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/VOL/S1/CRD1/Data.mat"
struct_name="Data"
field_name="ECG_before_warmup"
column=1

start_segment=0
sampling_rate=300 

ecg_signal=Funciones.extract_mat_column(file_path,struct_name,field_name,column)

duration_seconds = (len(ecg_signal)/sampling_rate)# Duración total basada en la longitud de la señal y la tasa de muestreo
print(len(ecg_signal))
print(duration_seconds)

# Se obtiene la lista de datos
lista_datos=Funciones.process_ecg_signal(start_segment, duration_seconds, sampling_rate, ecg_signal)

# Se descomprime la lista de parametros
df_peaks, df_peaks_seconds, df_isolation, ecg_signal, cleaned_ecg, ecg_signals = lista_datos
Funciones.visualizar_datos(lista_datos, start_segment,duration_seconds,sampling_rate)

#--------Inicializo el el serial------------------
serialPort= "/dev/tty.usbserial-1120" #Comando en terminal: ls /dev/tty.*
ser=serial.Serial(serialPort,baudrate=115200,timeout=1) #Inicializo puerto serial

#---------Obtengo los valores del arduino------------------

intervalo_ms=1/300 #Frecuencia de 300 hz

try:
    for recording_id in range(1, 11):  # Cambiar entre grabaciones S1 al S10
        print(f"Procesando grabación: S{recording_id}")
        
        # Ruta del archivo actual
        #file_path=f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/BAS/BAS2/S{recording_id}/CRD1/Data.mat"
        file_path=f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/VOL/S{recording_id}/CRD1/Data.mat"
        
        # Extraer señal de ECG
        ecg_signal = Funciones.extract_mat_column(file_path, struct_name, field_name, column=1)
        duration_seconds = len(ecg_signal) / sampling_rate
        print(f"Longitud de señal: {len(ecg_signal)}, Duración: {duration_seconds} segundos")
        
        # Procesar la señal
        lista_datos = Funciones.process_ecg_signal(0, duration_seconds, sampling_rate, ecg_signal)
        
        # Transmitir datos por puerto serial
        intervalo_ms = 1 / sampling_rate
        for ecg_value in ecg_signal:
            ecg_value_str = str(ecg_value)
            ser.write(f"{ecg_value_str}\n".encode())
            print(ecg_value_str)
            time.sleep(intervalo_ms)
        

except KeyboardInterrupt:
    print("Lectura interrumpida por el usuario.")
except Exception as e:
    print(f"Error al procesar datos: {e}")
finally:
    ser.close()
    print("Puerto serial cerrado.")