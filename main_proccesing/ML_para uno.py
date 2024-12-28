import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from Funciones import extract_mat_column as extract
from Funciones import process_ecg_signal
from Funciones import detect_anomalies
from Funciones import visualizar_datos

#Falta obtener modelo de todos las carpetas
#Falta pasarlo a funcion
#Falta normalizar datos de 0 a 2,5mV para la amplitud (No importa mucho)
#Falta usar prediccion con datos importados del resultados.mat

#Itera para que en cada ECG_recovery se use la extraccion de datos
field_name = "ECG_resting" 
struct_name = "Data"

cant_recording=10 #Cantidad de recordings de ECG del database de un deporte especifico

# Crear un DataFrame vacío con las columnas necesarias al inicio
df_resultados_isolation = pd.DataFrame(columns=["HR", "HRV", "Amplitud_T_prom", "Amplitud_ST_prom", "Intervalo_QTc_prom"])

#Empieza en S1
i=1

for ecg_recording in range(cant_recording):

    #Cambia S1 de file path por S2 y asi sucesivamente con i=i+1 para moverse de ubicacion
    file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/BAS/BAS1/S{1}/CRD1/Data.mat"

    print("Para "+str(field_name)+" --> S"+str(i)+": \n")

    # Para obtener solo la primera (0) y segunda columna (1)
    dat1 = extract(file_path, struct_name, field_name, column=0)
    ecg_signal = extract(file_path, struct_name, field_name, column=1)

    # Normalizar la señal ECG entre 0.25 mV y 15 mV
    #nuevo_min, nuevo_max = 0.25, 15
    #min_actual, max_actual = np.min(ecg_signal), np.max(ecg_signal)
    #ecg_signal_normalizado = nuevo_min + (ecg_signal - min_actual) * (nuevo_max - nuevo_min) / (max_actual - min_actual)

    # Escalar la señal
    sampling_rate = 300  # Tasa de muestreo en Hz

    # Seleccionar un segmento de la señal ECG
    start_segment = 0 # Tiempo de inicio en segundos (Normalmente al inicio esta raro la señal)
    duration_seconds = (len(ecg_signal)/sampling_rate)# Duración total basada en la longitud de la señal y la tasa de muestreo

    # Se obtiene la lista de datos
    lista_datos=process_ecg_signal(start_segment, duration_seconds, sampling_rate, ecg_signal)

    # Se descomprime la lista de parametros
    df_peaks, df_peaks_seconds, df_isolation, ecg_signal, cleaned_ecg, ecg_signals = lista_datos

    df_resultados_isolation = pd.concat([df_resultados_isolation, df_isolation], ignore_index=True)

    #visualizar_datos(lista_datos, start_segment,duration_seconds,sampling_rate)
    i=i+1

#Probamos el Isolation Forest---------------

# Mostrar el DataFrame final
print("\n DataFrame final con todos los resultados de Isolation Forest:")
print(df_resultados_isolation)

# Crear un diccionario para guardar resultados de anomalías
anomalies=detect_anomalies(df_resultados_isolation, contamination=0.1, n_estimators=100, random_state=42)

anomalies_df = pd.DataFrame(anomalies)

# Iterar sobre cada columna para eliminar filas marcadas como anomalías
for column in anomalies_df.columns:
    # Obtener los índices de las filas marcadas como anomalías
    anomaly_indices = anomalies_df.index[anomalies_df[column] == -1]

    # Verificar que los índices existen en el DataFrame original antes de eliminarlos
    anomaly_indices = anomaly_indices.intersection(df_resultados_isolation.index)

    # Eliminar las filas con anomalías
    df_resultados_isolation.drop(index=anomaly_indices, inplace=True)

# Mostrar el DataFrame final sin anomalías
print("\n DataFrame final sin anomalías:")
print(df_resultados_isolation)

# Prediceme la anomalia usando el S12 ....
file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/VOL/S{i}/CRD1"

# Para obtener solo la primera (0) y segunda columna (1)
dat1 = extract(file_path, struct_name, field_name, column=0)
ecg_signal = extract(file_path, struct_name, field_name, column=1)

lista_datos=process_ecg_signal(start_segment, duration_seconds, sampling_rate, ecg_signal)
df_peaks, df_peaks_seconds, df_isolation, ecg_signal, cleaned_ecg, ecg_signals = lista_datos

# Iterar sobre las columnas del DataFrame
print("S12 probado en el modelo obtenido de los demas recordings: \n")
for column in df_isolation.columns:
    
    model_filename = f"model_{column}.joblib"
    model = joblib.load(model_filename)  # Cargar el modelo

    # Realizar predicciones (1: No Anómalo, -1: Anómalo)
    predictions = model.predict(df_isolation[[column]])
        
    # Guardar los resultados en el diccionario de anomalías
    anomalies[column] = pd.Series(predictions)

    # Crear un DataFrame con los resultados de las anomalías
    anomaly_results = df_isolation[[column]].copy()  # Copiar la columna de datos originales
    anomaly_results['Anomaly'] = anomalies[column]  # Agregar la columna de anomalías
    
    # Mostrar los primeros 10 resultados de anomalía (1: No Anómalo, -1: Anómalo)
    print(f"Resultados de anomalía para {column}: \n", anomaly_results.head(20))


