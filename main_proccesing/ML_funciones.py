import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from Funciones import extract_mat_column as extract
from Funciones import process_ecg_signal
from Funciones import detect_anomalies
from Funciones import visualizar_datos
from Funciones import extract_mat_column_reverse

#i: inicializa en 1 para empezar con (S1)

#file_path: ruta con formato...
#file_path = f"/Users/miguel_05/Desktop/.../S{i}/CRD1/Data.mat"

#struct_name: nombre de la archivo que se desea importar
#Ej: struct_name = "Data"

#field_name: nombre del campo importado del .mat
#Ej: field_name = "ECG_resting" 

#cant_recordings: cantidad de recodgins de ECG
#Ej: cant_recording=10

#sampling rate: frecuencia de muestreo del .mat
#Ej: sampling_rate = 300  # Tasa de muestreo en Hz

#start_segment: tiempo inicial de la grafica (RECOMENDADO=0)

#Funcion que entrenara al modelo desde "start_segment" hasta el final de los datos
#usando un archivo .mat perteneciente a la columna "1", usando todos
#los database (S) de una carpeta especifica de deportes

def entrenar_modelo(i,struct_name,field_name,cant_recording,sampling_rate,start_segment):
    
    # Crear un DataFrame vacío con las columnas necesarias al inicio
    df_resultados_isolation = pd.DataFrame(columns=["HR", "HRV", "Amplitud_T_prom", "Amplitud_ST_prom", "Intervalo_QTc_prom"])

    #Logica para empezar por BAS, SOC, VOL

    #file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/BAS/BAS1/S{i}/CRD1/Data.mat"
    #file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/VOL/S{i}/CRD1/Data.mat"
    j=0
    while(j<4):
        for ecg_recording in range(cant_recording):

            #Cambia S1 de file path por S2 y asi sucesivamente con i=i+1 para moverse de ubicacion    
            print("Para "+str(field_name)+" --> S"+str(i)+": \n")

            #Dependiendo de cual iteracion sea se usa un file_path u otro
            if j==0:
                file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/BAS/BAS1/S{i}/CRD1/Data.mat"
                # Para obtener solo la primera (0) y segunda columna (1)
                dat1 = extract(file_path, struct_name, field_name, column=0)
                ecg_signal = extract(file_path, struct_name, field_name, column=1)

            elif j==1:
                file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/VOL/S{i}/CRD1/Data.mat"
                field_name= "ECG_recovery5"
                # Para obtener solo la primera (0) y segunda columna (1)
                dat1 = extract(file_path, struct_name, field_name, column=0)
                ecg_signal = extract(file_path, struct_name, field_name, column=1)

            elif j==2:
                file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/SOC/SOC1/S{i}/CRD1/Data.mat"
                field_name= "ECG_recovery5"
                # Para obtener solo la primera (0) y segunda columna (1)
                dat1 = extract(file_path, struct_name, field_name, column=0)
                ecg_signal = extract_mat_column_reverse(file_path, struct_name, field_name, column=1)

            elif j==3:
                file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/SOC/SOC2/S{i}/CRD1/Data.mat"
                field_name= "ECG_recovery5"
                # Para obtener solo la primera (0) y segunda columna (1)
                dat1 = extract(file_path, struct_name, field_name, column=0)
                ecg_signal = extract_mat_column_reverse(file_path, struct_name, field_name, column=1)

            # Normalizar la señal ECG entre 0.25 mV y 15 mV
            #nuevo_min, nuevo_max = 0.25, 15
            #min_actual, max_actual = np.min(ecg_signal), np.max(ecg_signal)
            #ecg_signal = nuevo_min + (ecg_signal - min_actual) * (nuevo_max - nuevo_min) / (max_actual - min_actual)

            duration_seconds = (len(ecg_signal)/sampling_rate)# Duración total basada en la longitud de la señal y la tasa de muestreo

            # Se obtiene la lista de datos
            lista_datos=process_ecg_signal(start_segment, duration_seconds, sampling_rate, ecg_signal)

            # Se descomprime la lista de parametros
            df_peaks, df_peaks_seconds, df_isolation, ecg_signal, cleaned_ecg, ecg_signals = lista_datos

            df_resultados_isolation = pd.concat([df_resultados_isolation, df_isolation], ignore_index=True)

            #ACTIVAR SI DESEA VISUALIZAR LOS DATOS QUE SON USADOS PARA EL ENTRENAMIENTO UNO POR UNO
            #visualizar_datos(lista_datos, start_segment,duration_seconds,sampling_rate)
            
            i=i+1 #Incrementa en uno
        i=1

        #Accion para ir concatenando el archivo
        j=j+1

    #----------Probamos el Isolation Forest---------------

    # Mostrar el DataFrame final
    print("\n DataFrame final con todos los resultados de Isolation Forest:")
    print(df_resultados_isolation)

    # Crear un diccionario para guardar resultados de anomalías en joblib
    detect_anomalies(df_resultados_isolation, contamination=0.1, n_estimators=100, random_state=42)

