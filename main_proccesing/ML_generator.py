import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import IsolationForest
from Funciones import extract_mat_column as extract
from Funciones import process_ecg_signal

from ML_funciones import entrenar_modelo

#Falta obtener modelo de todos las carpetas
#Falta pasarlo a funcion
#Falta normalizar datos de 0 a 2,5mV para la amplitud (No importa mucho)
#Falta usar prediccion con datos importados del resultados.mat

#Itera para que en cada ECG_recovery se use la extraccion de datos
field_name = "ECG_recovery0" 
struct_name = "Data"

cant_recording=8 #Cantidad de recordings de ECG del database de un deporte especifico

# Escalar la señal
sampling_rate = 300  # Tasa de muestreo en Hz

# Seleccionar un segmento de la señal ECG
start_segment = 0 # Tiempo de inicio en segundos (Normalmente al inicio esta raro la señal)

#Empieza en S1
i=1

entrenar_modelo(i,struct_name,field_name,cant_recording,sampling_rate,start_segment)

