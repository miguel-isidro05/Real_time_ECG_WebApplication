import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns

from sklearn.ensemble import IsolationForest
from Funciones import extract_mat_column as extract
from Funciones import process_ecg_signal
from Funciones import extract_mat_column_reverse

from ML_funciones import entrenar_modelo

anomalies={}

# Prediceme la anomalia usando el S12 ....
file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/BAS/BAS1/S9/CRD1/Data.mat"
field_name = "ECG_resting" 
struct_name = "Data"

start_segment=0
sampling_rate = 300  # Tasa de muestreo en Hz

#Cant recordings para cantidad de recordings que deseo
#Vario i para cambiar el recording
j=0
i=9
cant_recording=3
anomalies_all = []

while(j<5):

    #Dependiendo de cual iteracion sea se usa un file_path u otro
    if j==0:

        cant_recording=2 #S9 y S10
        print("Probado en el modelo obtenido de los recordings en BAS1: \n")

    elif j==1:

        cant_recording=3
        print("Probado en el modelo obtenido de los recordings en VOL: \n")

    elif j==2:

        cant_recording=2
        print("Probado en el modelo obtenido de los recordings en SOC1: \n")

    elif j==3:

        cant_recording=1
        print("Probado en el modelo obtenido de los recordings en SOC2: \n")

    elif j==4:

        cant_recording=2
        print("Probando en el modelo obtenido de los recordings en BAS2: \n")

    for ecg_recording in range(cant_recording):
        #Dependiendo de cual iteracion sea se usa un file_path u otro
        if j==0:
            file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/BAS/BAS1/S{i}/CRD1/Data.mat"
            field_name= "ECG_resting"

            # Para obtener solo la primera (0) y segunda columna (1)
            dat1 = extract(file_path, struct_name, field_name, column=0)
            ecg_signal = extract(file_path, struct_name, field_name, column=1)

        elif j==1:
            file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/VOL/S{i}/CRD1/Data.mat"
            field_name= "ECG_before_warmup"

            # Para obtener solo la primera (0) y segunda columna (1)
            dat1 = extract(file_path, struct_name, field_name, column=0)
            ecg_signal = extract(file_path, struct_name, field_name, column=1)

        elif j==2:
            file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/SOC/SOC1/S{i}/CRD1/Data.mat"
            field_name= "ECG_resting"

            # Para obtener solo la primera (0) y segunda columna (1)
            dat1 = extract(file_path, struct_name, field_name, column=0)
            ecg_signal = extract_mat_column_reverse(file_path, struct_name, field_name, column=1)

        elif j==3:
            file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/SOC/SOC2/S{i}/CRD1/Data.mat"
            field_name= "ECG_resting"

            # Para obtener solo la primera (0) y segunda columna (1)
            dat1 = extract(file_path, struct_name, field_name, column=0)
            ecg_signal = extract_mat_column_reverse(file_path, struct_name, field_name, column=1)

        elif j==4:
            file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/BAS/BAS2/S{i}/CRD1/Data.mat"
            field_name= "ECG_resting"

            # Para obtener solo la primera (0) y segunda columna (1)
            dat1 = extract(file_path, struct_name, field_name, column=0)
            ecg_signal = extract(file_path, struct_name, field_name, column=1)

        #Cambia S1 de file path por S2 y asi sucesivamente con i=i+1 para moverse de ubicacion    
        print("Para "+str(field_name)+" --> S"+str(i)+": \n")

        duration_seconds = (len(ecg_signal)/sampling_rate)# Duración total basada en la longitud de la señal y la tasa de muestreo

        # Se obtiene la lista de datos
        lista_datos=process_ecg_signal(start_segment, duration_seconds, sampling_rate, ecg_signal)
        df_peaks, df_peaks_seconds, df_isolation, ecg_signal, cleaned_ecg, ecg_signals = lista_datos

        #Obtiene la prediccion por cada parametro
        for column in df_isolation.columns:

            #Carga el modelo
            model_filename = f"Modelos_Entrenados/ECG_movement/model_{column}.joblib"
            model = joblib.load(model_filename)  # Cargar el modelo

            # Realizar predicciones (1: No Anómalo, -1: Anómalo)
            predictions = model.predict(df_isolation[[column]])
            anomalies_all.extend(predictions)

            # Guardar los resultados en el diccionario de anomalías
            anomalies[column] = pd.Series(predictions)

            # Crear un DataFrame con los resultados de las anomalías
            anomaly_results = df_isolation[[column]].copy()  # Copiar la columna de datos originales
            anomaly_results['Anomaly'] = anomalies[column]  # Agregar la columna de anomalías
            
            # Mostrar los primeros resultados de anomalía (1: No Anómalo, -1: Anómalo)   
            print(f"Resultados de anomalía para {column}: \n", anomaly_results.head(20))

        i=i+1 #Incrementa en uno
    i=9

    #Accion para ir concatenando el archivo
    j=j+1

# Supongamos que 'anomalies_all' contiene las predicciones
anomalies_all = np.array(anomalies_all)  # Convertir a numpy array
anomalies_all_sorted = np.array(anomalies_all)
anomalies_all_sorted = anomalies_all_sorted[anomalies_all_sorted == -1].tolist() + anomalies_all_sorted[anomalies_all_sorted == 1].tolist()

print(anomalies_all)

# Contar cuántos valores son anómalos (-1) y cuántos no lo son (1)
counts = {
    'Anomalous': np.sum(anomalies_all == -1),
    'Not Anomalous': np.sum(anomalies_all == 1)
}

# Crear un gráfico de barras
sns.barplot(x=list(counts.keys()), y=list(counts.values()))
plt.title("Conteo de Anomalías vs. No Anomalías")
plt.ylabel("Cantidad")
plt.show()



