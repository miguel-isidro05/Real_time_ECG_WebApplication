import scipy.io
import neurokit2 as nk
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

#Especificamente creado para extraer de .mat
def extract_mat_column(file_path, struct_name, field_name, column=None):
    # Cargar el archivo .mat
    mat_data = scipy.io.loadmat(file_path)
    
    # Acceder al struct y al campo
    struct = mat_data[struct_name]
    field = struct[field_name]
    
    # Extraer el array interno
    array_data = field[0][0]
    
    # Devolver la columna especificada o todas las columnas
    if column is not None:
        # Selecciona la columna especificada
        selected_column = np.array(array_data[:, column])
        
        # Filtrar los valores fuera del rango [-1.5, 2.5]
        selected_column = np.where((selected_column >= -1.5) & (selected_column <= 2.5), selected_column, 0)
        return selected_column
    
    # Filtrar todo el array
    filtered_array = np.where((array_data >= -1.5) & (array_data <= 2.5), array_data, 0)
    return np.array(filtered_array)

def process_ecg_signal(start_segment, duration_seconds, sampling_rate, ecg_signal):
    #Lista vacia
    lista_datos=[]
    
    #Cantidad de muestras
    total_samples = len(ecg_signal)

    # Calcular índices de inicio y fin del segmento
    start_index_segment = int(start_segment * sampling_rate)
    end_index_segment = int(duration_seconds * sampling_rate)
    segment = ecg_signal[start_index_segment:end_index_segment]
    
    # Limpiar la señal para detección de picos
    cleaned_ecg=segment

    #cleaned_ecg = nk.ecg_clean(segment, sampling_rate=sampling_rate,method="pantompkins1985")

    # Procesar la señal para encontrar picos y ondas PQRST
    ecg_signals, info = nk.ecg_process(cleaned_ecg, sampling_rate=sampling_rate)

    # Encontrar picos R
    peaks = nk.ecg_findpeaks(ecg_signals["ECG_Clean"], sampling_rate=sampling_rate)
    r_peaks = peaks["ECG_R_Peaks"]

    # Inicializar listas para almacenar los picos P, Q, S, y T
    p_peaks = []
    q_peaks = []
    s_peaks = []
    t_peaks = []

    # Detectar picos P, Q, S, y T en ventanas alrededor de cada pico R
    for r_peak in r_peaks:
        left = max(0, r_peak - int(0.2 * sampling_rate))
        right = min(len(ecg_signals["ECG_Clean"]), r_peak + int(0.4 * sampling_rate))
        segment = ecg_signals["ECG_Clean"][left:right]

        # Encuentra el pico Q como el mínimo local a la izquierda de R
        q_segment = ecg_signals["ECG_Clean"][left:r_peak]
        if len(q_segment) > 0:
            q_peak = left + np.argmin(q_segment)  # Buscar el mínimo dentro del segmento
            q_peaks.append(q_peak)

        # Encuentra el pico S como el mínimo local a la derecha de R
        # Se busca el mínimo en un rango a la derecha de R
        s_segment = ecg_signals["ECG_Clean"][r_peak:right]
        if len(s_segment) > 0:
            s_peak = r_peak + np.argmin(s_segment)  # Buscar el mínimo dentro del segmento
            s_peaks.append(s_peak)

        # Encuentra el pico P
        p_segment = ecg_signals["ECG_Clean"][left:q_peak]
        if len(p_segment) > 0:
            p_peak = left + np.argmax(p_segment)
            p_peaks.append(p_peak)

        # Encuentra el pico T
        t_segment = ecg_signals["ECG_Clean"][s_peak:right]
        if len(t_segment) > 0:
            t_peak = s_peak + np.argmax(t_segment)
            t_peaks.append(t_peak)

    print(f"Segment length: {len(segment)} samples")
    print(f"Detected R-peaks: {len(r_peaks)}")
    
    # Ajustar longitudes
    max_length = len(r_peaks)
    p_peaks = p_peaks[:max_length] + [np.nan] * (max_length - len(p_peaks))
    q_peaks = q_peaks[:max_length] + [np.nan] * (max_length - len(q_peaks))
    s_peaks = s_peaks[:max_length] + [np.nan] * (max_length - len(s_peaks))
    t_peaks = t_peaks[:max_length] + [np.nan] * (max_length - len(t_peaks))

    data = {
        "P_Peaks": p_peaks,
        "Q_Peaks": q_peaks,
        "R_Peaks": r_peaks,
        "S_Peaks": s_peaks,
        "T_Peaks": t_peaks

    }

    # Calcular intervalos RR
    rr_intervals = np.diff(r_peaks) / sampling_rate  # en segundos
    #---------------------------------------Obteniene amplitud de los picos T---------------------------------------

    # Lista para almacenar las amplitudes de los picos T
    t_amplitudes = []

    # Calcular amplitudes de los picos T
    for t_peak in t_peaks:
        if not np.isnan(t_peak) and 0 <= int(t_peak) < len(ecg_signals["ECG_Clean"]):  # Verifica que el índice sea válido
            # Valor de la señal en el pico T
            t_value = ecg_signals["ECG_Clean"][int(t_peak)]
            
            # Calcular línea base local (promedio de la señal en una ventana alrededor del pico T)
            left_base = max(0, int(t_peak) - int(0.1 * sampling_rate))  # 100 ms antes del pico T
            right_base = min(len(ecg_signals["ECG_Clean"]), int(t_peak) + int(0.1 * sampling_rate))  # 100 ms después del pico T
            baseline = np.mean(ecg_signals["ECG_Clean"][left_base:right_base])
            
            # Calcular la amplitud del pico T
            t_amplitude = t_value - baseline
            t_amplitudes.append(t_amplitude)
        else:
            t_amplitudes.append(np.nan)  # Si no hay pico T, coloca NaN

    #---------------------------------------Obteniene amplitud de los picos S---------------------------------------

    # Lista para almacenar las amplitudes de los picos S
    s_amplitudes = []

    # Calcular amplitudes de los picos S
    for s_peak in s_peaks:
        if not np.isnan(s_peak) and 0 <= int(s_peak) < len(ecg_signals["ECG_Clean"]):  # Verifica que el índice sea válido
            # Valor de la señal en el pico S
            s_value = ecg_signals["ECG_Clean"][int(s_peak)]
            
            # Calcular línea base local (promedio de la señal en una ventana alrededor del pico S)
            left_base = max(0, int(s_peak) - int(0.1 * sampling_rate))  # 100 ms antes del pico S
            right_base = min(len(ecg_signals["ECG_Clean"]), int(s_peak) + int(0.1 * sampling_rate))  # 100 ms después del pico S
            baseline = np.mean(ecg_signals["ECG_Clean"][left_base:right_base])
            
            # Calcular la amplitud del pico S
            s_amplitude = s_value - baseline
            s_amplitudes.append(s_amplitude)
        else:
            s_amplitudes.append(np.nan)  # Si el pico no es válido o está fuera de rango, coloca NaN

    #---------------------------------------Obteniene amplitud del segmento ST ---------------------------------------

    # Calcular el promedio de los valores en el segmento ST
    st_averages = []

    for s_peak, t_peak in zip(s_peaks, t_peaks):
        if not np.isnan(s_peak) and not np.isnan(t_peak):
            # Definir la ventana del segmento ST
            st_segment = ecg_signals["ECG_Clean"][int(s_peak):int(t_peak)]
            
            # Calcular el promedio del segmento ST
            st_average = np.mean(st_segment)
            st_averages.append(st_average)
        else:
            st_averages.append(np.nan)

    #---------------------------------------Obteniene amplitud del intervalo QT para QTc ---------------------------------------

    # Calcular el intervalo QT para cada latido
    qt_intervals = []
    for q_peak, t_peak in zip(q_peaks, t_peaks):
        if not np.isnan(q_peak) and not np.isnan(t_peak):  # Asegurarse de que ambos picos son válidos
            # Intervalo QT (en muestras)
            qt_interval = (t_peak - q_peak) / sampling_rate  # Convertir a segundos
            qt_intervals.append(qt_interval)
        else:
            qt_intervals.append(np.nan)  # Si no hay picos válidos, agregar NaN

    # Calcular el QT corregido (QTc) usando la fórmula de Bazett
    qtc_intervals = []
    for qt_interval in qt_intervals:
        if not np.isnan(qt_interval):
            qtc_interval = qt_interval / np.sqrt(rr_intervals.mean())  # QTc usando la fórmula de Bazett
            qtc_intervals.append(qtc_interval)
        else:
            qtc_intervals.append(np.nan)  # Si el intervalo QT es NaN, agregar NaN

    #---------------------------------------Obtener datos---------------------------------------
    # Tiempo máximo de la señal en segundos
    max_time = total_samples / sampling_rate

    # Mostrar el tiempo máximo
    #print(f"Total samples: {total_samples}")

    #print("Tiempo máximo de la señal ECG: "+str(max_time)+ " s")
    #print("\n")

    # Convertir el diccionario en un DataFrame
    df_peaks = pd.DataFrame(data)

    df_peaks['T_Amplitude'] = t_amplitudes #Agrega la amplitud de los picos T
    df_peaks['S_Amplitude'] = s_amplitudes #Agrega la amplitud de los picos S

    # Añadir los intervalos QTc al DataFrame
    df_peaks['QTc_Intervals'] = qtc_intervals

    # Mostrar las primeras filas del DataFrame
    #print(df_peaks)
    #print("\n") 

    # Convertir los índices de muestra a segundos
    df_peaks_seconds = df_peaks.copy()

    df_peaks_seconds['R_Peaks'] = df_peaks_seconds['R_Peaks'] / sampling_rate
    df_peaks_seconds['P_Peaks'] = df_peaks_seconds['P_Peaks'] / sampling_rate
    df_peaks_seconds['Q_Peaks'] = df_peaks_seconds['Q_Peaks'] / sampling_rate
    df_peaks_seconds['S_Peaks'] = df_peaks_seconds['S_Peaks'] / sampling_rate
    df_peaks_seconds['T_Peaks'] = df_peaks_seconds['T_Peaks'] / sampling_rate

    # Mostrar las primeras filas del DataFrame con los tiempos en segundos
    #print(df_peaks_seconds)
    #print("\n") 

    # Calcular HR
    hr = 60 / np.mean(rr_intervals)  # en latidos por minuto

    # Calcular la desviación estándar de los intervalos RR
    hrv = np.std(rr_intervals)

    # Calcular la amplitud promedio de las ondas T
    average_t_amplitude = np.nanmean(t_amplitudes)

    # Calcular la amplitud promedio de las ondas T
    average_ST_amplitude = np.nanmean(st_averages)

    # Calcular el QTc promedio
    average_qtc = np.nanmean(qtc_intervals)

    #Data isolation forest
    data_isolation_forest = {
        "HR": [hr],  # Frecuencia cardíaca
        "HRV": [hrv],  # Variabilidad de frecuencia cardíaca
        "Amplitud_T_prom": [average_t_amplitude],  # Promedio de amplitud de la onda T
        "Amplitud_ST_prom": [average_ST_amplitude],  # Promedio de amplitud del segmento ST
        "Intervalo_QTc_prom": [average_qtc]  # Intervalo QT corregido promedio
    }

    #Lo paso a un dataframe
    df_isolation = pd.DataFrame(data_isolation_forest)
    print(df_isolation)

    #Lista de todo las herramientas que necesito
    lista_datos.append(df_peaks)
    lista_datos.append(df_peaks_seconds)
    lista_datos.append(df_isolation)
    lista_datos.append(ecg_signal)
    lista_datos.append(cleaned_ecg)
    lista_datos.append(ecg_signals)

    return lista_datos

#----------------------------Modelo de Isolation Forest------------------

#Parameters:
#df_peaks_seconds: DataFrame que contiene los datos a analizar.
#contamination: Proporción de anomalías esperadas en los datos (por defecto 0.1).
#n_estimators: Número de árboles en el modelo de Isolation Forest (por defecto 100).
#random_state: Para garantizar reproducibilidad (por defecto 42).

def detect_anomalies(df_isolation, contamination=0.001, n_estimators=100, random_state=42):
    
    # Crear un diccionario para guardar los resultados de anomalías
    anomalies = {}

    # Iterar sobre las columnas del DataFrame
    for column in df_isolation.columns:
        
        # Parámetros del modelo
        model = IsolationForest(
            n_estimators=n_estimators,        # Número de árboles en el bosque
            contamination=contamination,      # Proporción de anomalías
            max_samples="auto",               # Usa todas las muestras
            max_features=1.0,                 # Usar todas las características
            bootstrap=False,                  # No usar reemplazo al crear los árboles
            random_state=random_state,        # Para reproducibilidad
        )
        
        # Entrenar el modelo
        model.fit(df_isolation[[column]])
        
        # Muestra los parametros similares para prediccion del modelo entrenado
        predictions = model.predict(df_isolation[[column]])
        
        # Guardar los resultados en el diccionario de anomalías
        anomalies[column] = pd.Series(predictions)

        # Crear un DataFrame con los resultados de las anomalías
        anomaly_results = df_isolation[[column]].copy()  # Copiar la columna de datos originales
        anomaly_results['Anomaly'] = anomalies[column]  # Agregar la columna de anomalías

        # Mostrar los primeros 10 resultados de anomalía (1: No Anómalo, -1: Anómalo)
        print(f"Resultados de anomalía para {column}: \n", anomaly_results.head(20))

        # Guardar el modelo entrenado
        model_filename = f"model_{column}.joblib"
        joblib.dump(model, model_filename)
        print(f"Modelo para {column} guardado como {model_filename}")
    

def visualizar_datos(lista_datos, start_segment, duration_seconds, sampling_rate):
    
    # Desempaquetar la lista de datos
    df_peaks, df_peaks_seconds, df_isolation, ecg_signal, cleaned_ecg, ecg_signals = lista_datos
    
    # Descomprimir los picos de ECG
    p_peaks = df_peaks["P_Peaks"]
    q_peaks = df_peaks["Q_Peaks"]
    r_peaks = df_peaks["R_Peaks"]
    s_peaks = df_peaks["S_Peaks"]
    t_peaks = df_peaks["T_Peaks"]

    # Definir los índices de la señal
    start_index = int(start_segment)
    duration_fs = int(duration_seconds)
    end_index = int(start_index + (sampling_rate * duration_fs))

    # Crear un vector de tiempo basado en la tasa de muestreo
    time_vector = np.arange(start_segment, start_segment + duration_seconds, 1 / int(sampling_rate))

    # Visualización de la señal ECG
    plt.figure(figsize=(15, 18))

    # Señal original
    plt.subplot(7, 1, 1)
    plt.plot(time_vector, ecg_signal[start_index:end_index], color='blue', label="ECG Original")
    plt.legend()

    # Señal limpia
    plt.subplot(7, 1, 2)
    plt.plot(time_vector, cleaned_ecg[start_index:end_index], label='ECG Limpio', color='blue')

    # Señal limpia de ecg_signals
    plt.subplot(7, 1, 3)
    plt.plot(time_vector, ecg_signals['ECG_Clean'][start_index:end_index], label='ECG Limpio', color='blue')

    # Función para graficar los picos con validación
    def plot_peaks(peaks, color, label):
        peaks_plot = [peak for peak in peaks if start_index <= peak < end_index]
        peaks_time = [time_vector[int(peak)] for peak in peaks_plot if 0 <= int(peak) < len(time_vector)]
        plt.scatter(peaks_time, ecg_signals['ECG_Clean'][peaks_plot], color=color, label=label, zorder=5)

    # Graficar los picos R
    plot_peaks(r_peaks, 'red', 'Picos R')

    # Graficar los picos P
    plot_peaks(p_peaks, 'green', 'Picos P')

    # Graficar los picos Q
    plot_peaks(q_peaks, 'purple', 'Picos Q')

    # Graficar los picos S
    plot_peaks(s_peaks, 'orange', 'Picos S')

    # Graficar los picos T
    plot_peaks(t_peaks, 'cyan', 'Picos T')

    plt.title("Señal ECG Limpia con Picos P, Q, R, S y T Detectados")
    plt.legend()

    # Visualización final
    plt.tight_layout()
    plt.show()
