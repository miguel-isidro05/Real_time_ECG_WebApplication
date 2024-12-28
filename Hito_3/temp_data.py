import os
import csv
import scipy.io as sio
from app import data_list  # Asegúrate de que 'data_list' esté importado correctamente

def save_data_temp(data_list):
    # Verificar si 'data_list' contiene datos
    if not data_list:
        raise Exception("No se encontraron datos para guardar.")
    
    # Extraer las variables relevantes de data_list
    time_data = [[entry['tiempo_formato']] for entry in data_list][::-1]  # Invertir si es necesario
    pot_data = [[entry['potenciometro']] for entry in data_list][::-1]  # Invertir si es necesario

    # -------- Preparación de archivos y carpetas ------------------

    carpeta = "signal_processing"
    archivo_mat = os.path.join(carpeta, "resultados_temp.mat")
    archivo_csv = os.path.join(carpeta, "resultados_temp.csv")

    # Verificar si la carpeta existe, si no, crearla
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    # -------- Guardar en archivo .mat ------------------
    mat_data = {
        'data_ecg': {
            'ecg': pot_data,  # Potenciómetro
            'tiempo': time_data  # Tiempo
        }
    }

    try:
        sio.savemat(archivo_mat, mat_data, appendmat=False)
        print("Datos guardados exitosamente en", archivo_mat)
    except Exception as e:
        raise Exception(f"Error al guardar los datos en .mat: {str(e)}")

    # -------- Guardar en archivo .csv ------------------
    try:
        with open(archivo_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["tiempo_formato", "potenciometro"])  # Encabezados

            # Escribir los datos
            for tiempo, pot in zip(time_data, pot_data):
                writer.writerow([tiempo[0], pot[0]])  # Desempaquetar listas para el CSV
        
        print("Datos guardados exitosamente en", archivo_csv)
    except Exception as e:
        raise Exception(f"Error al guardar los datos en .csv: {str(e)}")
