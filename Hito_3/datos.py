import os
import scipy.io as sio
from pymongo import MongoClient

# -------- Inicializo el cliente de MongoDB ------------------

# Link del cluster y base de datos de MongoDB a usar
client = MongoClient("mongodb+srv://test:test@cluster0.c7ax3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.myfirstdb

# Cargo la colección que se quiere usar del cliente
collections = db.myfirstcoll

def save_data():
    # Recuperar todos los datos de la base de datos
    data = list(collections.find({}))
    
    if not data:
        raise Exception("No se encontraron datos para guardar.")
    
    # Extraer las variables relevantes y convertirlas en columnas (listas de listas)
    time_data = [[entry['tiempo_formato']] for entry in data][::-1]  # Invertir si es necesario
    pot_data = [[entry['potenciometro']] for entry in data][::-1]  # Invertir si es necesario
    
    # -------- Preparación del archivo .mat ------------------

    # Especificar la carpeta y el archivo
    carpeta = "signal_processing"
    archivo_mat = os.path.join(carpeta, "resultados.mat")

    # Verificar si la carpeta existe
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)  # Si no existe, crea la carpeta

    # Crear la estructura del archivo .mat con el formato requerido
    mat_data = {
        'data_ecg': {
            'ecg': [[pot, time] for pot, time in zip(pot_data, time_data)]  # Combina los datos en un solo campo
        }
    }

    try:
        # Guardar los resultados en un archivo .mat
        sio.savemat(archivo_mat, mat_data, appendmat=False)
        print("Datos guardados exitosamente en", archivo_mat)
    except Exception as e:
        raise Exception(f"Error al guardar los datos: {str(e)}")
