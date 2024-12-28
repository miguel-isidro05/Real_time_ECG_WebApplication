from fastapi import FastAPI,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

import time
import asyncio
import serial
from pydantic import BaseModel
from pymongo import MongoClient
import datos  # Importa el archivo datos.py para guardar los datos en formato .mat
import temp_data
from Funciones import extract_mat_column as extract
from Funciones import process_ecg_signal
import joblib
import pandas as pd

#Configurare el timer de temporizador en signal_menu2 para que cuando ponga guardar datos me recorte en base al tiempo el data_list y eso mande a mongodb
#Timer para restablecer cada 10 s

#Prioridad, hacer que empieze y empieze el temporizador de 10s, termina 10s, de haber creado con datos el .mat, lo uso para predecir e imprimo resultados.

data_list=[]
data_list_cloud=[]
app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuración MongoDB
client = MongoClient("mongodb+srv://test:test@cluster0.c7ax3.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
db = client.myfirstdb
collections = db.myfirstcoll
register_db=db.user
pacient_db=db.pacient

result = collections.delete_many({})

# Modelo para recibir el estado del botón
class GraphState(BaseModel):
    state: int

class LoginUser(BaseModel):
    email: str
    password: str
    
# Modelo de datos
class User(BaseModel):
    name: str
    address: str
    email: str
    pass_: str

class Paciente(BaseModel):

    sexo: str
    deporte: str
    altura: str
    peso: str
    tasa: str
    fuma: str
    bebe: str

class ECGData(BaseModel):
    HR: float
    HRV: float
    Amplitud_T_prom: float
    Amplitud_ST_prom: float
    Intervalo_QTc_prom: float

# Variable global para almacenar los datos
stored_data = []

# Variable global para almacenar el último dato leído
latest_data = "Esperando datos..."
graphing_state = 0  # Estado inicial es 0 (no está graficando)
count=0
ser = None

# Función para calcular el tiempo transcurrido en formato HH:MM:SS.mmm
def format_elapsed_time(elapsed_seconds):
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = (elapsed_seconds - int(elapsed_seconds)) * 1000
    return f"{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03}"

# Función asíncrona para leer datos del puerto serial continuamente
async def read_serial():
    global latest_data
    global count
    global start_time

    while True:
        try:
            arduino_data = ser.readline()
            latest_data = arduino_data.decode("utf8", errors="ignore").strip()

            if latest_data:
                try:
                    # Dividir los datos por coma
                    valor_arduino = latest_data.split(",")

                    # Asegurar validación adecuada
                    if len(valor_arduino) == 2: 
                        pot_value = float(valor_arduino[0])
                        elapsed_time = int(valor_arduino[1]) / 1000  # Convertir a segundos
                        
                        if -1.5 <= pot_value <= 2.5:
                            formatted_time = format_elapsed_time(elapsed_time)

                            # Evitar duplicados
                            data = {
                                "id": f"Proyecto_biodiseño-{elapsed_time}",
                                "potenciometro": pot_value,
                                "tiempo_formato": formatted_time
                            }
                            # Agregar solo si es nuevo (si no existe ya en data_list).
                            if data not in data_list:
                                data_list.append(data)
                            
                            data_list_cloud.append(data)

                except ValueError:
                    latest_data = "Error en los datos recibidos"

        except Exception as e:
            latest_data = f"Error: {str(e)}"

        await asyncio.sleep(0.00001)  # Controla frecuencia de lectura.


# Función para guardar los datos en MongoDB cada 10 segundos
async def save_data_periodically():
    while True:
        try:
            if data_list:
                # Insertar los datos como un solo documento
                temp_data.save_data_temp(data_list)
                data_list.clear()  # Limpiar la lista después de guardarlos
                stored_data.clear()
                
                # Extraemos los datos resultados_temp
                file_path = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Signal_processing/SportDB2/VOL/S6/CRD1/Data.mat"
                field_name = "ECG_resting" 
                struct_name = "Data"
                sampling_rate= 300  #Coloca el sampleo en hrzt usados
                start_segment = 0 

                anomalies=[]
                # Para obtener solo la primera (0) y segunda columna (1)
                ecg_signal = extract(file_path, struct_name, field_name, column=1)

                #Mantengo duracion total de la señal
                duration_seconds = (len(ecg_signal)/sampling_rate)# Duración total basada en la longitud de la señal y la tasa de muestreo
                print("Duracion de la señal: ",duration_seconds)

                #Procesa los datos y lo deja en una lista
                lista_datos=process_ecg_signal(start_segment, duration_seconds, sampling_rate, ecg_signal)
                df_peaks, df_peaks_seconds, df_isolation, ecg_signal, cleaned_ecg, ecg_signals = lista_datos

                #Obtiene la prediccion por cada parametro
                for column in df_isolation.columns:

                    #Carga el modelo
                    model_filename = f"/Users/miguel_05/Desktop/Programacion/BioDesign-ECG-Project-main/Software/Hito_4/Hito_3/Modelos_Entrenados/ECG_resting/model_{column}.joblib"
                    model = joblib.load(model_filename)  # Cargar el modelo

                    # Realizar predicciones (1: No Anómalo, -1: Anómalo)
                    predictions = model.predict(df_isolation[[column]])

                    # Guardar los resultados en el diccionario de anomalías
                    anomalies[column] = pd.Series(predictions)

                    # Crear un DataFrame con los resultados de las anomalías
                    anomaly_results = df_isolation[[column]].copy()  # Copiar la columna de datos originales
                    anomaly_results['Anomaly'] = anomalies[column]  # Agregar la columna de anomalías
                    
                    # Mostrar los primeros resultados de anomalía (1: No Anómalo, -1: Anómalo)   
                    print(f"Resultados de anomalía para {column}: \n", anomaly_results.head(20))

                # Desempaquetar los valores del DataFrame
                if not df_isolation.empty:
                    HR = round(df_isolation["HR"].iloc[0], 3)
                    HRV = round(df_isolation["HRV"].iloc[0], 3)
                    Amplitud_T_prom = round(df_isolation["Amplitud_T_prom"].iloc[0], 3)
                    Amplitud_ST_prom = round(df_isolation["Amplitud_ST_prom"].iloc[0], 3)
                    Intervalo_QTc_prom = round(df_isolation["Intervalo_QTc_prom"].iloc[0], 3)
                    anomaly=anomaly_results['Anomaly'].iloc[0]

                    # Crear un objeto ECGData y agregarlo a stored_data
                    ecg_data = ECGData(
                        HR=HR,
                        HRV=HRV,
                        Amplitud_T_prom=Amplitud_T_prom,
                        Amplitud_ST_prom=Amplitud_ST_prom,
                        Intervalo_QTc_prom=Intervalo_QTc_prom,
                        Anomaly=anomaly
                    )
                    print(ecg_data)
                    stored_data.append(ecg_data.dict())  # Convertir a diccionario para almacenar
            else:
                print("No hay datos para guardar.")
                
        except Exception as e:
            print(f"Error al guardar los datos: {str(e)}")
        
        await asyncio.sleep(10)  # Esperar 10 segundos antes de guardar nuevamente

@app.get("/get_ecg_data")
async def get_ecg_data():
    # Verificar la respuesta antes de enviarla
    print("Datos a enviar:", stored_data)
    return {"data": stored_data}

# Variable global para controlar si las tareas ya se han iniciado
tasks_started = False

@app.on_event("startup")
async def startup_event():
    global ser, tasks_started

    if tasks_started:  # Evitar recrear tareas múltiples.
        return

    # Serial setup
    if not ser or not ser.is_open:
        try:
            ser = serial.Serial("/dev/tty.usbmodem11201", baudrate=115200, timeout=1)
        except serial.SerialException as e:
            latest_data = f"Error: {str(e)}"

    # Evitar duplicación de tareas
    if not tasks_started:
        asyncio.create_task(read_serial())  
        asyncio.create_task(save_data_periodically())
        tasks_started = True

# Montar archivos estáticos
app.mount("/assets", StaticFiles(directory="templates/assets"), name="assets")

# Configuración de plantillas HTML
templates = Jinja2Templates(directory="templates")

# Endpoint para servir la última lectura serial
@app.get("/data")
async def get_data():
    return {"data": latest_data}

# Endpoint principal para servir la página HTML
@app.get("/signal_menu.html", response_class=HTMLResponse)
async def read_root(request: Request):
    global ser, graphing_state, count, latest_data, data_list,data_list_cloud
    ser = None

    graphing_state = 0
    count = 0
    latest_data = "Esperando datos..."
    collections.delete_many({})
    data_list_cloud=[]
    data_list.clear()

    # Intentar abrir el puerto serial para cuando se ingresa a signal_menu
    if not ser or not ser.is_open:
        try:
            ser = serial.Serial("/dev/tty.usbmodem11201", baudrate=115200, timeout=1)

        except serial.SerialException as e:
            latest_data = f"Error: {str(e)}"

    return templates.TemplateResponse("signal_menu2.html", {"request": request})

# Rutas para servir tus archivos HTML (sirve para ingresar a mis diferentes archivo html responsives
@app.get("/login.html", response_class=HTMLResponse)
async def read_root(request: Request):
    #Aqui se coloca la ruta, lo que no es lo mismo que el nombr de ruta
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/menu_principal.html", response_class=HTMLResponse)
async def menu_principal(request: Request):
    return templates.TemplateResponse("menu_principal.html", {"request": request})

@app.get("/register.html", response_class=HTMLResponse)
async def register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/formulario.html", response_class=HTMLResponse)
async def signal_menu(request: Request):
    return templates.TemplateResponse("formulario.html", {"request": request})

# Endpoint para guardar los datos en formato .mat
@app.get("/guardar-datos")
async def save_data():
    try:
        # Insertar el data_list en la colección de MongoDB
        if data_list_cloud:
            # Insertar los datos como un solo documento
            collections.insert_many(data_list_cloud)
            datos.save_data()
            data_list_cloud.clear()  # Limpiar la lista después de guardarlos
            return {"message": "Datos guardados exitosamente en MongoDB"}
        else:
            return {"message": "No hay datos para guardar"}
    except Exception as e:
        return {"message": f"Error al guardar los datos: {str(e)}"}

graphing_state = 0  # Estado inicial es 0 (no está graficando)

@app.post("/update-graph-state")
async def update_graph_state(state: GraphState):
    global graphing_state
    graphing_state = state.state  # Actualiza la variable global con el valor recibido (1 o 0)
    return {"message": f"El estado del gráfico es ahora {'activo' if graphing_state == 1 else 'detenido'}"}

@app.post("/register")
async def register_user(user: User):
    if register_db.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email ya registrado")

    register_db.insert_one(user.dict(by_alias=True))
    return {"message": "Usuario registrado exitosamente"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
