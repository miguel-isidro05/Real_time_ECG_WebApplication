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

anomalies_all=0
# Convertir las predicciones a numpy array para facilidad
anomalies_all = np.array(anomalies_all)

# Número total de datos
total_datos = len(anomalies_all)

# Cantidad de anómalos
num_anomalos = 12

# Crear etiquetas: los primeros 3 serán anómalos (-1), el resto no anómalos (1)
true_labels = [-1] * num_anomalos + [1] * (total_datos - num_anomalos)

# Calcular y mostrar el F1-score
f1 = f1_score(true_labels, anomalies_all, average="binary", pos_label=-1)  # Anómalo como clase positiva
print(f"F1-score: {f1:.4f}")

# Mostrar un informe detallado
report = classification_report(true_labels, anomalies_all, target_names=["Normal (1)", "Anómalo (-1)"])
print("Reporte de clasificación:")
print(report)


# Cálculo de métricas
conf_matrix = confusion_matrix(true_labels, anomalies_all)
f1 = f1_score(true_labels, anomalies_all, average="binary", pos_label=-1)  # Clase 'anómalo (-1)' como positiva

# Crear un heatmap para la matriz de confusión
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal (1)", "Anómalo (-1)"], yticklabels=["Normal (1)", "Anómalo (-1)"])
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Real')
plt.title('Matriz de Confusión')
plt.show()