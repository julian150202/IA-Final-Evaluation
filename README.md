# Clasificación de riesgo de stroke con Regresión Logística

Este repositorio contiene una implementación **desde cero en C** de regresión logística para predecir la probabilidad de que un paciente sufra un **stroke (accidente cerebrovascular)**, usando el dataset público de salud `healthcare-dataset-stroke-data.csv`.  
Además incluye un **notebook en Jupyter** con el análisis exploratorio, uso de modelos de librerías de python, y ejecucion del modelo escrito en C.

---

## Contenido del repositorio

- `LogisticRegression.c`  
  Implementación de regresión logística en C:
  - Lectura del dataset desde archivo CSV.
  - Entrenamiento del modelo con descenso de gradiente.
  - Cálculo de métricas de desempeño (accuracy, etc.).

- `healthcare-dataset-stroke-data.csv`  
  Dataset de pacientes con información demográfica y clínica
  (edad, hipertensión, enfermedades cardíacas, hábito de fumar, etc.) y una etiqueta binaria
  que indica si el paciente ha sufrido un stroke.

- `Informe.ipynb`  
  Notebook de Jupyter con:
  - Descripción del problema.
  - Análisis exploratorio de datos (EDA).
  - Preprocesamiento y visualizaciones.
  - Resultados del modelo y discusión.


