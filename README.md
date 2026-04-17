# MLOps UdeM - Mushroom Classification

Proyecto de orquestación de un pipeline de Machine Learning para la clasificación de hongos comestibles y venenosos.

## Objetivo

Construir un flujo reproducible y automatizado que permita:

- adquirir y cargar datos
- procesar y preparar el dataset
- entrenar un modelo de clasificación
- registrar métricas, parámetros y artefactos con MLflow
- orquestar el flujo con Prefect

## Estructura del proyecto

MLOps_UdeM_Mushroom/
│
├── data/
├── notebooks/
├── src/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore

## Pipeline de Machine Learning

El flujo completo del pipeline es el siguiente:

1. **Adquisición de datos**
   - Carga del dataset desde la carpeta `data/`

2. **Procesamiento de datos**
   - Eliminación de duplicados
   - Preparación de variables
   - Separación en features (X) y target (y)
   - División en train y test

3. **Entrenamiento del modelo**
   - Codificación de variables categóricas (OneHotEncoder)
   - Balanceo de clases con SMOTE
   - Entrenamiento con Random Forest

4. **Evaluación**
   - Métricas: accuracy, precision, recall, F1-score
   - Matriz de confusión

5. **Tracking con MLflow**
   - Registro de parámetros
   - Registro de métricas
   - Guardado del modelo
   - Registro de artefactos

6. **Orquestación con Prefect**
   - Definición de tareas (tasks)
   - Ejecución del flujo completo (flow)
   - Automatización del pipeline

## Resultado

El modelo alcanzó un desempeño cercano al 100% en todas las métricas, sin falsos negativos, lo cual es crítico para este problema.
