# MLOps UdeM - Mushroom Classification

Proyecto de orquestación de un pipeline de Machine Learning para la clasificación de hongos comestibles y venenosos.

---

## 🎯 Objetivo

Construir un flujo reproducible y automatizado que permita:

- Adquirir y cargar datos  
- Procesar y preparar el dataset  
- Entrenar un modelo de clasificación  
- Registrar métricas, parámetros y artefactos con MLflow  
- Orquestar el flujo con Prefect  

---

## 📁 Estructura del proyecto

```text
MLOps_UdeM_Mushroom/
│
├── data/
├── notebooks/
├── src/
│   ├── data_preparation.py
│   ├── train.py
│   └── flow.py
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
