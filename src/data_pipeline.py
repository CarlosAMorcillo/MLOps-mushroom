from prefect import task, flow
import pandas as pd
import os

@task(name="Adquisición de Datos", retries=2, retry_delay_seconds=5)
def load_data(path: str):
    """Carga los datos de hongos desde el CSV del proyecto."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo en: {path}")
    return pd.read_csv(path, sep=';')

@task(name="Procesamiento y Calidad de Datos")
def clean_data(df):
    """
    Basado en el EDA: Limpieza de nulos, duplicados y validación.
    """
    # 1. Eliminar duplicados
    before_count = len(df)
    df = df.drop_duplicates()
    
    # 2. Manejo de nulos EDA en caso de tenerlos
    #     
    print(f"Procesamiento listo. Filas originales: {before_count} -> Filas limpias: {len(df)}")
    return df

@flow(name="Mushroom ETL Pipeline")
def mushroom_data_flow(data_path: str = "MLOps-mushroom/data/secondary_data.csv"):
    """Flujo principal que orquesta la limpieza de datos."""
    raw_data = load_data(data_path)
    cleaned_data = clean_data(raw_data)
    
    # Guardar el dato limpio para que el siguiente pipeline pueda usarlo
    # cleaned_data.to_csv("data/cleaned_data.csv", index=False)
    return cleaned_data

if __name__ == "__main__":
    # Esto ejecuta el pipeline localmente
    mushroom_data_flow()