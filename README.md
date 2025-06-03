# Prueba-analitica-Gestion-Fraude
**Prueba Analítica: Estimación precios de Vivienda**
## Descripción
Este proyecto desarrolla un modelo predictivo para estimar los precios de vivienda utilizando técnicas de machine learning. Se procesaron datos geográficos y categóricos, aplicando imputación, clustering y selección de variables para optimizar el rendimiento. El resultado final se presenta en un archivo CSV con las predicciones generadas.
## Estructura del proyecto
.gitignore (Archivos y carpetas ignoradas por Git)
requirements.txt (Lista de dependencias)
README.md (Este archivo de documentación)
Modelo_RFRegressor_RandomizedSearchCV.py (Script principal del modelo)
data/
entrenamiento_precios_vivienda.xlsx (Datos de entrenamiento)
testeo_precios_vivienda.xlsx (Datos de prueba)
PuntosInteres.csv (Puntos de interés geográficos)
base_evaluada.csv (Archivo de salida con predicciones)

## Requisitos o librerías
- numpy  
- pandas  
- scikit-learn  
- openpyxl  
# ¿Cómo ejecutar?
1. **Descarga del repositorio**: Clona el repositorio desde GitHub usando el comando `git clone <URL_DEL_REPOSITORIO>` o descarga el ZIP manualmente desde la página del repositorio.
2. **Preparación del entorno**:
  - Crea una carpeta `Data` en el directorio del proyecto y coloca dentro los archivos `entrenamiento_precios_vivienda.xlsx`, `testeo_precios_vivienda.xlsx`, `PuntosInteres.csv` y `base_evaluada.csv` provistos para la prueba.
  - Abre una terminal CMD en el directorio del proyecto.
  - Crea un entorno virtual con: `python -m venv .venv`.
  - Activa el entorno virtual con: `.venv\Scripts\activate`.
  - Actualiza pip con: `python -m pip install --upgrade pip`.
  - Instala las librerías requeridas con: `pip install -r requirements.txt`.
3. **Ejecución del modelo**: Ejecuta el script principal con: `python Modelo_RFRegressor_RandomizedSearchCV.py`. El modelo generará las predicciones y actualizará el archivo `base_evaluada.csv` en la carpeta `data/`.
# Metodología
## 1. Carga y Exploración Inicial de Datos
El proceso inició con la carga de tres conjuntos de datos: `entrenamiento_precios_vivienda.xlsx`, `testeo_precios_vivienda.xlsx` y `PuntosInteres.csv`. Se revisó la estructura de los datos, identificando columnas como `Latitud`, `Longitud`, `departamento_inmueble`, `municipio_inmueble`, `sector` y la variable objetivo `valor_inmueble`. Se detectaron valores nulos (ej., `'NAN'`, `'N/A'`) y ceros en coordenadas, tratándolos como faltantes, y se evaluaron preliminarmente correlaciones y alta cardinalidad.
## 2. Transformación de Variables
- **Imputación de Coordenadas**: Se utilizó `KNNImputer` con 5 vecinos para imputar valores faltantes en `Latitud` y `Longitud`, integrando variables categóricas codificadas (`OneHotEncoder`) como `departamento_inmueble`, `municipio_inmueble` y `sector`. Los ceros se convirtieron a `NaN` previamente.
- **Generación de Características con K-Means**: Se aplicó `MiniBatchKMeans` (100 clusters) a los puntos de interés, usando `Latitud`, `Longitud` y `cat_punto_interes` (codificado y escalado). Se generaron `dist_cluster`, `id_cluster` y conteos por categoría.
- **Limpieza**: Valores nulos en categóricas se reemplazaron por `NaN`.
- **Transformación de Objetivo**: `valor_inmueble` se convirtió de string a `float` (reemplazando comas por puntos) y se imputaron faltantes con la mediana.
## 3. Selección de Variables
- **Eliminación Inicial**: Se excluyeron `valor_inmueble`, `valor_uvr`, `valor_inmueble_en_uvr`, `Latitud` y `Longitud`.
- **Reducción**: Se eliminaron columnas con >30% de faltantes, numéricas con correlación >0.6 y categóricas con >10 categorías.
- **Selección por Importancia**: Un `RandomForestRegressor` preliminar (50 árboles) identificó las 50 características más relevantes.
## 4. Modelado
- **División**: Datos de entrenamiento se dividieron 70:30 (entrenamiento y validación) con `random_state=42`.
- **Pipeline**: Incluyó `OneHotEncoder` y `MinMaxScaler` para preprocesamiento, `SimpleImputer` y un `RandomForestRegressor`.
- **Optimización**: `RandomizedSearchCV` optimizó hiperparámetros (ej., `n_estimators`, `max_depth`) con MAPE, usando 8 iteraciones, 5 folds y `n_jobs=1`.
- **Evaluación**: Se calculó MAPE en el conjunto de validación.
## 5. Predicción y Resultados
- **Preparación**: `dtos_prueba` se alineó con las columnas de entrenamiento, conservando `Latitud` y `Longitud`.
- **Predicción**: Se generó `valor_total_avaluo` con el modelo optimizado.
- **Salida**: Predicciones se fusionaron con `base_evaluada.csv` (imputando nulos con mediana) y se guardaron en `data/base_evaluada.csv`.