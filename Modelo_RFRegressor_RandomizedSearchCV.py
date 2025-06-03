import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import make_scorer
from math import radians, sin, cos, sqrt, atan2
from sklearn.pipeline import Pipeline

# Función para calcular distancia haversine
def calc_dist_hav(lat1, lon1, lat2, lon2):
   R = 6371
   lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
   dlat = lat2 - lat1
   dlon = lon2 - lon1
   a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
   c = 2 * atan2(sqrt(a), sqrt(1-a))
   return R * c * 1000

# Cargar datos
dtos_entrena = pd.read_excel('data/entrenamiento_precios_vivienda.xlsx')
dtos_prueba = pd.read_excel('data/testeo_precios_vivienda.xlsx')
pnts_interes = pd.read_csv('data/PuntosInteres.csv', header=None, 
                          names=['Longitud', 'Latitud', 'cat_punto_interes', 'nom_punto_interes'],
                          sep=';', encoding='ISO-8859-1')

# Preparar datos para KNNImputer
caracts_imput = ['Latitud', 'Longitud', 'departamento_inmueble', 'municipio_inmueble', 'sector']
df_entrena_imput = dtos_entrena[caracts_imput].copy()
df_prueba_imput = dtos_prueba[caracts_imput].copy()

# Codificar variables categóricas para KNNImputer
codif = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
caracts_cat = ['departamento_inmueble', 'municipio_inmueble', 'sector']
cat_codif_entrena = codif.fit_transform(df_entrena_imput[caracts_cat])
cat_codif_prueba = codif.transform(df_prueba_imput[caracts_cat])
cat_codif_entrena_df = pd.DataFrame(cat_codif_entrena, columns=codif.get_feature_names_out(caracts_cat))
cat_codif_prueba_df = pd.DataFrame(cat_codif_prueba, columns=codif.get_feature_names_out(caracts_cat))

# Combinar características numéricas y codificadas
df_entrena_imput = pd.concat([df_entrena_imput[['Latitud', 'Longitud']], cat_codif_entrena_df], axis=1)
df_prueba_imput = pd.concat([df_prueba_imput[['Latitud', 'Longitud']], cat_codif_prueba_df], axis=1)

# Reemplazar ceros por NaN
df_entrena_imput.loc[(df_entrena_imput['Latitud'] == 0) | (df_entrena_imput['Longitud'] == 0), ['Latitud', 'Longitud']] = np.nan
df_prueba_imput.loc[(df_prueba_imput['Latitud'] == 0) | (df_prueba_imput['Longitud'] == 0), ['Latitud', 'Longitud']] = np.nan

# Imputar con KNNImputer
imput_knn = KNNImputer(n_neighbors=5)
imput_knn.fit(df_entrena_imput)
imputados_entrena = imput_knn.transform(df_entrena_imput)
imputados_prueba = imput_knn.transform(df_prueba_imput)

# Actualizar coordenadas
dtos_entrena[['Latitud', 'Longitud']] = imputados_entrena[:, :2]
dtos_prueba[['Latitud', 'Longitud']] = imputados_prueba[:, :2]

# Verificar que Latitud y Longitud sean numéricos y no tengan NaN
for df, nom in [(dtos_entrena, 'dtos_entrena'), (dtos_prueba, 'dtos_prueba')]:
   if 'Latitud' not in df.columns or 'Longitud' not in df.columns:
       raise KeyError(f"Error: 'Latitud' o 'Longitud' no están en {nom}")
   if df['Latitud'].isnull().any() or df['Longitud'].isnull().any():
       raise ValueError(f"Error: Hay valores NaN en 'Latitud' o 'Longitud' en {nom} después de la imputación")
   if not (df['Latitud'].dtype in [np.float64, np.int64] and df['Longitud'].dtype in [np.float64, np.int64]):
       raise TypeError(f"Error: 'Latitud' o 'Longitud' no son numéricos en {nom}")

# Codificar 'categoria_punto_interes' para K-Means
codif_cat = OneHotEncoder(sparse_output=False)
cats_codif = codif_cat.fit_transform(pnts_interes[['cat_punto_interes']])
cats_codif_df = pd.DataFrame(cats_codif, columns=codif_cat.get_feature_names_out(['cat_punto_interes']))

# Combinar latitud, longitud y categorías codificadas
caracts_kmeans_df = pd.concat([pnts_interes[['Latitud', 'Longitud']], cats_codif_df], axis=1)

# Escalar características para K-Means
escalador = StandardScaler()
caracts_kmeans_escaladas = escalador.fit_transform(caracts_kmeans_df)

# Aplicar K-Means
kmeans = MiniBatchKMeans(n_clusters=100, random_state=42)
pnts_interes['cluster'] = kmeans.fit_predict(caracts_kmeans_escaladas)

# Generar características de K-Means
def gen_caracts_kmeans(fila, puntos, kmeans, escalador, codif_cat, cols_caracts):
   centros_cluster = kmeans.cluster_centers_
   cats = codif_cat.get_feature_names_out(['cat_punto_interes'])
   cats_dummy = pd.Series(0, index=cats)
   caracts = pd.concat([pd.Series([fila['Latitud'], fila['Longitud']], index=['Latitud', 'Longitud']), cats_dummy])
   caracts = caracts[cols_caracts]
   caracts_escaladas = escalador.transform([caracts])
   distancias = np.linalg.norm(caracts_escaladas - centros_cluster, axis=1)
   dist_min = np.min(distancias)
   cluster_mas_cercano = np.argmin(distancias)
   pnts_cluster = puntos[puntos['cluster'] == cluster_mas_cercano]
   conteos = {f'num_{tipo.lower().replace(" ", "_")}_cluster': 0 
              for tipo in puntos['cat_punto_interes'].unique()}
   for tipo in pnts_cluster['cat_punto_interes'].unique():
       conteos[f'num_{tipo.lower().replace(" ", "_")}_cluster'] = len(
           pnts_cluster[pnts_cluster['cat_punto_interes'] == tipo])
   return pd.Series({'dist_cluster': dist_min, 'id_cluster': cluster_mas_cercano, **conteos})

# Aplicar características de K-Means
dtos_entrena = dtos_entrena.join(dtos_entrena.apply(
   lambda x: gen_caracts_kmeans(x, pnts_interes, kmeans, escalador, codif_cat, caracts_kmeans_df.columns), axis=1))
dtos_prueba = dtos_prueba.join(dtos_prueba.apply(
   lambda x: gen_caracts_kmeans(x, pnts_interes, kmeans, escalador, codif_cat, caracts_kmeans_df.columns), axis=1))

#Definición de variable respuesta
y = dtos_entrena['valor_total_avaluo'].str.replace(',', '.').astype(float)
X = dtos_entrena.copy()
cols_excluir = ['valor_total_avaluo', 'valor_uvr', 'valor_avaluo_en_uvr', 'Latitud', 'Longitud']
X.drop(columns=cols_excluir, axis=1, inplace=True)

# Eliminar columnas con >30% de valores faltantes
vals_nulos = ['NAN', 'N/A', 'N/D', 'NO APLICA', 'no aplica', '', 'None', '-', 'na', 'null', 'undefined']
cols_cat = X.select_dtypes(include=['object']).columns
cols_cat_prueba = dtos_prueba.select_dtypes(include=['object']).columns
X[cols_cat] = X[cols_cat].replace(vals_nulos, np.nan)
dtos_prueba[cols_cat_prueba] = dtos_prueba[cols_cat_prueba].replace(vals_nulos, np.nan)
ratio_faltantes = X.isnull().mean()
a_eliminar = ratio_faltantes[ratio_faltantes > 0.3].index
X = X.drop(columns=a_eliminar)
dtos_prueba = dtos_prueba.drop(columns=a_eliminar, errors='ignore')

# Eliminar columnas numéricas altamente correlacionadas
cols_num = X.select_dtypes(include=['int64', 'float64']).columns
matriz_corr = X[cols_num].corr().abs()
triang_sup = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(bool))
a_eliminar_corr = [col for col in triang_sup.columns if any(triang_sup[col] > 0.6)]
X = X.drop(columns=a_eliminar_corr)
dtos_prueba = dtos_prueba.drop(columns=a_eliminar_corr, errors='ignore')

# Eliminar columnas categóricas con alta cardinalidad
caracts_cat = X.select_dtypes(include=['object']).columns
cols_alta_card = [col for col in caracts_cat if X[col].nunique() > 10]
X = X.drop(columns=cols_alta_card)
dtos_prueba = dtos_prueba.drop(columns=cols_alta_card, errors='ignore')

# Identificar las columnas más importantes
# Actualizar características
caracts_num = X.select_dtypes(include=['int64', 'float64']).columns
caracts_cat = X.select_dtypes(include=['object']).columns

# Pipeline temporal para preprocesar datos
prep_temp = ColumnTransformer(
   transformers=[
       ('num', SimpleImputer(strategy='mean'), caracts_num),
       ('cat', Pipeline([
           ('imput', SimpleImputer(strategy='constant', fill_value='missing')),
           ('codif', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
       ]), caracts_cat)
   ])

# Ajustar y transformar datos
X_transf = prep_temp.fit_transform(X)
# Asegurarse de que y no tenga NaN
y = y.fillna(y.median())

# Entrenar un RandomForest simple para obtener importancias
rf_temp = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
rf_temp.fit(X_transf, y)

# Obtener importancias de características
noms_caracts = (caracts_num.tolist() + 
              prep_temp.named_transformers_['cat']
              .named_steps['codif']
              .get_feature_names_out(caracts_cat).tolist())
importancias = pd.Series(rf_temp.feature_importances_, index=noms_caracts)

# Seleccionar las 50 características más importantes
mejores_caracts = importancias.nlargest(50).index

# Mapear nombres de características codificadas a columnas originales
cols_selec = []
for car in mejores_caracts:
   if car in caracts_num:
       cols_selec.append(car)
   else:
       # Si es una característica codificada, extraer la columna original
       for col_cat in caracts_cat:
           if car.startswith(f"{col_cat}_"):
               if col_cat not in cols_selec:
                   cols_selec.append(col_cat)
                   break

# Filtrar X y dtos_prueba
X = X[cols_selec]
dtos_prueba = dtos_prueba[[col for col in cols_selec if col in dtos_prueba.columns]]

# Preparación de los conjuntos de datos
X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir función personalizada para MAPE
def calc_mape(y_verd, y_pred):
   # Evitar división por cero añadiendo un pequeño epsilon
   eps = 1e-10
   y_verd, y_pred = np.array(y_verd), np.array(y_pred)
   return np.mean(np.abs((y_verd - y_pred) / (y_verd + eps))) * 100

# Crear scorer para MAPE (menor es mejor)
mape_eval = make_scorer(calc_mape, greater_is_better=False)

# Preprocesamiento y pipeline con RandomizedSearchCV
pipe_rf = Pipeline(
   steps=[
       (
           "transf_cols",
           make_column_transformer(
               (
                   OneHotEncoder(categories="auto", drop='first', dtype=np.float64, handle_unknown="ignore"),
                   make_column_selector(dtype_include='object'),
               ),
               (
                   MinMaxScaler(feature_range=(0, 1)),
                   make_column_selector(dtype_include='float64'),
               ),
               remainder='passthrough',
           ),
       ),
       (
           "imput",
           SimpleImputer(strategy='mean')
       ),
       (
           "RegresorRF",
           RandomForestRegressor(random_state=0),
       ),
   ],
   verbose=True,
)

# Ajustar la grilla de hiperparámetros
grilla_params = {
   'RegresorRF__n_estimators': np.arange(10, 110, 20),
   'RegresorRF__criterion': ['absolute_error'], # Solo MAE, más alineado con MAPE
   'RegresorRF__max_depth': [3, 5, 10, 20, None],
   'RegresorRF__min_samples_split': [2, 5, 10],
   'RegresorRF__min_samples_leaf': [1, 2, 4]
}

# Configurar RandomizedSearchCV con MAPE como métrica
busq_rf = RandomizedSearchCV(
   pipe_rf,
   grilla_params,
   n_iter=8,
   cv=2,
   scoring=mape_eval, # Usar MAPE como métrica
   random_state=42,
   n_jobs=1
)

# Ajustar el modelo
busq_rf.fit(X_entrena, y_entrena)

# Predecir con el mejor estimador
y_pred = busq_rf.best_estimator_.predict(X_prueba)

# Calcular MAPE en el conjunto de prueba
mape = calc_mape(y_prueba, y_pred)
print(f"MAPE en el conjunto de prueba: {mape:.2f}%")

# Asegurar que la base de prueba tenga las columnas que el modelo espera
columnas_esperadas = X_entrena.columns
dtos_prueba = dtos_prueba[[col for col in columnas_esperadas if col in dtos_prueba.columns]]

# Manejar columnas faltantes
columnas_faltantes = [col for col in columnas_esperadas if col not in dtos_prueba.columns]
if columnas_faltantes:
   print(f"Advertencia: Las siguientes columnas faltan en dtos_prueba y se llenarán con NaN: {columnas_faltantes}")
   for col in columnas_faltantes:
       dtos_prueba[col] = np.nan

# Asegurar que los tipos de datos sean consistentes
# Identificar columnas numéricas y categóricas basadas en X_entrena
caracts_num = X_entrena.select_dtypes(include=['int64', 'float64']).columns
caracts_cat = X_entrena.select_dtypes(include=['object']).columns

# Convertir columnas numéricas a float, reemplazando valores no numéricos por NaN
for col in caracts_num:
   dtos_prueba[col] = pd.to_numeric(dtos_prueba[col], errors='coerce')

# Convertir columnas categóricas a string, reemplazando valores problemáticos
for col in caracts_cat:
   dtos_prueba[col] = dtos_prueba[col].astype(str).replace('nan', 'missing')

# Predecir directamente con el pipeline ajustado
dtos_prueba['valor_total_avaluo'] = busq_rf.best_estimator_.predict(dtos_prueba)

# Cargar base_evaluada.csv entregada
base_eval = pd.read_csv('data/base_evaluada.csv', sep=',')

# Cruzar y actualizar valores directamente (usando 'id' como clave)
base_eval = base_eval.merge(dtos_prueba[['id', 'valor_total_avaluo']], on='id', how='left')
base_eval['valor_total_avaluo '] = base_eval['valor_total_avaluo']
base_eval = base_eval.drop(columns=['valor_total_avaluo'])

# Verificar que no haya valores nulos en valor_total_avaluo
if base_eval['valor_total_avaluo '].isnull().any():
   base_eval['valor_total_avaluo '] = base_eval['valor_total_avaluo '].fillna(base_eval['valor_total_avaluo '].median())

# Guardar resultado en base_evaluada.csv (CSV, separado por comas, con encabezado)
base_eval.to_csv('data/base_evaluada.csv', index=False, sep=',')
print("Archivo base_evaluada.csv actualizado con éxito.")
print(base_eval.head())
