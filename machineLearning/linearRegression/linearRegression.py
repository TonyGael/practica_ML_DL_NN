# Objetivo:
# Predecir el precio promedio (precio_prom) de los departamentos
# en función de varias característicascomo el barrio, año, trimestre,
# cantidad de ambientes, estado y comuna.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Cargamos el dataset
df = pd.read_csv('precio-venta-deptos.csv')

# Realizamos una exploración básica del data frame para checar que haya cargado bien
print(df.head())

# Verificamos si hay valores nulos
print(df.isnull().sum())

# Limpiamos los valores nulos si los hallamos
# Aquí podrías manejar los nulos antes de continuar con el procesamiento
df.dropna(inplace=True)

# en la columna ambientes tenemos na cadena de texto, precisamos solamente un número entero
# con la cantidad de ambientes
#   barrio      año     trimestre   precio_prom     ambientes       estado  comuna
#   AGRONOMIA   2010    1           NaN             2 ambientes     Usado   15
# realizaremos la conversión de la columna ambientes de cadena a entero

# extraemos el número de la cadena y lo convertimos a texto
df['ambientes'] = df['ambientes'].str.extract(r'(\d+)').astype(int)
print(df.head())

# Convertimos las columnas categóricas con el método OneHotEncoder
# Usaremos list comprehension para ahorar lineas
categorical_cols = [col for col in ['barrio', 'estado', 'comuna'] if col in df.columns]

# APlicamos OneHotEncoding a las columnas categóricas
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_df = pd.DataFrame(encoder.fit_transform(df[categorical_cols]), columns=encoder.get_feature_names_out())

# Concatenamos las variables codificadas con el dataset original
df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# Seleccionamos las características (x) y la variable objetivo (y)
# usaremos comprensión de lista para ahorar lineas de código
features = ['año', 'trimestre', 'ambientes'] + list(encoded_df.columns)
x = df[features]
y = df['precio_prom']

# Ahora ya empezamos a trabajar con los conjuntos de datos a entrenar y analizar
# Dividimos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Ahora, asombosamente, entrenamos el modelo con regresión lineal!!!
model = LinearRegression()
model.fit(x_train, y_train)

# Realizamos las predicciones
y_pred = model.predict(x_test)

# Evaluamos el modelo para hallar el MSE ( mean squaed error): error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
# A continuación el coeficiente de determinación R^2
r2 = r2_score(y_test, y_pred)

# imprimimos los valores
print(f'Error cuadrático medio (MSE): {mse}')
print(f'Coeficiente de determinación (R^2): {r2}')

# Mostramos los coeficientes de cada variable para analizar su importancia
coeficientes = pd.DataFrame({
    'Feature': x.columns,
    'Coeficiente': model.coef_
})

print(coeficientes.sort_values(by='Coeficiente', ascending=False))