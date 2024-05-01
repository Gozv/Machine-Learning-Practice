import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Cargar el dataset
data = pd.read_csv("Salary_Data.csv")

# Separar las características y el target
X = data[["YearsExperience"]]
y = data[["Salary"]]

# Dividir por prueba y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Crear el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular métricas de rendimiento
# Error cuadrático
EC = ((y_test - y_pred) ** 2).sum()

# Error cuadrático medio
ECM = EC / len(y_test)

# Raíz del error cuadrático medio
RMSE = ECM ** 0.5

# Error absoluto medio
EAM = abs(y_test - y_pred).mean()

# Coeficiente de determinación R^2
residuos = y_test - y_pred
explained_variance = residuos.var()
total_variance = y_test.var()
R2 = 1 - (explained_variance / total_variance)

print("EC:", EC.iloc[0])
print("ECM:", ECM.iloc[0])
print("RMSE:", RMSE.iloc[0])
print("EAM:", EAM.iloc[0])
print("R^2:", R2.iloc[0])