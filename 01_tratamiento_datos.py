import pandas as pd

import numpy as np

df = pd.read_csv("dataset_1.csv", index_col=0)

#print(df)


set_gen = set(df["Género"].to_list())
set_edu = set(df["Nivel_Educación"].to_list())
set_ciu = set(df["Ciudad"].to_list())
#print(set_gen)
#print(set_edu)
#print(set_ciu)

# Tratamientos de valores negativos
df["Edad"] = df["Edad"].apply(lambda x: np.nan if x < 0 else x)
df["Ingresos"] = df["Ingresos"].apply(lambda x: np.nan if x < 0 else x)
df["Hijos"] = df["Hijos"].apply(lambda x: np.nan if x < 0 else x)

# 2 Imputar valores faltantes
for column in ["Edad", "Ingresos", "Hijos"]:
    median_value = df[column].median()
    df[column].fillna(median_value, inplace=True)
    
for column in ["Género", "Ciudad"]:
    mode_value = df[column].mode()[0]
    df[column].fillna(mode_value, inplace=True)
    
# 3 Mapeo de datos
education_mapping = {
    "Bachellors" : "Bachellor",
    "mastre" : "Master",
    "pHd" : "PhD",
    "no education" : "NE"
}

df["Nivel_Educación"].replace(education_mapping, inplace=True)
df["Nivel_Educación"].fillna("NE", inplace=True)

# 4 Casteo de tipos
df["Edad"] = df["Edad"].astype(int)
df["Hijos"] = df["Hijos"].astype(int)
df["Ingresos"] = df["Ingresos"].astype(float)
df["Altura"] = df["Altura"].astype(float)

print(df)
print(df.describe())
print(df.info())