import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Pronóstico Delitos Barranquilla", layout="wide")

st.title("Pronóstico de Delitos de Alto Impacto en Barranquilla")

# =========================
# CARGA DE DATOS
# =========================

try:
    df = pd.read_csv("Delitos_de_alto_impacto_en_Barranquilla.csv")
except Exception:
    st.error("No se pudo cargar el archivo CSV.")
    st.stop()

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower()

st.write("Columnas detectadas en el archivo:")
st.write(df.columns.tolist())

# =========================
# DETECTAR COLUMNA DE FECHA
# =========================

columna_fecha = None

for col in df.columns:
    if "fecha" in col or "año" in col or "ano" in col or "periodo" in col or "mes" in col:
        columna_fecha = col
        break

if columna_fecha is None:
    st.error("No se pudo detectar automáticamente la columna de fecha.")
    st.stop()

# Convertir a datetime
df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors="coerce")
df = df.dropna(subset=[columna_fecha])

# Ordenar
df = df.sort_values(columna_fecha)

# Establecer índice temporal
df.set_index(columna_fecha, inplace=True)

# =========================
# DETECTAR COLUMNA DE DELITOS
# =========================

columna_delito = None

for col in df.columns:
    if "alto" in col and "impacto" in col:
        columna_delito = col
        break

if columna_delito is None:
    # Si no encuentra por nombre, toma la primera columna numérica
    columnas_numericas = df.select_dtypes(include=np.number).columns
    if len(columnas_numericas) > 0:
        columna_delito = columnas_numericas[0]
    else:
        st.error("No se encontró una columna numérica para modelar.")
        st.stop()

# =========================
# AGRUPAR MENSUAL
# =========================

df_mensual = df.resample("M").sum(numeric_only=True)

serie = df_mensual[columna_delito].dropna()

st.subheader("Resumen del conjunto de datos")
st.write("Fecha mínima:", serie.index.min())
st.write("Fecha máxima:", serie.index.max())
st.write("Cantidad de meses disponibles:", len(serie))

if len(serie) < 12:
    st.warning("Se requieren al menos 12 meses de datos para ajustar el modelo.")
    st.stop()

# =========================
# MODELO
# =========================

modelo = ExponentialSmoothing(
    serie,
    trend="add",
    seasonal=None
).fit()

ultima_fecha = serie.index[-1]
meses_proyeccion = (2025 - ultima_fecha.year) * 12 + (12 - ultima_fecha.month)

if meses_proyeccion <= 0:
    st.warning("Los datos ya superan diciembre de 2025.")
    st.stop()

pronostico = modelo.forecast(meses_proyeccion)

# =========================
# VISUALIZACIÓN
# =========================

st.subheader("Serie histórica y pronóstico hasta diciembre 2025")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(serie.index, serie.values, label="Histórico")
ax.plot(pronostico.index, pronostico.values, linestyle="--", label="Pronóstico")
ax.set_xlabel("Fecha")
ax.set_ylabel("Delitos de alto impacto")
ax.legend()

st.pyplot(fig)

df_pronostico = pronostico.reset_index()
df_pronostico.columns = ["fecha", "pronostico"]

st.subheader("Tabla de pronóstico")
st.dataframe(df_pronostico)
