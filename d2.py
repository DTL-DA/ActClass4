import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Serie Temporal Delitos Barranquilla", layout="wide")

st.title("Serie Temporal y Pronóstico de Delitos")

# =========================
# CARGA
# =========================

df = pd.read_csv("Delitos_de_alto_impacto_en_Barranquilla.csv")
df.columns = df.columns.str.strip()

# =========================
# LIMPIEZA NUMÉRICA
# =========================

df["Casos/denuncias último periodo"] = (
    df["Casos/denuncias último periodo"]
    .astype(str)
    .str.replace(".", "", regex=False)
    .str.replace(",", ".", regex=False)
)

df["Casos/denuncias último periodo"] = pd.to_numeric(
    df["Casos/denuncias último periodo"],
    errors="coerce"
)

df = df.dropna(subset=["Casos/denuncias último periodo"])

# =========================
# EXTRAER AÑO FINAL
# =========================

df["Año"] = df["Años comparados"].str.split("-").str[1]

# =========================
# CONVERTIR MESES EN ESPAÑOL A NÚMERO
# =========================

meses = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12
}

df["Mes"] = df["Periodo meses comparado"].str.lower().map(meses)

df = df.dropna(subset=["Mes", "Año"])

# =========================
# CREAR FECHA CORRECTA
# =========================

df["Fecha"] = pd.to_datetime(
    dict(year=df["Año"].astype(int),
         month=df["Mes"].astype(int),
         day=1)
)

# =========================
# CREAR SERIE TEMPORAL
# =========================

serie = (
    df.groupby("Fecha")["Casos/denuncias último periodo"]
    .sum()
    .sort_index()
)

serie = serie.asfreq("MS")  # Frecuencia mensual inicio de mes

st.subheader("Resumen de la serie")
st.write("Fecha inicial:", serie.index.min())
st.write("Fecha final:", serie.index.max())
st.write("Cantidad de meses:", len(serie))

if serie.empty:
    st.error("La serie sigue vacía. Revisar datos originales.")
    st.stop()

# =========================
# MODELO
# =========================

modelo = ExponentialSmoothing(
    serie,
    trend="add",
    seasonal=None
).fit()

# =========================
# PRONÓSTICO HASTA DICIEMBRE 2025
# =========================

ultima_fecha = serie.index[-1]
meses_proyeccion = (2025 - ultima_fecha.year) * 12 + (12 - ultima_fecha.month)

if meses_proyeccion > 0:
    pronostico = modelo.forecast(meses_proyeccion)
else:
    pronostico = pd.Series()

# =========================
# GRÁFICO
# =========================

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(serie.index, serie.values, label="Histórico")

if not pronostico.empty:
    ax.plot(pronostico.index, pronostico.values, linestyle="--", label="Pronóstico")

ax.set_xlabel("Fecha")
ax.set_ylabel("Casos")
ax.legend()

st.pyplot(fig)

if not pronostico.empty:
    st.subheader("Pronóstico hasta diciembre 2025")
    st.dataframe(pronostico.reset_index().rename(columns={0: "Pronóstico"}))
