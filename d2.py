import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Pronóstico Delitos de Alto Impacto", layout="wide")

st.title("Pronóstico de Delitos de Alto Impacto en Barranquilla")

# Cargar archivo
archivo = st.file_uploader("Cargar archivo CSV", type=["csv"])

if archivo is not None:

    df = pd.read_csv(archivo)

    # Verificar columnas necesarias
    if "fecha" not in df.columns or "delitos_alto_impacto" not in df.columns:
        st.error("El archivo debe contener las columnas 'fecha' y 'delitos_alto_impacto'.")
        st.stop()

    # Convertir fecha a formato datetime
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna(subset=["fecha"])

    # Mostrar información básica
    st.subheader("Resumen del conjunto de datos")
    st.write("Fecha mínima:", df["fecha"].min())
    st.write("Fecha máxima:", df["fecha"].max())
    st.write("Cantidad total de registros:", len(df))

    # Agrupar por mes obligatoriamente
    df_mensual = (
        df
        .set_index("fecha")
        .resample("M")
        .sum(numeric_only=True)
    )

    if "delitos_alto_impacto" not in df_mensual.columns:
        st.error("No se encontró la columna 'delitos_alto_impacto' después de agrupar.")
        st.stop()

    serie = df_mensual["delitos_alto_impacto"].dropna()

    # Validación mínima
    if len(serie) < 12:
        st.warning("Se necesitan al menos 12 meses de datos para ajustar el modelo.")
        st.stop()

    # Ajustar modelo de suavizamiento exponencial
    modelo = ExponentialSmoothing(
        serie,
        trend="add",
        seasonal=None
    ).fit()

    # Calcular meses hasta diciembre 2025
    ultima_fecha = serie.index[-1]
    meses_proyeccion = (2025 - ultima_fecha.year) * 12 + (12 - ultima_fecha.month)

    if meses_proyeccion <= 0:
        st.warning("Los datos ya superan diciembre de 2025.")
        st.stop()

    pronostico = modelo.forecast(meses_proyeccion)

    # Gráfico
    st.subheader("Serie histórica y pronóstico hasta diciembre 2025")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(serie.index, serie.values, label="Histórico")
    ax.plot(pronostico.index, pronostico.values, label="Pronóstico", linestyle="--")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Delitos de alto impacto")
    ax.legend()

    st.pyplot(fig)

    # Mostrar tabla de pronóstico
    df_pronostico = pronostico.reset_index()
    df_pronostico.columns = ["fecha", "pronostico"]

    st.subheader("Tabla de pronóstico")
    st.dataframe(df_pronostico)
