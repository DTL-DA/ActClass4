import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io # Added this import

st.set_page_config(page_title="Dashboard Analitico", page_icon="📊", layout="wide")

def colores_pastel():
    return {
        'azul_pastel': '#A7C7E7',
        'verde_pastel': '#B5EAD7',
        'rosa_pastel': '#F7C6D9',
        'amarillo_pastel': '#FFF2B2',
        'morado_pastel': '#D7BDE2',
        'fondo': '#F8F9FA',
        'texto': '#2C3E50'
    }

# Metodologia QUEST
# Q: Question - Definir preguntas clave del analisis
# U: Understand - Comprender la estructura de los datos
# E: Explore - Explorar patrones y distribuciones
# S: Summarize - Resumir estadisticas descriptivas
# T: Transform - Transformar en visualizaciones accionables

st.title("Dashboard Analitico con Metodologia QUEST")
st.markdown("Este dashboard sigue la metodologia QUEST para un analisis estructurado y profesional.")

# Q: Question - Cargar el archivo adjunto (asumiendo CSV)
@st.cache_data
def cargar_datos(file_path): # Changed parameter to a variable name
    try:
        datos = pd.read_csv(file_path) # Used the parameter here
        return datos
    except Exception as e: # Added specific exception handling for better debugging
        st.error(f"No se encontro el archivo o hubo un error al cargarlo: {e}. Asegurese que el archivo '{file_path}' este en el directorio.")
        return pd.DataFrame()

# U: Understand - Mostrar estructura de datos
ruta_archivo = '/content/Comparativo_de_delitos_de_alto_impacto_en_la_ciudad_de_Barranquilla_20260221.csv'  # Corrected to the actual file path
datos = cargar_datos(ruta_archivo)

if not datos.empty:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Estructura de Datos")
        st.dataframe(datos.head(), use_container_width=True)
    with col2:
        st.subheader("Informacion General")
        buffer = io.StringIO()
        datos.info(buf=buffer)
        st.text(buffer.getvalue())
        st.metric("Filas Totales", len(datos))
        st.metric("Columnas", len(datos.columns))

    # E: Explore - Distribuciones basicas
    st.subheader("Exploracion de Datos")
    col1, col2 = st.columns(2)
    with col1:
        # Ensure 'datos' has numerical columns before attempting to plot histograms
        if len(datos.select_dtypes(include=['number']).columns) > 0:
            fig_hist = px.histogram(datos, nbins=20, template='plotly_white',
                                    color_discrete_sequence=[colores_pastel()['azul_pastel']])
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No se encontraron columnas numericas para el histograma.")
    with col2:
        numericas = datos.select_dtypes(include=['number']).columns.tolist()
        if numericas:
            selected_col = st.selectbox("Seleccionar columna numerica", numericas)
            fig_box = px.box(datos, y=selected_col, template='plotly_white',
                             color_discrete_sequence=[colores_pastel()['verde_pastel']])
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No se encontraron columnas numericas para el grafico de caja.")

    # S: Summarize - Estadisticas descriptivas
    st.subheader("Resumen Estadistico")
    if len(numericas) > 0:
        st.dataframe(datos[numericas].describe(), use_container_width=True)
    else:
        st.info("No se encontraron columnas numericas para el resumen estadistico.")

    # T: Transform - Visualizaciones avanzadas con colores pastel
    st.subheader("Visualizaciones Transformadas")
    colores = [colores_pastel()['azul_pastel'], colores_pastel()['verde_pastel'],
               colores_pastel()['rosa_pastel'], colores_pastel()['amarillo_pastel']]

    categoricas = datos.select_dtypes(include=['object']).columns.tolist()
    if categoricas and numericas:
        cat_col = st.selectbox("Columna categorica", categoricas)
        num_col = st.selectbox("Columna numerica para grafico", numericas)
        fig_bar = px.bar(datos, x=cat_col, y=num_col, color=cat_col,
                         color_discrete_sequence=colores, template='plotly_white')
        st.plotly_chart(fig_bar, use_container_width=True)
    elif categoricas and not numericas:
        st.info("No hay columnas numericas para crear graficos de barras combinando categoricas y numericas.")
    elif not categoricas and numericas:
        st.info("No hay columnas categoricas para crear graficos de barras combinando categoricas y numericas.")
    else:
        st.info("No hay columnas categoricas o numericas para crear graficos de barras.")

    if len(numericas) >= 2:
        fig_scatter = px.scatter(datos, x=numericas[0], y=numericas[1],
                                 trendline="ols", template='plotly_white',
                                 color_discrete_sequence=[colores_pastel()['morado_pastel']])
        st.plotly_chart(fig_scatter, use_container_width=True)
    elif len(numericas) < 2 and len(numericas) > 0:
        st.info("Se necesitan al menos dos columnas numericas para crear un grafico de dispersion.")
    else:
        st.info("No se encontraron columnas numericas para crear un grafico de dispersion.")

    st.markdown("Dashboard generado con colores pastel para una presentacion profesional.")
else:
    st.info("Por favor, proporcione el archivo CSV adjunto para visualizar el dashboard.") # Simplified the message
