import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="Delitos de Alto Impacto - Barranquilla", layout="wide")

# -------------------------------------------------------------------
# NAVEGACIÓN ENTRE PÁGINAS
# -------------------------------------------------------------------

pagina = st.sidebar.radio(
    "Navegación",
    ["Dashboard Analítico", "Documentación y Metodología"]
)

# -------------------------------------------------------------------
# FUNCIÓN DE CARGA AUTOMÁTICA (API SOCrata - Actualiza cada 24 horas)
# -------------------------------------------------------------------

@st.cache_data(ttl=86400)
def cargar_datos():
    url = "https://www.datos.gov.co/resource/4p95-h82w.csv?$limit=5000"
    try:
        df = pd.read_csv(url)
        return df
    except Exception:
        st.error("⚠️ Error al cargar datos desde el portal oficial de Datos Abiertos Colombia.")
        st.info("Verifique la conexión o la disponibilidad del portal.")
        st.stop()

# -------------------------------------------------------------------
# PÁGINA 1: DASHBOARD
# -------------------------------------------------------------------

if pagina == "Dashboard Analítico":

    df = cargar_datos()

    # Normalización nombres columnas
    df.columns = df.columns.str.lower().str.strip()

    # Renombrar según estructura REAL del dataset
    df = df.rename(columns={
        "casos_denuncias_anterior": "anterior",
        "actual": "actual",
        "a_os": "años"  # por si llega codificado raro
    })

    columnas_necesarias = ["años", "delito", "anterior", "actual"]

    for col in columnas_necesarias:
        if col not in df.columns:
            st.error(f"La columna '{col}' no existe en el dataset.")
            st.write("Columnas disponibles:", df.columns.tolist())
            st.stop()

    df["anterior"] = pd.to_numeric(df["anterior"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df = df.dropna()

    # ----------------------------------------------------------------
    # CONTEXTO
    # ----------------------------------------------------------------

    st.title("Análisis de Delitos de Alto Impacto en Barranquilla")

    st.markdown("## Periodo analizado")
    st.write("Comparaciones interanuales entre periodos 2019 – 2023")

    st.markdown("""
    Este panel presenta un análisis comparativo de los delitos de alto impacto en Barranquilla durante el periodo 2019–2023,
    utilizando el volumen de denuncias como variable central para comprender su evolución y comportamiento. 
    El objetivo es evaluar la relación entre la dinámica de las denuncias y la variación observada en cada delito,
    identificando patrones estructurales y tendencias emergentes. 

    A través de métricas agregadas, comparativos interanuales y visualizaciones analíticas, el usuario puede interpretar
    la magnitud y concentración del impacto delictivo. Las métricas superiores resumen los cambios totales entre periodos,
    mientras que los gráficos detallan la contribución individual por delito.

    Este análisis busca aportar una base técnica y empírica para la formulación y focalización de la estrategia de seguridad
    de la ciudad. En particular, sirve como insumo para la elaboración y ajuste del Plan Integral de Seguridad y Convivencia
    Ciudadana (PISCC).
    """)

    st.markdown("---")

    # ----------------------------------------------------------------
    # FILTROS
    # ----------------------------------------------------------------

    st.sidebar.header("Filtros")

    años = sorted(df["años"].dropna().unique())
    delitos = sorted(df["delito"].dropna().unique())

    año_seleccionado = st.sidebar.multiselect(
        "Seleccionar año",
        options=años,
        default=años
    )

    delito_seleccionado = st.sidebar.multiselect(
        "Seleccionar delito",
        options=delitos,
        default=delitos
    )

    df_filtrado = df[
        (df["años"].isin(año_seleccionado)) &
        (df["delito"].isin(delito_seleccionado))
    ]

    # ----------------------------------------------------------------
    # MÉTRICAS GENERALES
    # ----------------------------------------------------------------

    total_anterior = df_filtrado["anterior"].sum()
    total_actual = df_filtrado["actual"].sum()

    variacion_abs = total_actual - total_anterior
    variacion_pct = (variacion_abs / total_anterior * 100) if total_anterior != 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total período anterior", f"{int(total_anterior):,}")
    col2.metric("Total último período", f"{int(total_actual):,}")
    col3.metric("Variación absoluta total", f"{int(variacion_abs):,}")
    col4.metric("Variación porcentual total", f"{variacion_pct:.2f}%")

    st.markdown("---")

    # ----------------------------------------------------------------
    # TABLA Y GRÁFICO
    # ----------------------------------------------------------------

    tabla = df_filtrado.groupby("delito").agg({
        "anterior": "sum",
        "actual": "sum"
    }).reset_index()

    tabla["Variación absoluta"] = tabla["actual"] - tabla["anterior"]

    st.subheader("Variación absoluta por delito")

    fig = px.bar(
        tabla,
        x="delito",
        y="Variación absoluta",
        title="Cambio absoluto en número de denuncias por delito"
    )

    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Tabla resumen consolidada por delito")
    st.dataframe(tabla, use_container_width=True)

    st.caption(f"Última actualización automática del panel: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# -------------------------------------------------------------------
# PÁGINA 2: DOCUMENTACIÓN
# -------------------------------------------------------------------

elif pagina == "Documentación y Metodología":

    st.title("Documentación del Panel")

    st.markdown("## Fuente de datos")
    st.write("""
    - Portal: Datos Abiertos Colombia  
    - Entidad publicadora: Alcaldía Distrital de Barranquilla  
    - Dataset: Comparativo de delitos de alto impacto en Barranquilla  
    - Acceso mediante API pública Socrata  
    - Link: https://www.datos.gov.co/d/4p95-h82w (presionar en "Exportar")
    """)

    st.markdown("## Fecha de acceso")
    st.write(f"""
    Los datos son consultados automáticamente cada vez que el panel se ejecuta.
    Última consulta registrada: **{datetime.now().strftime('%d de %B de %Y, %H:%M:%S')}**
    """)

    st.markdown("## Cómo se actualizan los datos")
    st.write("""
    El panel está conectado directamente a la API oficial del portal de Datos Abiertos Colombia.
    La información se actualiza automáticamente cada 24 horas mediante un sistema de cache.
    Si la entidad publica nuevas versiones del dataset, el panel reflejará automáticamente
    los cambios sin necesidad de intervención manual.
    """)

    st.markdown("## Metodología aplicada (Marco QUEST)")
    st.write("""
    - Q: Definición de la pregunta sobre relación entre denuncias y variación delictiva.
    - U: Comprensión de la estructura del dataset.
    - E: Exploración comparativa entre periodos 2019–2023.
    - S: Identificación de patrones estructurales y emergentes.
    - T: Comunicación visual mediante dashboard interactivo.
    """)

    st.markdown("---")
    st.info("Este panel prioriza transparencia, trazabilidad, actualización automática y soporte técnico para la formulación del PISCC.")
