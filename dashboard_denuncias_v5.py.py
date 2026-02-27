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
# FUNCIÓN DE CARGA DE DATOS
# -------------------------------------------------------------------

@st.cache_data
def cargar_datos():
    url = "https://www.datos.gov.co/resource/4p95-h82w.csv?$limit=5000"
    try:
        df = pd.read_csv(url)
        return df
    except Exception:
        st.error("⚠️ Error al cargar datos desde el portal oficial de Datos Abiertos Colombia.")
        st.stop()

# -------------------------------------------------------------------
# PÁGINA 1: DASHBOARD ANALÍTICO
# -------------------------------------------------------------------

if pagina == "Dashboard Analítico":

    df = cargar_datos()

    # Normalizar nombres de columnas
    df.columns = df.columns.str.lower().str.strip()

    # Renombrar columnas reales del dataset
    df = df.rename(columns={
        "casos_denuncias_anterior": "anterior",
        "actual": "actual",
        "a_os": "años"
    })

    columnas_necesarias = ["años", "delito", "anterior", "actual"]

    for col in columnas_necesarias:
        if col not in df.columns:
            st.error(f"La columna '{col}' no existe en el dataset.")
            st.write("Columnas disponibles:", df.columns.tolist())
            st.stop()

    # Convertir a numérico
    df["anterior"] = pd.to_numeric(df["anterior"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df = df.dropna()

    # ------------------------------------------------------------
    # FILTROS
    # ------------------------------------------------------------

    st.sidebar.header("Filtros")

    años = sorted(df["años"].unique())
    delitos = sorted(df["delito"].unique())

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

    # ------------------------------------------------------------
    # MÉTRICAS GENERALES
    # ------------------------------------------------------------

    st.title("Análisis de Delitos de Alto Impacto en Barranquilla")

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

    # ------------------------------------------------------------
    # TABLA Y GRÁFICO
    # ------------------------------------------------------------

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

    st.caption(f"Última consulta: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# -------------------------------------------------------------------
# PÁGINA 2: DOCUMENTACIÓN Y METODOLOGÍA
# -------------------------------------------------------------------

elif pagina == "Documentación y Metodología":

    st.title("Documentación del Panel")

    # ------------------------------------------------------------
    # CONTEXTO ESTRATÉGICO
    # ------------------------------------------------------------

    st.markdown("""
    ## ¿Qué relación existe entre el volumen de denuncias y la variación observada en los delitos de alto impacto?
    """)

    st.markdown("""
    Este panel analiza el comportamiento de los delitos de alto impacto en Barranquilla durante el periodo 2019–2023,
    utilizando el volumen de denuncias como variable central. El objetivo es evaluar la relación entre la dinámica de
    las denuncias y la variación observada en cada delito, aportando una base técnica para la formulación de la
    estrategia de seguridad y la elaboración del Plan Integral de Seguridad y Convivencia Ciudadana (PISCC).
    """)

    st.markdown("---")

    # ------------------------------------------------------------
    # PERIODO ANALIZADO
    # ------------------------------------------------------------

    st.markdown("## Periodo analizado")
    st.write("Comparaciones interanuales entre periodos 2019 – 2023")

    # ------------------------------------------------------------
    # FUENTE
    # ------------------------------------------------------------

    st.markdown("## Fuente de datos")
    st.write("""
    - Portal: Datos Abiertos Colombia  
    - Entidad publicadora: Alcaldía Distrital de Barranquilla  
    - Dataset: Comparativo de delitos de alto impacto en Barranquilla  
    - Link oficial: https://www.datos.gov.co/d/4p95-h82w  
    """)

    # ------------------------------------------------------------
    # METODOLOGÍA
    # ------------------------------------------------------------

    st.markdown("## Metodología aplicada (Marco QUEST)")
    st.write("""
    - Q: Definición de la pregunta sobre relación entre denuncias y variación delictiva.  
    - U: Comprensión de la estructura del dataset oficial.  
    - E: Exploración comparativa entre periodos.  
    - S: Análisis de variaciones absolutas y concentración del impacto.  
    - T: Comunicación visual mediante dashboard interactivo.  
    """)

    # ------------------------------------------------------------
    # FECHA DE ACCESO
    # ------------------------------------------------------------

    st.markdown("## Fecha de consulta")
    st.write(f"""
    Última consulta realizada: **{datetime.now().strftime('%d de %B de %Y, %H:%M:%S')}**
    """)

    st.markdown("---")
    st.info("Este panel prioriza transparencia, trazabilidad y soporte técnico para la formulación del PISCC.")
