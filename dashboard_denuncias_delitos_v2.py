import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Delitos de Alto Impacto - Barranquilla", layout="wide")

# -------------------------------------------------------------------
# TÍTULO Y PREGUNTA PROBLEMA
# -------------------------------------------------------------------

st.title("Análisis de Delitos de Alto Impacto en Barranquilla")

st.markdown("""
## Pregunta problema

**¿Qué relación existe entre el volumen de denuncias y la variación observada en los delitos de alto impacto?**
""")

st.markdown("---")

# -------------------------------------------------------------------
# CARGA DE DATOS
# -------------------------------------------------------------------

archivo = st.file_uploader("Delitos_de_alto_impacto_en_Barranquilla", type=["csv"])

if archivo is not None:

    df = pd.read_csv(archivo)
    df.columns = df.columns.str.strip()

    # Limpieza numérica
    df["Casos/denuncias  anterior periodo"] = pd.to_numeric(
        df["Casos/denuncias  anterior periodo"], errors="coerce"
    )

    df["Casos/denuncias último periodo"] = pd.to_numeric(
        df["Casos/denuncias último periodo"], errors="coerce"
    )

    df["Variación %"] = (
        df["Variación %"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    df["Variación %"] = pd.to_numeric(df["Variación %"], errors="coerce")

    df["Variación absoluta"] = pd.to_numeric(
        df["Variación absoluta"], errors="coerce"
    )

    df = df.dropna()

    # -------------------------------------------------------------------
    # FILTROS
    # -------------------------------------------------------------------

    st.sidebar.header("Filtros")

    # Filtro por periodo
    periodos = df["Periodo meses comparado"].unique()
    periodo_seleccionado = st.sidebar.multiselect(
        "Seleccionar periodo",
        options=periodos,
        default=periodos
    )

    # Filtro por delito
    delitos = df["Delito"].unique()
    delito_seleccionado = st.sidebar.multiselect(
        "Seleccionar delito",
        options=delitos,
        default=delitos
    )

    df_filtrado = df[
        (df["Periodo meses comparado"].isin(periodo_seleccionado)) &
        (df["Delito"].isin(delito_seleccionado))
    ]

    # -------------------------------------------------------------------
    # INDICADORES GENERALES
    # -------------------------------------------------------------------

    total_actual = df_filtrado["Casos/denuncias último periodo"].sum()
    total_anterior = df_filtrado["Casos/denuncias  anterior periodo"].sum()

    col1, col2 = st.columns(2)

    col1.metric("Total denuncias último periodo", f"{int(total_actual):,}")
    col2.metric("Total denuncias periodo anterior", f"{int(total_anterior):,}")

    st.markdown("---")

    # -------------------------------------------------------------------
    # 1️⃣ BARRAS COMPARANDO PERIODOS
    # -------------------------------------------------------------------

    st.subheader("Comparación de denuncias entre periodos")

    df_comparacion = df_filtrado.melt(
        id_vars="Delito",
        value_vars=[
            "Casos/denuncias  anterior periodo",
            "Casos/denuncias último periodo"
        ],
        var_name="Periodo",
        value_name="Denuncias"
    )

    fig1 = px.bar(
        df_comparacion,
        x="Delito",
        y="Denuncias",
        color="Periodo",
        barmode="group",
        title="Comparación de denuncias por delito"
    )

    fig1.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig1, use_container_width=True)

    # -------------------------------------------------------------------
    # 2️⃣ VARIACIÓN PORCENTUAL
    # -------------------------------------------------------------------

    st.subheader("Variación porcentual por delito")

    fig2 = px.bar(
        df_filtrado,
        x="Delito",
        y="Variación %",
        title="Variación porcentual entre periodos"
    )

    fig2.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------------
    # 3️⃣ RELACIÓN VOLUMEN vs VARIACIÓN (Scatter con tendencia)
    # -------------------------------------------------------------------

    st.subheader("Relación entre volumen de denuncias y variación")

    fig3 = px.scatter(
        df_filtrado,
        x="Casos/denuncias último periodo",
        y="Variación %",
        size="Casos/denuncias último periodo",
        color="Delito",
        trendline="ols",
        title="Volumen vs Variación porcentual"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # -------------------------------------------------------------------
    # 4️⃣ DIAGRAMA DE CORRELACIÓN (MATRIZ)
    # -------------------------------------------------------------------

    st.subheader("Matriz de correlación")

    variables_corr = df_filtrado[[
        "Casos/denuncias  anterior periodo",
        "Casos/denuncias último periodo",
        "Variación %",
        "Variación absoluta"
    ]]

    matriz_corr = variables_corr.corr()

    fig4 = px.imshow(
        matriz_corr,
        text_auto=True,
        title="Mapa de calor de correlaciones",
        aspect="auto"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # -------------------------------------------------------------------
    # 5️⃣ COEFICIENTE DE CORRELACIÓN PRINCIPAL
    # -------------------------------------------------------------------

    st.subheader("Correlación principal")

    correlacion = df_filtrado["Casos/denuncias último periodo"].corr(
        df_filtrado["Variación %"]
    )

    st.write(
        f"Coeficiente de correlación entre volumen y variación porcentual: **{correlacion:.3f}**"
    )

else:
    st.info("Cargue el archivo CSV para visualizar el análisis.")
