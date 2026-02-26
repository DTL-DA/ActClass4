import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Delitos de Alto Impacto - Barranquilla", layout="wide")

# -------------------------------------------------------------------
# CARGA DIRECTA DEL DATASET (SIN UPLOADER)
# -------------------------------------------------------------------

df = pd.read_csv("Delitos_de_alto_impacto_en_Barranquilla.csv")
df.columns = df.columns.str.strip()

# Limpieza de datos
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
# TÍTULO Y PREGUNTA PROBLEMA
# -------------------------------------------------------------------

st.title("Análisis de Delitos de Alto Impacto en Barranquilla")

st.markdown("""
## Pregunta problema

**¿Qué relación existe entre el volumen de denuncias y la variación observada en los delitos de alto impacto?**
""")

st.markdown("---")

# -------------------------------------------------------------------
# FILTRO POR AÑOS
# -------------------------------------------------------------------

st.sidebar.header("Filtros")

años = df["Años comparados"].unique()

año_seleccionado = st.sidebar.multiselect(
    "Seleccionar año",
    options=años,
    default=años
)

df_filtrado = df[df["Años comparados"].isin(año_seleccionado)]

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
# 1️⃣ BARRAS COMPARATIVAS ENTRE PERIODOS
# -------------------------------------------------------------------

st.subheader("Comparación de denuncias entre años")

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
    title="Denuncias por delito: comparación entre periodos"
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
# 3️⃣ RELACIÓN ENTRE VOLUMEN Y VARIACIÓN
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
# 4️⃣ MATRIZ DE CORRELACIÓN
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
# 5️⃣ COEFICIENTE CLAVE
# -------------------------------------------------------------------

correlacion = df_filtrado["Casos/denuncias último periodo"].corr(
    df_filtrado["Variación %"]
)

st.markdown(f"""
### Correlación principal

Coeficiente de correlación entre volumen y variación porcentual:  
**{correlacion:.3f}**
""")
