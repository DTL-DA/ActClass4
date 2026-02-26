import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Delitos de Alto Impacto - Barranquilla", layout="wide")

# -------------------------------------------------------------------
# CARGA DEL DATASET
# -------------------------------------------------------------------

df = pd.read_csv("Delitos_de_alto_impacto_en_Barranquilla.csv")
df.columns = df.columns.str.strip()

# Limpieza
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
# TÍTULO
# -------------------------------------------------------------------

st.title("Análisis de Delitos de Alto Impacto en Barranquilla")

st.markdown("""
## Pregunta problema

**¿Qué relación existe entre el volumen de denuncias y la variación observada en los delitos de alto impacto?**
""")

st.markdown("---")

# -------------------------------------------------------------------
# FILTRO POR AÑO
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
# TOTALES GENERALES
# -------------------------------------------------------------------

total_anterior = df_filtrado["Casos/denuncias  anterior periodo"].sum()
total_actual = df_filtrado["Casos/denuncias último periodo"].sum()

variacion_absoluta_total = total_actual - total_anterior

if total_anterior != 0:
    variacion_porcentual_total = (variacion_absoluta_total / total_anterior) * 100
else:
    variacion_porcentual_total = 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total casos (Período anterior)", f"{int(total_anterior):,}")
col2.metric("Total casos (Último período)", f"{int(total_actual):,}")
col3.metric("Variación absoluta (Total)", f"{int(variacion_absoluta_total):,}")
col4.metric("Variación porcentual (Total)", f"{variacion_porcentual_total:.2f}%")

st.markdown("---")

# -------------------------------------------------------------------
# TABLA CONSOLIDADA POR DELITO
# -------------------------------------------------------------------

st.subheader("Resumen consolidado por delito")

tabla_resumen = df_filtrado.groupby("Delito").agg({
    "Casos/denuncias  anterior periodo": "sum",
    "Casos/denuncias último periodo": "sum"
}).reset_index()

tabla_resumen["Variación absoluta"] = (
    tabla_resumen["Casos/denuncias último periodo"] -
    tabla_resumen["Casos/denuncias  anterior periodo"]
)

tabla_resumen["Variación %"] = (
    tabla_resumen["Variación absoluta"] /
    tabla_resumen["Casos/denuncias  anterior periodo"] * 100
)

st.dataframe(tabla_resumen, use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------------
# VARIACIÓN ABSOLUTA POR DELITO
# -------------------------------------------------------------------

st.subheader("Variación absoluta por delito")

fig_var_abs = px.bar(
    tabla_resumen,
    x="Delito",
    y="Variación absoluta",
    title="Cambio absoluto en número de denuncias por delito"
)

fig_var_abs.update_layout(xaxis_tickangle=45)

st.plotly_chart(fig_var_abs, use_container_width=True)

st.markdown("---")

# -------------------------------------------------------------------
# COMPARATIVO ENTRE PERIODOS
# -------------------------------------------------------------------

st.subheader("Comparativo por delito: período anterior vs último período")

df_melt = df_filtrado.melt(
    id_vars="Delito",
    value_vars=[
        "Casos/denuncias  anterior periodo",
        "Casos/denuncias último periodo"
    ],
    var_name="Periodo",
    value_name="Denuncias"
)

fig_comp = px.bar(
    df_melt,
    x="Delito",
    y="Denuncias",
    color="Periodo",
    barmode="group",
    title="Comparación entre periodos"
)

fig_comp.update_layout(xaxis_tickangle=45)

st.plotly_chart(fig_comp, use_container_width=True)
