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

df = df.dropna()

# -------------------------------------------------------------------
# TÍTULO
# -------------------------------------------------------------------

st.title("Análisis de Delitos de Alto Impacto en Barranquilla")

st.markdown("""
## ¿Qué relación existe entre el volumen de denuncias y la variación observada en los delitos de alto impacto?
""")

st.markdown("---")

# -------------------------------------------------------------------
# FILTROS
# -------------------------------------------------------------------

st.sidebar.header("Filtros")

# Filtro por año
años = sorted(df["Años comparados"].unique())

año_seleccionado = st.sidebar.multiselect(
    "Seleccionar año",
    options=años,
    default=años
)

df_filtrado = df[df["Años comparados"].isin(año_seleccionado)]

# Filtro por delito
delitos = sorted(df_filtrado["Delito"].unique())

delito_seleccionado = st.sidebar.multiselect(
    "Seleccionar delito",
    options=delitos,
    default=delitos
)

df_filtrado = df_filtrado[df_filtrado["Delito"].isin(delito_seleccionado)]

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
# TABLA RESUMEN BASE
# -------------------------------------------------------------------

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

st.markdown("---")
# -------------------------------------------------------------------
# PRINCIPIO DE PARETO 80/20
# -------------------------------------------------------------------

st.markdown("---")
st.subheader("Análisis de concentración – Principio de Pareto (80/20)")

# Usamos valor absoluto porque interesa magnitud del cambio
tabla_pareto = tabla_resumen.copy()
tabla_pareto["Impacto absoluto"] = tabla_pareto["Variación absoluta"].abs()

# Ordenar de mayor a menor impacto
tabla_pareto = tabla_pareto.sort_values(
    by="Impacto absoluto", ascending=False
)

# Calcular porcentaje del total
total_impacto = tabla_pareto["Impacto absoluto"].sum()

tabla_pareto["% del total"] = (
    tabla_pareto["Impacto absoluto"] / total_impacto * 100
)

tabla_pareto["% acumulado"] = tabla_pareto["% del total"].cumsum()

# Clasificación 80/20
tabla_pareto["Grupo"] = tabla_pareto["% acumulado"].apply(
    lambda x: "Delitos que explican el 80%" if x <= 80 else "Resto de delitos"
)

# Agrupar para gráfica de torta
resumen_pareto = tabla_pareto.groupby("Grupo")["Impacto absoluto"].sum().reset_index()

fig_pareto = px.pie(
    resumen_pareto,
    values="Impacto absoluto",
    names="Grupo",
    title="Concentración del cambio total en denuncias (Principio de Pareto)"
)

st.plotly_chart(fig_pareto, use_container_width=True)

# Mostrar delitos que explican el 80%
st.markdown("### Delitos que concentran el 80% del cambio")

delitos_criticos = tabla_pareto[tabla_pareto["Grupo"] == "Delitos que explican el 80%"]

st.dataframe(
    delitos_criticos[["Delito", "Variación absoluta", "% acumulado"]],
    use_container_width=True
)
# -------------------------------------------------------------------
# TABLA RESUMEN FINAL
# -------------------------------------------------------------------

st.subheader("Tabla resumen consolidada por delito")

st.dataframe(tabla_resumen, use_container_width=True)
