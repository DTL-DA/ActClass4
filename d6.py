import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Delitos de Alto Impacto - Barranquilla", layout="wide")

# -------------------------------------------------------------------
# Título y pregunta problema
# -------------------------------------------------------------------

st.title("Análisis de Delitos de Alto Impacto en Barranquilla")

st.markdown("""
### Pregunta problema

¿Qué relación existe entre el volumen de denuncias y la variación observada en los delitos de alto impacto?
""")

st.markdown("---")

# -------------------------------------------------------------------
# Cargar datos
# -------------------------------------------------------------------

archivo = st.file_uploader("Cargar archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)

    # Limpieza básica
    df.columns = df.columns.str.strip()

    # Convertir columnas numéricas
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

    # Eliminar filas vacías
    df = df.dropna(subset=["Casos/denuncias último periodo"])

    # -------------------------------------------------------------------
    # Indicadores generales
    # -------------------------------------------------------------------

    total_denuncias = df["Casos/denuncias último periodo"].sum()
    variacion_promedio = df["Variación %"].mean()

    col1, col2 = st.columns(2)

    col1.metric("Total denuncias último periodo", f"{int(total_denuncias):,}")
    col2.metric("Variación porcentual promedio", f"{variacion_promedio:.2f}%")

    st.markdown("---")

    # -------------------------------------------------------------------
    # 1. Ranking por volumen de denuncias
    # -------------------------------------------------------------------

    st.subheader("Ranking por volumen de denuncias")

    df_volumen = df.sort_values(
        by="Casos/denuncias último periodo", ascending=False
    )

    fig1 = px.bar(
        df_volumen,
        x="Delito",
        y="Casos/denuncias último periodo",
        title="Volumen de denuncias por tipo de delito",
    )

    fig1.update_layout(xaxis_tickangle=45)

    st.plotly_chart(fig1, use_container_width=True)

    # -------------------------------------------------------------------
    # 2. Variación porcentual por delito
    # -------------------------------------------------------------------

    st.subheader("Variación porcentual por delito")

    fig2 = px.bar(
        df,
        x="Delito",
        y="Variación %",
        title="Variación porcentual entre periodos",
    )

    fig2.update_layout(xaxis_tickangle=45)

    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------------------------------------------
    # 3. Relación entre volumen y variación
    # -------------------------------------------------------------------

    st.subheader("Relación entre volumen de denuncias y variación porcentual")

    fig3 = px.scatter(
        df,
        x="Casos/denuncias último periodo",
        y="Variación %",
        size="Casos/denuncias último periodo",
        color="Delito",
        hover_name="Delito",
        title="Volumen vs Variación %",
    )

    st.plotly_chart(fig3, use_container_width=True)

    # -------------------------------------------------------------------
    # 4. Participación porcentual de cada delito
    # -------------------------------------------------------------------

    st.subheader("Participación porcentual de cada delito")

    df["Participación %"] = (
        df["Casos/denuncias último periodo"] / total_denuncias * 100
    )

    fig4 = px.pie(
        df,
        values="Casos/denuncias último periodo",
        names="Delito",
        title="Distribución de denuncias por delito",
    )

    st.plotly_chart(fig4, use_container_width=True)

    # -------------------------------------------------------------------
    # 5. Correlación estadística
    # -------------------------------------------------------------------

    st.subheader("Análisis de correlación")

    correlacion = df["Casos/denuncias último periodo"].corr(df["Variación %"])

    st.write(
        f"Coeficiente de correlación entre volumen de denuncias y variación porcentual: {correlacion:.3f}"
    )

    if correlacion > 0:
        st.write("Existe una relación positiva entre volumen y variación.")
    elif correlacion < 0:
        st.write("Existe una relación negativa entre volumen y variación.")
    else:
        st.write("No se observa relación lineal significativa.")

else:
    st.info("Cargue el archivo CSV para visualizar el análisis.")
