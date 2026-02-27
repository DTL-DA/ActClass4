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
# PÁGINA 1: DASHBOARD
# -------------------------------------------------------------------

if pagina == "Dashboard Analítico":

    # CARGA DEL DATASET
    df = pd.read_csv("Delitos_de_alto_impacto_en_Barranquilla.csv")
    df.columns = df.columns.str.strip()

    df["Casos/denuncias  anterior periodo"] = pd.to_numeric(
        df["Casos/denuncias  anterior periodo"], errors="coerce"
    )

    df["Casos/denuncias último periodo"] = pd.to_numeric(
        df["Casos/denuncias último periodo"], errors="coerce"
    )

    df = df.dropna()

    # TÍTULO
    st.title("Análisis de Delitos de Alto Impacto en Barranquilla")

    st.markdown("""
    ## ¿Qué relación existe entre el volumen de denuncias y la variación observada en los delitos de alto impacto?
       """)

    st.markdown("---")

    # FILTROS
    st.sidebar.header("Filtros")

    años = df["Años comparados"].unique()
    delitos = df["Delito"].unique()

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
        (df["Años comparados"].isin(año_seleccionado)) &
        (df["Delito"].isin(delito_seleccionado))
    ]

    # TOTALES GENERALES
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
    # TABLA RESUMEN
    # -------------------------------------------------------------------

    tabla_resumen = df_filtrado.groupby("Delito").agg({
        "Casos/denuncias  anterior periodo": "sum",
        "Casos/denuncias último periodo": "sum"
    }).reset_index()

    tabla_resumen["Variación absoluta"] = (
        tabla_resumen["Casos/denuncias último periodo"] -
        tabla_resumen["Casos/denuncias  anterior periodo"]
    )

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

    st.subheader("Tabla resumen consolidada por delito")
    st.dataframe(tabla_resumen, use_container_width=True)

    # -------------------------------------------------------------------
    # DIAGRAMA DE CORRELACIÓN
    # -------------------------------------------------------------------

    st.subheader("Matriz de correlación")

    st.markdown("""
    ## ¿Los delitos que tenían más denuncias antes siguen teniendo más ahora?
       """)
    
    df_filtrado["Variación absoluta"] = (
        df_filtrado["Casos/denuncias último periodo"] -
        df_filtrado["Casos/denuncias  anterior periodo"]
    )

    df_filtrado["Variación %"] = (
        df_filtrado["Variación absoluta"] /
        df_filtrado["Casos/denuncias  anterior periodo"].replace(0, 1)
    ) * 100
    
    variables_corr = df_filtrado[[
        "Casos/denuncias  anterior periodo",
        "Casos/denuncias último periodo",
        "Variación %",
        "Variación absoluta"
    ]].astype(float)
    
    matriz_corr = variables_corr.corr()
    
    fig4 = px.imshow(
        matriz_corr,
        text_auto=True,
        title="Mapa de calor de correlaciones",
        aspect="auto"
    )
    
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    ## Es alta y positiva (cerca de 1), significa:  
    
       Los delitos más frecuentes siguen siendo los más frecuentes.
       
       La estructura delictiva es estable.  
       
       El impacto delictivo está concentrado en delitos estructurales.
       """)

    st.markdown("""
    ## Si fuera baja:
    
       Hubo reconfiguración del patrón delictivo
       """)

    # -------------------------------------------------------------------
    # PRINCIPIO DE PARETO 80/20
    # -------------------------------------------------------------------

    st.markdown("---")
    st.subheader("Análisis de concentración – Principio de Pareto (80/20)")

    st.markdown("""
    ## ¿Se cumple el principio de "Pareto" 80/20 en la variación de los delitos de alto impacto?
       """)

    tabla_pareto = tabla_resumen.copy()
    tabla_pareto["Impacto absoluto"] = tabla_pareto["Variación absoluta"].abs()

    tabla_pareto = tabla_pareto.sort_values(
        by="Impacto absoluto", ascending=False
    )

    total_impacto = tabla_pareto["Impacto absoluto"].sum()

    tabla_pareto["% del total"] = (
        tabla_pareto["Impacto absoluto"] / total_impacto * 100
    )

    tabla_pareto["% acumulado"] = tabla_pareto["% del total"].cumsum()

    tabla_pareto["Grupo"] = tabla_pareto["% acumulado"].apply(
        lambda x: "Delitos que explican el 80%" if x <= 80 else "Resto de delitos"
    )

    resumen_pareto = tabla_pareto.groupby("Grupo")["Impacto absoluto"].sum().reset_index()

    fig_pareto = px.pie(
        resumen_pareto,
        values="Impacto absoluto",
        names="Grupo",
        title="Concentración del cambio total en denuncias (Principio de Pareto)"
    )

    st.plotly_chart(fig_pareto, use_container_width=True)

    st.markdown("### Delitos que concentran el 80% del cambio")

    delitos_criticos = tabla_pareto[
        tabla_pareto["Grupo"] == "Delitos que explican el 80%"
    ]

    st.dataframe(
        delitos_criticos[["Delito", "Variación absoluta", "% acumulado"]],
        use_container_width=True
    )


    st.markdown("""
    ## Si el gráfico muestra que pocos delitos concentran el 80%:

       El fenómeno es estructural y focalizable
       
       La política pública puede priorizar esos delitos

       Contrario a esto:

       Si el 80% del cambio necesita muchos delitos:

       El fenómeno es generalizado
       
       Se requieren estrategias transversales
       """)
# -------------------------------------------------------------------
# PÁGINA 2: DOCUMENTACIÓN
# -------------------------------------------------------------------

elif pagina == "Documentación y Metodología":

    st.title("Documentación del Panel")

    st.markdown("## Fuente de datos")
    st.write("""
    - Dataset: Delitos de Alto Impacto en Barranquilla  
    - Fuente institucional: Registros administrativos de seguridad ciudadana  
    - Archivo utilizado: Delitos_de_alto_impacto_en_Barranquilla.csv  
    """)

    st.markdown("## Fecha de acceso a los datos")
    st.write(f"""
    **Sábado, ‎21 ‎de ‎febrero ‎de ‎2026, ‏‎11:21:53 a. m.**
    """)

    st.markdown("## Periodo analizado")
    st.write("""
    Comparaciones interanuales entre periodos 2019 – 2023
    """)
    st.markdown("""
    Este panel presenta un análisis comparativo de los delitos de alto impacto en Barranquilla durante el periodo 2019–2023, utilizando el volumen de denuncias.
    como variable central para comprender su evolución y comportamiento. El objetivo es evaluar la relación entre la dinámica de las denuncias y la variación
    observada en cada delito, identificando patrones estructurales y tendencias emergentes. A través de métricas agregadas, comparativos interanuales y
    visualizaciones analíticas, el usuario puede interpretar la magnitud y concentración del impacto delictivo. Las métricas superiores resumen los cambios
    totales entre periodos, mientras que los gráficos detallan la contribución individual por delito. Este análisis busca aportar una base técnica y empírica
    para la formulación y focalización de la estrategia de seguridad de la ciudad. En particular, sirve como insumo para la elaboración y ajuste del Plan Integral
    de Seguridad y Convivencia Ciudadana (PISCC).
    """)
    st.markdown("## Metodología aplicada")
    st.write("""
    El análisis sigue el marco QUEST:
    
    - Q: Definición de la pregunta sobre relación entre denuncias y variación delictiva.
    - U: Comprensión de estructura del dataset.
    - E: Exploración comparativa de periodos.
    - S: Análisis de variaciones absolutas y concentración del impacto.
    - T: Comunicación visual mediante dashboard interactivo.
    """)

    st.markdown("## Fuente:")
    st.write("""

     Proviene de Datos Abiertos Colombia, publicado por la Alcaldía Distrital de Barranquilla.
   
   -Link de acceso: https://www.datos.gov.co/d/4p95-h82w. presionar en "EXPORTAR".
  
   
    """)

    st.markdown("---")
    st.info("Este panel prioriza la transparencia, trazabilidad y actualización continua de datos.")
