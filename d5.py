# app.py
# Tablero interactivo en Streamlit para delitos de alto impacto en Barranquilla (pronóstico 2026)
# Incluye consolidación anual sumando meses por año y vista previa antes de publicar

!pip install streamlit plotly statsmodels
!pip -q install streamlit pandas numpy plotly statsmodels pyngrok

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Pronóstico de delitos de alto impacto (Barranquilla)", layout="wide")

pregunta = "¿Cómo será el pronóstico de los delitos de alto impacto en la ciudad de Barranquilla en 2026?"

meses_es = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12
}

def normalizar_txt(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def ultimo_dia_mes(anio: int, mes: int) -> pd.Timestamp:
    return (pd.Timestamp(year=anio, month=mes, day=1) + pd.offsets.MonthEnd(0))

def extraer_mes_final(periodo: str):
    if not isinstance(periodo, str):
        return None
    s = normalizar_txt(periodo)
    encontrados = [m for m in meses_es.keys() if re.search(rf"\b{m}\b", s)]
    if not encontrados:
        return None
    return meses_es[encontrados[-1]]

def encontrar_columna(df: pd.DataFrame, patrones: list[str]):
    cols_norm = {c: normalizar_txt(c) for c in df.columns}
    for c, cn in cols_norm.items():
        if all(p in cn for p in patrones):
            return c
    return None

@st.cache_data(show_spinner=False)
def cargar_csv(archivo) -> pd.DataFrame:
    return pd.read_csv(archivo, encoding="utf-8", sep=",")

@st.cache_data(show_spinner=False)
def preparar_serie(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    c_delito = encontrar_columna(df, ["delito"])
    c_anios = encontrar_columna(df, ["años comparados"]) or encontrar_columna(df, ["anios comparados"])
    c_periodo = encontrar_columna(df, ["periodo meses comparado"])
    c_prev = encontrar_columna(df, ["casos/denuncias", "anterior", "periodo"])
    c_last = encontrar_columna(df, ["casos/denuncias", "último", "periodo"]) or encontrar_columna(df, ["casos/denuncias", "ultimo", "periodo"])

    necesarios = [c_delito, c_anios, c_periodo, c_prev, c_last]
    if any(x is None for x in necesarios):
        raise ValueError("No se encontraron todas las columnas necesarias. Revisa nombres del CSV.")

    df = df[[c_delito, c_anios, c_periodo, c_prev, c_last]].rename(columns={
        c_delito: "Delito",
        c_anios: "AniosComparados",
        c_periodo: "Periodo",
        c_prev: "AcumuladoPrevio",
        c_last: "AcumuladoUltimo"
    })

    df["Mes"] = df["Periodo"].apply(extraer_mes_final)
    df = df.dropna(subset=["Mes"]).copy()
    df["Mes"] = df["Mes"].astype(int)

    def split_anios(s):
        m = re.search(r"(\d{4})\s*-\s*(\d{4})", str(s))
        if not m:
            return None, None
        return int(m.group(1)), int(m.group(2))

    anios = df["AniosComparados"].apply(split_anios)
    df["AnioPrevio"] = anios.apply(lambda x: x[0])
    df["AnioUltimo"] = anios.apply(lambda x: x[1])
    df = df.dropna(subset=["AnioPrevio", "AnioUltimo"]).copy()
    df["AnioPrevio"] = df["AnioPrevio"].astype(int)
    df["AnioUltimo"] = df["AnioUltimo"].astype(int)

    df["AcumuladoPrevio"] = pd.to_numeric(df["AcumuladoPrevio"], errors="coerce")
    df["AcumuladoUltimo"] = pd.to_numeric(df["AcumuladoUltimo"], errors="coerce")
    df = df.dropna(subset=["AcumuladoPrevio", "AcumuladoUltimo"]).copy()

    df_prev = df[["Delito", "AnioPrevio", "Mes", "AcumuladoPrevio"]].rename(columns={
        "AnioPrevio": "Anio",
        "AcumuladoPrevio": "Acumulado"
    })
    df_last = df[["Delito", "AnioUltimo", "Mes", "AcumuladoUltimo"]].rename(columns={
        "AnioUltimo": "Anio",
        "AcumuladoUltimo": "Acumulado"
    })
    out = pd.concat([df_prev, df_last], ignore_index=True)

    # Corrección crítica: deduplicar por (Delito, Año, Mes)
    # Un mismo año puede aparecer en dos comparaciones; se conserva el mayor acumulado del mes.
    out = out.groupby(["Delito", "Anio", "Mes"], as_index=False)["Acumulado"].max()

    out["Fecha"] = out.apply(lambda r: ultimo_dia_mes(int(r["Anio"]), int(r["Mes"])), axis=1)
    out = out.sort_values(["Delito", "Anio", "Fecha"]).reset_index(drop=True)

    # Mensual desde acumulado dentro del mismo año
    out["Mensual"] = out.groupby(["Delito", "Anio"])["Acumulado"].diff()
    out["Mensual"] = out["Mensual"].fillna(out["Acumulado"])
    out["Mensual"] = out["Mensual"].clip(lower=0)

    # Total anual consolidado (suma de meses)
    out["TotalAnual"] = out.groupby(["Delito", "Anio"])["Mensual"].transform("sum")

    return out

def ajustar_pronostico_ets(serie: pd.Series, periodos: int, m_estacional: int | None):
    y = serie.dropna().astype(float)
    n = len(y)

    if n < 6:
        idx = pd.date_range(y.index.max() + pd.offsets.MonthEnd(1), periods=periodos, freq="ME")
        return pd.Series([y.iloc[-1]] * periodos, index=idx), "Naive (último valor)"

    if m_estacional is not None and n >= 2 * m_estacional:
        try:
            modelo = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=m_estacional).fit(optimized=True)
            return modelo.forecast(periodos), f"ETS (Holt-Winters aditivo, m={m_estacional})"
        except Exception:
            pass

    try:
        modelo = ExponentialSmoothing(y, trend="add", seasonal=None).fit(optimized=True)
        return modelo.forecast(periodos), "ETS (Holt aditivo)"
    except Exception:
        modelo = ExponentialSmoothing(y, trend=None, seasonal=None).fit(optimized=True)
        return modelo.forecast(periodos), "ETS (suavizado simple)"

st.title("Tablero de pronóstico de delitos de alto impacto en Barranquilla")
st.write(pregunta)

st.subheader("Metodología QUEST")

st.write("1. Q: Pregunta")
st.write("El objetivo es pronosticar los delitos de alto impacto para 2026 con base en el histórico.")

st.write("2. U: Entender")
st.write("El archivo compara pares de años y reporta acumulados por periodos. Se reconstruye una serie mensual y se consolida anual sumando los meses del año.")

st.write("3. E: Explorar")
st.write("Se valida coherencia del índice temporal, se revisan duplicados por mes y se inspeccionan patrones estacionales.")

st.write("4. S: Seleccionar")
st.write("Se utiliza suavizado exponencial (ETS) por parsimonia y buen desempeño en tendencia y estacionalidad. Se conserva promedio móvil como referencia visual.")

st.write("5. T: Trazar y probar")
st.write("Se presenta una vista previa de la serie antes de publicar, y luego se genera el pronóstico de 2026. El tablero permite filtrar delitos y rango de años.")

st.divider()

with st.sidebar:
    st.header("Carga y filtros")

    # Use the local CSV file as a default if no file is uploaded
    archivo = st.file_uploader("Carga el CSV", type=["csv"])
    if archivo is None:
        try:
            # Try to open the file from the local file system using its path
            archivo = "/content/Delitos_de_alto_impacto_en_Barranquilla.csv"
            with open(archivo, "rb") as f:
                archivo_bytes = f.read()
            # Streamlit's file_uploader returns an UploadedFile object, which can be simulated
            # For simple pandas.read_csv, a string path is sufficient.
            # However, to maintain the structure for Streamlit, we will assign the path directly
            # and modify cargar_csv to handle string paths as well.

            # Instead of simulating UploadedFile, let's just make the cargcsv function accept string or UploadedFile
            pass # archivo variable will now be the string path
        except FileNotFoundError:
            st.error("No se pudo encontrar el archivo CSV por defecto.")
            st.stop()
        except Exception as e:
            st.error(f"Error al cargar el archivo por defecto: {e}")
            st.stop()


    df_raw = cargar_csv(archivo)
    df_ts = preparar_serie(df_raw)

    delitos = sorted(df_ts["Delito"].dropna().unique().tolist())
    anio_min = int(df_ts["Anio"].min())
    anio_max = int(df_ts["Anio"].max())

    delito_sel = st.multiselect("Delito", options=delitos, default=delitos[:1] if delitos else [])
    anios_sel = st.slider("Rango de años", min_value=anio_min, max_value=anio_max, value=(max(anio_min, anio_max - 5), anio_max), step=1)

    modo = st.selectbox("Modo de análisis", ["Mensual", "Anual consolidado"])
    k_ma = st.slider("Ventana del promedio móvil (k)", min_value=2, max_value=24, value=7, step=1)
    mostrar_pronostico = st.checkbox("Mostrar pronóstico 2026", value=True)

df_f = df_ts.copy()
if delito_sel:
    df_f = df_f[df_f["Delito"].isin(delito_sel)]
df_f = df_f[(df_f["Anio"] >= anios_sel[0]) & (df_f["Anio"] <= anios_sel[1])].copy()

if df_f.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

st.subheader("Vista previa antes de publicar")

tab1, tab2 = st.tabs(["Serie mensual reconstruida", "Serie anual consolidada"])

with tab1:
    st.caption("Serie mensual con deduplicación por mes y mensualidad reconstruida desde acumulados.")
    vista_m = df_f[["Delito", "Anio", "Mes", "Fecha", "Acumulado", "Mensual"]].sort_values(["Delito", "Fecha"])
    st.dataframe(vista_m.tail(60), use_container_width=True)

    fig_m = px.line(vista_m, x="Fecha", y="Mensual", color="Delito", title="Serie mensual (Mensual)")
    st.plotly_chart(fig_m, use_container_width=True)

with tab2:
    st.caption("Serie anual consolidada calculada como suma de los meses del año.")
    anual = (
        df_f.groupby(["Delito", "Anio"], as_index=False)["Mensual"].sum()
        .rename(columns={"Mensual": "TotalAnual"})
    )
    anual["Fecha"] = pd.to_datetime(anual["Anio"].astype(str)) + pd.offsets.YearEnd(0)
    st.dataframe(anual.tail(30), use_container_width=True)

    fig_a = px.line(anual, x="Fecha", y="TotalAnual", color="Delito", title="Serie anual consolidada (Total anual)")
    st.plotly_chart(fig_a, use_container_width=True)

st.divider()

st.subheader("Histórico y promedio móvil")

if modo == "Mensual":
    base = df_f.sort_values(["Delito", "Fecha"]).set_index("Fecha")
    panel = []
    for d in sorted(base["Delito"].unique()):
        tmp = base[base["Delito"] == d].copy()
        idx = pd.date_range(tmp.index.min(), tmp.index.max(), freq="ME")
        s = tmp["Mensual"].reindex(idx).fillna(0.0)
        panel.append(pd.DataFrame({"Delito": d, "Fecha": s.index, "Valor": s.values}))
    serie_plot = pd.concat(panel, ignore_index=True)
else:
    anual = (
        df_f.groupby(["Delito", "Anio"], as_index=False)["Mensual"].sum()
        .rename(columns={"Mensual": "Valor"})
    )
    anual["Fecha"] = pd.to_datetime(anual["Anio"].astype(str)) + pd.offsets.YearEnd(0)
    serie_plot = anual[["Delito", "Fecha", "Valor"]].copy()

serie_plot = serie_plot.sort_values(["Delito", "Fecha"]).copy()
serie_plot["MA"] = serie_plot.groupby("Delito")["Valor"].transform(lambda s: s.rolling(k_ma, min_periods=1).mean())

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.line(serie_plot, x="Fecha", y="Valor", color="Delito", title=f"Histórico ({modo})"), use_container_width=True)
with c2:
    st.plotly_chart(px.line(serie_plot, x="Fecha", y="MA", color="Delito", title=f"Promedio móvil (k={k_ma})",
                            line_shape="spline"), use_container_width=True)

if mostrar_pronostico:
    st.subheader("Pronóstico 2026 con suavizado exponencial (ETS)")

    modelos = []
    pron_panel = []

    for d in sorted(serie_plot["Delito"].unique()):
        tmp = serie_plot[serie_plot["Delito"] == d].sort_values("Fecha").set_index("Fecha")["Valor"]

        if modo == "Mensual":
            tmp = tmp.asfreq("ME").fillna(0.0)
            fc, nombre = ajustar_pronostico_ets(tmp, periodos=12, m_estacional=12)
            idx_obj = pd.date_range("2026-01-31", "2026-12-31", freq="ME")
            fc = fc.reindex(idx_obj)
        else:
            # En anual consolidado, la estacionalidad no aplica igual; se pronostica 1 año (2026)
            tmp = tmp.asfreq("YE")
            fc, nombre = ajustar_pronostico_ets(tmp, periodos=1, m_estacional=None)
            idx_obj = pd.to_datetime(["2026-12-31"])
            fc = fc.reindex(idx_obj)

        pron_panel.append(pd.DataFrame({"Delito": d, "Fecha": fc.index, "Pronostico": fc.values}))
        modelos.append({"Delito": d, "Modelo": nombre, "Observaciones": int(tmp.dropna().shape[0])})

    df_modelos = pd.DataFrame(modelos)
    df_pron = pd.concat(pron_panel, ignore_index=True)

    st.dataframe(df_modelos, use_container_width=True)
    st.plotly_chart(px.line(df_pron, x="Fecha", y="Pronostico", color="Delito", title=f"Pronóstico 2026 ({modo})"), use_container_width=True)

    combinado = pd.concat(
        [
            serie_plot.assign(Tipo="Histórico").rename(columns={"Valor": "Serie"}),
            df_pron.assign(Tipo="Pronóstico").rename(columns={"Pronostico": "Serie"})
        ],
        ignore_index=True
    )
    st.plotly_chart(
        px.line(combinado, x="Fecha", y="Serie", color="Delito", line_dash="Tipo", title="Histórico y pronóstico"),
        use_container_width=True
    )

    st.subheader("Descarga")
    st.download_button(
        "Descargar pronóstico 2026 (CSV)",
        data=df_pron.to_csv(index=False).encode("utf-8"),
        file_name="pronostico_delitos_barranquilla_2026.csv",
        mime="text/csv"
    )

st.caption(
    "Nota: el archivo original reporta acumulados por periodos y pares de años. "
    "La serie mensual se reconstruye por diferencias de acumulados dentro de cada año, "
    "y el anual consolidado se obtiene sumando los meses del año."
)


 
   


       
