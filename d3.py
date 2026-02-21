import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


def to_number(x):
    if pd.isna(x):
        return np.nan

    v = str(x).strip()
    if v == "" or v.lower() in ("na", "n/a", "none", "nan"):
        return np.nan

    v = re.sub(r"[^\d\-,\.]", "", v)

    if "," in v and "." in v:
        if v.rfind(",") > v.rfind("."):
            v = v.replace(".", "")
            v = v.replace(",", ".")
        else:
            v = v.replace(",", "")
    elif "," in v and "." not in v:
        v = v.replace(",", ".")

    try:
        return float(v)
    except ValueError:
        return np.nan


def to_percent(x):
    if pd.isna(x):
        return np.nan

    v = str(x).strip()
    if v == "" or v.lower() in ("na", "n/a", "none", "nan"):
        return np.nan

    v = v.replace("%", "").strip()
    v = re.sub(r"[^\d\-,\.]", "", v).replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return np.nan


MESES = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def extraer_mes(texto: str) -> int | None:
    if pd.isna(texto):
        return None
    t = str(texto).strip().lower()
    for mes, n in MESES.items():
        if mes in t:
            return n
    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_lower = {c: re.sub(r"\s+", " ", c.strip().lower()) for c in df.columns}

    def find_col(all_terms):
        for orig, low in cols_lower.items():
            if all(term in low for term in all_terms):
                return orig
        return None

    c_periodo = find_col(["periodo"]) or find_col(["período"])
    c_years = find_col(["años"]) or find_col(["anos"]) or find_col(["comparados"])
    c_delito = find_col(["delito"])
    c_prev = find_col(["anterior", "periodo"]) or find_col(["anterior", "período"])
    c_last = find_col(["último", "periodo"]) or find_col(["ultimo", "periodo"])
    c_var_pct = find_col(["variación", "%"]) or find_col(["variacion", "%"])
    c_var_abs = find_col(["variación", "absoluta"]) or find_col(["variacion", "absoluta"])
    c_fuente = find_col(["fuente"])

    rename = {}
    if c_periodo:
        rename[c_periodo] = "periodo"
    if c_years:
        rename[c_years] = "anios_comparados"
    if c_delito:
        rename[c_delito] = "delito"
    if c_prev:
        rename[c_prev] = "casos_prev"
    if c_last:
        rename[c_last] = "casos_last"
    if c_var_pct:
        rename[c_var_pct] = "var_pct"
    if c_var_abs:
        rename[c_var_abs] = "var_abs"
    if c_fuente:
        rename[c_fuente] = "fuente"

    df = df.rename(columns=rename)

    if "casos_prev" in df.columns:
        df["casos_prev"] = df["casos_prev"].apply(to_number)
    if "casos_last" in df.columns:
        df["casos_last"] = df["casos_last"].apply(to_number)
    if "var_abs" in df.columns:
        df["var_abs"] = df["var_abs"].apply(to_number)
    if "var_pct" in df.columns:
        df["var_pct"] = df["var_pct"].apply(to_percent)

    if "casos_prev" in df.columns and "casos_last" in df.columns:
        if "var_abs" not in df.columns:
            df["var_abs"] = df["casos_last"] - df["casos_prev"]
        else:
            m = df["var_abs"].isna()
            df.loc[m, "var_abs"] = df.loc[m, "casos_last"] - df.loc[m, "casos_prev"]

        if "var_pct" not in df.columns:
            df["var_pct"] = np.where(
                df["casos_prev"].fillna(0) == 0,
                np.nan,
                (df["casos_last"] - df["casos_prev"]) / df["casos_prev"] * 100,
            )
        else:
            m = df["var_pct"].isna()
            df.loc[m, "var_pct"] = np.where(
                df.loc[m, "casos_prev"].fillna(0) == 0,
                np.nan,
                (df.loc[m, "casos_last"] - df.loc[m, "casos_prev"]) / df.loc[m, "casos_prev"] * 100,
            )

    if "periodo" in df.columns:
        df["mes"] = df["periodo"].apply(extraer_mes)

    return df


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return standardize_columns(df)


def fmt_int(x):
    if pd.isna(x):
        return "—"
    try:
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return str(x)


def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{float(x):.1f}%"


def construir_serie_mensual(df_std: pd.DataFrame) -> pd.DataFrame:
    """
    El campo 'periodo' es acumulado (enero a fin de mes). Se reconstruye una serie mensual
    diferenciando acumulados por delito y año.

    Se deriva el año del "último periodo" como el segundo año de 'anios_comparados' (por ejemplo, 2022-2023).
    """
    req = {"anios_comparados", "delito", "mes", "casos_prev", "casos_last"}
    if not req.issubset(set(df_std.columns)):
        return pd.DataFrame()

    rows = []
    for _, r in df_std.dropna(subset=["anios_comparados", "delito", "mes"]).iterrows():
        anios = str(r["anios_comparados"])
        m = int(r["mes"])
        parts = re.split(r"\s*-\s*", anios)
        if len(parts) != 2:
            continue
        try:
            y_prev = int(parts[0])
            y_last = int(parts[1])
        except ValueError:
            continue

        rows.append({"anio": y_prev, "mes": m, "delito": r["delito"], "acumulado": r["casos_prev"]})
        rows.append({"anio": y_last, "mes": m, "delito": r["delito"], "acumulado": r["casos_last"]})

    long = pd.DataFrame(rows).dropna(subset=["acumulado"])
    if long.empty:
        return long

    long = long.sort_values(["delito", "anio", "mes"])
    long["mensual"] = long.groupby(["delito", "anio"])["acumulado"].diff()
    long["mensual"] = long["mensual"].fillna(long["acumulado"])
    long["mensual"] = long["mensual"].clip(lower=0)

    long["fecha"] = pd.to_datetime(dict(year=long["anio"].astype(int), month=long["mes"].astype(int), day=1))
    return long[["fecha", "anio", "mes", "delito", "mensual"]]


def pronostico_estacional(
    serie: pd.DataFrame,
    fecha_inicio: str,
    fecha_fin: str,
    ventana_anios: int = 3,
) -> pd.DataFrame:
    """
    Pronóstico sencillo: promedio por mes del año usando los últimos 'ventana_anios' años disponibles.
    Devuelve media y bandas con ±1 desviación estándar (por mes).

    serie: DataFrame con columnas ['fecha', 'valor'] (mensual).
    """
    if serie.empty:
        return pd.DataFrame()

    s = serie.copy()
    s["anio"] = s["fecha"].dt.year
    s["mes"] = s["fecha"].dt.month

    max_anio = int(s["anio"].max())
    anio_min = max_anio - (ventana_anios - 1)
    base = s[s["anio"].between(anio_min, max_anio)].copy()

    patron = base.groupby("mes")["valor"].agg(media="mean", desviacion="std").reset_index()

    fechas_futuras = pd.date_range(start=fecha_inicio, end=fecha_fin, freq="MS")
    out = pd.DataFrame({"fecha": fechas_futuras})
    out["mes"] = out["fecha"].dt.month

    out = out.merge(patron, on="mes", how="left")
    out["pronostico"] = out["media"]
    out["inferior"] = out["media"] - out["desviacion"].fillna(0)
    out["superior"] = out["media"] + out["desviacion"].fillna(0)

    out = out.drop(columns=["media", "desviacion"])
    out["inferior"] = out["inferior"].clip(lower=0)

    return out


st.set_page_config(page_title="Tablero analítico de delitos", layout="wide")
st.title("Tablero analítico de delitos de alto impacto en Barranquilla")

DEFAULT_FILE = "Delitos_de_alto_impacto_en_Barranquilla.csv"

with st.sidebar:
    st.header("Datos")
    uploaded = st.file_uploader("Sube el CSV (opcional)", type=["csv"])

    if uploaded is not None:
        df = standardize_columns(pd.read_csv(uploaded))
    else:
        if not os.path.exists(DEFAULT_FILE):
            st.warning(
                f"No se encontró el archivo '{DEFAULT_FILE}' en el repositorio. "
                "Súbelo desde aquí o agrégalo a la raíz del proyecto."
            )
            st.stop()
        df = load_csv(DEFAULT_FILE)

    st.divider()
    st.header("Filtros")

    anios = sorted(df["anios_comparados"].dropna().unique().tolist()) if "anios_comparados" in df.columns else []
    periodos = sorted(df["periodo"].dropna().unique().tolist()) if "periodo" in df.columns else []
    delitos = sorted(df["delito"].dropna().unique().tolist()) if "delito" in df.columns else []

    sel_anios = st.multiselect("Años comparados", anios, default=anios[:1] if anios else [])
    sel_periodos = st.multiselect("Período", periodos, default=periodos[:1] if periodos else [])
    sel_delitos = st.multiselect("Delito", delitos, default=[])

    st.divider()
    st.header("Distribuciones")
    variable_dist = st.selectbox(
        "Variable para boxplot y violín",
        options=["var_pct", "var_abs", "casos_last", "casos_prev"],
        format_func={
            "var_pct": "Variación porcentual",
            "var_abs": "Variación absoluta",
            "casos_last": "Casos del último período",
            "casos_prev": "Casos del período anterior",
        }.get,
    )

df_f = df.copy()
if sel_anios and "anios_comparados" in df_f.columns:
    df_f = df_f[df_f["anios_comparados"].isin(sel_anios)]
if sel_periodos and "periodo" in df_f.columns:
    df_f = df_f[df_f["periodo"].isin(sel_periodos)]
if sel_delitos and "delito" in df_f.columns:
    df_f = df_f[df_f["delito"].isin(sel_delitos)]

needed = {"delito", "casos_prev", "casos_last", "var_abs", "var_pct"}
missing = [c for c in needed if c not in df_f.columns]
if missing:
    st.error(
        "Faltan columnas necesarias para el tablero: " + ", ".join(missing) + ". "
        "Revisa el CSV o ajusta el mapeo en standardize_columns()."
    )
    st.stop()

total_prev = df_f["casos_prev"].sum(skipna=True)
total_last = df_f["casos_last"].sum(skipna=True)
total_abs = total_last - total_prev
total_pct = np.nan if (pd.isna(total_prev) or total_prev == 0) else (total_abs / total_prev) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de casos (período anterior)", fmt_int(total_prev))
c2.metric("Total de casos (último período)", fmt_int(total_last))
c3.metric("Variación absoluta (total)", fmt_int(total_abs))
c4.metric("Variación porcentual (total)", fmt_pct(total_pct))

st.divider()

agg = (
    df_f.groupby("delito", as_index=False)
    .agg(
        casos_prev=("casos_prev", "sum"),
        casos_last=("casos_last", "sum"),
        var_abs=("var_abs", "sum"),
    )
)
agg["var_pct"] = np.where(
    agg["casos_prev"].fillna(0) == 0,
    np.nan,
    (agg["casos_last"] - agg["casos_prev"]) / agg["casos_prev"] * 100,
)
agg = agg.sort_values("casos_last", ascending=False)

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Comparativo por delito: período anterior vs último período")

    long = agg.melt(
        id_vars=["delito"],
        value_vars=["casos_prev", "casos_last"],
        var_name="periodo_tipo",
        value_name="casos",
    )
    long["periodo_tipo"] = long["periodo_tipo"].map({"casos_prev": "Anterior", "casos_last": "Último"})

    fig_bar = px.bar(
        long,
        x="delito",
        y="casos",
        color="periodo_tipo",
        barmode="group",
        labels={"delito": "Delito", "casos": "Casos o denuncias", "periodo_tipo": "Período"},
    )
    fig_bar.update_layout(xaxis_tickangle=-25, height=520, legend_title_text="Período")
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.subheader("Variación absoluta por delito")

    fig_abs = px.bar(
        agg.sort_values("var_abs", ascending=True),
        x="var_abs",
        y="delito",
        orientation="h",
        labels={"var_abs": "Variación absoluta", "delito": "Delito"},
    )
    fig_abs.update_layout(height=520)
    st.plotly_chart(fig_abs, use_container_width=True)

st.divider()

d1, d2 = st.columns([1, 1])

with d1:
    st.subheader("Diagrama de correlación (matriz)")

    corr_cols = ["casos_prev", "casos_last", "var_abs", "var_pct"]
    corr_df = df_f[corr_cols].dropna()
    if len(corr_df) < 3:
        st.info("No hay suficientes datos para calcular correlaciones con los filtros actuales.")
    else:
        corr = corr_df.corr(numeric_only=True)
        fig_corr = px.imshow(corr, text_auto=True, labels=dict(color="Correlación"))
        fig_corr.update_layout(height=420)
        st.plotly_chart(fig_corr, use_container_width=True)

with d2:
    st.subheader("Dispersión: casos del último período vs variación porcentual")

    fig_scatter = px.scatter(
        agg,
        x="casos_last",
        y="var_pct",
        size=np.clip(agg["casos_last"].fillna(0), 0, None) + 1,
        hover_name="delito",
        labels={"casos_last": "Casos (último período)", "var_pct": "Variación porcentual"},
    )
    fig_scatter.update_layout(height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

b1, b2 = st.columns([1, 1])

with b1:
    st.subheader("Boxplot por delito")

    if variable_dist in df_f.columns:
        fig_box = px.box(
            df_f.dropna(subset=[variable_dist]),
            x="delito",
            y=variable_dist,
            points="outliers",
            labels={"delito": "Delito", variable_dist: "Valor"},
        )
        fig_box.update_layout(xaxis_tickangle=-25, height=460)
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("La variable seleccionada no está disponible con los filtros actuales.")

with b2:
    st.subheader("Gráfico de violín por delito")

    if variable_dist in df_f.columns:
        fig_violin = px.violin(
            df_f.dropna(subset=[variable_dist]),
            x="delito",
            y=variable_dist,
            box=True,
            points="outliers",
            labels={"delito": "Delito", variable_dist: "Valor"},
        )
        fig_violin.update_layout(xaxis_tickangle=-25, height=460)
        st.plotly_chart(fig_violin, use_container_width=True)
    else:
        st.info("La variable seleccionada no está disponible con los filtros actuales.")

st.divider()

st.subheader("Pronóstico en serie de tiempo (enero de 2023 a enero de 2026)")

serie_m = construir_serie_mensual(df)
if serie_m.empty:
    st.info("No se pudo construir la serie mensual. Verifica que el CSV tenga 'Años comparados' y 'Período' con meses.")
else:
    total_m = (
        serie_m.groupby("fecha", as_index=False)["mensual"]
        .sum()
        .rename(columns={"mensual": "valor"})
        .sort_values("fecha")
    )

    real = total_m[(total_m["fecha"] >= "2023-01-01") & (total_m["fecha"] <= "2023-12-01")].copy()

    fc = pronostico_estacional(
        serie=total_m,
        fecha_inicio="2024-01-01",
        fecha_fin="2026-01-01",
        ventana_anios=3,
    )

    graf = pd.concat(
        [
            real.assign(tipo="Real", y=real["valor"]),
            fc.assign(tipo="Pronóstico", y=fc["pronostico"]),
        ],
        ignore_index=True,
    )

    fig_ts = px.line(
        graf,
        x="fecha",
        y="y",
        color="tipo",
        labels={"fecha": "Fecha", "y": "Total mensual (estimado a partir de acumulados)", "tipo": "Serie"},
    )
    fig_ts.update_layout(height=520)

    if not fc.empty:
        banda = fc[["fecha", "inferior", "superior"]].copy()
        fig_band_up = px.area(banda, x="fecha", y="superior")
        fig_band_lo = px.area(banda, x="fecha", y="inferior")
        for tr in fig_band_up.data:
            fig_ts.add_trace(tr)
        for tr in fig_band_lo.data:
            fig_ts.add_trace(tr)

    st.plotly_chart(fig_ts, use_container_width=True)

    with st.expander("Ver tabla de serie y pronóstico"):
        tabla_real = real.rename(columns={"valor": "real"}).set_index("fecha")
        tabla_fc = fc.set_index("fecha")[["pronostico", "inferior", "superior"]]
        tabla = tabla_real.join(tabla_fc, how="outer")
        st.dataframe(tabla, use_container_width=True, height=360)

st.divider()

st.subheader("Tabla resumen por delito")

view = agg[["delito", "casos_prev", "casos_last", "var_abs", "var_pct"]].copy()
view["casos_prev"] = view["casos_prev"].round(0).astype("Int64")
view["casos_last"] = view["casos_last"].round(0).astype("Int64")
view["var_abs"] = view["var_abs"].round(0).astype("Int64")
view["var_pct"] = view["var_pct"].round(2)

st.dataframe(view, use_container_width=True, height=420)

csv_export = view.to_csv(index=False).encode("utf-8")
with st.sidebar:
    st.divider()
    st.header("Exportación")
    st.download_button(
        "Descargar resumen (CSV)",
        data=csv_export,
        file_name="resumen_delitos_por_delito.csv",
        mime="text/csv",
    )
  
