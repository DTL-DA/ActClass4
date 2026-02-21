import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# -----------------------------
# Limpieza y estandarización
# -----------------------------
def to_number(x):
    """
    Convierte valores tipo:
      '1.234' '1,234' '1.234,56' '1,234.56' ' 23 ' 'NA' '$ 5.000'
    a float. Devuelve NaN si no se puede.

    IMPORTANTÍSIMO: define 'v' siempre, evitando NameError.
    """
    if pd.isna(x):
        return np.nan

    v = str(x).strip()  # <-- 'v' definido SIEMPRE
    if v == "" or v.lower() in ("na", "n/a", "none", "nan"):
        return np.nan

    # deja solo dígitos, signo, punto y coma
    v = re.sub(r"[^\d\-,\.]", "", v)

    # Si tiene coma y punto: decidir cuál es decimal por la última aparición
    if "," in v and "." in v:
        # LATAM: 1.234,56 -> decimal es coma (aparece después)
        if v.rfind(",") > v.rfind("."):
            v = v.replace(".", "")
            v = v.replace(",", ".")
        # US: 1,234.56 -> decimal es punto
        else:
            v = v.replace(",", "")

    # Si solo tiene coma, asumir coma decimal
    elif "," in v and "." not in v:
        v = v.replace(",", ".")

    try:
        return float(v)
    except ValueError:
        return np.nan


def to_percent(x):
    """
    Convierte '-8%', '6 %', '0,5%' a float (escala 0-100).
    """
    if pd.isna(x):
        return np.nan
    v = str(x).strip()
    if v == "" or v.lower() in ("na", "n/a", "none", "nan"):
        return np.nan
    v = v.replace("%", "").strip()
    v = re.sub(r"[^\d\-,\.]", "", v)
    v = v.replace(",", ".")
    try:
        return float(v)
    except ValueError:
        return np.nan


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas a nombres estándar (robusto a variaciones).
    Esperados finales:
      periodo, anios_comparados, delito, casos_prev, casos_last, var_pct, var_abs, fuente (opcional)
    """
    df = df.copy()

    # normaliza nombres
    cols_lower = {c: re.sub(r"\s+", " ", c.strip().lower()) for c in df.columns}

    def find_col(all_terms):
        for orig, low in cols_lower.items():
            if all(term in low for term in all_terms):
                return orig
        return None

    # columnas típicas del dataset
    c_periodo = find_col(["periodo"]) or find_col(["período"])
    c_years = find_col(["años"]) or find_col(["anos"]) or find_col(["comparados"])
    c_delito = find_col(["delito"])
    c_prev = find_col(["anterior", "periodo"]) or find_col(["anterior", "período"])
    c_last = find_col(["último", "periodo"]) or find_col(["ultimo", "periodo"])
    c_var_pct = find_col(["variación", "%"]) or find_col(["variacion", "%"])
    c_var_abs = find_col(["variación", "absoluta"]) or find_col(["variacion", "absoluta"])
    c_fuente = find_col(["fuente"])

    rename = {}
    if c_periodo: rename[c_periodo] = "periodo"
    if c_years: rename[c_years] = "anios_comparados"
    if c_delito: rename[c_delito] = "delito"
    if c_prev: rename[c_prev] = "casos_prev"
    if c_last: rename[c_last] = "casos_last"
    if c_var_pct: rename[c_var_pct] = "var_pct"
    if c_var_abs: rename[c_var_abs] = "var_abs"
    if c_fuente: rename[c_fuente] = "fuente"

    df = df.rename(columns=rename)

    # conversión numérica
    if "casos_prev" in df.columns:
        df["casos_prev"] = df["casos_prev"].apply(to_number)
    if "casos_last" in df.columns:
        df["casos_last"] = df["casos_last"].apply(to_number)
    if "var_abs" in df.columns:
        df["var_abs"] = df["var_abs"].apply(to_number)
    if "var_pct" in df.columns:
        df["var_pct"] = df["var_pct"].apply(to_percent)

    # recalcular variaciones si faltan o hay NaN
    if "casos_prev" in df.columns and "casos_last" in df.columns:
        if "var_abs" not in df.columns:
            df["var_abs"] = df["casos_last"] - df["casos_prev"]
        else:
            mask = df["var_abs"].isna()
            df.loc[mask, "var_abs"] = df.loc[mask, "casos_last"] - df.loc[mask, "casos_prev"]

        if "var_pct" not in df.columns:
            df["var_pct"] = np.where(
                df["casos_prev"].fillna(0) == 0,
                np.nan,
                (df["casos_last"] - df["casos_prev"]) / df["casos_prev"] * 100,
            )
        else:
            mask = df["var_pct"].isna()
            df.loc[mask, "var_pct"] = np.where(
                df.loc[mask, "casos_prev"].fillna(0) == 0,
                np.nan,
                (df.loc[mask, "casos_last"] - df.loc[mask, "casos_prev"]) / df.loc[mask, "casos_prev"] * 100,
            )

    return df


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return standardize_columns(df)


def fmt_int(x):
    if pd.isna(x):
        return "—"
    try:
        # separador de miles estilo ES (.)
        return f"{int(round(float(x))):,}".replace(",", ".")
    except Exception:
        return str(x)


def fmt_pct(x):
    if pd.isna(x):
        return "—"
    return f"{float(x):.1f}%"


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Tablero Analítico - Delitos", layout="wide")
st.title("📊 Tablero Analítico - Delitos de Alto Impacto (Barranquilla)")

DEFAULT_FILE = "Delitos_de_alto_impacto_en_Barranquilla.csv"

with st.sidebar:
    st.header("📁 Datos")
    uploaded = st.file_uploader("Sube el CSV (opcional)", type=["csv"])

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        df = standardize_columns(df_raw)
        st.success("CSV cargado desde el cargador.")
    else:
        if not os.path.exists(DEFAULT_FILE):
            st.warning(
                f"No encontré '{DEFAULT_FILE}' en el repo.\n\n"
                "➡️ Sube el archivo arriba o añádelo al repositorio en la raíz."
            )
            st.stop()
        df = load_csv(DEFAULT_FILE)
        st.caption(f"Usando archivo local: {DEFAULT_FILE}")

    st.divider()
    st.header("🎛️ Filtros")

    # filtros (si existen)
    anios = sorted(df["anios_comparados"].dropna().unique().tolist()) if "anios_comparados" in df.columns else []
    periodos = sorted(df["periodo"].dropna().unique().tolist()) if "periodo" in df.columns else []
    delitos = sorted(df["delito"].dropna().unique().tolist()) if "delito" in df.columns else []

    sel_anios = st.multiselect("Años comparados", anios, default=anios[:1] if anios else [])
    sel_periodos = st.multiselect("Periodo", periodos, default=periodos[:1] if periodos else [])
    sel_delitos = st.multiselect("Delito", delitos, default=[])

# aplicar filtros
df_f = df.copy()
if sel_anios and "anios_comparados" in df_f.columns:
    df_f = df_f[df_f["anios_comparados"].isin(sel_anios)]
if sel_periodos and "periodo" in df_f.columns:
    df_f = df_f[df_f["periodo"].isin(sel_periodos)]
if sel_delitos and "delito" in df_f.columns:
    df_f = df_f[df_f["delito"].isin(sel_delitos)]

# columnas mínimas necesarias para el tablero
needed = {"delito", "casos_prev", "casos_last", "var_abs", "var_pct"}
missing = [c for c in needed if c not in df_f.columns]
if missing:
    st.error(
        "Faltan columnas necesarias para graficar.\n\n"
        f"Columnas faltantes: {missing}\n\n"
        "Revisa el CSV o ajusta el mapeo en standardize_columns()."
    )
    st.stop()

# KPIs
total_prev = df_f["casos_prev"].sum(skipna=True)
total_last = df_f["casos_last"].sum(skipna=True)
total_abs = total_last - total_prev
total_pct = np.nan if (total_prev == 0 or pd.isna(total_prev)) else (total_abs / total_prev) * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total casos (Periodo anterior)", fmt_int(total_prev))
k2.metric("Total casos (Último periodo)", fmt_int(total_last))
k3.metric("Variación absoluta (Total)", fmt_int(total_abs))
k4.metric("Variación % (Total)", fmt_pct(total_pct))

st.divider()

# agregación por delito
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

# -----------------------------
# Visualizaciones
# -----------------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("📌 Comparativo por delito (Anterior vs Último)")

    long = agg.melt(
        id_vars=["delito"],
        value_vars=["casos_prev", "casos_last"],
        var_name="periodo_tipo",
        value_name="casos",
    )
    long["periodo_tipo"] = long["periodo_tipo"].map(
        {"casos_prev": "Anterior", "casos_last": "Último"}
    )

    fig_bar = px.bar(
        long,
        x="delito",
        y="casos",
        color="periodo_tipo",
        barmode="group",
        labels={"delito": "Delito", "casos": "Casos/Denuncias", "periodo_tipo": "Periodo"},
    )
    fig_bar.update_layout(xaxis_tickangle=-25, height=520, legend_title_text="Periodo")
    st.plotly_chart(fig_bar, use_container_width=True)

with right:
    st.subheader("📈 Variación absoluta por delito")

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

c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("🎯 Dispersión: casos (último) vs variación %")

    fig_scatter = px.scatter(
        agg,
        x="casos_last",
        y="var_pct",
        size=np.clip(agg["casos_last"].fillna(0), 0, None) + 1,
        hover_name="delito",
        labels={"casos_last": "Casos (Último periodo)", "var_pct": "Variación %"},
    )
    fig_scatter.update_layout(height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

with c2:
    st.subheader("🧾 Tabla resumen (por delito)")
    view = agg[["delito", "casos_prev", "casos_last", "var_abs", "var_pct"]].copy()

    # types más amigables para ver
    view["casos_prev"] = view["casos_prev"].round(0).astype("Int64")
    view["casos_last"] = view["casos_last"].round(0).astype("Int64")
    view["var_abs"] = view["var_abs"].round(0).astype("Int64")
    view["var_pct"] = view["var_pct"].round(2)

    st.dataframe(view, use_container_width=True, height=420)

# export
csv_export = view.to_csv(index=False).encode("utf-8")
with st.sidebar:
    st.divider()
    st.header("⬇️ Exportar")
    st.download_button(
        "Descargar resumen (CSV)",
        data=csv_export,
        file_name="resumen_delitos_por_delito.csv",
        mime="text/csv",
    )

st.caption("Tip: usa los filtros de la izquierda para comparar periodos, años y delitos.")
