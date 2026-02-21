import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Tablero Analítico - Delitos", layout="wide")

DEFAULT_CSV = "Delitos_de_alto_impacto_en_Barranquilla.csv"

# ----------------------------
# Helpers
# ----------------------------
def _normalize_colname(c: str) -> str:
    c = str(c).strip()
    c = re.sub(r"\s+", " ", c)
    return c

def _to_numeric_series(s: pd.Series) -> pd.Series:
    # Convierte strings tipo "1.234" o "1,234" o "1 234" a numérico cuando sea posible
    if s.dtype.kind in "biufc":
        return s
    x = s.astype(str).str.strip()
    x = x.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    # quita separadores de miles comunes
    x = x.str.replace(r"[^\d\-,\.]", "", regex=True)
    # si hay comas y puntos, intenta estandarizar
    # Caso frecuente: 1.234 -> miles. Caso: 1,234 -> miles. Caso: 12,5 -> decimal.
    # Estrategia:
    # - si tiene más de 1 separador, quitar miles
    # - luego reemplazar coma decimal por punto si aplica
    def fix(v: str) -> str:
        if v is None or v is np.nan:
            return v
        if v.count(",") > 0 and v.count(".") > 0:
            # asumir que el separador decimal es el último que aparece
            if v.rfind(",") > v.rfind("."):
                v = v.replace(".", "")
                v = v.replace(",", ".")
            else:
                v = v.replace(",", "")
        else:
            # si solo hay comas: puede ser miles o decimal
            if v.count(",") == 1 and len(v.split(",")[-1]) in (1, 2):
                v = v.replace(",", ".")  # decimal
            else:
                v = v.replace(",", "")   # miles
            # si solo hay puntos: casi siempre miles en LATAM, pero podría ser decimal.
            # si hay 1 punto y 1-2 decimales -> decimal; si no, miles
            if v.count(".") == 1 and len(v.split(".")[-1]) in (1, 2):
                pass
            else:
                v = v.replace(".", "")
        return v

    x = x.apply(fix)
    return pd.to_numeric(x, errors="coerce")

def detect_period_col(df: pd.DataFrame) -> str | None:
    # Busca una columna que parezca "Año", "Mes", "Fecha", "Periodo"
    candidates = []
    for c in df.columns:
        cl = c.lower()
        if any(k in cl for k in ["fecha", "period", "año", "ano", "mes", "trimestre", "semana"]):
            candidates.append(c)

    # Prioridad: fecha/periodo explícito
    for prefer in ["fecha", "period", "periodo"]:
        for c in candidates:
            if prefer in c.lower():
                return c

    # Si hay Año y Mes separados, devolvemos None y armamos un periodo después
    return candidates[0] if candidates else None

def build_period_if_year_month(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    cols_lower = {c.lower(): c for c in df.columns}
    year_col = None
    month_col = None

    for key in ["año", "ano", "year"]:
        if key in cols_lower:
            year_col = cols_lower[key]
            break
    for key in ["mes", "month"]:
        if key in cols_lower:
            month_col = cols_lower[key]
            break

    if year_col and month_col:
        tmp = df.copy()
        y = _to_numeric_series(tmp[year_col]).astype("Int64")
        m = _to_numeric_series(tmp[month_col]).astype("Int64")
        # crea una fecha al primer día del mes
        tmp["_Periodo"] = pd.to_datetime(
            dict(year=y.fillna(1).astype(int), month=m.fillna(1).astype(int), day=1),
            errors="coerce",
        )
        return tmp, "_Periodo"
    return df, None

def safe_parse_datetime(s: pd.Series) -> pd.Series:
    # Intenta parsear datetime en varios formatos
    return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)

def pick_category_cols(df: pd.DataFrame, numeric_cols: list[str]) -> list[str]:
    return [c for c in df.columns if c not in numeric_cols]

# ----------------------------
# Carga de datos
# ----------------------------
st.title("Tablero Analítico de Delitos (CSV)")

with st.sidebar:
    st.header("📁 Datos")
    uploaded = st.file_uploader("Sube el CSV (opcional)", type=["csv"])
    sep = st.selectbox("Separador CSV", options=[",", ";", "\t", "|"], index=0)
    encoding = st.selectbox("Encoding", options=["utf-8", "latin-1"], index=0)
    na_values = st.text_input("Valores NA (separados por coma)", value="")

@st.cache_data(show_spinner=False)
def load_data_from_bytes(data: bytes, sep: str, encoding: str, na_values_raw: str) -> pd.DataFrame:
    na_vals = [v.strip() for v in na_values_raw.split(",") if v.strip()] if na_values_raw else None
    return pd.read_csv(io.BytesIO(data), sep=sep, encoding=encoding, na_values=na_vals)

@st.cache_data(show_spinner=False)
def load_data_from_path(path: str, sep: str, encoding: str, na_values_raw: str) -> pd.DataFrame:
    na_vals = [v.strip() for v in na_values_raw.split(",") if v.strip()] if na_values_raw else None
    return pd.read_csv(path, sep=sep, encoding=encoding, na_values=na_vals)

try:
    if uploaded is not None:
        df = load_data_from_bytes(uploaded.getvalue(), sep, encoding, na_values)
        source_label = "Archivo subido"
    else:
        df = load_data_from_path(DEFAULT_CSV, sep, encoding, na_values)
        source_label = DEFAULT_CSV
except Exception as e:
    st.error(
        "No pude cargar el CSV. Prueba cambiando el separador/encoding o sube el archivo.\n\n"
        f"Detalle: {e}"
    )
    st.stop()

# Normaliza nombres de columnas
df = df.copy()
df.columns = [_normalize_colname(c) for c in df.columns]

# Intenta construir periodo si hay año/mes
df, period_from_year_month = build_period_if_year_month(df)

# Detecta y convierte columnas numéricas (intenta sobre object también)
numeric_guess = []
for c in df.columns:
    if df[c].dtype.kind in "biufc":
        numeric_guess.append(c)
    else:
        # prueba: si al convertir a numérico se pierden pocos datos, considerarla numérica
        conv = _to_numeric_series(df[c])
        if conv.notna().sum() >= max(5, int(0.6 * len(df))):
            df[c] = conv
            numeric_guess.append(c)

numeric_cols = [c for c in numeric_guess if df[c].dtype.kind in "biufc"]

# Detecta columna de periodo/fecha
period_col = period_from_year_month or detect_period_col(df)
if period_col and period_col in df.columns:
    if not np.issubdtype(df[period_col].dtype, np.datetime64):
        parsed = safe_parse_datetime(df[period_col])
        # si se parsea razonablemente bien, usarla como datetime
        if parsed.notna().sum() >= max(5, int(0.5 * len(df))):
            df[period_col] = parsed
else:
    period_col = None

cat_cols = pick_category_cols(df, numeric_cols)

# ----------------------------
# Vista previa
# ----------------------------
with st.expander("🔎 Vista previa del dataset", expanded=False):
    st.caption(f"Fuente: {source_label} | Filas: {len(df):,} | Columnas: {len(df.columns)}")
    st.dataframe(df.head(50), use_container_width=True)

if not numeric_cols:
    st.warning("No detecté columnas numéricas para graficar. Revisa el separador/encoding o el contenido del CSV.")
    st.stop()

# ----------------------------
# Filtros (sidebar)
# ----------------------------
with st.sidebar:
    st.header("🎛️ Filtros")
    # filtro por categorías (máx 2 para no saturar)
    selectable_cat_cols = [c for c in cat_cols if df[c].nunique(dropna=True) <= 200]
    cat1 = st.selectbox("Categoría 1", options=["(ninguna)"] + selectable_cat_cols, index=0)
    cat2 = st.selectbox("Categoría 2", options=["(ninguna)"] + [c for c in selectable_cat_cols if c != cat1], index=0)

    # filtro por periodo si existe
    if period_col:
        st.subheader("⏱️ Rango de tiempo")
        dmin = pd.to_datetime(df[period_col]).min()
        dmax = pd.to_datetime(df[period_col]).max()
        if pd.notna(dmin) and pd.notna(dmax):
            start, end = st.date_input(
                "Selecciona rango",
                value=(dmin.date(), dmax.date()),
                min_value=dmin.date(),
                max_value=dmax.date(),
            )
        else:
            start, end = None, None
    else:
        start, end = None, None

    metric_col = st.selectbox("Métrica numérica", options=numeric_cols, index=0)
    agg = st.selectbox("Agregación", options=["Suma", "Promedio", "Mediana", "Máximo", "Mínimo"], index=0)

def apply_filters(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()

    if period_col and start and end:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        out = out[(out[period_col] >= s) & (out[period_col] <= e)]

    # filtros de categorías (si eligieron)
    for c in [cat1, cat2]:
        if c and c != "(ninguna)":
            vals = out[c].dropna().unique()
            default_pick = vals[: min(len(vals), 10)]
            chosen = st.sidebar.multiselect(f"Valores en {c}", options=sorted(vals), default=list(default_pick))
            if chosen:
                out = out[out[c].isin(chosen)]

    return out

df_f = apply_filters(df)

# ----------------------------
# KPIs
# ----------------------------
def aggregate_series(s: pd.Series, how: str) -> float:
    if how == "Suma":
        return float(s.sum(skipna=True))
    if how == "Promedio":
        return float(s.mean(skipna=True))
    if how == "Mediana":
        return float(s.median(skipna=True))
    if how == "Máximo":
        return float(s.max(skipna=True))
    if how == "Mínimo":
        return float(s.min(skipna=True))
    return float(s.sum(skipna=True))

kpi_total = aggregate_series(df_f[metric_col], agg)
kpi_count = int(df_f[metric_col].notna().sum())
kpi_rows = len(df_f)

c1, c2, c3 = st.columns(3)
c1.metric(label=f"{agg} de {metric_col}", value=f"{kpi_total:,.2f}")
c2.metric(label=f"Registros con {metric_col}", value=f"{kpi_count:,}")
c3.metric(label="Filas (post-filtros)", value=f"{kpi_rows:,}")

st.divider()

# ----------------------------
# Gráficos
# ----------------------------
# 1) Serie temporal (si hay periodo)
if period_col:
    st.subheader("📈 Evolución en el tiempo")

    # Agrupa por periodo (día/mes/año según granularidad)
    # Si parece mensual (muchos valores con día=1), agrupa por mes.
    dt = df_f[[period_col, metric_col]].dropna()
    if not dt.empty:
        # decide frecuencia
        is_monthly_like = (dt[period_col].dt.day == 1).mean() > 0.7
        freq = "MS" if is_monthly_like else "D"
        ts = (
            dt.set_index(period_col)[metric_col]
            .resample(freq)
            .sum()
            .reset_index()
            .rename(columns={metric_col: "valor"})
        )
        fig_ts = px.line(ts, x=period_col, y="valor", markers=True)
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No hay datos suficientes para la serie temporal con los filtros actuales.")

# 2) Top categorías (si hay al menos una categoría elegible)
st.subheader("📊 Comparación por categorías")

group_cols = [c for c in [cat1, cat2] if c and c != "(ninguna)"]
if group_cols:
    gb = df_f.groupby(group_cols, dropna=False)[metric_col]
    if agg == "Suma":
        g = gb.sum(min_count=1)
    elif agg == "Promedio":
        g = gb.mean()
    elif agg == "Mediana":
        g = gb.median()
    elif agg == "Máximo":
        g = gb.max()
    elif agg
