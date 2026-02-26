import streamlit as st
import pandas as pd
import plotly.express as px
import io
import csv

def mostrar_analisis_delitos():
    st.title(" Análisis de Tendencias: Delitos de Alto Impacto")

    # 1. CARGA Y LIMPIEZA (Lógica de tu Colab)
    file_path = 'Delitos_de_alto_impacto_en_Barranquilla.csv'
    encoding = 'latin1'

    try:
        with open(file_path, 'r', encoding=encoding) as f:
            raw_content = f.read()

        processed_rows = []
        for raw_line in raw_content.splitlines():
            # Limpieza de comillas anidadas ""VALUE"" -> "VALUE"
            inner_csv_line = raw_line[1:-1] if raw_line.startswith('"') and raw_line.endswith('"') else raw_line
            inner_csv_line = inner_csv_line.replace('""', '"')
            
            reader = csv.reader(io.StringIO(inner_csv_line), delimiter=',', quotechar='"')
            try:
                processed_rows.append(next(reader))
            except:
                continue

        df_delitos = pd.DataFrame(processed_rows[1:], columns=processed_rows[0])

        # 2. PROCESAMIENTO PARA LA GRÁFICA
        # Corregir nombres de columnas por el encoding latin1
        df_delitos.columns = [c.replace('AÃ±os', 'Anios').replace('Ãº', 'u') for c in df_delitos.columns]
        
        # Convertir a numérico la columna de casos
        target_col = 'Casos/denuncias ultimo periodo'
        df_delitos[target_col] = pd.to_numeric(df_delitos[target_col], errors='coerce').fillna(0)

        # Agrupar como en tu Colab
        time_series_data = df_delitos.groupby(['Anios comparados', 'Categoria_Delito'])[target_col].sum().reset_index()
        
        # Filtrar 'otros' y categorías vacías para limpiar la visualización
        time_series_data = time_series_data[~time_series_data['Categoria_Delito'].isin(['otros', '', None])]

        # 3. CREACIÓN DE LA VISUALIZACIÓN (Estilo Colab pero Interactivo)
        # Usamos Plotly con eje X categórico para evitar los "huecos" o picos
        fig = px.line(
            time_series_data, 
            x='Anios comparados', 
            y=target_col, 
            color='Categoria_Delito',
            markers=True,
            title='Evolución de Delitos por Periodos Comparados',
            labels={target_col: 'Total de Casos', 'Anios comparados': 'Periodo de Comparación'}
        )

        # Forzar el eje X a ser categórico para que no intente rellenar fechas
        fig.update_xaxes(type='category', tickangle=45)
        fig.update_layout(
            legend_title_text='Categoría',
            hovermode="x unified",
            template="plotly_dark" # Para que combine con el estilo de tu dashboard
        )

        # 4. RENDERIZADO EN STREAMLIT
        st.plotly_chart(fig, use_container_width=True)

        # Mostrar tabla resumen para auditoría
        with st.expander("Ver tabla de datos agregados"):
            st.dataframe(time_series_data)

    except FileNotFoundError:
        st.error(f"No se encontró el archivo: {file_path}. Asegúrate de que esté en la raíz del proyecto.")
    except Exception as e:
        st.error(f"Error al procesar los datos: {e}")

# Ejecutar la función
if __name__ == "__main__":
    mostrar_analisis_delitos()
