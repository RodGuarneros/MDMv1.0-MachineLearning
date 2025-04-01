#######################
# Importar  librerias #
#######################
import streamlit as st
st.set_page_config(
    page_title="Aprendizaje Automático para los Municipios de México",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)
import gc
import pandas as pd
import altair as alt
import plotly.express as px
import io
import geopandas as gpd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.io as pio
import altair_viewer as altviewer
import logging
import folium
import zipfile
from streamlit import components
from sklearn.linear_model import LinearRegression
import folium
from streamlit_folium import folium_static  # Importar folium_static para Streamlit
from scipy.stats import gaussian_kde
import pymongo
from pymongo import MongoClient
from gridfs import GridFS
from io import BytesIO
from dotenv import load_dotenv
import os
from bson import ObjectId
from concurrent.futures import ThreadPoolExecutor

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Optimización 1: Mejorar la conexión y consultas a MongoDB

# 1. Establecer conexión una sola vez (singleton)
@st.cache_resource
def get_mongodb_connection():
    """
    Establece una única conexión a MongoDB reutilizable
    """
    mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
    client = MongoClient(mongo_uri, maxPoolSize=10)  # Aumentar el tamaño del pool de conexiones
    return client

# 2. Optimizar las funciones de consulta para hacer menos operaciones
@st.cache_data(ttl=3600)  # Cache por 1 hora
def incrementar_contador_visitas():
    """
    Incrementa el contador de visitas de manera más eficiente
    """
    try:
        client = get_mongodb_connection()
        db = client['Municipios_Rodrigo']
        collection = db['visita']
        
        # Usar updateOne con upsert en lugar de find_one_and_update para mejor rendimiento
        result = collection.update_one(
            {"_id": "contador"},
            {"$inc": {"contador": 1}},
            upsert=True
        )
        
        # Obtener el valor actualizado
        doc = collection.find_one({"_id": "contador"})
        return doc.get('contador', 0)
    except Exception as e:
        st.error(f"Error al acceder a la base de datos: {e}")
        return 0  # Devolver 0 en caso de error para evitar interrupciones

# 3. Optimizar la carga de datos con proyecciones y filtros específicos
@st.cache_data(ttl=3600*6)  # Cache por 6 horas
def bajando_procesando_datos():
    """
    Optimiza la carga de datos principales usando proyecciones y filtrado del lado del servidor
    """
    client = get_mongodb_connection()
    db = client['Municipios_Rodrigo']
    collection = db['datos_finales']

    # Especificar solo los campos necesarios (proyección)
    projection = {
        "_id": 1,
        "Lugar": 1,
        "Estado2": 1,
        "Madurez": 1,
        "Ranking": 1,
        "Etapa_Madurez": 1,
        "Índice_Compuesto": 1,
        "cvegeo": 1,
        "Latitud": 1,
        "Longitud": 1,
        # Incluir aquí todas las variables numéricas y categóricas relevantes
        # que realmente se utilicen en las visualizaciones
    }

    # Obtener datos con proyección para reducir el tamaño de transferencia
    datos_raw = list(collection.find({}, projection))
    
    # Convertir a DataFrame sin necesidad del map si la estructura es simple
    datos = pd.DataFrame(datos_raw)
    
    # Limpieza de datos más eficiente
    for column in datos.select_dtypes(include=['object']).columns:
        if column in datos and datos[column].notna().any():
            try:
                datos[column] = datos[column].astype(str)
            except:
                pass  # Si falla la conversión, mantener el tipo original
    
    # Optimización de categorías
    categorias_orden = ['Optimización', 'Definición', 'En desarrollo', 'Inicial']
    if 'Madurez' in datos.columns:
        datos['Madurez'] = pd.Categorical(
            datos['Madurez'].astype(str).str.strip(),
            categories=categorias_orden,
            ordered=True  # Hacerlo ordenado mejora el rendimiento de ordenación
        )
    
    return datos

# 4. Optimizar la carga del conjunto de datos completo
@st.cache_data(ttl=3600*12)  # Cache por 12 horas
def bajando_procesando_datos_completos():
    """
    Optimiza la carga del dataset completo
    """
    client = get_mongodb_connection()
    db = client['Municipios_Rodrigo']
    collection = db['completo']

    # Proyectar solo los campos necesarios
    projection = {
        # Incluir solo los campos que realmente se usan
        "Lugar": 1,
        # Añadir otros campos relevantes
    }

    # Obtener los datos con una sola operación
    datos_raw = list(collection.find({}, projection))
    
    dataset_complete = pd.DataFrame(datos_raw)
    
    # Limpieza más eficiente de columnas
    dataset_complete.columns = dataset_complete.columns.str.strip()
    
    return dataset_complete

# 5. Uso de multithreading para cargas paralelas
def cargar_todos_los_datos():
    """
    Carga todos los conjuntos de datos en paralelo
    """
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Lanzar las tareas en paralelo
        future_datos = executor.submit(bajando_procesando_datos)
        future_completo = executor.submit(bajando_procesando_datos_completos)
        future_normalizador = executor.submit(bajando_procesando_X_entrenamiento)
        
        # Obtener resultados
        datos = future_datos.result()
        dataset_complete = future_completo.result()
        df = future_normalizador.result()
        
    return datos, dataset_complete, df




# Optimización 2: Mejorar el procesamiento y visualización de datos

# 1. Optimizar la carga y procesamiento de archivos GeoJSON
@st.cache_data(ttl=3600*24)  # Cache por 24 horas - los datos geográficos rara vez cambian
def obtener_datos_geograficos():
    """
    Obtiene los datos geográficos de MongoDB de manera más eficiente
    """
    mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
    client = MongoClient(mongo_uri)
    db = client['Municipios_Rodrigo']
    fs = GridFS(db)
    
    file = fs.find_one({'filename': 'municipios.geojson'})
    if not file:
        return None
    
    # Cargar directamente como GeoDataFrame
    gdf = gpd.read_file(BytesIO(file.read()))
    
    # Optimizar las columnas del GeoDataFrame
    gdf['CVEGEO'] = gdf['CVEGEO'].astype(str)
    
    # Reducir el tamaño del GeoDataFrame manteniendo solo columnas necesarias
    columns_to_keep = ['CVEGEO', 'geometry']
    gdf = gdf[columns_to_keep]
    
    return gdf

# 2. Optimizar el procesamiento de datos para las visualizaciones
@st.cache_data
def preparar_datos_para_visualizacion(datos, geojson):
    """
    Prepara los datos para visualización de manera eficiente
    """
    if 'cvegeo' in datos.columns:
        # Crear copia solo de las columnas necesarias
        datos_viz = datos[['Lugar', 'Madurez', 'cvegeo', 'Ranking', 'Etapa_Madurez', 'Índice_Compuesto']].copy()
        
        # Conversión eficiente de tipos
        datos_viz['cvegeo'] = datos_viz['cvegeo'].astype(str).str.zfill(5)
        datos_viz.rename(columns={'cvegeo': 'CVEGEO'}, inplace=True)
        
        # Fusionar datos con geometría de manera eficiente
        if geojson is not None:
            # Usar merge con parámetros optimizados
            dataset_geometry = datos_viz.merge(
                geojson[['CVEGEO', 'geometry']], 
                on='CVEGEO', 
                how='left',
                suffixes=('', '_geo')  # Evitar columnas duplicadas
            )
            return dataset_geometry
    return datos

# 3. Optimizar funciones de visualización con caching adecuado
@st.cache_data
def crear_mapa_choropleth2(dataset, lugar=None, municipio_inicial="Abalá, Yucatán"):
    """
    Crea un mapa choropleth interactivo mostrando clústeres y filtrando por lugar.
    Destaca el municipio seleccionado en color naranja y tamaño más grande.
    """
    # Convertir el dataset a GeoDataFrame si aún no lo es
    gdf = gpd.GeoDataFrame(dataset, geometry='geometry')
    
    # Filtrar por 'Lugar' si se pasa como parámetro
    lugar_a_buscar = lugar if lugar else municipio_inicial
    if lugar_a_buscar:
        gdf_filtrado = gdf[gdf['Lugar'] == lugar_a_buscar]
        if gdf_filtrado.empty:
            print(f"No se encontraron datos para el lugar: {lugar_a_buscar}")
            return None
        gdf = gdf_filtrado

    # Obtener el centroide del municipio seleccionado
    centro = gdf.geometry.centroid.iloc[0]
    
    # Crear el mapa base centrado en el municipio
    m = folium.Map(
        location=[centro.y, centro.x],
        zoom_start=12,  # Aumentamos el zoom inicial
        tiles="CartoDB dark_matter"
    )
    
    # Ajustar los límites del mapa al municipio seleccionado
    bounds = gdf.geometry.total_bounds
    m.fit_bounds([
        [bounds[1], bounds[0]],  # esquina suroeste
        [bounds[3], bounds[2]]   # esquina noreste
    ])

    # Mapa de colores personalizado para los clústeres
    mapa_colores = {
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7',
        'Definición': '#CC6CE7',
        'Optimización': '#51C622',
    }

    # CAMBIO: Modificar la función para obtener color para usar naranja para el municipio seleccionado
    def obtener_color(feature, lugar_seleccionado):
        if feature['properties']['Lugar'] == lugar_seleccionado:
            return '#FFA500'  # Color naranja para el municipio seleccionado
        return mapa_colores.get(feature['properties']['Madurez'], '#FFFFFF')

    # CAMBIO: Modificar el style_function para aplicar estilo especial al municipio seleccionado
    def style_function(feature):
        is_selected = feature['properties']['Lugar'] == lugar_a_buscar
        return {
            'fillColor': '#FFA500' if is_selected else mapa_colores.get(feature['properties']['Madurez'], '#FFFFFF'),
            'color': 'white' if is_selected else 'black',  # Borde blanco para resaltar
            'weight': 3 if is_selected else 1,  # Borde más grueso para el seleccionado
            'fillOpacity': 0.9 if is_selected else 0.7,  # Más opaco el seleccionado
        }

    # Añadir la capa GeoJson con los colores personalizados y tooltips
    folium.GeoJson(
        gdf,
        name="Choropleth de Clústers",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['Lugar', 'Madurez'],
            aliases=['Lugar', 'Grado de Madurez'],
            localize=True,
            sticky=True  # Hace que el tooltip sea permanente
        ),
        highlight_function=lambda x: {'weight': 5, 'fillOpacity': 1.0}  # Resalta al pasar el mouse
    ).add_to(m)

    # Añadir control de capas
    folium.LayerControl().add_to(m)

    # Añadir leyenda con estilo mejorado
    legend = """
    <div style="position: fixed; 
                bottom: 50px; left: 50px; 
                background-color: white;
                border: 2px solid grey;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                z-index: 1000;">
        <b>Grado de Madurez</b><br>
        <i style="background: #D20103; width: 15px; height: 15px; display: inline-block; margin-right: 5px;"></i> En desarrollo<br>
        <i style="background: #5DE2E7; width: 15px; height: 15px; display: inline-block; margin-right: 5px;"></i> Inicial<br>
        <i style="background: #CC6CE7; width: 15px; height: 15px; display: inline-block; margin-right: 5px;"></i> Definición<br>
        <i style="background: #51C622; width: 15px; height: 15px; display: inline-block; margin-right: 5px;"></i> Optimización<br>
        <i style="background: #FFA500; width: 15px; height: 15px; display: inline-block; margin-right: 5px;"></i> Municipio seleccionado<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    return m

# 4. Optimizar la generación de gráficos con Plotly
@st.cache_data
def plot_bar_chart(data, lugar_columna, indice_columna, lugar_seleccionado):
    """
    Versión optimizada de la función de gráfico de barras
    """
    # Reducir el tamaño de los datos
    cols_needed = [lugar_columna, indice_columna, 'Ranking', 'Etapa_Madurez']
    plot_data = data[cols_needed].copy()
    
    # Convertir a numérico de manera eficiente
    plot_data[indice_columna] = pd.to_numeric(plot_data[indice_columna], errors='coerce')
    
    # Ordenar eficientemente
    plot_data = plot_data.sort_values(by=indice_columna, ascending=True)
    
    # Crear colores de manera eficiente
    bar_colors = ['red' if lugar == lugar_seleccionado else 'dodgerblue' 
                 for lugar in plot_data[lugar_columna]]
    
    # Crear la figura de manera más eficiente
    fig = go.Figure()
    
    # Añadir traza con formato optimizado
    fig.add_trace(go.Bar(
        x=plot_data[indice_columna],
        y=plot_data[lugar_columna],
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=0.5)
        ),
        customdata=np.stack((
            plot_data["Ranking"],
            plot_data["Etapa_Madurez"],
            plot_data[indice_columna]
        ), axis=-1),
        hovertemplate=(
            "Municipio: %{y}<br>" +
            "Índice de Madurez: %{customdata[2]:.10f}<br>" +
            "Lugar en el Ranking: %{customdata[0]}<br>" +
            "Madurez: %{customdata[1]}<extra></extra>"
        )
    ))
    
    # Optimizar anotaciones
    annotations = []
    for i, (lugar, ranking, valor) in enumerate(zip(
            plot_data[lugar_columna], 
            plot_data["Ranking"], 
            plot_data[indice_columna])):
        if i % 5 == 0 or lugar == lugar_seleccionado:  # Reducir cantidad de anotaciones
            annotations.append(dict(
                xref='paper', yref='y',
                x=0, y=lugar,
                text=lugar,
                showarrow=False,
                font=dict(
                    color='red' if lugar == lugar_seleccionado else 'white',
                    size=10,
                    family="Arial"
                ),
                xanchor='right',
                xshift=-10
            ))
            annotations.append(dict(
                x=valor, y=lugar,
                text=f"{int(ranking)} ({valor:.6f})",  # Reducir decimales mostrados
                showarrow=False,
                font=dict(color='white', size=7),
                xanchor='left',
                xshift=5
            ))
    
    # Ajustar altura de manera más eficiente
    num_lugares = len(plot_data)
    height = max(400, min(800, num_lugares * 15))  # Limitar la altura máxima
    
    # Actualizar layout de manera más eficiente
    fig.update_layout(
        title=dict(
            text=f"Índice de Madurez por Municipio (Resaltado: {lugar_seleccionado})",
            font=dict(color='#FFD86C')
        ),
        xaxis_title=dict(text="Índice de Madurez", font=dict(color='#FFD86C')),
        yaxis_title=dict(text="Municipio", font=dict(color='#FFD86C')),
        height=height,
        margin=dict(l=200, r=20, t=70, b=50),
        showlegend=False,
        xaxis=dict(
            range=[0, plot_data[indice_columna].max() * 1.1],
            tickformat='.6f',  # Reducir precisión para mejorar rendimiento
            showgrid=False
        ),
        yaxis=dict(showticklabels=False, showgrid=False),
        annotations=annotations,
        bargap=0.2,
        plot_bgcolor='rgba(0, 0, 0, 0.1)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    return fig



# Optimización 3: Mejorar el rendimiento de gráficos estadísticos

# 1. Optimizar el histograma para mejor rendimiento
@st.cache_data
def plot_histogram(df, numeric_column, categorical_column, nbins=30):
    """
    Versión optimizada del histograma
    """
    # Utilizar solo las columnas necesarias para reducir uso de memoria
    df_subset = df[[numeric_column, categorical_column]].copy()
    
    # Mapa de colores predefinido para mejorar rendimiento
    color_map = {
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7',
        'Definición': '#CC6CE7',
        'Optimización': '#51C622',
    }

    # Usar nbins para controlar la granularidad del histograma
    fig = px.histogram(
        df_subset, 
        x=numeric_column, 
        color=categorical_column,
        color_discrete_map=color_map,
        opacity=0.6,
        nbins=nbins,  # Controlar número de bins para mejor rendimiento
        title=f'Histograma de la variable "{numeric_column}" y <br>la categoría "{categorical_column}"'
    )
    
    # Actualizar ejes
    fig.update_yaxes(title_text="Frecuencia absoluta")
    
    # Calcular estadísticas de manera más eficiente
    # Usar .agg para calcular todas las estadísticas de una vez
    stats = df_subset[numeric_column].agg(['mean', 'median', 'std']).to_dict()
    try:
        # La moda puede ser costosa, manejarla por separado
        stats['Moda'] = df_subset[numeric_column].mode().iloc[0]
    except:
        stats['Moda'] = "N/A"
    
    # Crear texto para las anotaciones de manera más eficiente
    stats_text = "<br>".join([f"<b>{key.capitalize()}</b>: {value:.2f}" for key, value in stats.items()])

    # Simplificar el conteo por categoría
    category_counts = df_subset[categorical_column].value_counts()
    counts_text = "<br>".join([f"<b>{category}</b>: {count}" for category, count in category_counts.items()])
    
    # Crear el texto de anotaciones de una sola vez
    annotations_text = f"{stats_text}<br><br><b>Conteo por categoría:</b><br>{counts_text}"
    
    # Configurar una sola anotación en lugar de múltiples
    annotations = [dict(
        x=1.1,
        y=0.9,
        xref='paper',
        yref='paper',
        text=annotations_text,
        showarrow=False,
        font=dict(color='white', size=12),
        align='center',
        bgcolor='rgba(0, 0, 0, 0.7)',
        bordercolor='white',
        borderwidth=1,
        opacity=0.8
    )]
    
    # Actualizar el diseño de manera más eficiente
    fig.update_layout(
        title_font=dict(color='#FFD86C', size=16),
        title_x=0.05,
        showlegend=True,
        width=1350,
        height=500,
        margin=dict(l=50, r=50, t=80, b=200),
        annotations=annotations,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.3,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title_font=dict(color='#FFD86C'),
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        yaxis=dict(
            title_font=dict(color='#FFD86C'),
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
    )

    return fig

# 2. Optimizar la creación del histograma con densidad
@st.cache_data
def plot_histogram_with_density(df, numeric_column, selected_value=None, nbins=40):
    """
    Versión optimizada del histograma con densidad
    """
    # Usar solo los datos necesarios
    hist_data = df[numeric_column].dropna().astype(float)
    
    # Crear el histograma con configuración optimizada
    fig = px.histogram(
        hist_data,
        nbins=nbins,  # Controlar número de bins
        opacity=0.6,
        title=f'Distribución del índice de madurez digital',
        labels={'value': 'Valores del Índice', 'count': 'Frecuencia'}
    )
    
    # Aplicar estilo a las barras
    fig.update_traces(marker_line_color='white', marker_line_width=1.5)

    # Calcular la densidad de manera más eficiente
    if len(hist_data) > 1:  # Verificar que hay suficientes datos
        try:
            kde = gaussian_kde(hist_data)
            # Reducir el número de puntos para la línea de densidad
            density_x = np.linspace(hist_data.min(), hist_data.max(), 200)
            density_y = kde(density_x)
            # Escalar la densidad para que coincida con el histograma
            scale_factor = len(hist_data) * (hist_data.max() - hist_data.min()) / nbins
            density_y_scaled = density_y * scale_factor

            # Agregar la línea de densidad
            fig.add_trace(
                go.Scatter(
                    x=density_x,
                    y=density_y_scaled,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Densidad'
                )
            )
        except:
            # Si falla el KDE, continuar sin la línea de densidad
            pass
    
    # Añadir punto seleccionado de manera eficiente
    if selected_value is not None:
        try:
            # Convertir el valor seleccionado a numérico
            selected_row = df[df['Lugar'] == selected_value]
            if not selected_row.empty:
                selected_value_float = selected_row[numeric_column].values[0]
                fig.add_trace(
                    go.Scatter(
                        x=[selected_value_float],
                        y=[0],
                        mode='markers+text',
                        marker=dict(color='red', size=10, line=dict(color='white', width=1)),
                        text=f'{selected_value_float:.2f}',
                        textposition='top center',
                        name='Lugar seleccionado'
                    )
                )
        except:
            pass  # Si falla, continuar sin el punto resaltado
    
    # Calcular estadísticas de manera eficiente
    stats = hist_data.agg(['mean', 'std', 'median']).to_dict()
    try:
        stats['mode'] = hist_data.mode().iloc[0]
    except:
        stats['mode'] = "N/A"
    
    # Texto de anotaciones
    annotation_text = (
        f"<b>Estadísticos:</b><br>"
        f"Media: {stats['mean']:.2f}<br>"
        f"Mediana: {stats['median']:.2f}<br>"
        f"Moda: {stats['mode']:.2f}<br>"
        f"Desv. Est.: {stats['std']:.2f}"
    )
    
    # Añadir anotaciones
    fig.add_annotation(
        dict(
            x=1, y=0.95, xref='paper', yref='paper',
            text=annotation_text,
            showarrow=False,
            font=dict(size=12, color='white'),
            align='left',
            bgcolor='rgba(0, 0, 0, 0.7)',
            bordercolor='rgba(255, 255, 255, 0.7)',
            borderwidth=2
        )
    )

    # Estilo del gráfico optimizado
    fig.update_layout(
        title_font=dict(color='#FFD86C'),
        xaxis_title_font=dict(color='#FFD86C'),
        yaxis_title_font=dict(color='#FFD86C'),
        legend=dict(title_text='Leyenda', font=dict(color='#FFD86C')),
        xaxis=dict(
            showgrid=False,
            title='Valores del Índice'
        ),
        yaxis=dict(
            showgrid=False,
            title='Frecuencia'
        ),
        plot_bgcolor='rgba(0, 0, 0, 0.1)',
    )

    return fig

# 3. Optimizar el boxplot
@st.cache_data
def generate_boxplot_with_annotations(df, variable, lugar_seleccionado):
    """
    Versión modificada del boxplot para destacar municipio seleccionado en naranja y tamaño más grande
    """
    stats = {
        'Media': np.mean(df[variable]),
        'Mediana': np.median(df[variable]),
        'Moda': df[variable].mode().iloc[0],
        'Desviación estándar': np.std(df[variable])
    }
    
    fig = px.box(
        df,
        y=variable,
        points=False,  # No mostrar puntos en el boxplot
        title=f'Diagrama para la variable<br>"{variable}"',
        template='plotly_dark'
    )

    if lugar_seleccionado:
        df_lugar = df[df['Lugar'] == lugar_seleccionado]
        if not df_lugar.empty:
            # CAMBIO: Usar naranja y tamaño más grande
            fig.add_scatter(
                x=[0] * len(df_lugar),
                y=df_lugar[variable],
                mode='markers',
                marker=dict(
                    color='#FFA500',  # Cambio a naranja
                    size=15,         # Tamaño más grande
                    line=dict(color='white', width=1)  # Borde blanco para visibilidad
                ),
                name=f'Lugar seleccionado: {lugar_seleccionado}',
                hovertemplate='<b>%{customdata[0]}</b><br>'+variable+': %{y:.2f}<extra></extra>',
                customdata=df_lugar[['Municipio']]
            )

    df_rest = df[df['Lugar'] != lugar_seleccionado]
    fig.add_scatter(
        x=[0] * len(df_rest),
        y=df_rest[variable],
        mode='markers',
        marker=dict(
            color='rgba(255, 165, 0, 0.3)',  # Más transparente para contrastar con el seleccionado
            size=7,
            line=dict(color='rgba(255, 165, 0, 0.7)', width=0.5)
        ),
        name='Otros lugares',
        hovertemplate='<b>%{customdata[0]}</b><br>'+variable+': %{y:.2f}<extra></extra>',
        customdata=df_rest[['Municipio']]
    )

    # Texto de las anotaciones agrupado
    annotations_text = "<br>".join([f"<b>{stat_name}</b>: {stat_value:.2f}" for stat_name, stat_value in stats.items()])
    
    # Añadir anotaciones agrupadas
    annotations = [
        dict(
            x=0.5,  # Centrar
            y=-0.3,  # Ubicar debajo de la leyenda
            xref='paper',
            yref='paper',
            text=annotations_text,
            showarrow=False,
            font=dict(color='white', size=12),
            align='center',
            bgcolor='rgba(0, 0, 0, 0.7)',  # Fondo oscuro
            bordercolor='white',  # Borde blanco
            borderwidth=2,  # Ancho del borde
            opacity=0.8  # Opacidad del recuadro
        )
    ]

    fig.update_layout(
        title_font=dict(color='#FFD86C', size=16),
        title_x=0.2,  # Centrar título
        showlegend=True,
        width=1350,
        height=500,  # Altura ajustada
        margin=dict(l=55, r=55, t=80, b=200),  # Márgenes ajustados para leyenda y anotaciones
        annotations=annotations,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.3,  # Posicionar leyenda debajo de la gráfica
            xanchor='center',
            x=0.5,   # Centrar leyenda
            bgcolor='rgba(0,0,0,0)'
        ),
        yaxis=dict(
            title=variable,
            title_font=dict(color='#FFD86C'),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.2)'
        ),
        xaxis=dict(
            showticklabels=False,
            zeroline=False,
            showgrid=False
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig



# Optimización 4: Mejorar gráficos 3D y 2D

# 1. Optimizar la generación del gráfico 3D
@st.cache_data
def generar_grafico_3d_con_lugar(df, df_normalizado, dataset_complete, lugar_seleccionado=None, max_points=2000):
    """
    Versión optimizada del gráfico 3D que destaca el municipio seleccionado con color naranja y mayor tamaño
    """
    # Mapa de colores predefinido
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }
    
    # Extraer solo las columnas PCA necesarias de manera eficiente
    if isinstance(df_normalizado, pd.DataFrame):
        # Si ya es DataFrame, extraer columnas
        if df_normalizado.shape[1] >= 4:
            pca_cols = df_normalizado.iloc[:, 1:4].values
        else:
            # Fallback si no hay suficientes columnas
            return None
    else:
        # Si es numpy array
        df_pca2 = df_normalizado
        if df_pca2.shape[1] >= 4:
            pca_cols = df_pca2[:, 1:4]
        else:
            # Fallback si no hay suficientes columnas
            return None

    # Crear DataFrame para Plotly de manera eficiente
    pca_df = pd.DataFrame(pca_cols, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df['Madurez'] = df['Etapa_Madurez'] if 'Etapa_Madurez' in df.columns else df['Madurez']
    pca_df['Lugar'] = dataset_complete['Lugar']
    
    # Limitar número de puntos para mejor rendimiento
    if len(pca_df) > max_points:
        # Asegurar que el lugar seleccionado permanezca en el muestreo
        lugar_df = None
        if lugar_seleccionado:
            lugar_df = pca_df[pca_df['Lugar'] == lugar_seleccionado]
        
        # Muestrear el resto
        resto_df = pca_df[pca_df['Lugar'] != lugar_seleccionado]
        # Calcular cuántos puntos muestrear
        muestra_size = max_points - (0 if lugar_df is None else len(lugar_df))
        if muestra_size > 0 and len(resto_df) > muestra_size:
            # Estratificar por Madurez para mantener proporciones
            resto_muestra = resto_df.groupby('Madurez', group_keys=False).apply(
                lambda x: x.sample(min(int(muestra_size * len(x) / len(resto_df)) + 1, len(x)), random_state=42)
            )
            # Combinar con el lugar seleccionado
            if lugar_df is not None and not lugar_df.empty:
                pca_df = pd.concat([lugar_df, resto_muestra])
            else:
                pca_df = resto_muestra

    # Crear el gráfico 3D optimizado
    fig = px.scatter_3d(
        pca_df, 
        x='PCA1', y='PCA2', z='PCA3',
        color='Madurez',
        labels={'PCA1': 'Componente PC1', 
                'PCA2': 'Componente PC2', 
                'PCA3': 'Componente PC3'},
        hover_data=['Lugar'],
        category_orders={'Madurez': ['Optimización', 'Definición', 'En desarrollo', 'Inicial']},
        color_discrete_map=color_map
    )
    
    # CAMBIO: Destacar municipio seleccionado con naranja y mayor tamaño
    if lugar_seleccionado:
        lugar_df = pca_df[pca_df['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            # Agregar solo los puntos específicos del lugar seleccionado
            fig.add_trace(
                go.Scatter3d(
                    x=lugar_df['PCA1'],
                    y=lugar_df['PCA2'],
                    z=lugar_df['PCA3'],
                    mode='markers',
                    marker=dict(
                        size=15,  # Tamaño más grande
                        color='#FFA500',  # Color naranja
                        opacity=1,
                        symbol='circle',
                        line=dict(
                            width=1,
                            color='white'  # Borde blanco para destacar
                        )
                    ),
                    name=f"Lugar: {lugar_seleccionado}",
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    text=[lugar_seleccionado]
                )
            )

    # Actualizar estilo de los marcadores
    fig.update_traces(
        marker=dict(
            size=6,
            opacity=0.7,
            line=dict(width=0.2, color='gray')
        ),
        selector=dict(type='scatter3d')
    )

    # Actualizar layout
    fig.update_layout(
        title="Municipios por grado de madurez multidimensional",
        title_x=0.05,
        showlegend=True,
        legend=dict(
            title=dict(text='Madurez'),
            itemsizing='constant',
            font=dict(color='white'),
        ),
        scene=dict(
            xaxis_title="Componente PC1",
            yaxis_title="Componente PC2",
            zaxis_title="Componente PC3",
            xaxis=dict(
                titlefont=dict(color='white'),
                gridcolor='white',
                zerolinecolor='white'
            ),
            yaxis=dict(
                titlefont=dict(color='white'),
                gridcolor='white',
                zerolinecolor='white'
            ),
            zaxis=dict(
                titlefont=dict(color='white'),
                gridcolor='white',
                zerolinecolor='white'
            ),
            bgcolor='rgb(0, 0, 0)',
        ),
        font=dict(color='white'),
        paper_bgcolor='rgb(0, 0, 0)',
        plot_bgcolor='rgb(0, 0, 0)',
    )

    return fig


# 2. Optimizar la generación de gráficos 2D
@st.cache_data
def generar_grafico_2d(df, df_normalizado, dataset_complete, lugar_seleccionado=None, max_points=2000, x_col='PCA1', y_col='PCA2', title=None):
    """
    Versión optimizada y generalizada para gráficos 2D con municipio seleccionado en naranja y mayor tamaño
    """
    # Mapa de colores predefinido
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }
    
    # Preparar datos del PCA de manera eficiente
    if isinstance(df_normalizado, pd.DataFrame):
        # Si ya es DataFrame, extraer columnas (al menos necesitamos 3)
        if df_normalizado.shape[1] >= 4:
            pca_cols = df_normalizado.iloc[:, 1:4].values
        else:
            # Fallback si no hay suficientes columnas
            return None
    else:
        # Si es numpy array
        df_pca2 = df_normalizado
        if df_pca2.shape[1] >= 4:
            pca_cols = df_pca2[:, 1:4]
        else:
            # Fallback si no hay suficientes columnas
            return None

    # Crear DataFrame para Plotly
    pca_df = pd.DataFrame(pca_cols, columns=['PCA1', 'PCA2', 'PCA3'])
    
    # Asegurar que 'Madurez' está presente, con fallback a 'Etapa_Madurez'
    if 'Madurez' in df.columns:
        pca_df['Madurez'] = df['Madurez']
    elif 'Etapa_Madurez' in df.columns:
        pca_df['Madurez'] = df['Etapa_Madurez']
    else:
        # Si ninguno está disponible, usar un valor predeterminado
        pca_df['Madurez'] = 'Desconocido'
    
    # Añadir columna de Lugar
    if 'Lugar' in dataset_complete.columns:
        pca_df['Lugar'] = dataset_complete['Lugar']
    
    # Limitar número de puntos para mejor rendimiento
    if len(pca_df) > max_points:
        # Asegurar que el lugar seleccionado permanezca
        lugar_df = None
        if lugar_seleccionado:
            lugar_df = pca_df[pca_df['Lugar'] == lugar_seleccionado]
        
        # Muestrear el resto
        resto_df = pca_df[pca_df['Lugar'] != lugar_seleccionado]
        # Cuántos puntos muestrear
        muestra_size = max_points - (0 if lugar_df is None else len(lugar_df))
        if muestra_size > 0 and len(resto_df) > muestra_size:
            # Estratificar por Madurez
            resto_muestra = resto_df.groupby('Madurez', group_keys=False).apply(
                lambda x: x.sample(min(int(muestra_size * len(x) / len(resto_df)) + 1, len(x)), random_state=42)
            )
            # Combinar con lugar seleccionado
            if lugar_df is not None and not lugar_df.empty:
                pca_df = pd.concat([lugar_df, resto_muestra])
            else:
                pca_df = resto_muestra

    # Título por defecto si no se proporciona
    if title is None:
        title = f"{x_col} vs. {y_col} (2D)"

    # Crear gráfico 2D optimizado
    fig = px.scatter(
        pca_df, 
        x=x_col, y=y_col,
        color='Madurez',
        labels={x_col: f'Componente {x_col}', 
                y_col: f'Componente {y_col}'},
        hover_data=['Lugar'],
        category_orders={'Madurez': list(color_map.keys())},
        color_discrete_map=color_map
    )
    
    # CAMBIO: Resaltar lugar seleccionado con naranja y mayor tamaño
    if lugar_seleccionado:
        lugar_df = pca_df[pca_df['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=lugar_df[x_col],
                    y=lugar_df[y_col],
                    mode='markers',
                    marker=dict(
                        size=15,  # Tamaño más grande
                        color='#FFA500',  # Color naranja
                        symbol='circle',
                        line=dict(
                            width=1,
                            color='white'  # Borde blanco para mayor visibilidad
                        )
                    ),
                    name=f"Lugar: {lugar_seleccionado}",
                    hovertemplate='<b>Lugar: %{text}</b><extra></extra>',
                    text=[lugar_seleccionado]
                )
            )

    # Estilo de marcadores
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(
                width=0.5,
                color='gray'
            )
        ),
        selector=dict(mode='markers')
    )

    # Layout optimizado
    fig.update_layout(
        title=title,
        title_x=0.5,
        showlegend=True,
        legend=dict(
            title=dict(text='Madurez'),
            itemsizing='constant'
        ),
        paper_bgcolor='rgb(0, 0, 0)',
        plot_bgcolor='rgb(0, 0, 0)',
        font=dict(color='white')
    )

    return fig

# 3. Función optimizada para generar los tres gráficos 2D a la vez
@st.cache_data
def generar_todos_graficos_2d(df, df_normalizado, dataset_complete, lugar_seleccionado=None, max_points=2000):
    """
    Genera los tres gráficos 2D en una sola pasada para mejorar rendimiento
    """
    graphs = {}
    
    # Generar cada gráfico con componentes específicos
    graphs['grafico2d1'] = generar_grafico_2d(
        df, df_normalizado, dataset_complete, lugar_seleccionado, 
        max_points, 'PCA1', 'PCA2', "PC1 vs. PC2 (2D)"
    )
    
    graphs['grafico2d2'] = generar_grafico_2d(
        df, df_normalizado, dataset_complete, lugar_seleccionado, 
        max_points, 'PCA1', 'PCA3', "PC1 vs. PC3 (2D)"
    )
    
    graphs['grafico2d3'] = generar_grafico_2d(
        df, df_normalizado, dataset_complete, lugar_seleccionado, 
        max_points, 'PCA2', 'PCA3', "PC2 vs. PC3 (2D)"
    )
    
    return graphs





# Optimización 5: Gráficos de análisis de clústers y correlaciones

# 1. Optimizar boxplot por clúster
@st.cache_data
def boxplot_por_cluster(df, variable, max_points=2000):
    """
    Versión optimizada del boxplot por clúster
    """
    # Mapa de colores predefinido
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }

    # Usar solo las columnas necesarias
    df_subset = df[['Madurez', 'Lugar', variable]].copy()
    
    # Calcular estadísticas eficientemente mediante groupby
    stats = df_subset.groupby('Madurez')[variable].agg(['mean', 'median', 'std']).reset_index()
    stats.columns = ['Madurez', 'mean_' + variable, 'median_' + variable, 'std_' + variable]
    
    # Unir estadísticas al DataFrame principal
    df_with_stats = pd.merge(df_subset, stats, on='Madurez', how='left')
    
    # Limitar puntos si hay demasiados
    if len(df_with_stats) > max_points:
        df_with_stats = df_with_stats.groupby('Madurez', group_keys=False).apply(
            lambda x: x.sample(min(int(max_points * len(x) / len(df_with_stats)), len(x)), random_state=42)
        )
    
    # Crear el boxplot optimizado
    fig = px.box(
        df_with_stats,
        y=variable,
        points='all',  # Mostrar todos los puntos
        title=f'Diagrama de caja de la variable\n"{variable}"',
        labels={variable: variable},
        template='plotly_dark',
        color='Madurez',
        color_discrete_map=color_map,
        hover_data={
            'Madurez': True, 
            'Lugar': True,
            'mean_' + variable: ':.2f',
            'median_' + variable: ':.2f',
            'std_' + variable: ':.2f',
        }
    )

    # Mejorar estilo de marcadores
    fig.update_traces(
        marker=dict(
            opacity=0.6,
            size=5,  # Puntos más pequeños para mejor rendimiento
            line=dict(color='rgba(255, 165, 0, 0.5)', width=0.5)
        ),
        jitter=0.5,  # Añadir dispersión para evitar superposición
        boxpoints='outliers'  # Solo mostrar outliers como puntos para mejorar rendimiento
    )
    
    # Optimizar layout
    fig.update_layout(
        template='plotly_dark',
        boxmode='group',  # Agrupar cajas por categoría
        boxgap=0.5,  # Espacio entre grupos
        boxgroupgap=0.2  # Espacio entre cajas del mismo grupo
    )
    
    return fig

# 2. Optimizar histograma por clúster
@st.cache_data
def plot_histogram_clusters(df, numeric_column, nbins=30):
    """
    Versión optimizada del histograma por clúster
    """
    # Mapa de colores
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }
    
    # Usar solo columnas necesarias
    df_subset = df[['Madurez', numeric_column]].copy()
    
    # Crear el histograma optimizado
    fig = px.histogram(
        df_subset, 
        x=numeric_column, 
        color='Madurez',
        color_discrete_map=color_map,
        opacity=0.6,
        nbins=nbins,
        title=f'Histograma de la variable "{numeric_column}"'
    )
    
    # Actualizar ejes
    fig.update_xaxes(title_text="Rangos de valor")
    fig.update_yaxes(title_text="Frecuencia absoluta")
    
    # Calcular estadísticas de manera más eficiente usando groupby
    stats_df = df_subset.groupby('Madurez')[numeric_column].agg(['mean', 'median', 'std']).reset_index()
    
    # Calcular la moda por grupo separadamente (es más costosa)
    mode_by_group = {}
    for level in df_subset['Madurez'].unique():
        subset = df_subset[df_subset['Madurez'] == level]
        try:
            mode_by_group[level] = subset[numeric_column].mode().iloc[0]
        except (IndexError, KeyError):
            mode_by_group[level] = None
    
    # Crear anotaciones de manera más eficiente
    annotations = []
    positions = [
        {'x': 1.15, 'y': 0.95},
        {'x': 1.15, 'y': 0.75},
        {'x': 1.15, 'y': 0.55},
        {'x': 1.15, 'y': 0.35}
    ]
    
    # Iterar sobre los grupos para crear anotaciones
    for i, (level, stats) in enumerate(stats_df.iterrows()):
        if i < len(positions):
            # Obtener estadísticas del grupo
            level_name = stats['Madurez']
            mean = stats['mean']
            median = stats['median']
            std = stats['std']
            mode = mode_by_group.get(level_name, "N/A")
            
            # Crear anotación
            annotations.append(dict(
                x=positions[i]['x'],
                y=positions[i]['y'],
                xref='paper',
                yref='paper',
                text=f'<b>{level_name}</b><br>Media: {mean:.2f}<br>Mediana: {median:.2f}<br>Desv. estándar: {std:.2f}',
                showarrow=False,
                font=dict(size=10, color='black'),
                bgcolor=color_map[level_name],
                borderpad=4,
                opacity=0.8,
                align="left",
                width=150
            ))
    
    # Añadir anotaciones al gráfico
    for annotation in annotations:
        fig.add_annotation(annotation)
    
    # Actualizar layout
    fig.update_layout(
        template='plotly_dark',
        showlegend=False,
        margin=dict(r=250),
        height=400
    )
    
    return fig

# 3. Optimizar scatter plot con regresión
@st.cache_data
# Modificación del scatter plot con regresión (continuación)
def generate_scatter_with_annotations(df, x_variable, y_variable, categorical_variable, lugar_seleccionado=None, max_points=2000):
    """
    Versión modificada del scatter plot para destacar municipio seleccionado en naranja y tamaño más grande
    """
    # Mapa de colores
    color_map = {
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7',
        'Definición': '#CC6CE7',
        'Optimización': '#51C622',
    }

    # Usar solo las columnas necesarias y limpiar NaN
    df_clean = df[['Lugar', x_variable, y_variable, categorical_variable]].dropna(subset=[x_variable, y_variable])
    
    # Limitar número de puntos para mejor rendimiento
    if len(df_clean) > max_points:
        # Primero preservar el municipio seleccionado
        df_seleccionado = df_clean[df_clean['Lugar'] == lugar_seleccionado] if lugar_seleccionado else pd.DataFrame()
        df_resto = df_clean[df_clean['Lugar'] != lugar_seleccionado] if lugar_seleccionado else df_clean
        
        # Muestrear estratificado por categoría
        muestra_size = max_points - len(df_seleccionado)
        if muestra_size > 0 and len(df_resto) > muestra_size:
            df_resto = df_resto.groupby(categorical_variable, group_keys=False).apply(
                lambda x: x.sample(min(int(muestra_size * len(x) / len(df_resto)), len(x)), random_state=42)
            )
        
        # Combinar municipio seleccionado con el resto muestreado
        if not df_seleccionado.empty:
            df_clean = pd.concat([df_seleccionado, df_resto])
        else:
            df_clean = df_resto

    # Crear el scatter plot básico
    fig = px.scatter(
        df_clean,
        x=x_variable,
        y=y_variable,
        hover_data={'Lugar': True, categorical_variable: True},
        color=categorical_variable,
        color_discrete_map=color_map,
        opacity=0.7
    )

    # Calcular regresión de manera eficiente
    X = df_clean[[x_variable]].values
    y = df_clean[y_variable].values
    model = LinearRegression()
    model.fit(X, y)

    intercept = model.intercept_
    slope = model.coef_[0]
    r_squared = model.score(X, y)
    
    # Calcular R² ajustado
    n = len(df_clean)
    p = 1
    r_squared_adj = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))

    # Ecuación de regresión
    regression_equation = f"y = {slope:.4f}x + {intercept:.4f}"

    # Añadir línea de regresión
    x_min, x_max = df_clean[x_variable].min(), df_clean[x_variable].max()
    x_range = np.linspace(x_min, x_max, 50)
    y_predicted = slope * x_range + intercept
    
    fig.add_scatter(
        x=x_range,
        y=y_predicted,
        mode='lines',
        name='Regresión',
        line=dict(color='orange', dash='dash', width=1.5)
    )

    # CAMBIO: Resaltar el municipio seleccionado con color naranja y tamaño más grande
    if lugar_seleccionado:
        df_lugar = df_clean[df_clean['Lugar'] == lugar_seleccionado]
        if not df_lugar.empty:
            fig.add_scatter(
                x=df_lugar[x_variable],
                y=df_lugar[y_variable],
                mode='markers',
                marker=dict(
                    size=15,  # Tamaño más grande
                    color='#FFA500',  # Color naranja
                    line=dict(
                        width=1,
                        color='white'  # Borde blanco para mayor visibilidad
                    )
                ),
                name=f'Municipio: {lugar_seleccionado}',
                hovertemplate=f'<b>Municipio: {lugar_seleccionado}</b><br>' +
                              f'<b>{x_variable}</b>: %{{x:.2f}}<br>' +
                              f'<b>{y_variable}</b>: %{{y:.2f}}<extra></extra>'
            )

    # Optimizar layout
    fig.update_layout(
        plot_bgcolor='rgb(30,30,30)',
        paper_bgcolor='rgb(30,30,30)',
        font_color='white',
        title=dict(
            text=f"Scatter Plot: '{x_variable}' vs '{y_variable}'",
            font=dict(color='white')
        ),
        xaxis=dict(
            title=f"Variable: {x_variable}",
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            showgrid=True,
            gridcolor='rgba(100,100,100,0.2)'
        ),
        yaxis=dict(
            title=f"Variable: {y_variable}",
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            showgrid=True,
            gridcolor='rgba(100,100,100,0.2)'
        ),
        # Añadir anotaciones de manera más eficiente
        annotations=[
            dict(
                xref='paper', yref='paper',
                x=0.95, y=1.05,
                text=f'R² Ajustada: {r_squared_adj:.4f}',
                showarrow=False,
                font=dict(color='orange', size=12)
            ),
            dict(
                xref='paper', yref='paper',
                x=0.05, y=1.05,
                text=f'Regresión: {regression_equation}',
                showarrow=False,
                font=dict(color='orange', size=12)
            )
        ],
        legend=dict(
            title=categorical_variable,
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    # Optimizar hover template para el resto de puntos
    fig.update_traces(
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                    f'<b>{x_variable}</b>: %{{x:.2f}}<br>' +
                    f'<b>{y_variable}</b>: %{{y:.2f}}<extra></extra>',
        selector=dict(mode='markers')
    )

    return fig

# 4. Optimizar mapa con clústers
@st.cache_data
# Modificación del mapa completo
def generar_mapa_con_lugar(df, lugar=None, max_points=3000):
    """
    Versión optimizada del mapa con clústers que destaca el municipio seleccionado en naranja y tamaño más grande
    """
    # Mapa de colores
    color_map = {
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7',
        'Definición': '#CC6CE7',
        'Optimización': '#51C622',
    }

    # Usar solo las columnas necesarias
    cols_needed = ['Lugar', 'Madurez', 'Latitud', 'Longitud']
    if not all(col in df.columns for col in cols_needed):
        return None  # Salir si faltan columnas
        
    plot_data = df[cols_needed].copy()
    
    # Asegurarse de que las coordenadas son numéricas
    plot_data['Latitud'] = pd.to_numeric(plot_data['Latitud'], errors='coerce')
    plot_data['Longitud'] = pd.to_numeric(plot_data['Longitud'], errors='coerce')
    
    # Eliminar filas con coordenadas faltantes
    plot_data = plot_data.dropna(subset=['Latitud', 'Longitud'])
    
    # Convertir Madurez a categoría
    if 'Madurez' in plot_data.columns:
        plot_data['Madurez'] = plot_data['Madurez'].astype('category')
    
    # Reducir el número de puntos para mejor rendimiento, preservando el lugar seleccionado
    if len(plot_data) > max_points:
        lugar_df = None
        if lugar:
            lugar_df = plot_data[plot_data['Lugar'] == lugar]
            
        resto_df = plot_data[plot_data['Lugar'] != lugar]
        if len(resto_df) > max_points:
            # Muestrear estratificado por Madurez
            muestra_size = max_points - (0 if lugar_df is None else len(lugar_df))
            resto_muestra = resto_df.groupby('Madurez', group_keys=False).apply(
                lambda x: x.sample(min(int(muestra_size * len(x) / len(resto_df)), len(x)), random_state=42)
            )
            
            if lugar_df is not None and not lugar_df.empty:
                plot_data = pd.concat([lugar_df, resto_muestra])
            else:
                plot_data = resto_muestra

    # Crear el mapa con Plotly
    fig = px.scatter_mapbox(
        plot_data,
        lat="Latitud",
        lon="Longitud",
        color="Madurez",
        opacity=0.7,
        hover_data=["Madurez", "Lugar"],
        zoom=4,
        center={"lat": 23.6345, "lon": -102.5528},
        title="Mapa de Clústers por Madurez Digital en México",
        color_discrete_map=color_map,
        size_max=8  # Tamaño máximo de los marcadores
    )

    # CAMBIO: Resaltar lugar seleccionado con naranja y tamaño mucho más grande
    if lugar:
        lugar_df = plot_data[plot_data['Lugar'] == lugar]
        if not lugar_df.empty:
            # Añadir punto destacado
            fig.add_trace(
                go.Scattermapbox(
                    lat=lugar_df["Latitud"],
                    lon=lugar_df["Longitud"],
                    mode='markers',
                    marker=dict(
                        size=20,  # Tamaño mucho más grande
                        color='#FFA500',  # Color naranja
                        opacity=1,
                        symbol='circle',  # Símbolo circular
                        sizemode='diameter',  # Modo de tamaño
                        sizeref=1  # Referencia de tamaño
                    ),
                    name=f"Seleccionado: {lugar}",
                    text=lugar_df["Lugar"],
                    hoverinfo='text',
                    hovertemplate='<b>Lugar: %{text}</b><br>Seleccionado<extra></extra>'
                )
            )

    # Configurar estilo y diseño
    fig.update_layout(
        mapbox_style="carto-darkmatter",
        height=600,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        legend=dict(
            title="Nivel de Madurez",
            itemsizing="constant",
            traceorder="normal",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig



# 5. Optimizar recuento de clústers (continuación)
@st.cache_data
def recuento(df):
    """
    Versión optimizada del recuento de clústers
    """
    # Mapa de colores predefinido
    color_map = {
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7',
        'Definición': '#CC6CE7',
        'Optimización': '#51C622',
    }
    
    # Verificar la presencia de la columna 'Madurez'
    if 'Madurez' not in df.columns:
        return None
    
    # Contar registros por nivel de madurez de manera eficiente
    total_municipios = len(df)
    counts = df['Madurez'].value_counts().reset_index()
    counts.columns = ['Madurez', 'Cantidad']
    
    # Calcular frecuencia relativa
    counts['Frecuencia relativa'] = counts['Cantidad'] / total_municipios
    
    # Orden personalizado para las categorías
    category_order = ['Inicial', 'En desarrollo', 'Definición', 'Optimización']
    counts['Madurez'] = pd.Categorical(counts['Madurez'], categories=category_order, ordered=True)
    counts = counts.sort_values('Madurez')
    
    # Crear gráfico de barras optimizado
    fig = px.bar(
        counts, 
        x='Madurez', 
        y='Frecuencia relativa', 
        title="Frecuencia relativa por nivel de madurez",
        labels={
            'Frecuencia relativa': 'Frecuencia relativa', 
            'Madurez': 'Nivel de madurez'
        },
        color='Madurez', 
        color_discrete_map=color_map,
        height=280
    )
    
    # Añadir etiquetas de valores
    fig.update_traces(
        texttemplate='%{y:.1%}',
        textposition='outside',
        marker_line_color='white',
        marker_line_width=1,
        opacity=0.85
    )
    
    # Optimizar diseño
    fig.update_layout(
        xaxis=dict(
            title=dict(text="Nivel de madurez", font=dict(color="white")),
            tickfont=dict(color="white")
        ),
        yaxis=dict(
            title=dict(text="Frecuencia relativa", font=dict(color="white")),
            tickformat='.1%',  # Formato de porcentaje
            tickfont=dict(color="white"),
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=50, b=30)
    )
    
    return fig





# Optimización 6: Mejorar la interfaz de usuario y carga inicial

# 1. Optimizar la carga inicial de la aplicación
def optimizar_carga_inicial():
    """
    Muestra un spinner mientras se cargan los datos principales
    """
    with st.spinner('Cargando datos municipales...'):
        # Establecer una sola conexión a MongoDB
        cliente_mongo = get_mongodb_connection()
        
        # Mostrar progreso de carga
        progress_bar = st.progress(0)
        
        # Cargar datos geográficos (10%)
        geojson = obtener_datos_geograficos()
        progress_bar.progress(10)
        
        # Cargar datos principales (40%)
        datos = bajando_procesando_datos()
        progress_bar.progress(40)
        
        # Procesar variables numéricas y categóricas (50%)
        variable_list_numerica, variable_list_categoricala, variable_list_municipio = procesar_listas_variables(datos)
        progress_bar.progress(50)
        
        # Cargar dataset completo (70%)
        dataset_complete = bajando_procesando_datos_completos()
        progress_bar.progress(70)
        
        # Cargar datos de normalización (90%)
        df = bajando_procesando_X_entrenamiento()
        df_normalizado = bajando_procesando_df_normalizado()
        progress_bar.progress(90)
        
        # Preparar datos para visualización (100%)
        dataset_complete_geometry = preparar_datos_para_visualizacion(datos, geojson)
        progress_bar.progress(100)
        
        # Eliminar la barra de progreso
        progress_bar.empty()
        
        return datos, dataset_complete, df, df_normalizado, geojson, dataset_complete_geometry, variable_list_numerica, variable_list_categoricala, variable_list_municipio

# 2. Optimizar procesamiento de listas de variables
@st.cache_data
def procesar_listas_variables(input_datos):
    """
    Procesa y filtra listas de variables de manera eficiente
    """
    # Procesar variables numéricas
    variable_list_numerica = list(input_datos.select_dtypes(include=['int64', 'float64']).columns)
    
    # Procesar variables categóricas
    variable_list_categoricala = list(input_datos.select_dtypes(include=['object', 'category']).columns)
    
    # Lista de municipios
    variable_list_municipio = list(input_datos['Lugar'].unique())
    
    # Columnas para excluir
    columns_to_exclude_numeric = [
        'Cluster2', 'Unnamed: 0', 'Unnamed: 0.2', 'cve_edo', 'cve_municipio', 
        'cvegeo', 'Estratos ICM', 'Estrato IDDM', 'Municipio', 'df1_ENTIDAD', 
        'df1_KEY MUNICIPALITY', 'df2_Clave Estado', 'df2_Clave Municipio', 
        'df3_Clave Estado', 'df3_Clave Municipio', 'df4_Clave Estado', 
        'df4_Clave Municipio'
    ]
    
    columns_to_exclude_categorical = [
        '_id', 'Lugar', 'Estado2', 'df2_Región', 'df3_Región', 
        'df3_Tipo de población', 'df4_Región', 'Municipio'
    ]
    
    # Filtrar variables
    variable_list_numeric = [col for col in variable_list_numerica if col not in columns_to_exclude_numeric]
    variable_list_categorical = [col for col in variable_list_categoricala if col not in columns_to_exclude_categorical]
    
    return variable_list_numeric, variable_list_categorical, variable_list_municipio

# 3. Optimizar títulos dinámicos
@st.cache_data
def titulo_dinamico(variable, tipo="variable"):
    """
    Genera títulos dinámicos optimizados
    """
    if tipo == "variable":
        return f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">La variable mostrada es: "{variable}".</span>'
    elif tipo == "municipio":
        return f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">Municipio de "{variable}".</span>'
    elif tipo == "madurez":
        return f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">Análisis de Madurez Digital de "{variable}".</span>'
    else:
        return f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">{variable}</span>'

# 4. Optimizar carga de gráficos para cada pestaña
def cargar_graficos_tab1(variable_seleccionada_municipio, datos, dataset_complete_geometry):
    """
    Carga solo los gráficos necesarios para la pestaña 1
    """
    # La pestaña 1 es principalmente texto, no hay gráficos pesados para cargar
    pass

def cargar_graficos_tab2(variable_seleccionada_municipio, variable_seleccionada_numerica, datos, dataset_complete_geometry):
    """
    Carga solo los gráficos necesarios para la pestaña 2
    """
    with st.spinner('Generando visualizaciones del municipio...'):
        progress_bar = st.progress(0)
        
        # Cargar mapa choropleth (25%)
        fig_municipio = crear_mapa_choropleth2(dataset_complete_geometry, lugar=variable_seleccionada_municipio)
        progress_bar.progress(25)
        
        # Cargar gráfico de barras (50%)
        fig_ranking = plot_bar_chart(datos, 'Lugar', 'Índice_Compuesto', variable_seleccionada_municipio)
        progress_bar.progress(50)
        
        # Cargar cuadro resumen (60%)
        cuadro_resumen = crear_display(datos, variable_seleccionada_municipio)
        progress_bar.progress(60)
        
        # Cargar histograma (80%)
        fig_hist = plot_histogram(datos, variable_seleccionada_numerica, 'Madurez')
        progress_bar.progress(80)
        
        # Cargar histograma de índice (90%)
        fig_hist_index = plot_histogram_with_density(datos, 'Índice_Compuesto', variable_seleccionada_municipio)
        progress_bar.progress(90)
        
        # Cargar boxplot (100%)
        fig_boxplot = generate_boxplot_with_annotations(datos, variable_seleccionada_numerica, variable_seleccionada_municipio)
        progress_bar.progress(100)
        
        # Eliminar la barra de progreso
        progress_bar.empty()
        
        return fig_municipio, fig_ranking, cuadro_resumen, fig_hist, fig_hist_index, fig_boxplot

def cargar_graficos_tab3(variable_seleccionada_municipio, datos, df_normalizado, dataset_complete):
    """
    Carga solo los gráficos necesarios para la pestaña 3
    """
    with st.spinner('Generando visualizaciones 3D y PCA...'):
        progress_bar = st.progress(0)
        
        # Cargar gráfico 3D (50%)
        grafico3d = generar_grafico_3d_con_lugar(datos, df_normalizado, dataset_complete, variable_seleccionada_municipio)
        progress_bar.progress(50)
        
        # Cargar gráficos 2D (100%)
        graficos_2d = generar_todos_graficos_2d(datos, df_normalizado, dataset_complete, variable_seleccionada_municipio)
        progress_bar.progress(100)
        
        # Eliminar la barra de progreso
        progress_bar.empty()
        
        return grafico3d, graficos_2d

def cargar_graficos_tab4(variable_seleccionada_numerica, datos):
    """
    Carga solo los gráficos necesarios para la pestaña 4
    """
    with st.spinner('Generando análisis estadísticos por grupo...'):
        progress_bar = st.progress(0)
        
        # Cargar recuento de clústers (30%)
        recuento_clusters = recuento(datos)
        progress_bar.progress(30)
        
        # Cargar boxplot por clústers (60%)
        boxplots_clusters = boxplot_por_cluster(datos, variable_seleccionada_numerica)
        progress_bar.progress(60)
        
        # Cargar histograma por clústers (100%)
        histograma_por_clusters = plot_histogram_clusters(datos, variable_seleccionada_numerica)
        progress_bar.progress(100)
        
        # Eliminar la barra de progreso
        progress_bar.empty()
        
        return recuento_clusters, boxplots_clusters, histograma_por_clusters

def cargar_graficos_tab5(variable_seleccionada_numerica, variable_seleccionada_paracorrelacion, variable_seleccionada_categorica, datos):
    """
    Carga solo los gráficos necesarios para la pestaña 5
    """
    with st.spinner('Generando análisis correlacional...'):
        # Cargar scatter plot con regresión
        fig_scatter = generate_scatter_with_annotations(
            datos, 
            variable_seleccionada_numerica, 
            variable_seleccionada_paracorrelacion, 
            variable_seleccionada_categorica
        )
        return fig_scatter

def cargar_graficos_tab6(variable_seleccionada_municipio, datos):
    """
    Carga solo los gráficos necesarios para la pestaña 6
    """
    with st.spinner('Generando mapa geográfico...'):
        # Cargar mapa con lugar
        fig_map_final = generar_mapa_con_lugar(datos, lugar=variable_seleccionada_municipio)
        return fig_map_final

# 5. Mejora general de la interfaz de usuario
def optimizar_interfaz():
    """
    Aplicar mejoras generales a la interfaz de usuario
    """
    # Reducir el CSS a lo esencial
    st.markdown("""
    <style>
    [data-testid="block-container"] {
        padding-left: 2rem;
        padding-right: 2rem;
        padding-top: -10rem;
        padding-bottom: 0rem;
        margin-bottom: -7rem;
    }
    [data-testid="stMetric"] {
        background-color: #393939;
        text-align: center;
        padding: 10px 0;
    }
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Configuración de la página optimizada
    
    # Habilitar tema oscuro para Altair
    alt.themes.enable("dark")



# Optimización 7: Implementar carga lazy y mejoras en el manejo de memoria

# 1. Implementar SessionState para Streamlit
class SessionState:
    """
    Clase para mantener estado entre recargas de Streamlit
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_session_state(**kwargs):
    """
    Obtiene o crea un SessionState
    """
    # Verificar si ya existe un estado en la sesión
    session_state = st.session_state
    
    # Actualizar con los valores proporcionados
    for key, val in kwargs.items():
        if key not in session_state:
            session_state[key] = val
    
    return session_state

# 2. Implementar carga lazy para datos pesados
def lazy_load(key, loader_func, *args, **kwargs):
    """
    Carga un recurso solo cuando se necesita y lo almacena en session_state
    
    Args:
        key: Clave para almacenar en session_state
        loader_func: Función para cargar el recurso
        args, kwargs: Argumentos para loader_func
    
    Returns:
        El recurso cargado
    """
    session_state = get_session_state()
    
    # Verificar si el recurso ya está cargado
    if key not in session_state:
        # Cargar el recurso y almacenarlo
        session_state[key] = loader_func(*args, **kwargs)
    
    return session_state[key]

# 3. Implementar limpieza de memoria para recursos no utilizados
def cleanup_unused_resources(active_tab):
    """
    Libera recursos que no se están utilizando en la pestaña actual
    
    Args:
        active_tab: Índice de la pestaña activa
    """
    session_state = get_session_state()
    
    # Mapeo de recursos por pestaña
    tab_resources = {
        0: [],  # Tab 1: No tiene recursos pesados
        1: ['fig_municipio', 'fig_ranking', 'cuadro_resumen', 'fig_hist', 'fig_hist_index', 'fig_boxplot'],
        2: ['grafico3d', 'grafico2d1', 'grafico2d2', 'grafico2d3'],
        3: ['recuento_clusters', 'boxplots_clusters', 'histograma_por_clusters'],
        4: ['fig_scatter'],
        5: ['fig_map_final']
    }
    
    # Recursos compartidos que no deben limpiarse
    shared_resources = ['datos', 'dataset_complete', 'df', 'df_normalizado', 'geojson', 'dataset_complete_geometry']
    
    # Obtener recursos activos para la pestaña actual
    active_resources = tab_resources.get(active_tab, []) + shared_resources
    
    # Obtener todos los recursos de todas las pestañas
    all_tab_resources = []
    for resources in tab_resources.values():
        all_tab_resources.extend(resources)
    
    # Liberar recursos no utilizados
    for key in list(session_state.keys()):
        if key in all_tab_resources and key not in active_resources:
            # Solo limpiar recursos gráficos, no datos base
            if key not in shared_resources:
                del session_state[key]

# 4. Implementar gestión de memoria para DataFrame grandes
def optimize_dataframe(df):
    """
    Optimiza un DataFrame para reducir uso de memoria
    
    Args:
        df: DataFrame a optimizar
    
    Returns:
        DataFrame optimizado
    """
    # Si no es un DataFrame, devolver tal cual
    if not isinstance(df, pd.DataFrame):
        return df
    
    # Hacer una copia para no modificar el original
    result = df.copy()
    
    # Optimizar tipos numéricos
    for col in result.select_dtypes(include=['int']).columns:
        # Determinar el rango de valores
        col_min, col_max = result[col].min(), result[col].max()
        
        # Elegir el tipo más pequeño que pueda contener los valores
        if col_min >= 0:
            if col_max < 2**8:
                result[col] = result[col].astype(np.uint8)
            elif col_max < 2**16:
                result[col] = result[col].astype(np.uint16)
            elif col_max < 2**32:
                result[col] = result[col].astype(np.uint32)
        else:
            if col_min > -2**7 and col_max < 2**7:
                result[col] = result[col].astype(np.int8)
            elif col_min > -2**15 and col_max < 2**15:
                result[col] = result[col].astype(np.int16)
            elif col_min > -2**31 and col_max < 2**31:
                result[col] = result[col].astype(np.int32)
    
    # Optimizar tipos float
    for col in result.select_dtypes(include=['float']).columns:
        # Convertir a float32 si es posible
        result[col] = result[col].astype(np.float32)
    
    # Optimizar tipos categóricos
    for col in result.select_dtypes(include=['object']).columns:
        # Verificar si la columna tiene pocos valores únicos
        if result[col].nunique() < len(result) * 0.5:  # Menos del 50% de valores únicos
            result[col] = result[col].astype('category')
    
    return result

# 5. Implementar carga por bloques para datos muy grandes
def load_in_chunks(collection, query=None, projection=None, chunk_size=1000):
    """
    Carga datos de MongoDB en bloques para reducir uso de memoria
    
    Args:
        collection: Colección de MongoDB
        query: Consulta para filtrar documentos
        projection: Proyección para seleccionar campos
        chunk_size: Tamaño de cada bloque
    
    Returns:
        DataFrame con todos los datos
    """
    # Preparar consulta y proyección
    query = {} if query is None else query
    projection = None if projection is None else projection
    
    # Obtener cursor
    cursor = collection.find(query, projection)
    
    # Inicializar lista para almacenar chunks
    chunks = []
    
    # Cargar documentos por bloques
    while True:
        # Obtener siguiente bloque
        chunk = list(cursor.limit(chunk_size).skip(len(chunks) * chunk_size))
        
        # Si no hay más documentos, salir del bucle
        if not chunk:
            break
        
        # Convertir ObjectId a str en cada documento
        chunk = list(map(convert_objectid_to_str, chunk))
        
        # Añadir bloque a la lista
        chunks.append(pd.DataFrame(chunk))
    
    # Concatenar todos los bloques
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    else:
        return pd.DataFrame()

# 6. Implementar un sistema de cache con límite de tamaño
class LRUCache:
    """
    Cache LRU (Least Recently Used) con límite de tamaño
    """
    def __init__(self, max_size=10):
        self.cache = {}
        self.max_size = max_size
        self.order = []
    
    def get(self, key):
        """
        Obtiene un valor del cache
        
        Args:
            key: Clave a buscar
        
        Returns:
            Valor asociado a la clave o None si no existe
        """
        if key in self.cache:
            # Actualizar orden
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        """
        Añade un valor al cache
        
        Args:
            key: Clave
            value: Valor a almacenar
        """
        # Si la clave ya existe, actualizar orden
        if key in self.cache:
            self.order.remove(key)
        
        # Si el cache está lleno, eliminar el elemento menos usado
        elif len(self.cache) >= self.max_size:
            oldest_key = self.order.pop(0)
            del self.cache[oldest_key]
        
        # Añadir nueva clave-valor
        self.cache[key] = value
        self.order.append(key)
    
    def clear(self):
        """
        Limpia el cache
        """
        self.cache = {}
        self.order = []

# 7. Implementar un sistema de prefetch para pestañas adyacentes
def prefetch_tab_data(current_tab, datos, dataset_complete, df, df_normalizado, 
                      dataset_complete_geometry, variable_seleccionada_municipio,
                      variable_seleccionada_numerica, variable_seleccionada_paracorrelacion,
                      variable_seleccionada_categorica):
    """
    Precarga datos para pestañas adyacentes en segundo plano
    
    Args:
        current_tab: Índice de la pestaña actual
        ... otros parámetros necesarios para cargar datos
    """
    session_state = get_session_state()
    
    # Determinar pestañas adyacentes (actual +/- 1)
    adjacent_tabs = [
        t for t in [current_tab - 1, current_tab + 1] 
        if 0 <= t <= 5  # Solo pestañas válidas
    ]
    
    # Función para cargar datos de una pestaña en segundo plano
    def load_tab_data(tab_index):
        if tab_index == 1 and not all(k in session_state for k in ['fig_municipio', 'fig_ranking']):
            # Precargar solo los gráficos principales de la pestaña 2
            fig_municipio = crear_mapa_choropleth2(dataset_complete_geometry, lugar=variable_seleccionada_municipio)
            fig_ranking = plot_bar_chart(datos, 'Lugar', 'Índice_Compuesto', variable_seleccionada_municipio)
            
            session_state['fig_municipio'] = fig_municipio
            session_state['fig_ranking'] = fig_ranking
        
        elif tab_index == 2 and 'grafico3d' not in session_state:
            # Precargar solo el gráfico 3D de la pestaña 3
            grafico3d = generar_grafico_3d_con_lugar(datos, df_normalizado, dataset_complete, variable_seleccionada_municipio)
            session_state['grafico3d'] = grafico3d
        
        elif tab_index == 3 and 'recuento_clusters' not in session_state:
            # Precargar recuento de clústers de la pestaña 4
            recuento_clusters = recuento(datos)
            session_state['recuento_clusters'] = recuento_clusters
        
        elif tab_index == 4 and 'fig_scatter' not in session_state:
            # Precargar scatter plot de la pestaña 5
            fig_scatter = generate_scatter_with_annotations(
                datos, 
                variable_seleccionada_numerica, 
                variable_seleccionada_paracorrelacion, 
                variable_seleccionada_categorica
            )
            session_state['fig_scatter'] = fig_scatter
    
    # Usar ThreadPoolExecutor para cargar en segundo plano
    with ThreadPoolExecutor(max_workers=len(adjacent_tabs)) as executor:
        for tab in adjacent_tabs:
            executor.submit(load_tab_data, tab)






# Estructura principal optimizada de la aplicación (continuación)

def main():
    try:
        # Establecer configuración de página
        optimizar_interfaz()
        
        # Inicializar estado de sesión
        session_state = get_session_state(
            contador_visitas=0,
            active_tab=0,
            data_loaded=False
        )
        
        # Sidebar
        with st.sidebar:
            # Logo e información del proyecto
            st.markdown("""
            <h5 style='text-align: center;'> 
                Centro de Investigación e Innovación en TICs (INFOTEC)
                <hr>
                Aplicación elaborada por <br><br>
                <a href='https://www.linkedin.com/in/guarneros' style='color: #51C622; text-decoration: none;'>Rodrigo Guarneros Gutiérrez</a>        
                <br><br> 
                Para obtener el grado de Maestro en Ciencia de Datos e Información.
                <hr> 
                Asesor: <a href='https://www.infotec.mx/es_mx/Infotec/mario-graff-guerrero' style='color: #51C622; text-decoration: none;'> Ph.D. Mario Graff Guerrero </a>
            </h5>
            """, unsafe_allow_html=True)

            st.sidebar.image("fuentes/nube.png", use_column_width=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Cargar datos principales si aún no están cargados
            if not session_state.data_loaded:
                with st.spinner("Cargando datos iniciales..."):
                    # Incrementar contador de visitas una sola vez
                    session_state.contador_visitas = lazy_load(
                        'contador_visitas', 
                        incrementar_contador_visitas
                    )
                    
                    # Cargar datasets básicos
                    datos = lazy_load('datos', bajando_procesando_datos)
                    datos = optimize_dataframe(datos)  # Optimizar uso de memoria
                    
                    # Procesar variables
                    variable_list_numeric, variable_list_categorical, variable_list_municipio = lazy_load(
                        'variable_lists',
                        procesar_listas_variables,
                        datos
                    )
                    
                    session_state.data_loaded = True
            else:
                # Recuperar datos ya cargados
                datos = session_state.datos
                variable_list_numeric = session_state.variable_lists[0]
                variable_list_categorical = session_state.variable_lists[1]
                variable_list_municipio = session_state.variable_lists[2]
            
            # Selectores
            st.markdown("Principales características por Municipio:", unsafe_allow_html=True)
            variable_seleccionada_municipio = st.selectbox(
                'Selecciona el municipio de tu interés:', 
                sorted(variable_list_municipio, reverse=False),
                key='municipio_selector'
            )

            st.markdown("<hr>", unsafe_allow_html=True)
            
            st.markdown("Análisis Estadístico por Variable:", unsafe_allow_html=True)
            variable_seleccionada_numerica = st.selectbox(
                'Selecciona la variable numérica de interés:', 
                sorted(variable_list_numeric, reverse=False),
                key='variable_numerica_selector'
            )
            
            variable_seleccionada_categorica = st.selectbox(
                'Selecciona la variable categórica de interés:', 
                sorted(variable_list_categorical, reverse=False),
                key='variable_categorica_selector'
            )
            
            variable_seleccionada_paracorrelacion = st.selectbox(
                'Selecciona la variable que quieras correlacionar con la primera selección:', 
                sorted(variable_list_numeric, reverse=False),
                key='variable_correlacion_selector'
            )

            st.markdown("<hr>", unsafe_allow_html=True)

            # Expanders con información
            with st.expander('Enfoque de esta aplicación', expanded=False):
                st.write('''
                    - Se basa en un enfoque de <span style="color:#51C622">"Programación Orientada a Objetos"</span>.
                    - Los 2,456 municipios se pueden modelar a partir de sus atributos y funciones para aprovechar la revolución digital. 
                    - El principal objetivo es: <span style="color:#51C622">Ajustar un modelo de aprendizaje automático para clasificar a las localidades de México por su vocación para la transformación digital y despliegue de servicios TIC, en función de variables fundamentales de infraestructura, demográficas y socio-económicas.</span>
                    - Este aplicativo incluye atributos a nivel municipal tales como:
                        1. Número de viviendas. 
                        2. Grado educativo (Analfabetismo, Porcentaje de personas con educación básica, etc.).
                        3. Edad promedio, 
                        4. Penetración de Internet, entre otas.
                    - Con base en estas características, se pueden generar diferentes combinaciones y visualizaciones de interés para conocer mejor aspectos como:
                        1. La distribución estadística de las variables. 
                        2. Relación entre las variables. 
                        3. La distribución geográfica de las variables.
                    - La ventaja de un panel de control como este consiste en sus <span style="color:#51C622">economías de escala y la capacidad que tiene para presentar insights más profundos respecto a la población y sus funciones o actividades, tales como capacidad adquisitiva, preferencias, crédito al consumo, acceso a servicios de conectividad, empleo, sequías y hasta modelos predictivos.</span> 
                    ''', unsafe_allow_html=True)

            with st.expander('Fuentes y detalles técnicos', expanded=False):
                st.write('''
                    - Fuente: [Consejo Nacional de Población (CONAPO), consultado el 3 de febrero de 2024.](https://www.gob.mx/conapo).
                    - Tecnologías y lenguajes: Python 3.10, Streamlit 1.30.0, CSS 3.0, HTML5, Google Colab y GitHub. 
                    - Autor: Rodrigo Guarneros ([LinkedIn](https://www.linkedin.com/in/guarneros/) y [X](https://twitter.com/RodGuarneros)).
                    - Comentarios al correo electrónico rodrigo.guarneros@gmail.com
                    ''', unsafe_allow_html=True)

            st.image('fuentes/cc.png', caption= '\u00A9 Copy Rights Rodrigo Guarneros, 2024', use_column_width=True)
            st.markdown("Esta aplicación web se rige por los derechos de propiedad de [Creative Commons CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Si quieres hacer algunos ajustes o adaptar esta aplicación te puedo ayudar, [escríbeme](rodrigo.guarneros@gmail.com).", unsafe_allow_html=True)
            st.markdown(f"Visitas al sitio: **{session_state.contador_visitas}**", unsafe_allow_html=True)

        # Cargar datos adicionales solo cuando se necesiten
        def load_additional_data():
            with st.spinner("Cargando datos adicionales..."):
                # Cargar datasets adicionales
                dataset_complete = lazy_load('dataset_complete', bajando_procesando_datos_completos)
                dataset_complete = optimize_dataframe(dataset_complete)  # Optimizar uso de memoria
                
                df = lazy_load('df', bajando_procesando_X_entrenamiento)
                df = optimize_dataframe(df)  # Optimizar uso de memoria
                
                df_normalizado = lazy_load('df_normalizado', bajando_procesando_df_normalizado)
                df_normalizado = optimize_dataframe(df_normalizado)  # Optimizar uso de memoria
                
                # Cargar datos geográficos
                geojson = lazy_load('geojson', obtener_datos_geograficos)
                
                # Preparar datos para visualización
                dataset_complete_geometry = lazy_load(
                    'dataset_complete_geometry',
                    preparar_datos_para_visualizacion,
                    datos, geojson
                )
                
                return dataset_complete, df, df_normalizado, geojson, dataset_complete_geometry
        
        # Definir las pestañas
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Presentación", "Municipio", "Madurez Digital", 
            "Estadísiticas por Grupo", "Análisis Relacional", "Geografía"
        ])
        
        # Detectar la pestaña activa
        # Nota: Esto es una aproximación ya que Streamlit no tiene una API directa
        # para detectar la pestaña seleccionada actualmente
        if 'tab_clicked' not in session_state:
            session_state.tab_clicked = 0  # Pestaña 1 por defecto
            
        # Definir funciones para cada pestaña
        def render_tab1():
            with tab1:
                # La pestaña 1 es principalmente texto, carga rápida
                with st.expander('¿Para qué sirve esta aplicación?', expanded=False):
                    st.markdown(f'Provee un punto de referencia estadísticamente robusto, claro y preciso —con un criterio basado en aprendizaje automático y poder computacional, sin intervención humana, solo considerando las principales características de los municipios—, para efectos de que puedas ver dónde está cada municipio de México en su trayectoria hacia la <span style="color:#51C622">"Madurez Digital"</span> y qué características debe considerar para favorecer su transición a la siguiente fase del ciclo de transformación digital.', unsafe_allow_html=True)
                    # ... (resto del contenido de este expander)
                
                with st.expander('¿Qué es la madurez digital?', expanded=False):
                    st.markdown(f'En la inteligencia de negocios existen modelos de maduración para las organizaciones y empresas con el objeto de evaluar la toma decisiones basada en datos (Gartner 2004, AMR Research, Service Oriented Business Intelligence Maturirty Model (SOBIMM), entre otros descritos por <a href="https://aulavirtual.infotec.mx/pluginfile.php/115302/mod_label/intro/Medici%C3%B3n%20de%20Madurez%20en%20la%20Implementaci%C3%B3n%20de%20Inteligencia%20de%20Negocios.pdf" target="_blank"><b>Urbina Nájera y Medina-Barrera (2021)</b></a>), la Unión Europea desarrolló la metodología para evaluar la madurez digital de los gobiernos locales (<a href="https://data.europa.eu/en/news-events/news/lordimas-digital-maturity-assessment-tool-regions-and-cities" target="_blank"><b>LORDIMAS 2023, Digital Maturity Index for local governments</b></a>), no existe un enfoque único para evaluar la madurez digital de las regiones o localidades donde el ciudadano sea el objeto de estudio. No obstante, algunos países reconocen el papel de los servicios digitales y financieros como elementos fundamentales para hacer negocios y generar bienestar en una región. Por ello, han definido en sus estándares de desarrollo una canasta básica de bienes y servicios digitales.', unsafe_allow_html=True)
                    # ... (imagen y resto del contenido)
                
                with st.expander('¿Cómo utilizar esta aplicación?', expanded=False):
                    st.markdown(f'Como se puede ver, se cuenta con 5 secciones adicionales:', unsafe_allow_html=True)
                    # ... (resto del contenido)
                
                # Actualizar pestaña activa
                session_state.active_tab = 0
        
        def render_tab2():
            with tab2:
                # Cargar datos adicionales si es necesario
                dataset_complete, df, df_normalizado, geojson, dataset_complete_geometry = load_additional_data()
                
                # Título dinámico
                st.markdown(titulo_dinamico(variable_seleccionada_municipio, tipo="municipio"), unsafe_allow_html=True)
                
                # Expander con descripción
                with st.expander('Descripción', expanded=False):
                    st.markdown(f'Esta sección incluye cuatro visualizaciones relevantes para conocer mejor al municipio seleccionado y el lugar que tiene en la clasificación realizada por nuestra máquina de inferencia estadística. Se sugiere analizar en el siguiente orden:', unsafe_allow_html=True)
                    # ... (resto del contenido descriptivo)
                
                # Cargar gráficos solo si no están en session_state
                if not all(k in session_state for k in ['fig_municipio', 'fig_ranking', 'cuadro_resumen']):
                    with st.spinner("Generando visualizaciones..."):
                        # Generar los gráficos necesarios para esta pestaña
                        session_state.fig_municipio = crear_mapa_choropleth2(
                            dataset_complete_geometry, 
                            lugar=variable_seleccionada_municipio
                        )
                        
                        session_state.fig_ranking = plot_bar_chart(
                            datos, 
                            'Lugar', 
                            'Índice_Compuesto', 
                            variable_seleccionada_municipio
                        )
                        
                        session_state.cuadro_resumen = crear_display(
                            datos, 
                            variable_seleccionada_municipio
                        )
                
                # Cargar histogramas y boxplot bajo demanda
                if not all(k in session_state for k in ['fig_hist', 'fig_hist_index', 'fig_boxplot']):
                    with st.spinner("Generando visualizaciones estadísticas..."):
                        session_state.fig_hist = plot_histogram(
                            datos, 
                            variable_seleccionada_numerica, 
                            variable_seleccionada_categorica
                        )
                        
                        session_state.fig_hist_index = plot_histogram_with_density(
                            datos, 
                            'Índice_Compuesto', 
                            variable_seleccionada_municipio
                        )
                        
                        session_state.fig_boxplot = generate_boxplot_with_annotations(
                            datos, 
                            variable_seleccionada_numerica, 
                            variable_seleccionada_municipio
                        )
                
                # Crear dos columnas principales
                col_izq, col_der = st.columns([6, 6])
                
                # Columna izquierda: solo el ranking
                with col_izq:
                    st.plotly_chart(session_state.fig_ranking, width=400, use_container_width=True)
                
                # Columna derecha: mapa y gráficos en secuencia vertical
                with col_der:
                    st.plotly_chart(session_state.cuadro_resumen, width=400, use_container_width=True)
                    # Mapa ajustado al ancho de la columna
                    folium_static(session_state.fig_municipio, width=455, height=180)
                    # Análisis expander
                    with st.expander('Análisis', expanded=False):
                        st.markdown(f'Esta distribución bimodal sugiere dos grupos diferenciados en términos de madurez digital, una brecha digital significativa entre los municipios:', unsafe_allow_html=True)
                        # ... (resto del análisis)
                    
                    st.plotly_chart(session_state.fig_hist_index, use_container_width=True)
                    st.plotly_chart(session_state.fig_hist, use_container_width=True)
                    st.plotly_chart(session_state.fig_boxplot, use_container_width=True)
                
                # Actualizar pestaña activa
                session_state.active_tab = 1
                
                # Prefetch para pestañas adyacentes
                prefetch_tab_data(
                    1, 
                    datos, dataset_complete, df, df_normalizado, dataset_complete_geometry,
                    variable_seleccionada_municipio, variable_seleccionada_numerica,
                    variable_seleccionada_paracorrelacion, variable_seleccionada_categorica
                )
        
        def render_tab3():
            with tab3:
                # Cargar datos adicionales si es necesario
                dataset_complete, df, df_normalizado, geojson, dataset_complete_geometry = load_additional_data()
                
                st.markdown(titulo_dinamico(variable_seleccionada_municipio, tipo="madurez"), unsafe_allow_html=True)
                st.markdown(
                    """
                    <div style="text-align: justify;">
                        Maximiza la página para visualizar los tres Componentes Principales y sus patrones identificados. Visualiza cómo se complementan entre sí: <br>
                        - PC1 <span style="color:#51C622; font-weight:bold;">- Actividad financiera (volumen/intensidad);</span> <br>
                        - PC2 <span style="color:#51C622; font-weight:bold;">- Servicios digitales (infraestructura/acceso), y</span> <br>
                        - PC3 <span style="color:#51C622; font-weight:bold;">- Adopción financiera (diversificación/inclusión).</span> <br>
                        Con esta metodología se proporciona una visión muy completa del desarrollo financiero y digital de los municipios.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Configuración de las columnas
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Cargar gráfico 3D
                    if 'grafico3d' not in session_state:
                        with st.spinner("Generando gráfico 3D..."):
                            session_state.grafico3d = generar_grafico_3d_con_lugar(
                                datos, df_normalizado, dataset_complete, variable_seleccionada_municipio
                            )
                    
                    # Cargar gráfico 2D
                    if 'grafico2d1' not in session_state:
                        with st.spinner("Generando gráficos 2D..."):
                            session_state.grafico2d1 = generar_grafico_2d(
                                datos, df_normalizado, dataset_complete, variable_seleccionada_municipio,
                                x_col='PCA1', y_col='PCA2', title="PC1 vs. PC2 (2D)"
                            )
                    
                    # Expander con explicación
                    with st.expander('El significado de cada Componente Principal', expanded=False):
                        st.markdown(f'<span style="color:#51C622">Los componentes principales (PC1, PC2 y PC3) buscan maximizar la suma de las distancias al cuadrado entre los puntos proyectados y el origen</span>. Su resultado es una combinación lineal de todas las variables que los conforman. Así, la descomposición en valores singulares (SVD) nos permite visualizar en la gráfica la proyección de cada una de las combinaciones lineales en los municipios, representados en un espacio vectorial que va de -1 a 1 en cada eje del gráfico tridimensional.', unsafe_allow_html=True)
                        # ... (resto de la explicación)
                    
                    # Mostrar gráfico 3D
                    st.plotly_chart(session_state.grafico3d, use_container_width=True, height=500)
                    
                    with st.expander('Patrones en los clústers', expanded=False):
                        st.markdown(f'La separación entre clústers tiene mejor visibilidad en tres dimensiones, en general se puede decir que:', unsafe_allow_html=True)
                        # ... (resto del contenido)
                    
                    st.plotly_chart(session_state.grafico2d1, use_container_width=True, height=250)
                
                with col2:
                    # Cargar gráficos 2D adicionales
                    if 'grafico2d2' not in session_state:
                        with st.spinner("Generando gráficos 2D adicionales..."):
                            session_state.grafico2d2 = generar_grafico_2d(
                                datos, df_normalizado, dataset_complete, variable_seleccionada_municipio,
                                x_col='PCA1', y_col='PCA3', title="PC1 vs. PC3 (2D)"
                            )
                    
                    if 'grafico2d3' not in session_state:
                        with st.spinner("Generando último gráfico 2D..."):
                            session_state.grafico2d3 = generar_grafico_2d(
                                datos, df_normalizado, dataset_complete, variable_seleccionada_municipio,
                                x_col='PCA2', y_col='PCA3', title="PC2 vs. PC3 (2D)"
                            )
                    
                    with st.expander('Estructura de los clústers', expanded=False):
                        st.markdown(f'Esta segmentación, resultado de las similitudes en las 81 características de los municipios que propone la reducción dimensional, sugiere una clara estratificación de los municipios basada principalmente en su nivel de desarrollo financiero y económico, con subdivisiones adicionales basadas en infraestructura y acceso a servicios financieros especializados.', unsafe_allow_html=True)
                        # ... (resto del contenido)
                    
                    st.plotly_chart(session_state.grafico2d2, use_container_width=True, height=250)
                    
                    with st.expander('Perfil del municipio en cada clúster', expanded=False):
                        st.markdown(f'El Clúster Inicial (turquesa) tiene las siguientes características:', unsafe_allow_html=True)
                        # ... (resto del contenido)
                    
                    st.plotly_chart(session_state.grafico2d3, use_container_width=True, height=250)
                
                # Actualizar pestaña activa
                session_state.active_tab = 2
                
                # Prefetch para pestañas adyacentes
                prefetch_tab_data(
                    2, 
                    datos, dataset_complete, df, df_normalizado, dataset_complete_geometry,
                    variable_seleccionada_municipio, variable_seleccionada_numerica,
                    variable_seleccionada_paracorrelacion, variable_seleccionada_categorica
                )
        
        def render_tab4():
            with tab4:
                st.markdown("¿Qué patrones se encuentran en cada clúster?")
                
                # Cargar gráficos de clústers
                if 'recuento_clusters' not in session_state:
                    with st.spinner("Generando análisis de clústers..."):
                        session_state.recuento_clusters = recuento(datos)
                
                if 'boxplots_clusters' not in session_state:
                    with st.spinner("Generando boxplots por clústers..."):
                        session_state.boxplots_clusters = boxplot_por_cluster(datos, variable_seleccionada_numerica)
                
                if 'histograma_por_clusters' not in session_state:
                    with st.spinner("Generando histogramas por clústers..."):
                        session_state.histograma_por_clusters = plot_histogram_clusters(datos, variable_seleccionada_numerica)
                
                with st.expander('Recuento por nivel de madurez', expanded=False):
                    # Crear las columnas
                    col1, col2 = st.columns(2)
                    
                    # Columna 1: Información de recuento
                    with col1:
                        st.markdown("""
                        <div class="madurez-card">
                            <br>
                            <br>                
                            <p><span class="madurez-count">Optimización:</span> <b style="color:#51C622">647</b> municipios</p>
                            <p><span class="madurez-count">Definición:</span> <b style="color:#51C622">551</b> municipios</p>
                            <p><span class="madurez-count">En desarrollo:</span> <b style="color:#51C622">627</b> municipios</p>
                            <p><span class="madurez-count">Inicial:</span> <b style="color:#51C622">631</b> municipios</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Columna 2: Gráfico de barras
                    with col2:
                        st.plotly_chart(session_state.recuento_clusters, use_container_width=True, height=250)
                
                # Mostrar gráficos de análisis
                st.plotly_chart(session_state.boxplots_clusters, use_container_width=True)
                st.plotly_chart(session_state.histograma_por_clusters, use_container_width=True)
                
                # Actualizar pestaña activa
                session_state.active_tab = 3
                
                # Prefetch para pestañas adyacentes
                prefetch_tab_data(
                    3, 
                    datos, dataset_complete, df, df_normalizado, dataset_complete_geometry,
                    variable_seleccionada_municipio, variable_seleccionada_numerica,
                    variable_seleccionada_paracorrelacion, variable_seleccionada_categorica
                )
        
        def render_tab5():
            with tab5:
                st.markdown(titulo_dinamico(variable_seleccionada_numerica), unsafe_allow_html=True)
                
                with st.expander('Análisis', expanded=False):
                    st.markdown(f'Los diagramas de dispersión permiten visualizar las relaciones lineales y no lineales de las variables.', unsafe_allow_html=True)
                    # ... (resto del análisis)
                
                # Cargar scatter plot
                if 'fig_scatter' not in session_state:
                    with st.spinner("Generando gráfico de correlación..."):
                        session_state.fig_scatter = generate_scatter_with_annotations(
                            datos, 
                            variable_seleccionada_numerica, 
                            variable_seleccionada_paracorrelacion, 
                            variable_seleccionada_categorica
                        )
                
                st.plotly_chart(session_state.fig_scatter, use_container_width=True, height=500)
                
                # Actualizar pestaña activa
                session_state.active_tab = 4
                
                # Prefetch para pestañas adyacentes
                prefetch_tab_data(
                    4, 
                    datos, dataset_complete, df, df_normalizado, dataset_complete_geometry,
                    variable_seleccionada_municipio, variable_seleccionada_numerica,
                    variable_seleccionada_paracorrelacion, variable_seleccionada_categorica
                )
        
        def render_tab6():
            with tab6:
                with st.expander('Análisis', expanded=False):
                    st.markdown(f'La clasificación proporcionada por el aprendizaje automático no supervisado sugiere que <span style="color:#51C622"> la madurez digital de los municipios no es aleatoria, sino que sigue patrones relacionados con factores financieros, socio-económicos y geográficos</span>. Cuando se realizaba el entrenamiento de los modelos y se evaluaban, se revisaron los pesos de cada variable en cada componente principal; donde llama la atención que son estadísticamente relevantes variables geográficas como la latitud, longitud y el número de vecinos cercanos en un radio de 5 km. Sugiriendo que la proximidad geográfica entre los municipios influye en su madurez digital debido a la infraestructura compartida y la movilidad de sus factores productivos.', unsafe_allow_html=True)
                    # ... (resto del análisis)
                
                # Cargar mapa final
                if 'fig_map_final' not in session_state:
                    with st.spinner("Generando mapa geográfico..."):
                        session_state.fig_map_final = generar_mapa_con_lugar(datos, lugar=variable_seleccionada_municipio)
                
                st.plotly_chart(session_state.fig_map_final, use_container_width=True, height=500)
                
                # Actualizar pestaña activa
                session_state.active_tab = 5
        
        # Renderizar la pestaña seleccionada
        # Detectamos qué pestaña está activa monitoreando los clics
        # Este es un truco para detectar la pestaña activa, ya que Streamlit no tiene una API directa
        tab1_clicked = tab1.selectbox('', [''], key='tab1_select', label_visibility="collapsed")
        tab2_clicked = tab2.selectbox('', [''], key='tab2_select', label_visibility="collapsed")
        tab3_clicked = tab3.selectbox('', [''], key='tab3_select', label_visibility="collapsed")
        tab4_clicked = tab4.selectbox('', [''], key='tab4_select', label_visibility="collapsed")
        tab5_clicked = tab5.selectbox('', [''], key='tab5_select', label_visibility="collapsed")
        tab6_clicked = tab6.selectbox('', [''], key='tab6_select', label_visibility="collapsed")
        
        # Renderizar inicialmente la pestaña 1
        render_tab1()
        
        # Detectar cambios en los selectores y actualizar componentes relevantes
        if st.session_state.get('previous_municipio') != variable_seleccionada_municipio:
            # Municipio cambió, limpiar gráficos relacionados
            keys_to_clear = [
                'fig_municipio', 'fig_ranking', 'cuadro_resumen', 'fig_hist_index', 
                'grafico3d', 'grafico2d1', 'grafico2d2', 'grafico2d3',
                'fig_map_final'
            ]
            
            for key in keys_to_clear:
                if key in session_state:
                    del session_state[key]
            
            # Guardar el nuevo valor para la próxima comparación
            session_state.previous_municipio = variable_seleccionada_municipio
        
        if st.session_state.get('previous_numerica') != variable_seleccionada_numerica:
            # Variable numérica cambió, limpiar gráficos relacionados
            keys_to_clear = [
                'fig_hist', 'fig_boxplot', 'boxplots_clusters',
                'histograma_por_clusters', 'fig_scatter'
            ]
            
            for key in keys_to_clear:
                if key in session_state:
                    del session_state[key]
            
            # Guardar el nuevo valor para la próxima comparación
            session_state.previous_numerica = variable_seleccionada_numerica
        
        if st.session_state.get('previous_categorica') != variable_seleccionada_categorica:
            # Variable categórica cambió, limpiar histograma
            if 'fig_hist' in session_state:
                del session_state['fig_hist']
            
            if 'fig_scatter' in session_state:
                del session_state['fig_scatter']
            
            # Guardar el nuevo valor para la próxima comparación
            session_state.previous_categorica = variable_seleccionada_categorica
        
        if st.session_state.get('previous_correlacion') != variable_seleccionada_paracorrelacion:
            # Variable de correlación cambió, limpiar scatter plot
            if 'fig_scatter' in session_state:
                del session_state['fig_scatter']
            
            # Guardar el nuevo valor para la próxima comparación
            session_state.previous_correlacion = variable_seleccionada_paracorrelacion
        
        # Limpiar recursos no utilizados según la pestaña activa
        cleanup_unused_resources(session_state.active_tab)
        
        # Forzar recolección de basura para liberar memoria
        gc.collect()
        
    except Exception as e:
        st.error(f"Error en la aplicación: {str(e)}")
        logger.exception("Error en la aplicación:")

if __name__ == "__main__":
    main()