#############################
# IMPORTAR LIBRERÍAS Y CONFIG #
#############################
import streamlit as st
st.set_page_config(
    page_title="Aprendizaje Automático para los Municipios de México",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import io
import geopandas as gpd
import numpy as np
import json
import folium
import zipfile
from streamlit import components
from sklearn.linear_model import LinearRegression
from streamlit_folium import folium_static
from scipy.stats import gaussian_kde
import pymongo
from pymongo import MongoClient
from gridfs import GridFS
from io import BytesIO
from dotenv import load_dotenv
import os
from bson import ObjectId
from concurrent.futures import ThreadPoolExecutor
import gc
import logging

# Configuración de logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Habilitar tema oscuro para Altair
alt.themes.enable("dark")

#############################
# CSS STYLING
#############################
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: -10rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}
[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}
[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}
[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}
[data-testid="stMetricDeltaIcon-Up"],
[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

#############################
# FUNCIONES AUXILIARES
#############################
def convert_objectid_to_str(document):
    for key, value in document.items():
        if isinstance(value, ObjectId):
            document[key] = str(value)
    return document

#############################
# CONEXIÓN CON MONGO Y CACHE
#############################
@st.cache_resource
def connect_to_mongo(mongo_uri):
    client = MongoClient(mongo_uri)
    return client['Municipios_Rodrigo']

def incrementar_contador_visitas():
    try:
        mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
        client = MongoClient(mongo_uri)
        db = client['Municipios_Rodrigo']
        collection = db['visita']
        visita = collection.find_one_and_update(
            {"_id": "contador"},
            {"$inc": {"contador": 1}},
            upsert=True,
            return_document=pymongo.ReturnDocument.AFTER
        )
        return visita['contador']
    except Exception as e:
        st.error(f"Hubo un error al acceder a la base de datos: {e}")
        raise

contador_visitas = incrementar_contador_visitas()

@st.cache_data
def bajando_procesando_datos():
    mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
    client = MongoClient(mongo_uri)
    db = client['Municipios_Rodrigo']
    collection = db['datos_finales']
    datos_raw = collection.find()
    datos = pd.DataFrame(list(map(convert_objectid_to_str, datos_raw)))
    for column in datos.select_dtypes(include=['object']).columns:
        datos[column] = datos[column].apply(lambda x: x.encode('Latin1').decode('Latin1') if isinstance(x, str) else x)
    categorias_orden = ['Optimización', 'Definición', 'En desarrollo', 'Inicial']
    datos['Madurez'] = pd.Categorical(datos['Madurez'], categories=categorias_orden, ordered=False)
    return datos

@st.cache_data
def bajando_procesando_datos_completos():
    mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
    client = MongoClient(mongo_uri)
    db = client['Municipios_Rodrigo']
    collection = db['completo']
    datos_raw = collection.find()
    dataset_complete = pd.DataFrame(list(map(convert_objectid_to_str, datos_raw)))
    for column in dataset_complete.select_dtypes(include=['object']).columns:
        dataset_complete[column] = dataset_complete[column].apply(lambda x: x.encode('Latin1').decode('Latin1') if isinstance(x, str) else x)
    dataset_complete.columns = dataset_complete.columns.str.strip()
    return dataset_complete

@st.cache_data
def bajando_procesando_X_entrenamiento():
    mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
    client = MongoClient(mongo_uri)
    db = client['Municipios_Rodrigo']
    collection = db['X_for_training_normalizer']
    datos_raw = collection.find()
    df = pd.DataFrame(list(map(convert_objectid_to_str, datos_raw)))
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def bajando_procesando_df_normalizado():
    mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
    client = MongoClient(mongo_uri)
    db = client['Municipios_Rodrigo']
    collection = db['df_pca_norm']
    datos_raw = collection.find()
    df_normalizado = pd.DataFrame(list(map(convert_objectid_to_str, datos_raw)))
    df_normalizado.columns = df_normalizado.columns.astype(str).str.strip()
    return df_normalizado

# Cargar y procesar datos base
datos = bajando_procesando_datos()
input_datos = datos.copy()

# Procesar columnas adicionales
input_datos['Operadores Escala Pequeña BAF'] = input_datos['operadores_escal_pequeña_baf']
input_datos.drop(columns=['operadores_escal_pequeña_baf'], inplace=True)
input_datos['Penetración BAF (Fibra)'] = input_datos['penetracion_baf_fibra']
input_datos.drop(columns=['penetracion_baf_fibra'], inplace=True)

dataset_complete = bajando_procesando_datos_completos()
df = bajando_procesando_X_entrenamiento()
df_normalizado = bajando_procesando_df_normalizado()

# Procesar variables para la selección
variable_list_numerica = list(input_datos.select_dtypes(include=['int64', 'float64']).columns)
variable_list_categoricala = list(input_datos.select_dtypes(include=['object', 'category']).columns)
variable_list_municipio = list(input_datos['Lugar'].unique())
columns_to_exclude_numeric = ['Cluster2','Unnamed: 0', 'Unnamed: 0.2', 'cve_edo', 'cve_municipio', 'cvegeo', 'Estratos ICM', 'Estrato IDDM', 'Municipio', 'df1_ENTIDAD', 'df1_KEY MUNICIPALITY', 'df2_Clave Estado', 'df2_Clave Municipio', 'df3_Clave Estado', 'df3_Clave Municipio', 'df4_Clave Estado', 'df4_Clave Municipio']
columns_to_exclude_categorical = ['_id','Lugar', 'Estado2', 'df2_Región', 'df3_Región', 'df3_Tipo de población', 'df4_Región', 'Municipio']
variable_list_numeric = [col for col in variable_list_numerica if col not in columns_to_exclude_numeric]
variable_list_categorical = [col for col in variable_list_categoricala if col not in columns_to_exclude_categorical]

# GEOJSON Y GEODATAFRAME
@st.cache_data
def consultando_base_de_datos(_db):
    fs = GridFS(_db)
    file = fs.find_one({'filename': 'municipios.geojson'})
    if file:
        return file.read()
    return None

def geojson_to_geodataframe(geojson_data):
    return gpd.read_file(BytesIO(geojson_data))

mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
db_geo = connect_to_mongo(mongo_uri)
geojson_data = consultando_base_de_datos(db_geo)
if geojson_data:
    geojson = geojson_to_geodataframe(geojson_data)
    input_datos.rename(columns={'cvegeo': 'CVEGEO'}, inplace=True)
    input_datos['CVEGEO'] = input_datos['CVEGEO'].astype(str).str.zfill(5)
    geojson['CVEGEO'] = geojson['CVEGEO'].astype(str)
    dataset_complete_geometry = input_datos.merge(geojson[['CVEGEO', 'geometry']], on='CVEGEO', how='left')
else:
    geojson = None
    dataset_complete_geometry = None

#############################
# FUNCIONES DE GRÁFICOS
#############################
@st.cache_data
def crear_mapa_choropleth2(dataset, estado=None, cluster=None, lugar=None, municipio_inicial="MunicipioX"):
    gdf = gpd.GeoDataFrame(dataset, geometry='geometry')
    if estado:
        gdf = gdf[gdf['Estado'] == estado]
    if cluster is not None:
        gdf = gdf[gdf['Clústers'] == cluster]
    lugar_a_buscar = lugar if lugar else municipio_inicial
    if lugar_a_buscar:
        gdf_filtrado = gdf[gdf['Lugar'] == lugar_a_buscar]
        if gdf_filtrado.empty:
            print(f"No se encontraron datos para el lugar: {lugar_a_buscar}")
            return None
        gdf = gdf_filtrado
    centro = gdf.geometry.centroid.iloc[0]
    m = folium.Map(location=[centro.y, centro.x], zoom_start=12, tiles="CartoDB dark_matter")
    bounds = gdf.geometry.total_bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    mapa_colores = {
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7',
        'Definición': '#CC6CE7',
        'Optimización': '#51C622',
    }
    def obtener_color(cluster_value):
        return mapa_colores.get(cluster_value, '#FFFFFF')
    folium.GeoJson(
        gdf,
        name="Choropleth de Clústers",
        style_function=lambda feature: {
            'fillColor': obtener_color(feature['properties']['Madurez']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['Lugar', 'Madurez'],
            aliases=['Lugar', 'Grado de Madurez'],
            localize=True,
            sticky=True
        ),
        highlight_function=lambda x: {'fillOpacity': 0.9}
    ).add_to(m)
    folium.LayerControl().add_to(m)
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
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))
    return m

@st.cache_data
def plot_bar_chart(data, lugar_columna, indice_columna, lugar_seleccionado):
    plot_data = data.copy()
    plot_data[indice_columna] = pd.to_numeric(plot_data[indice_columna], errors='coerce')
    plot_data = plot_data.sort_values(by=indice_columna, ascending=True)
    bar_colors = ['red' if lugar == lugar_seleccionado else 'dodgerblue' for lugar in plot_data[lugar_columna]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_data[indice_columna],
        y=plot_data[lugar_columna],
        orientation='h',
        marker=dict(color=bar_colors, line=dict(color='white', width=0.5)),
        customdata=np.stack((plot_data["Ranking"], plot_data["Etapa_Madurez"], plot_data[indice_columna]), axis=-1),
        hovertemplate=(
            "Municipio: %{y}<br>" +
            "Índice de Madurez: %{customdata[2]:.10f}<br>" +
            "Lugar en el Ranking: %{customdata[0]}<br>" +
            "Madurez: %{customdata[1]}<extra></extra>"
        )
    ))
    annotations = []
    for lugar, ranking, valor in zip(plot_data[lugar_columna], plot_data["Ranking"], plot_data[indice_columna]):
        annotations.append(dict(
            xref='paper', yref='y',
            x=0, y=lugar,
            text=lugar,
            showarrow=False,
            font=dict(color='red' if lugar == lugar_seleccionado else 'white', size=10, family="Arial"),
            xanchor='right',
            xshift=-10
        ))
        annotations.append(dict(
            x=valor, y=lugar,
            text=f"{int(ranking)} ({valor:.10f})",
            showarrow=False,
            font=dict(color='white', size=7),
            xanchor='left',
            xshift=5
        ))
    num_lugares = len(plot_data)
    height = max(400, num_lugares * 18)
    fig.update_layout(
        title=dict(text=f"Índice de Madurez por Municipio (Resaltado: {lugar_seleccionado})", font=dict(color='#FFD86C')),
        xaxis_title=dict(text="Índice de Madurez", font=dict(color='#FFD86C')),
        yaxis_title=dict(text="Municipio", font=dict(color='#FFD86C')),
        height=height,
        margin=dict(l=200, r=20, t=70, b=50),
        showlegend=False,
        xaxis=dict(range=[0, plot_data[indice_columna].max() * 1.1], tickformat='.10f', showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        annotations=annotations,
        bargap=0.2,
        plot_bgcolor='rgba(0, 0, 0, 0.1)',
        paper_bgcolor='rgba(0, 0, 0, 0)'
    )
    return fig

def crear_display(data, lugar_seleccionado):
    mapa_colores = {
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7',
        'Definición': '#CC6CE7',
        'Optimización': '#51C622',
    }
    lugar_row = data[data['Lugar'] == lugar_seleccionado]
    if lugar_row.empty:
        return None
    lugar_ranking = lugar_row['Ranking'].iloc[0]
    etapa_madurez = lugar_row['Etapa_Madurez'].iloc[0]
    color_rect = mapa_colores.get(etapa_madurez, 'dodgerblue')
    fig = go.Figure()
    fig.add_shape(
        type="path",
        path="M 0,0 Q 0,0 0.1,0 L 0.9,0 Q 1,0 1,0.1 L 1,0.9 Q 1,1 0.9,1 L 0.1,1 Q 0,1 0,0.9 Z",
        fillcolor=color_rect,
        line=dict(width=0),
        xref="paper", yref="paper",
        layer="below",
        opacity=1
    )
    fig.add_annotation(
        text="Lugar en el Ranking de 2,456 municipios en México",
        x=0.5,
        y=0.80,
        showarrow=False,
        font=dict(family="Arial", size=12, color="#050505"),
        align="center"
    )
    fig.add_annotation(
        text=str(int(lugar_ranking)),
        x=0.5,
        y=0.35,
        showarrow=False,
        font=dict(family="Arial", size=37, color="#050505"),
        align="center"
    )
    fig.update_layout(
        width=200,
        height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=color_rect,
        plot_bgcolor=color_rect,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1])
    )
    return fig

@st.cache_data
def plot_histogram(df, numeric_column, categorical_column):
    color_map = {
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7',
        'Definición': '#CC6CE7',
        'Optimización': '#51C622',
    }
    fig = px.histogram(
        df, 
        x=numeric_column, 
        color=categorical_column,
        color_discrete_map=color_map,
        opacity=0.6,
        title=f'Histograma de la variable "{numeric_column}" y <br>la categoría "{categorical_column}"'
    )
    fig.update_yaxes(title_text="Frecuencia absoluta")
    stats = {
        'Media': df[numeric_column].mean(),
        'Mediana': df[numeric_column].median(),
        'Moda': df[numeric_column].mode()[0],
        'Desviación estándar': df[numeric_column].std()
    }
    stats_text = "<br>".join([f"<b>{key}</b>: {value:.2f}" for key, value in stats.items()])
    category_counts = df[categorical_column].value_counts()
    counts_text = "<br>".join([f"<b>{category}</b>: {count}" for category, count in category_counts.items()])
    annotations_text = f"{stats_text}<br><br><b>Conteo por categoría:</b><br>{counts_text}"
    annotations = [
        dict(
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
        )
    ]
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

def plot_histogram_with_density(df, numeric_column, selected_value=None):
    fig = px.histogram(
        df,
        x=numeric_column,
        opacity=0.6,
        title=f'Distribución del índice de madurez digital',
        nbins=50,
        labels={'x': 'Valores del Índice', 'y': 'Frecuencia'}
    )
    fig.update_traces(marker_line_color='white', marker_line_width=1.5)
    hist_data = df[numeric_column].dropna().astype(float)
    kde = gaussian_kde(hist_data)
    density_x = np.linspace(hist_data.min(), hist_data.max(), 1000)
    density_y = kde(density_x)
    density_y_scaled = density_y * len(hist_data) * (hist_data.max() - hist_data.min()) / 50
    fig.add_trace(
        go.Scatter(
            x=density_x,
            y=density_y_scaled,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Dens'
        )
    )
    if selected_value is not None:
        try:
            selected_value_float = float(selected_value)
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
        except ValueError:
            print(f"Error: El valor seleccionado '{selected_value}' no es numérico y no se puede destacar.")
    mean = hist_data.mean()
    std = hist_data.std()
    median = hist_data.median()
    mode = hist_data.mode()[0]
    annotation_text = (
        f"<b>Estadísticos:</b><br>"
        f"Media: {mean:.2f}<br>"
        f"Mediana: {median:.2f}<br>"
        f"Moda: {mode:.2f}<br>"
        f"Desv. Est.: {std:.2f}"
    )
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
        plot_bgcolor='rgba(0, 0, 0, 0.1)'
    )
    return fig

def generate_boxplot_with_annotations(df, variable, lugar_seleccionado):
    stats = {
        'Media': np.mean(df[variable]),
        'Mediana': np.median(df[variable]),
        'Moda': df[variable].mode().iloc[0],
        'Desviación estándar': np.std(df[variable])
    }
    fig = px.box(
        df,
        y=variable,
        points=False,
        title=f'Diagrama para la variable<br>"{variable}"',
        template='plotly_dark'
    )
    if lugar_seleccionado:
        df_lugar = df[df['Lugar'] == lugar_seleccionado]
        fig.add_scatter(
            x=[0] * len(df_lugar),
            y=df_lugar[variable],
            mode='markers',
            marker=dict(
                color='rgba(0, 255, 0, 0.7)',
                size=10,
                line=dict(color='rgba(0, 255, 0, 1)', width=2)
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
            color='rgba(255, 165, 0, 0.5)',
            size=7,
            line=dict(color='rgba(255, 165, 0, 0.7)', width=1)
        ),
        name='Otros lugares',
        hovertemplate='<b>%{customdata[0]}</b><br>'+variable+': %{y:.2f}<extra></extra>',
        customdata=df_rest[['Municipio']]
    )
    annotations_text = "<br>".join([f"<b>{stat_name}</b>: {stat_value:.2f}" for stat_name, stat_value in stats.items()])
    annotations = [
        dict(
            x=0.5,
            y=-0.3,
            xref='paper',
            yref='paper',
            text=annotations_text,
            showarrow=False,
            font=dict(color='white', size=12),
            align='center',
            bgcolor='rgba(0, 0, 0, 0.7)',
            bordercolor='white',
            borderwidth=2,
            opacity=0.8
        )
    ]
    fig.update_layout(
        title_font=dict(color='#FFD86C', size=16),
        title_x=0.2,
        showlegend=True,
        width=1350,
        height=500,
        margin=dict(l=55, r=55, t=80, b=200),
        annotations=annotations,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.3,
            xanchor='center',
            x=0.5,
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

def generar_grafico_3d_con_lugar(df, df_normalizado, dataset_complete, lugar_seleccionado=None):
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }
    df_pca2 = df_normalizado.to_numpy()[:, 1:4]
    pca_df = pd.DataFrame(df_pca2, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df['Madurez'] = df['Etapa_Madurez']
    pca_df['Lugar'] = dataset_complete['Lugar']
    fig = px.scatter_3d(
        pca_df, 
        x='PCA1', y='PCA2', z='PCA3',
        color='Madurez',
        labels={'PCA1': 'Componente PC1', 'PCA2': 'Componente PC2', 'PCA3': 'Componente PC3'},
        hover_data=['Lugar'],
        category_orders={'Madurez': ['Optimización', 'Definición', 'En desarrollo', 'Inicial']},
        color_discrete_map=color_map
    )
    if lugar_seleccionado:
        lugar_df = pca_df[pca_df['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            fig.add_trace(
                px.scatter_3d(lugar_df, x='PCA1', y='PCA2', z='PCA3', hover_data=['Lugar'],
                              color_discrete_map={'Madurez': 'green'}).data[0]
            )
            fig.update_traces(marker=dict(size=20, color='green', opacity=1), selector=dict(name=lugar_seleccionado))
    fig.update_traces(
        marker=dict(size=6, opacity=0.7, line=dict(width=0.02, color='gray')),
        selector=dict(type='scatter3d')
    )
    fig.update_layout(
        title="Municipios por grado de madurez multidimensional",
        title_x=0.05,
        showlegend=True,
        legend=dict(
            title=dict(text='Madurez'),
            itemsizing='constant',
            font=dict(color='white')
        ),
        scene=dict(
            xaxis_title="Componente PC1",
            yaxis_title="Componente PC2",
            zaxis_title="Componente PC3",
            xaxis=dict(titlefont=dict(color='white'), gridcolor='white', zerolinecolor='white'),
            yaxis=dict(titlefont=dict(color='white'), gridcolor='white', zerolinecolor='white'),
            zaxis=dict(titlefont=dict(color='white'), gridcolor='white', zerolinecolor='white'),
            bgcolor='rgb(0, 0, 0)',
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            zaxis_showgrid=True
        ),
        font=dict(color='white'),
        paper_bgcolor='rgb(0, 0, 0)',
        plot_bgcolor='rgb(0, 0, 0)'
    )
    return fig

def generar_grafico_2d(df, df_normalizado, dataset_complete, lugar_seleccionado=None):
    df['Madurez'] = df['Madurez'].str.strip()
    df_pca2 = df_normalizado.to_numpy()[:, 1:4]
    pca_df = pd.DataFrame(df_pca2, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df['Madurez'] = df['Madurez'].astype('category')
    pca_df['Lugar'] = dataset_complete['Lugar']
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }
    fig = px.scatter(pca_df, x='PCA1', y='PCA2',
                     color='Madurez',
                     labels={'PCA1': 'Componente PC1', 'PCA2': 'Componente PC2'},
                     hover_data=['Lugar'],
                     category_orders={'Madurez': ['Optimización', 'Definición', 'En desarrollo', 'Inicial']},
                     color_discrete_map=color_map)
    if lugar_seleccionado:
        lugar_df = pca_df[pca_df['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            fig.add_trace(
                px.scatter(lugar_df, x='PCA1', y='PCA2', hover_data=['Lugar'],
                           color_discrete_map={'Madurez': 'green'}).data[0]
            )
            fig.update_traces(marker=dict(size=10, color='green', opacity=1), selector=dict(name=lugar_seleccionado))
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='gray')),
        selector=dict(mode='markers')
    )
    fig.update_layout(
        title="PC2 vs. PC1 (2D)",
        title_x=0.3,
        showlegend=True,
        legend=dict(title=dict(text='Madurez'), itemsizing='constant'),
        paper_bgcolor='rgb(0, 0, 0)',
        plot_bgcolor='rgb(0, 0, 0)',
        font=dict(color='white')
    )
    return fig

def generar_grafico_2d2(df, df_normalizado, dataset_complete, lugar_seleccionado=None):
    df['Madurez'] = df['Madurez'].astype('category')
    df_pca2 = df_normalizado.to_numpy()[:, 1:4]
    pca_df = pd.DataFrame(df_pca2, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df['Etapa_Madurez'] = df['Madurez']
    pca_df['Lugar'] = dataset_complete['Lugar']
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }
    fig = px.scatter(
        pca_df,
        x='PCA1',
        y='PCA3',
        labels={'PCA1': 'Componente PC1', 'PCA3': 'Componente PC3'},
        hover_data=['Lugar'],
        color='Etapa_Madurez',
        color_discrete_map=color_map
    )
    if lugar_seleccionado:
        lugar_df = pca_df[pca_df['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=lugar_df['PCA1'],
                    y=lugar_df['PCA3'],
                    mode='markers',
                    marker=dict(size=12, color='orange', symbol='diamond'),
                    name=f"Lugar: {lugar_seleccionado}"
                )
            )
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='gray'))
    )
    fig.update_layout(
        title="PC1 vs. PC3 (2D)",
        title_x=0.5,
        showlegend=True,
        legend=dict(title=dict(text='Etapa de Madurez'), itemsizing='constant'),
        paper_bgcolor='rgb(0, 0, 0)',
        plot_bgcolor='rgb(0, 0, 0)',
        font=dict(color='white')
    )
    return fig

def generar_grafico_2d3(df, df_normalizado, dataset_complete, lugar_seleccionado=None):
    df['Madurez'] = df['Madurez'].str.strip()
    df_pca2 = df_normalizado.to_numpy()[:, 1:4]
    pca_df = pd.DataFrame(df_pca2, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df['Madurez'] = df['Madurez'].astype('category')
    pca_df['Lugar'] = dataset_complete['Lugar']
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }
    fig = px.scatter(pca_df, x='PCA2', y='PCA3',
                     color='Madurez',
                     labels={'PCA2': 'Componente PC2', 'PCA3': 'Componente PC3'},
                     hover_data=['Lugar'],
                     category_orders={'Madurez': ['Optimización', 'Definición', 'En desarrollo', 'Inicial']},
                     color_discrete_map=color_map)
    if lugar_seleccionado:
        lugar_df = pca_df[pca_df['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            fig.add_trace(
                px.scatter(lugar_df, x='PCA2', y='PCA3', hover_data=['Lugar'],
                           color_discrete_map={'Madurez': 'green'}).data[0]
            )
            fig.update_traces(marker=dict(size=10, color='green', opacity=1), selector=dict(name=lugar_seleccionado))
    fig.update_traces(
        marker=dict(size=8, opacity=0.7, line=dict(width=0.02, color='gray'))
    )
    fig.update_layout(
        title="PC3 vs. PC2 (2D)",
        title_x=0.3,
        showlegend=True,
        legend=dict(title=dict(text='Madurez'), itemsizing='constant', font=dict(color='white')),
        font=dict(color='white'),
        paper_bgcolor='rgb(0, 0, 0)',
        plot_bgcolor='rgb(0, 0, 0)'
    )
    return fig

def recuento(df):
    total_municipios = len(df)
    counts = df['Madurez'].value_counts()
    df_counts = counts.reset_index()
    df_counts.columns = ['Madurez', 'Cantidad']
    df_counts['Frecuencia relativa'] = df_counts['Cantidad'] / total_municipios
    color_map = {
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7',
        'Definición': '#CC6CE7',
        'Optimización': '#51C622',
    }
    fig = px.bar(df_counts, 
                 x='Madurez', 
                 y='Frecuencia relativa', 
                 title="Frecuencia relativa por nivel de madurez",
                 labels={'Frecuencia relativa': 'Frecuencia relativa', 'Madurez': 'Nivel de madurez'},
                 color='Madurez', 
                 color_discrete_map=color_map,
                 category_orders={'Madurez': ['Inicial', 'En desarrollo', 'Definición', 'Optimización']},
                 height=280)
    return fig

def titulo_dinamico(variable):
    styled_title = f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">La variable mostrada es: "{variable}".</span>'
    return styled_title

def titulo_dinamico2(variable):
    styled_title = f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">Municipio de "{variable}".</span>'
    return styled_title

def titulo_dinamico3(variable):
    styled_title = f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">Análisis de Madurez Digital de "{variable}".</span>'
    return styled_title

#########################################
# PARALLELIZACIÓN DE CARGA DE GRÁFICOS
#########################################
def parallel_load_tab2(datos, dataset_complete_geometry, variable_seleccionada_municipio, variable_seleccionada_numerica, variable_seleccionada_categorica):
    with ThreadPoolExecutor() as executor:
        future_municipio = executor.submit(crear_mapa_choropleth2, dataset_complete_geometry, None, None, variable_seleccionada_municipio, municipio_inicial="Abalá, Yucatán")
        future_ranking = executor.submit(plot_bar_chart, datos, 'Lugar', 'Índice_Compuesto', variable_seleccionada_municipio)
        future_cuadro = executor.submit(crear_display, datos, variable_seleccionada_municipio)
        future_hist_index = executor.submit(plot_histogram_with_density, datos, 'Índice_Compuesto', variable_seleccionada_municipio)
        future_hist = executor.submit(plot_histogram, datos, variable_seleccionada_numerica, variable_seleccionada_categorica)
        future_boxplot = executor.submit(generate_boxplot_with_annotations, datos, variable_seleccionada_numerica, variable_seleccionada_municipio)
        fig_municipio = future_municipio.result()
        fig_ranking = future_ranking.result()
        cuadro_resumen = future_cuadro.result()
        fig_hist_index = future_hist_index.result()
        fig_hist = future_hist.result()
        fig_boxplot = future_boxplot.result()
    return fig_municipio, fig_ranking, cuadro_resumen, fig_hist_index, fig_hist, fig_boxplot

def parallel_load_tab3(datos, df, df_normalizado, dataset_complete, variable_seleccionada_municipio):
    with ThreadPoolExecutor() as executor:
        future_3d = executor.submit(generar_grafico_3d_con_lugar, datos, df_normalizado, dataset_complete, variable_seleccionada_municipio)
        future_2d1 = executor.submit(generar_grafico_2d, datos, df_normalizado, dataset_complete, variable_seleccionada_municipio)
        future_2d2 = executor.submit(generar_grafico_2d2, datos, df_normalizado, dataset_complete, variable_seleccionada_municipio)
        future_2d3 = executor.submit(generar_grafico_2d3, df, df_normalizado, dataset_complete, variable_seleccionada_municipio)
        grafico3d = future_3d.result()
        grafico2d1 = future_2d1.result()
        grafico2d2 = future_2d2.result()
        grafico2d3 = future_2d3.result()
    return grafico3d, grafico2d1, grafico2d2, grafico2d3

#########################################
# SIDEBAR Y CONFIGURACIÓN DE TABS
#########################################
with st.sidebar:
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
    st.markdown("Principales características por Municipio:", unsafe_allow_html=True)
    variable_seleccionada_municipio = st.selectbox('Selecciona el municipio de tu interés:', sorted(variable_list_municipio, reverse=False))
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("Análisis Estadístico por Variable:", unsafe_allow_html=True)
    variable_seleccionada_numerica = st.selectbox('Selecciona la variable numérica de interés:', sorted(variable_list_numeric, reverse=False))
    variable_seleccionada_categorica = st.selectbox('Selecciona la variable categórica de interés:', sorted(variable_list_categorical, reverse=False))
    variable_seleccionada_paracorrelacion = st.selectbox('Selecciona la variable que quieras correlaccionar con la primera selección:', sorted(variable_list_numeric, reverse=False))
    st.markdown("<hr>", unsafe_allow_html=True)
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
    st.markdown(f"Visitas al sitio: **{contador_visitas}**", unsafe_allow_html=True)

#########################################
# CARGA DE GRÁFICOS PARA LOS TABS CON PARALLELIZACIÓN
#########################################
fig_municipio, fig_ranking, cuadro_resumen, fig_hist_index, fig_hist, fig_boxplot = parallel_load_tab2(
    input_datos, dataset_complete_geometry, variable_seleccionada_municipio, 
    variable_seleccionada_numerica, variable_seleccionada_categorica
)
grafico3d, grafico2d1, grafico2d2, grafico2d3 = parallel_load_tab3(
    input_datos, df, df_normalizado, dataset_complete, variable_seleccionada_municipio
)
fig_scatter = generate_scatter_with_annotations(input_datos, variable_seleccionada_numerica, variable_seleccionada_paracorrelacion, variable_seleccionada_categorica)
fig_map_final = generar_mapa_con_lugar(input_datos, lugar=variable_seleccionada_municipio)
recuento_clusters = recuento(input_datos)
Titulo_dinamico = titulo_dinamico(variable=variable_seleccionada_numerica)
Titulo_dinamico2 = titulo_dinamico2(variable=variable_seleccionada_municipio)
Titulo_dinamico3 = titulo_dinamico3(variable=variable_seleccionada_municipio)

#########################################
# TABS DEL DASHBOARD
#########################################
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Presentación", "Municipio", "Madurez Digital", "Estadísiticas por Grupo", "Análisis Relacional", "Geografía"])

with tab1:
    with st.expander('¿Para qué sirve esta aplicación?', expanded=False):
        st.markdown(f'Provee un punto de referencia estadísticamente robusto, claro y preciso —con un criterio basado en aprendizaje automático y poder computacional, sin intervención humana, solo considerando las principales características de los municipios—, para efectos de que puedas ver dónde está cada municipio de México en su trayectoria hacia la <span style="color:#51C622">"Madurez Digital"</span> y qué características debe considerar para favorecer su transición a la siguiente fase del ciclo de transformación digital.', unsafe_allow_html=True)
        st.markdown(f'Permíteme compartir tres elementos que motivaron la creación de esta aplicación:', unsafe_allow_html=True)
        st.markdown(f'1. <span style="color:#51C622">La madurez digital</span> es multifactorial, incluye una combinación precisa de factores adicionales a los tradicionales como el acceso a Internet, los servicios de conectividad o dispositivos (socio-económicos, infraestructura y demográficos).', unsafe_allow_html=True)
        st.markdown(f'''
        <div style="text-align: center; padding-left: 40px;">
            Uno de mis libros favoritos, escrito por 
            <span style="color:#51C622">Antoine Augustin Cournot</span> (1897, página 
            <span style="color:#51C622">24</span>) 
            <a href="http://bibliotecadigital.econ.uba.ar/download/Pe/181738.pdf" target="_blank">
                <em>Researches Into the Mathematical Principles of the Theory of Wealth Economic</em>
            </a>, destaca la necesidad de un punto de referencia para medir las variaciones de un sistema.
        </div>
        ''', unsafe_allow_html=True)
        st.markdown(f'3. La <span style="color:#C2185B">Inteligencia Artificial Generativa</span> sostiene que la ciencia necesita puntos de referencia para medir fenómenos de manera precisa.', unsafe_allow_html=True)
        st.markdown(f'En este contexto, esta aplicación constituye un marco de referencia para ubicar a los municipios en su ciclo de madurez digital.', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: right;">Rodrigo Guarneros Gutiérrez<br><span style="color:#51C622">Ciudad de México, 20.12.2024</span></div>', unsafe_allow_html=True)
    with st.expander('¿Qué es la madurez digital?', expanded=False):
        st.markdown(f'Existen diversos modelos para evaluar la madurez digital. Esta aplicación se basa en el análisis de datos y aprendizaje automático para clasificar a los municipios en 4 etapas.', unsafe_allow_html=True)
        st.image("fuentes/MDM_madurez1.png", caption="Modelo de Madurez Digital", use_column_width=True)
        st.markdown(f'<b style="color:#51C622">Etapa 1 (Inicial):</b> Desempeño bajo en variables relevantes.', unsafe_allow_html=True)
        st.markdown(f'<b style="color:#51C622">Etapa 2 (Desarrollo):</b> Avance en servicios digitales.', unsafe_allow_html=True)
        st.markdown(f'<b style="color:#51C622">Etapa 3 (Definición):</b> Penetración promedio y ecosistema financiero vibrante.', unsafe_allow_html=True)
        st.markdown(f'<b style="color:#51C622">Etapa 4 (Optimización):</b> Plenitud digital y balance en características.', unsafe_allow_html=True)
    with st.expander('¿Cómo utilizar esta aplicación?', expanded=False):
        st.markdown(f'La aplicación cuenta con 5 secciones: Municipio, Madurez Digital, Estadísticas por Grupo, Correlaciones y Geografía.', unsafe_allow_html=True)
        st.image("fuentes/como_utilizar_1.png", caption="Página de Inicio.", use_container_width=True)
        st.image("fuentes/como_utilizar_2.png", caption="Selección de variables para análisis correlacional.", use_container_width=True)
        st.image("fuentes/como_utilizar_3.png", caption="Enfoque de la aplicación y fuentes.", use_container_width=True)

with tab2:
    st.markdown(Titulo_dinamico2, unsafe_allow_html=True)
    with st.expander('Descripción', expanded=False):
        st.markdown(f'Esta sección muestra visualizaciones relevantes para el municipio seleccionado, como ranking, mapa, histogramas y diagramas de caja.', unsafe_allow_html=True)
    col_izq, col_der = st.columns([6, 6])
    with col_izq:
        st.plotly_chart(fig_ranking, width=400, use_container_width=True)
    with col_der:
        st.plotly_chart(cuadro_resumen, width=400, use_container_width=True)
        folium_static(fig_municipio, width=455, height=180)
        with st.expander('Análisis', expanded=False):
            st.markdown(f'El histograma muestra una distribución bimodal, indicando la existencia de dos grupos diferenciados en madurez digital.', unsafe_allow_html=True)
        st.plotly_chart(fig_hist_index, use_container_width=True)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.plotly_chart(fig_boxplot, use_container_width=True)

with tab3:
    st.markdown(Titulo_dinamico3, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: justify;">
            Maximiza la página para visualizar los tres Componentes Principales y sus patrones identificados:
            <br>- PC1: Actividad financiera;
            <br>- PC2: Servicios digitales;
            <br>- PC3: Adopción financiera.
        </div>
        """, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.expander('El significado de cada Componente Principal', expanded=False):
            st.markdown(f'<span style="color:#51C622">Los componentes principales maximizan la varianza de los datos y permiten visualizar combinaciones lineales de las variables.</span>', unsafe_allow_html=True)
        st.plotly_chart(grafico3d, use_container_width=True, height=500)
        with st.expander('Patrones en los clústers', expanded=False):
            st.markdown(f'Los clústers se muestran de forma continua, con transiciones suaves entre etapas de madurez.', unsafe_allow_html=True)
        st.plotly_chart(grafico2d1, use_container_width=True, height=250)
    with col2:
        with st.expander('Estructura de los clústers', expanded=False):
            st.markdown(f'La visualización en 2D muestra fronteras relativamente definidas entre los clústers.', unsafe_allow_html=True)
        st.plotly_chart(grafico2d2, use_container_width=True, height=250)
        with st.expander('Perfil del municipio en cada clúster', expanded=False):
            st.markdown(f'Interpretación de cada clúster según la posición en los componentes principales.', unsafe_allow_html=True)
        st.plotly_chart(grafico2d3, use_container_width=True, height=250)

with tab4:
    st.markdown("¿Qué patrones se encuentran en cada clúster?")
    with st.expander('Recuento por nivel de madurez', expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="madurez-card">
                <br><br>                
                <p><span class="madurez-count">Optimización:</span> <b style="color:#51C622">647</b> municipios</p>
                <p><span class="madurez-count">Definición:</span> <b style="color:#51C622">551</b> municipios</p>
                <p><span class="madurez-count">En desarrollo:</span> <b style="color:#51C622">627</b> municipios</p>
                <p><span class="madurez-count">Inicial:</span> <b style="color:#51C622">631</b> municipios</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.plotly_chart(recuento_clusters, use_container_width=True, height=250)
    st.plotly_chart(boxplots_clusters, use_container_width=True)
    st.plotly_chart(histograma_por_clusters, use_container_width=True)

with tab5:
    st.markdown(Titulo_dinamico, unsafe_allow_html=True)
    with st.expander('Análisis', expanded=False):
        st.markdown(f'El scatter plot permite visualizar las relaciones entre variables, con la R² ajustada y la línea de regresión indicadas.', unsafe_allow_html=True)
    st.plotly_chart(fig_scatter, use_container_width=True, height=500)

with tab6:
    with st.expander('Análisis', expanded=False):
        st.markdown(f'El mapa final muestra la distribución geográfica de la madurez digital y la concentración de municipios en cada clúster.', unsafe_allow_html=True)
    st.plotly_chart(fig_map_final, use_container_width=True, height=500)

# Finalmente, opcionalmente, se puede forzar la recolección de basura
gc.collect()
