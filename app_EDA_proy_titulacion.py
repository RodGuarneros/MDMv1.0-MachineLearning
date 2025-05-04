#######################
# Importar  librerias #
#######################
import streamlit as st
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


# Page configuration
st.set_page_config(
    page_title="Aprendizaje Automático para los Municipios de México",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
####################

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

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)


######################################
# Integración y preparación de Datos #
######################################

# Cargar las variables de entorno
# Conectar con MongoDB

# Cargar variables de entorno

# Función para convertir ObjectId a str

def convert_objectid_to_str(document):
    for key, value in document.items():
        if isinstance(value, ObjectId):
            document[key] = str(value)
    return document

# Función para mostrar el formulario solo una vez
def incrementar_contador_visitas():
    try:
        # Obtener la URI de MongoDB desde los secretos
        mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
        
        # Conexión a MongoDB usando la URI desde los secretos
        client = MongoClient(mongo_uri)
        db = client['Municipios_Rodrigo']
        collection = db['visita']
        
        # Intentar obtener el contador de visitas
        visita = collection.find_one_and_update(
            {"_id": "contador"},  # Usamos un único documento con id 'contador'
            {"$inc": {"contador": 1}},  # Incrementamos el contador
            upsert=True,  # Si no existe el documento, lo crea
            return_document=pymongo.ReturnDocument.AFTER  # Usamos el valor correcto (AFTER)
        )
        
        return visita['contador']  # Devuelve el valor del contador de visitas

    except Exception as e:
        st.error(f"Hubo un error al acceder a la base de datos: {e}")
        raise

# Incrementar contador de visitas
contador_visitas = incrementar_contador_visitas()

# Función para cargar y procesar los datos con cache
@st.cache_data
def bajando_procesando_datos():
    mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
    client = MongoClient(mongo_uri)
    db = client['Municipios_Rodrigo']
    collection = db['datos_finales']

    # Obtener datos y convertir a DataFrame
    datos_raw = collection.find()
    datos = pd.DataFrame(list(map(convert_objectid_to_str, datos_raw)))

    # Asegurarse de que los datos estén en Latin1
    for column in datos.select_dtypes(include=['object']).columns:
        datos[column] = datos[column].apply(lambda x: x.encode('Latin1').decode('Latin1') if isinstance(x, str) else x)

    categorias_orden = ['Optimización', 'Definición', 'En desarrollo', 'Inicial']
    # Limpiar y normalizar la variable Madurez
    # datos['Madurez'] = datos['Madurez'].str.strip()
    
    # Convertir a categoría con orden específico
    datos['Etapa_Madurez'] = pd.Categorical(
        datos['Etapa_Madurez'],
        categories=categorias_orden)
    
    return datos



# Llamar a la función para cargar y procesar los datos
datos = bajando_procesando_datos()
input_datos = datos

# Procesar otras columnas como se mencionaba en el código original
datos['Operadores Escala Pequeña BAF'] = datos['operadores_escal_pequeña_baf']
datos.drop(columns=['operadores_escal_pequeña_baf'], inplace=True)
datos['Penetración BAF (Fibra)'] = datos['penetracion_baf_fibra']
datos.drop(columns=['penetracion_baf_fibra'], inplace=True)

# OBTENIENDO EL DATASET COMPLETO:
@st.cache_data
def bajando_procesando_datos_completos():
    # Obtener la URI de MongoDB desde los secretos
    mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
    
    # Conexión a MongoDB usando la URI desde los secretos
    client = MongoClient(mongo_uri)
    db = client['Municipios_Rodrigo']
    collection = db['completo']

    # Obtener todos los documentos de la colección y convertir ObjectId a str
    datos_raw = collection.find()

    dataset_complete = pd.DataFrame(list(map(convert_objectid_to_str, datos_raw)))
    for column in dataset_complete.select_dtypes(include=['object']).columns:
        dataset_complete[column] = dataset_complete[column].apply(lambda x: x.encode('Latin1').decode('Latin1') if isinstance(x, str) else x)

    # Limpiar los nombres de las columnas eliminando espacios
    dataset_complete.columns = dataset_complete.columns.str.strip()

    return dataset_complete

dataset_complete = bajando_procesando_datos_completos()

# OBTENIENDO X PARA EL TRAINING NORMALIZER:
@st.cache_data
def bajando_procesando_X_entrenamiento():
    # Obtener la URI de MongoDB desde los secretos
    mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
    
    # Conexión a MongoDB usando la URI desde los secretos
    client = MongoClient(mongo_uri)
    db = client['Municipios_Rodrigo']
    collection = db['X_for_training_normalizer']

    # Obtener todos los documentos de la colección y convertir ObjectId a str
    datos_raw = collection.find()
    df = pd.DataFrame(list(map(convert_objectid_to_str, datos_raw)))

    # Limpiar los nombres de las columnas eliminando espacios
    df.columns = df.columns.str.strip()

    return df

df = bajando_procesando_X_entrenamiento()

# OBTENIENDO DF PCA NORMALIZER:
@st.cache_data
def bajando_procesando_df_normalizado():
    # Obtener la URI de MongoDB desde los secretos
    mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
    
    # Conexión a MongoDB usando la URI desde los secretos
    client = MongoClient(mongo_uri)
    db = client['Municipios_Rodrigo']
    collection = db['df_pca_norm']

    # Obtener todos los documentos de la colección y convertir ObjectId a str
    datos_raw = collection.find()
    df_normalizado = pd.DataFrame(list(map(convert_objectid_to_str, datos_raw)))

    # Limpiar los nombres de las columnas eliminando espacios
    df_normalizado.columns = df_normalizado.columns.astype(str).str.strip()

    return df_normalizado

df_normalizado = bajando_procesando_df_normalizado()

# Procesamiento de variables numéricas y categóricas
variable_list_numerica = list(input_datos.select_dtypes(include=['int64', 'float64']).columns)
variable_list_categoricala = list(input_datos.select_dtypes(include=['object', 'category']).columns)
variable_list_municipio = list(input_datos['Lugar'].unique())  # Municipio seleccionado

columns_to_exclude_numeric = ['Cluster2','Unnamed: 0', 'Unnamed: 0.2', 'Unnamed: 0.2', 'cve_edo', 'cve_municipio', 'cvegeo', 'Estratos ICM', 'Estrato IDDM', 'Municipio', 'df1_ENTIDAD', 'df1_KEY MUNICIPALITY', 'df2_Clave Estado', 'df2_Clave Municipio', 'df3_Clave Estado', 'df3_Clave Municipio', 'df4_Clave Estado', 'df4_Clave Municipio']
columns_to_exclude_categorical = ['_id','Lugar', 'Estado2', 'df2_Región', 'df3_Región', 'df3_Tipo de población', 'df4_Región', 'Municipio']

# Numéricas
variable_list_numeric = [col for col in variable_list_numerica if col not in columns_to_exclude_numeric]
# Categóricas
variable_list_categorical = [col for col in variable_list_categoricala if col not in columns_to_exclude_categorical]

# Conectar a MongoDB con caché para los polígonos
@st.cache_resource
def connect_to_mongo(mongo_uri):
    client = MongoClient(mongo_uri)
    return client['Municipios_Rodrigo']

# Obtener el archivo GeoJSON desde MongoDB GridFS con caché
@st.cache_data
def consultando_base_de_datos(_db):  # Cambiar 'db' a '_db' para evitar el error
    fs = GridFS(_db)
    file = fs.find_one({'filename': 'municipios.geojson'})
    if file:
        return file.read()
    return None

# Convertir los datos a GeoDataFrame
def geojson_to_geodataframe(geojson_data):
    return gpd.read_file(BytesIO(geojson_data))

# Conectar a MongoDB
mongo_uri = st.secrets["MONGO"]["MONGO_URI"]  # Usar la URI de MongoDB desde los secretos
db = connect_to_mongo(mongo_uri)

# Obtener el archivo GeoJSON
geojson_data = consultando_base_de_datos(db)

# Convertir a GeoDataFrame si los datos fueron encontrados
geojson = geojson_to_geodataframe(geojson_data) if geojson_data else None

# Si tienes un DataFrame `datos`, realiza la fusión con el GeoDataFrame
if geojson is not None:
    datos.rename(columns={'cvegeo': 'CVEGEO'}, inplace=True)
    datos['CVEGEO'] = datos['CVEGEO'].astype(str).str.zfill(5)
    geojson['CVEGEO'] = geojson['CVEGEO'].astype(str)

    # Fusionar los datos con la geometría
    dataset_complete_geometry = datos.merge(geojson[['CVEGEO', 'geometry']], on='CVEGEO', how='left')


###################################################################################################################
###################################################################################################################
###################################################################################################################

# Sidebar
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
    
    # variable_seleccionada_estado = st.selectbox('Selecciona el estado de tu interés:', sorted(variable_list_estado, reverse=False))
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


######################
# Mapa por Municipio #
######################
def crear_mapa_choropleth2(dataset, estado=None, cluster=None, lugar=None, municipio_inicial="MunicipioX"):
    """
    Crea un mapa choropleth interactivo mostrando clústeres y filtrando por estado, clúster o lugar.
    
    Parámetros:
    - dataset: El dataset con los datos geoespaciales.
    - estado: El estado por el cual filtrar (opcional).
    - clúster: El número de clúster por el cual filtrar (opcional).
    - lugar: El nombre del lugar (municipio) para filtrar (opcional).
    - municipio_inicial: El nombre del municipio inicial para centrar el mapa si no se pasa un lugar.
    """
    # Convertir el dataset a GeoDataFrame si aún no lo es
    gdf = gpd.GeoDataFrame(dataset, geometry='geometry')

    # Filtrar por 'Estado' si se pasa como parámetro
    if estado:
        gdf = gdf[gdf['Estado'] == estado]
    
    # Filtrar por 'Cluster' si se pasa como parámetro
    if cluster is not None:
        gdf = gdf[gdf['Clústers'] == cluster]
    
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

    # Función para obtener el color según el valor del 'Clústers'
    def obtener_color(cluster_value):
        return mapa_colores.get(cluster_value, '#FFFFFF')

    # Añadir la capa GeoJson con los colores personalizados y tooltips
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
            sticky=True  # Hace que el tooltip sea permanente
        ),
        highlight_function=lambda x: {'fillOpacity': 0.9}  # Resalta al pasar el mouse
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
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend))

    return m

# Llamar a la función con Streamlit y folium_static
# Aquí asumo que 'dataset_complete_geometry' es tu dataset y 'variable_seleccionada_municipio' es el lugar seleccionado
fig_municipio = crear_mapa_choropleth2(dataset_complete_geometry, lugar=variable_seleccionada_municipio, municipio_inicial="Abalá, Yucatán")

##############
## Ranking ###
##############
def plot_bar_chart(data, lugar_columna, indice_columna, lugar_seleccionado):
    """
    Genera una gráfica de barras horizontal con precisión completa en los valores del índice.
    
    Args:
        data (pd.DataFrame): El DataFrame que contiene los datos.
        lugar_columna (str): Nombre de la columna con los lugares.
        indice_columna (str): Nombre de la columna con los índices a graficar.
        lugar_seleccionado (str): Lugar que será resaltado en rojo.
    """
    # Crear una copia del DataFrame y asegurar tipo numérico
    plot_data = data.copy()
    plot_data[indice_columna] = pd.to_numeric(plot_data[indice_columna], errors='coerce')
    
    # Ordenar por índice compuesto de menor a mayor
    plot_data = plot_data.sort_values(by=indice_columna, ascending=True)
    
    # Crear la lista de colores para las barras
    bar_colors = ['red' if lugar == lugar_seleccionado else 'dodgerblue' 
                 for lugar in plot_data[lugar_columna]]
    
    # Crear la gráfica usando graph_objects
    fig = go.Figure()
    
    # Añadir las barras con formato de hover personalizado y bordes blancos
    fig.add_trace(go.Bar(
        x=plot_data[indice_columna],
        y=plot_data[lugar_columna],
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(
                color='white',  # Color del borde
                width=0.5      # Ancho del borde
            )
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
    
    # Crear anotaciones para personalizar los nombres y rankings
    annotations = []
    for lugar, ranking, valor in zip(plot_data[lugar_columna], 
                                   plot_data["Ranking"], 
                                   plot_data[indice_columna]):
        # Nombre del lugar
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
        # Ranking y valor preciso del índice
        annotations.append(dict(
            x=valor, y=lugar,
            text=f"{int(ranking)} ({valor:.10f})",
            showarrow=False,
            font=dict(
                color='white',
                size=7
            ),
            xanchor='left',
            xshift=5
        ))
    
    # Ajustar la altura dinámica
    num_lugares = len(plot_data)
    height = max(400, num_lugares * 18)
    
    # Actualizar el layout
    fig.update_layout(
        title=dict(
            text=f"Índice de Madurez por Municipio (Resaltado: {lugar_seleccionado})",
            font=dict(color='#FFD86C')  # Color dorado para el título
        ),
        xaxis_title=dict(
            text="Índice de Madurez",
            font=dict(color='#FFD86C')  # Color dorado para el título del eje x
        ),
        yaxis_title=dict(
            text="Municipio",
            font=dict(color='#FFD86C')  # Color dorado para el título del eje y
        ),
        height=height,
        margin=dict(l=200, r=20, t=70, b=50),
        showlegend=False,
        xaxis=dict(
            range=[0, plot_data[indice_columna].max() * 1.1],
            tickformat='.10f',  # Mostrar más decimales en el eje x
            showgrid=False     # Opcional: remover la cuadrícula
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False     # Opcional: remover la cuadrícula
        ),
        annotations=annotations,
        bargap=0.2,  # Espacio entre barras; valores altos las hacen más delgadas
        plot_bgcolor='rgba(0, 0, 0, 0.1)',  # Fondo transparente
        paper_bgcolor='rgba(0, 0, 0, 0)'    # Fondo transparente
    )
    
    return fig

fig_ranking = plot_bar_chart(data=datos,lugar_columna='Lugar', indice_columna='Índice_Compuesto', lugar_seleccionado=variable_seleccionada_municipio)


########################
#  Posición en ranking #
########################
def crear_display(data, lugar_seleccionado):
    """
    Crea una figura simple que muestra el ranking del lugar seleccionado,
    con color basado en su etapa de madurez.
    
    Args:
        data (pd.DataFrame): El DataFrame que contiene los datos.
        lugar_seleccionado (str): Lugar que será resaltado.
    
    Returns:
        go.Figure: Figura que contiene el ranking con el color correspondiente.
    """
    
    # Mapa de colores personalizado para los clústeres
    mapa_colores = {
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7',
        'Definición': '#CC6CE7',
        'Optimización': '#51C622',
    }
    
    # Filtrar la fila del lugar seleccionado
    lugar_row = data[data['Lugar'] == lugar_seleccionado]
    if lugar_row.empty:
        return None
    
    # Obtener el valor del ranking y la etapa de madurez
    lugar_ranking = lugar_row['Ranking'].iloc[0]
    etapa_madurez = lugar_row['Etapa_Madurez'].iloc[0]
    
    # Determinar el color según la etapa de madurez
    color_rect = mapa_colores.get(etapa_madurez, 'dodgerblue')
    
    # Crear la figura
    fig = go.Figure()
    
    # Añadir el rectángulo de fondo con esquinas redondeadas
    fig.add_shape(
        type="path",
        path="M 0,0 Q 0,0 0.1,0 L 0.9,0 Q 1,0 1,0.1 L 1,0.9 Q 1,1 0.9,1 L 0.1,1 Q 0,1 0,0.9 Z",
        fillcolor=color_rect,
        line=dict(width=0),
        xref="paper", yref="paper",  # Hace que el tamaño del rectángulo sea relativo
        layer="below",  # Se asegura de que esté al fondo
        opacity=1
    )
    
    # Añadir el texto del ranking
    fig.add_annotation(
        text="Lugar en el Ranking de 2,456 municipios en México",
        x=0.5,
        y=0.80,
        showarrow=False,
        font=dict(
            family="Arial",
            size=12,
            color="#050505"
        ),
        align="center"
    )
    
    # Añadir el número del ranking
    fig.add_annotation(
        text=str(int(lugar_ranking)),
        x=0.5,
        y=0.35,
        showarrow=False,
        font=dict(
            family="Arial",
            size=37,
            color="#050505"
        ),
        align="center"
    )
    
    # Actualizar el layout para que sea más compacto
    fig.update_layout(
        width=200,
        height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=color_rect,
        plot_bgcolor=color_rect,
        showlegend=False,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            range=[0, 1]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            range=[0, 1]
        )
    )
    
    return fig

cuadro_resumen = crear_display(datos, lugar_seleccionado=variable_seleccionada_municipio)

##############
# Histograma #
##############
def plot_histogram(df, numeric_column, categorical_column):
    """
    Elaborada por Rodrigo Guarneros
    """
    # Definir un mapa de colores personalizado para los clústeres
    color_map = {
        'En desarrollo': '#D20103',    # Cluster 0 -> Rojo
        'Inicial': '#5DE2E7',          # Cluster 1 -> Turquesa
        'Definición': '#CC6CE7',       # Cluster 2 -> Amarillo
        'Optimización': '#51C622',     # Cluster 3 -> Verde oscuro
    }

    # Crear el histograma
    fig = px.histogram(
        df, 
        x=numeric_column, 
        color=categorical_column,
        color_discrete_map=color_map,
        opacity=0.6,
        title=f'Histograma de la variable "{numeric_column}" y <br>la categoría "{categorical_column}"'
    )
    
    # Actualizar títulos de los ejes
    fig.update_yaxes(title_text="Frecuencia absoluta")
    
    # Calcular estadísticos descriptivos
    stats = {
        'Media': df[numeric_column].mean(),
        'Mediana': df[numeric_column].median(),
        'Moda': df[numeric_column].mode()[0],
        'Desviación estándar': df[numeric_column].std()
    }
    
    # Crear texto agrupado para las anotaciones
    stats_text = "<br>".join([f"<b>{key}</b>: {value:.2f}" for key, value in stats.items()])

    # Añadir conteo total por categoría
    category_counts = df[categorical_column].value_counts()
    counts_text = "<br>".join([f"<b>{category}</b>: {count}" for category, count in category_counts.items()])
    
    # Recuadro de anotaciones
    annotations_text = f"{stats_text}<br><br><b>Conteo por categoría:</b><br>{counts_text}"
    
    # Configurar la posición del recuadro de anotaciones
    annotations = [
        dict(
            x=1.1,  # Centrar horizontalmente
            y=0.9,  # Ubicar debajo de la leyenda
            xref='paper',
            yref='paper',
            text=annotations_text,
            showarrow=False,
            font=dict(color='white', size=12),
            align='center',
            bgcolor='rgba(0, 0, 0, 0.7)',  # Fondo oscuro
            bordercolor='white',  # Borde blanco
            borderwidth=1,  # Ancho del borde
            opacity=0.8  # Opacidad del recuadro
        )
    ]
    
    # Actualizar diseño para incluir leyenda y recuadro
    fig.update_layout(
        title_font=dict(color='#FFD86C', size=16),
        title_x=0.05,  # Centrar título
        showlegend=True,
        width=1350,
        height=500,  # Altura ajustada para espacio de anotaciones
        margin=dict(l=50, r=50, t=80, b=200),  # Márgenes ajustados
        annotations=annotations,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.3,  # Posicionar leyenda debajo de la gráfica
            xanchor='center',
            x=0.5,   # Centrar leyenda
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


fig_hist = plot_histogram(input_datos, variable_seleccionada_numerica, variable_seleccionada_categorica)


#####################
## Histograma Dens ##
#####################

def plot_histogram_with_density(df, numeric_column, selected_value=None):
    """
    Crea un histograma con línea de densidad, bordes en las barras, y destaca un punto específico.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        numeric_column (str): Nombre de la columna numérica para el histograma.
        selected_value (float, optional): Valor seleccionado para resaltar en el gráfico.

    Returns:
        plotly.graph_objects.Figure: Objeto de figura del histograma.
    """
    # Crear el histograma con bordes blancos
    fig = px.histogram(
        df,
        x=numeric_column,
        opacity=0.6,
        title=f'Distribución del índice de madurez digital',
        nbins=50,  # Aumentar el número de bins
        labels={'x': 'Valores del Índice', 'y': 'Frecuencia'}  # Añadido aquí
    )
    fig.update_traces(marker_line_color='white', marker_line_width=1.5)

    # Calcular la densidad usando KDE
    hist_data = df[numeric_column].dropna().astype(float)
    kde = gaussian_kde(hist_data)
    density_x = np.linspace(hist_data.min(), hist_data.max(), 1000)
    density_y = kde(density_x)
    density_y_scaled = density_y * len(hist_data) * (hist_data.max() - hist_data.min()) / 50

    # Agregar la línea de densidad
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
    
    # Calcular estadísticos descriptivos
    mean = hist_data.mean()
    std = hist_data.std()
    median = hist_data.median()
    mode = hist_data.mode()[0]
    
    # Crear el texto de las anotaciones
    annotation_text = (
        f"<b>Estadísticos:</b><br>"
        f"Media: {mean:.2f}<br>"
        f"Mediana: {median:.2f}<br>"
        f"Moda: {mode:.2f}<br>"
        f"Desv. Est.: {std:.2f}"
    )
    
    # Añadir las anotaciones
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

    # Estilo del gráfico
    fig.update_layout(
        title_font=dict(color='#FFD86C'),
        xaxis_title_font=dict(color='#FFD86C'),
        yaxis_title_font=dict(color='#FFD86C'),
        legend=dict(title_text='Leyenda', font=dict(color='#FFD86C')),
        xaxis=dict(
            showgrid=False,
            title='Valores del Índice'  # Actualizado aquí también
        ),
        yaxis=dict(
            showgrid=False,
            title='Frecuencia'  # Actualizado aquí también
        ),
        plot_bgcolor='rgba(0, 0, 0, 0.1)',
    )

    return fig


fig_hist_index = plot_histogram_with_density(input_datos, numeric_column='Índice_Compuesto', selected_value=variable_seleccionada_municipio)

######################
##### BOX PLOT #######
######################
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
        points=False,  # No mostrar puntos en el boxplot
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

    
fig_boxplot = generate_boxplot_with_annotations(input_datos, variable_seleccionada_numerica, variable_seleccionada_municipio)    

#################
## 3D plot PCA ##
#################
def generar_grafico_3d_con_lugar(df, df_normalizado, dataset_complete, lugar_seleccionado=None):
    # Primero, asegurarse que los valores de Madurez estén limpios y sean consistentes
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }
    
    # Normalización de PCA
    df_pca2 = df_normalizado.to_numpy()
    df_pca2 = df_pca2[:, 1:4]

    # Crear DataFrame para Plotly
    pca_df = pd.DataFrame(df_pca2, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df['Etapa_Madurez'] = df['Etapa_Madurez']  # Usar la versión categorizada
    pca_df['Lugar'] = dataset_complete['Lugar']

    # Crear el gráfico asegurando el orden y los colores
    fig = px.scatter_3d(
        pca_df, 
        x='PCA1', y='PCA2', z='PCA3',
        color='Etapa_Madurez',
        labels={'PCA1': 'Componente PC1', 
                'PCA2': 'Componente PC2', 
                'PCA3': 'Componente PC3'},
        hover_data=['Lugar'],
        #category_orders={'Etapa_Madurez': ['Optimización', 'Definición', 'En desarrollo', 'Inicial']},
        color_discrete_map=color_map
    )
    # Manejar lugar seleccionado
    if lugar_seleccionado:
        lugar_df = pca_df[pca_df['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            # Agregar los puntos del lugar seleccionado al gráfico y cambiar su color y tamaño
            fig.add_trace(
                px.scatter_3d(lugar_df, 
                             x='PCA1', y='PCA2', z='PCA3', hover_data=['Lugar'],
                             color_discrete_map={'Madurez': 'green'}).data[0]
            )
            fig.update_traces(marker=dict(size=20, color='green', opacity=1), 
                            selector=dict(name=lugar_seleccionado))

    # Actualizar estilo de los marcadores
    fig.update_traces(
        marker=dict(
            size=6,
            opacity=0.7,
            line=dict(
                width=0.02,
                color='gray'
            )
        )
    )

    # Actualizar layout
    fig.update_layout(
        title="Municipios por grado de madurez multidimensional",
        title_x=0.05,  # Centrar el título
        showlegend=True,  # Asegurar que la leyenda esté visible
        legend=dict(
            title=dict(text='Madurez'),  # Título de la leyenda
            itemsizing='constant',  # Tamaño constante para los elementos de la leyenda
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
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            zaxis_showgrid=True
        ),
        font=dict(color='white'),
        paper_bgcolor='rgb(0, 0, 0)',
        plot_bgcolor='rgb(0, 0, 0)',
    )

    return fig


grafico3d = generar_grafico_3d_con_lugar(datos, df_normalizado, dataset_complete, lugar_seleccionado=variable_seleccionada_municipio)

###################
### Gráfico 2D 1###
###################


def generar_grafico_2d(df, df_normalizado, dataset_complete, lugar_seleccionado=None):
    
    # Normalización de PCA
    df_pca2 = df_normalizado.to_numpy()
    df_pca2 = df_pca2[:, 1:4]

    # Crear DataFrame para Plotly
    pca_df = pd.DataFrame(df_pca2, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df['Etapa_Madurez'] = df['Madurez'].astype('category')
    pca_df['Lugar'] = dataset_complete['Lugar']

    # Definir un mapa de colores estricto
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }

    # Crear el gráfico de dispersión 2D
    fig = px.scatter(pca_df, 
                     x='PCA1', y='PCA2',
                     color='Etapa_Madurez',
                     labels={'PCA1': 'Componente PC1', 
                            'PCA2': 'Componente PC2'},
                     hover_data=['Lugar'],
                    #  category_orders={'Etapa_Madurez': ['Optimización', 'Definición', 'En desarrollo', 'Inicial']},  # Orden explícito
                     color_discrete_map=color_map)

    # Manejar lugar seleccionado
    if lugar_seleccionado:
        lugar_df = pca_df[pca_df['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            # Agregar los puntos del lugar seleccionado al gráfico y cambiar su color y tamaño
            fig.add_trace(
                px.scatter(lugar_df, 
                           x='PCA1', y='PCA2', hover_data=['Lugar'],
                           color_discrete_map={'Madurez': 'green'}).data[0]
            )
            fig.update_traces(marker=dict(size=10, color='green', opacity=1), 
                             selector=dict(name=lugar_seleccionado))

    # Actualizar estilo de los marcadores
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(
                width=0.02,
                color='gray'
            )
        )
    )

    # Actualizar layout
    fig.update_layout(
        title="PC2 vs. PC1 (2D)",
        title_x=0.3,  # Centrar el título
        showlegend=True,  # Asegurar que la leyenda esté visible
        legend=dict(
            title=dict(text='Madurez'),  # Título de la leyenda
            itemsizing='constant',  # Tamaño constante para los elementos de la leyenda
            font=dict(color='white'),
        ),
        font=dict(color='white'),
        paper_bgcolor='rgb(0, 0, 0)',
        plot_bgcolor='rgb(0, 0, 0)',
    )

    return fig


grafico2d1 = generar_grafico_2d(df, df_normalizado, dataset_complete, lugar_seleccionado=variable_seleccionada_municipio)

###################
### Gráfico 2D 2###
###################


def generar_grafico_2d2(df, df_normalizado, dataset_complete, lugar_seleccionado=None):
    # Limpiar posibles espacios o caracteres invisibles en 'Madurez'
    df['Etapa_Madurez'] = df['Etapa_Madurez'].astype('category')
    
    # Normalización de PCA
    df_pca2 = df_normalizado.to_numpy()
    df_pca2 = df_pca2[:, 1:4]  # Selección de las primeras tres componentes principales

    # Crear DataFrame para Plotly
    pca_df = pd.DataFrame(df_pca2, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df['Etapa_Madurez'] = df['Etapa_Madurez']
    pca_df['Lugar'] = dataset_complete['Lugar']

    # Definir un mapa de colores más contrastante
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }

    # Crear el gráfico asegurando consistencia en los colores
    fig = px.scatter(
        pca_df,
        x='PCA1',
        y='PCA3',
        labels={'PCA1': 'Componente PC1', 'PCA3': 'Componente PC3'},
        hover_data=['Lugar'],  # Información adicional en el hover
        color='Etapa_Madurez',  # <- Especificar la columna para asignar colores
        # category_orders={'Etapa_Madurez': ['Optimización', 'Definición', 'En desarrollo', 'Inicial']},
        color_discrete_map=color_map  # Asignar colores específicos a las categorías
    )

    # Manejar lugar seleccionado
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

    # Ajustar el estilo del gráfico
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(
                width=0.5,
                color='gray'
            )
        )
    )

    # Actualizar el layout del gráfico
    fig.update_layout(
        title="PC1 vs. PC3 (2D)",
        title_x=0.5,  # Centrar el título
        showlegend=True,
        legend=dict(
            title=dict(text='Etapa de Madurez'),
            itemsizing='constant'
        ),
        paper_bgcolor='rgb(0, 0, 0)',
        plot_bgcolor='rgb(0, 0, 0)',
        font=dict(color='white')
    )

    return fig




grafico2d2 = generar_grafico_2d2(datos, df_normalizado, dataset_complete, lugar_seleccionado=variable_seleccionada_municipio)

###################
### Gráfico 2D 3###
###################


def generar_grafico_2d3(df, df_normalizado, dataset_complete, lugar_seleccionado=None):
    
    # Normalización de PCA
    df_pca2 = df_normalizado.to_numpy()
    df_pca2 = df_pca2[:, 1:4]

    # Crear DataFrame para Plotly
    pca_df = pd.DataFrame(df_pca2, columns=['PCA1', 'PCA2', 'PCA3'])
    pca_df['Etapa_Madurez'] = df['Madurez'].astype('category')
    pca_df['Lugar'] = dataset_complete['Lugar']

    # Definir un mapa de colores estricto
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }

    # Crear el gráfico de dispersión 2D
    fig = px.scatter(pca_df, 
                     x='PCA2', y='PCA3',
                     color='Etapa_Madurez',
                     labels={'PCA2': 'Componente PC2', 
                            'PCA3': 'Componente PC3'},
                     hover_data=['Lugar'],
                    #  category_orders={'Etapa_Madurez': ['Optimización', 'Definición', 'En desarrollo', 'Inicial']},  # Orden explícito
                     color_discrete_map=color_map)

    # Manejar lugar seleccionado
    if lugar_seleccionado:
        lugar_df = pca_df[pca_df['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            # Agregar los puntos del lugar seleccionado al gráfico y cambiar su color y tamaño
            fig.add_trace(
                px.scatter(lugar_df, 
                           x='PCA2', y='PCA3', hover_data=['Lugar'],
                           color_discrete_map={'Madurez': 'green'}).data[0]
            )
            fig.update_traces(marker=dict(size=10, color='green', opacity=1), 
                             selector=dict(name=lugar_seleccionado))

    # Actualizar estilo de los marcadores
    fig.update_traces(
        marker=dict(
            size=8,
            opacity=0.7,
            line=dict(
                width=0.02,
                color='gray'
            )
        )
    )

    # Actualizar layout
    fig.update_layout(
        title="PC3 vs. PC2 (2D)",
        title_x=0.3,  # Centrar el título
        showlegend=True,  # Asegurar que la leyenda esté visible
        legend=dict(
            title=dict(text='Madurez'),  # Título de la leyenda
            itemsizing='constant',  # Tamaño constante para los elementos de la leyenda
            font=dict(color='white'),
        ),
        font=dict(color='white'),
        paper_bgcolor='rgb(0, 0, 0)',
        plot_bgcolor='rgb(0, 0, 0)',
    )

    return fig


grafico2d3 = generar_grafico_2d3(df, df_normalizado, dataset_complete, lugar_seleccionado=variable_seleccionada_municipio)


#########################
### Box plots by group ##
#########################

def boxplot_por_cluster(df, variable):
    """
    Genera un único boxplot con todos los puntos, coloreados según el clúster.
    El tooltip muestra el 'lugar', la media, mediana y desviación estándar del clúster correspondiente.
    
    Parameters:
        df (pd.DataFrame): El DataFrame de entrada.
        variable (str): La columna para analizar.
    """
    # Definir el nuevo mapa de colores para los clusters
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }

    # Calcular estadísticas para cada cluster y agregarlas al DataFrame
    stats = df.groupby('Madurez')[variable].agg(['mean', 'median', 'std']).reset_index()
    stats.rename(columns={'mean': 'mean_' + variable, 'median': 'median_' + variable, 'std': 'std_' + variable}, inplace=True)
    df = pd.merge(df, stats, on='Madurez', how='left')

    # Crear el boxplot con todos los puntos
    fig = px.box(
        df,
        y=variable,
        points='all',
        title=f'Diagrama de caja de la variable\n"{variable}"',
        labels={variable: variable},
        template='plotly_dark',
        color='Madurez',  # Colorear los puntos según el cluster
        color_discrete_map=color_map,  # Usar el mapa de colores actualizado
        hover_data={
            'Madurez': True, 
            'Lugar': True,  # Mostrar el lugar en el tooltip
            'mean_' + variable: True,  # Mostrar la media en el tooltip
            'median_' + variable: True,  # Mostrar la mediana en el tooltip
            'std_' + variable: True,  # Mostrar la desviación estándar en el tooltip
        }
    )

    # Actualizar las trazas para los bordes de los puntos (naranja claro)
    fig.update_traces(marker=dict(
        opacity=0.6,
        line=dict(color='rgba(255, 165, 0, 0.5)', width=1)  # Borde en color naranja claro
    ))
    # Mostrar el gráfico
    return fig

boxplots_clusters = boxplot_por_cluster(datos, variable_seleccionada_numerica)

################################
### Histrograma por cluster ####
################################

def plot_histogram(df, numeric_column):
    """
    Crea un histograma de superposición para cada clúster, usando colores basados en el mapa de colores proporcionado,
    y agrega recuadros con las estadísticas distribuidos de manera organizada.
    
    Parameters:
        df (pd.DataFrame): El DataFrame de entrada.
        numeric_column (str): La columna numérica para el histograma.
    """
    # Mapa de colores para los clusters
    color_map = {
        'Optimización': '#51C622',
        'Definición': '#CC6CE7',
        'En desarrollo': '#D20103',
        'Inicial': '#5DE2E7'
    }
    
    # Crear el histograma con superposición por cluster
    fig = px.histogram(df, 
                      x=numeric_column, 
                      color='Madurez',
                      color_discrete_map=color_map,
                      opacity=0.6,
                      title=f'Histograma de la variable "{numeric_column}"')
    
    # Actualizar los ejes
    fig.update_xaxes(title_text="Rangos de valor")
    fig.update_yaxes(title_text="Frecuencia absoluta")
    
    # Calcular estadísticas por cada nivel de madurez
    annotations = []
    
    # Definir posiciones para los recuadros (dos columnas)
    positions = [
        {'x': 1.15, 'y': 1.33},  # Primera columna, arriba
        {'x': 1.15, 'y': 1},  # Primera columna, abajo
        {'x': 1.15, 'y': 0.50},  # Segunda columna, arriba
        {'x': 1.15, 'y': 0.02}   # Segunda columna, abajo
    ]
    
    for i, level in enumerate(df['Madurez'].unique()):
        # Filtrar los datos por nivel de madurez
        subset = df[df['Madurez'] == level]
        
        # Calcular las estadísticas
        mean = subset[numeric_column].mean()
        median = subset[numeric_column].median()
        mode = subset[numeric_column].mode()[0]
        std = subset[numeric_column].std()
        
        # Crear un recuadro con las estadísticas
        annotations.append(dict(
            x=positions[i]['x'],
            y=positions[i]['y'],
            xref='paper',
            yref='paper',
            text=f'<b>{level}</b><br>Media: {mean:.2f}<br>Mediana: {median:.2f}<br>Moda: {mode:.2f}<br>Desviación estándar: {std:.2f}',
            showarrow=False,
            font=dict(size=10, color='black'),
            bgcolor=color_map[level],
            borderpad=4,
            opacity=0.8,
            align="left",
            width=150
        ))
    
    # Añadir todas las anotaciones al gráfico
    for annotation in annotations:
        fig.add_annotation(annotation)
    
    # Actualizar el layout para dar espacio a las anotaciones
    fig.update_layout(
        template='plotly_dark',
        showlegend=False,
        margin=dict(r=250),  # Aumentar el margen derecho para las anotaciones
        height=400  # Aumentar la altura para mejor distribución
    )
    
    return fig

histograma_por_clusters = plot_histogram(datos, variable_seleccionada_numerica)

##############
### Scatter ##
##############

def generate_scatter_with_annotations(df, x_variable, y_variable, categorical_variable):
    """
    Generates a scatter plot with annotations including adjusted R² and regression line equation.
    Handles missing values in the dataset.
    """
    # Drop rows with missing values in relevant columns
    df_clean = df.dropna(subset=[x_variable, y_variable])

    # Define a custom color map for clusters
    color_map = {
        'En desarrollo': '#D20103',    # Cluster 0 -> Rojo
        'Inicial': '#5DE2E7',    # Cluster 1 -> Turquesa
        'Definición': '#CC6CE7',    # Cluster 2 -> Amarillo
        'Optimización': '#51C622',    # Cluster 3 -> Verde oscuro
    }

    # Create the scatter plot
    fig = px.scatter(
        df_clean,
        x=x_variable,
        y=y_variable,
        hover_data={'Lugar': True, categorical_variable: True},
        color=categorical_variable,
        color_discrete_map=color_map
    )

    # Compute adjusted R² and regression line equation
    X = df_clean[[x_variable]].values
    y = df_clean[y_variable].values
    model = LinearRegression()
    model.fit(X, y)

    intercept = model.intercept_
    slope = model.coef_[0]
    r_squared = model.score(X, y)
    n = len(df_clean)
    p = 1  # Only one independent variable
    r_squared_adj = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))

    # Regression line equation
    regression_equation = f"y = {slope:.2f}x + {intercept:.2f}"

    # Add the regression line to the scatter plot
    x_range = np.linspace(df_clean[x_variable].min(), df_clean[x_variable].max(), 100)
    y_predicted = slope * x_range + intercept
    fig.add_scatter(
        x=x_range,
        y=y_predicted,
        mode='lines',
        name='Regression Line',
        line=dict(color='orange', dash='dash')
    )

    # Update layout
    fig.update_layout(
        plot_bgcolor='rgb(30,30,30)',  # Dark background
        paper_bgcolor='rgb(30,30,30)',  # Dark paper background
        font_color='white',  # White font
        title=dict(
            text=f"Scatter Plot: '{x_variable}' vs '{y_variable}'",
            font=dict(color='white')  # Title font color
        ),
        xaxis=dict(
            title=f"Variable: {x_variable}",
            titlefont=dict(color='white'),
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            title=f"Variable: {y_variable}",
            titlefont=dict(color='white'),
            tickfont=dict(color='white')
        ),
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=0.95,
                y=1.05,
                text=f'R² Ajustada: {r_squared_adj:.4f}',
                showarrow=False,
                font=dict(color='orange')
            ),
            dict(
                xref='paper',
                yref='paper',
                x=0.05,
                y=1.05,
                text=f'Regresión: {regression_equation}',
                showarrow=False,
                font=dict(color='orange')
            )
        ]
    )

    # Customize hover template
    fig.update_traces(
        hovertemplate='<b>Municipio</b>: %{customdata[0]}<br>' +
                      f'<b>{x_variable}</b>: %{{x}}<br>' +
                      f'<b>{y_variable}</b>: %{{y}}<br>'
    )
    fig.update_traces(
        marker=dict(opacity=0.9, line=dict(color='rgba(255, 165, 0, 0.5)', width=1))
    )

    return fig

# Call the function to generate scatter plot
fig_scatter = generate_scatter_with_annotations(input_datos, variable_seleccionada_numerica, variable_seleccionada_paracorrelacion, variable_seleccionada_categorica)

##################################
###### Mapa completo #############
##################################

def generar_mapa_con_lugar(df, lugar=None):
    # Definir el mapa de colores para los clústeres

    color_map = {
        'En desarrollo': '#D20103',    # Cluster 0 -> Rojo
        'Inicial': '#5DE2E7',    # Cluster 1 -> Turquesa
        'Definición': '#CC6CE7',    # Cluster 2 -> Amarillo
        'Optimización': '#51C622',    # Cluster 3 -> Verde oscuro
    }

    # Asegurarse de que 'Cluster2' sea categórico
    df['Madurez'] = df['Madurez'].astype('category')

    # Crear el mapa con Plotly usando scatter_mapbox
    fig = px.scatter_mapbox(
        df,
        lat="Latitud",
        lon="Longitud",
        color="Madurez",  # Usamos 'Cluster2' para definir el color
        opacity=0.8,
        hover_data=["Madurez", "Lugar"],  # Mostrar información al pasar el cursor
        zoom=4,  # Nivel de zoom inicial
        center={"lat": 23.6345, "lon": -102.5528},  # Coordenadas centrales de México
        title="Mapa de Clústers por Madurez Digital en México",
        color_discrete_map=color_map  # Aplicar el mapa de colores definido
    )

    # Resaltar el lugar seleccionado si se proporciona el parámetro 'lugar'
    if lugar:
        lugar_df = df[df['Lugar'] == lugar]
        if not lugar_df.empty:
            # Añadir un marcador especial para el lugar seleccionado
            fig.add_trace(
                px.scatter_mapbox(
                    lugar_df,
                    lat="Latitud",
                    lon="Longitud",
                    color_discrete_map={0: '#ffa500', 1: '#ffa500', 2: '#ffa500', 3: 'ffa500'},  # Resaltar en amarillo
                    size_max=10,  # Tamaño máximo de marcador
                    size=[8],  # Tamaño del marcador en puntos
                    hover_data=["Madurez", "Lugar"]
                ).data[0]
            )

    # Configurar el estilo del mapa a uno oscuro y ajustar diseño
    fig.update_layout(
        mapbox_style="carto-darkmatter",  # Estilo oscuro del mapa
        height=600,  # Altura del mapa
        margin={"r": 0, "t": 50, "l": 0, "b": 0},  # Márgenes del gráfico
        legend=dict(
            title="Nivel de Madurez",  # Título de la leyenda
            itemsizing="constant",  # Tamaño constante en ítems
            traceorder="normal"  # Orden normal de la leyenda
        )
    )

    # Retornar la figura para usar en Streamlit
    return fig

fig_map_final = generar_mapa_con_lugar(input_datos, lugar = variable_seleccionada_municipio)


###################################
#### Recuento de Clusters #########
###################################
def recuento(df):
    # Contar el total de registros en la columna 'Lugar'
    total_municipios = len(df)

    # Contar el número de registros por cada nivel de madurez
    counts = df['Madurez'].value_counts()

    # Calcular la frecuencia relativa
    df_counts = counts.reset_index()
    df_counts.columns = ['Madurez', 'Cantidad']
    df_counts['Frecuencia relativa'] = df_counts['Cantidad'] / total_municipios

    # Definir el color map personalizado
    color_map = {
        'En desarrollo': '#D20103',    # Cluster 0 -> Rojo
        'Inicial': '#5DE2E7',          # Cluster 1 -> Turquesa
        'Definición': '#CC6CE7',       # Cluster 2 -> Amarillo
        'Optimización': '#51C622',     # Cluster 3 -> Verde oscuro
    }

    # Crear el gráfico de barras para la frecuencia relativa usando Plotly
    fig = px.bar(df_counts, 
                 x='Madurez', 
                 y='Frecuencia relativa', 
                 title="Frecuencia relativa por nivel de madurez",
                 labels={'Frecuencia relativa': 'Frecuencia relativa', 'Nivel de madurez': 'Nivel de madurez'},
                 color='Madurez', 
                 color_discrete_map=color_map,  # Usar el color map personalizado
                 category_orders={'Madurez': ['Inicial', 'En desarrollo', 'Definición', 'Optimización']},  # Cambiar el orden
                 height=280)  # Reducir la altura del gráfico
    
    return fig

recuento_clusters = recuento(datos)


##################################
### Título Dinámico Variable #####
##################################
def titulo_dinamico(variable):

    # Set a yellow color for the title
    styled_title = f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">La variable mostrada es: "{variable}".</span>'

    return styled_title

Titulo_dinamico = titulo_dinamico(variable=variable_seleccionada_numerica)


###################################
### Título Dinámico Municipio #####
###################################
def titulo_dinamico2(variable):

    # Set a yellow color for the title
    styled_title = f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">Municipio de "{variable}".</span>'

    return styled_title

Titulo_dinamico2 = titulo_dinamico2(variable=variable_seleccionada_municipio)

###########################################
### Título Dinámico Municipio Madurez #####
###########################################
def titulo_dinamico3(variable):

    # Set a yellow color for the title
    styled_title = f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">Análisis de Madurez Digital de "{variable}".</span>'

    return styled_title

Titulo_dinamico3 = titulo_dinamico3(variable=variable_seleccionada_municipio)



# Dashboard Main Panel
# calculos_df
# Define the tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Presentación", "Municipio", "Madurez Digital","Estadísiticas por Grupo", "Análisis Relacional","Geografía"])



# Aquí asumo que ya tienes definidos los gráficos:
# fig_municipio: tu gráfico de Folium (mapa)
# fig_boxplot: tu gráfico de Plotly (box plot)
# fig_hist: tu gráfico de Plotly (histograma)

# Crear pestaña con las visualizaciones

with tab1:
    # Expander con información adicional
    with st.expander('¿Para qué sirve esta aplicación?', expanded=False):
        st.markdown(f'Provee un punto de referencia estadísticamente robusto, claro y preciso —con un criterio basado en aprendizaje automático y poder computacional, sin intervención humana, solo considerando las principales características de los municipios—, para efectos de que puedas ver dónde está cada municipio de México en su trayectoria hacia la <span style="color:#51C622">"Madurez Digital"</span> y qué características debe considerar para favorecer su transición a la siguiente fase del ciclo de transformación digital.', unsafe_allow_html=True)

        st.markdown(f'Permíteme compartir tres elementos que motivaron la creación de esta aplicación:', unsafe_allow_html=True)
        
        st.markdown(f'1. <span style="color:#51C622">La madurez digital</span> es multifactorial, incluye una combinación precisa de factores adicionales a los tradicionales como el acceso a Internet, los servicios de conectividad o dispositivos (socio-económicos, infraestructura y demográficos). Para algunos países, la plenitud digital requiere de la definición incluso de una canasta básica de productos digitales que cualquier hogar o ciudadano debe tener.', unsafe_allow_html=True)

        st.markdown(f'''
        <div style="text-align: center; padding-left: 40px;">
            Uno de mis libros favoritos, escrito por 
            <span style="color:#51C622">Antoine Augustin Cournot</span> (1897, página 
            <span style="color:#51C622">24</span>) 
            <a href="http://bibliotecadigital.econ.uba.ar/download/Pe/181738.pdf" target="_blank">
                <em>Researches Into the Mathematical Principles of the Theory of Wealth Economic</em>
            </a>, destaca la necesidad de un punto de referencia para efectos de evaluar las variaciones relativas y absolutas de los elementos en cualquier sistema (pone como ejemplo, al sistema solar y el papel del modelo de Kepler como punto de referencia para medir las variaciones de cada planeta y el sol, haciéndonos conscientes de los verdaderos movimientos de cada cuerpo planetario).
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'3. La <span style="color:#C2185B">Inteligencia Artificial Generativa (Consulta realizada a Search Labs, <span style="color:#C2185B">Diciembre 2024</span></span>: <i>“does science need reference points?”</i>), también sostiene que <i>“…la ciencia necesita puntos de referencia porque proveen un punto fijo de comparación para medir de manera precisa y describir un fenómeno”</i>. Entre estos fenómenos están, por ejemplo, el movimiento planetario, las preferencias de consumidores, las ventas, la distribución del ingreso, la competencia en un mercado y la madurez digital.', unsafe_allow_html=True)

        st.markdown(f'En este contexto, esta aplicación consiste en el marco de referencia para saber con precisión dónde están los municipios en su ciclo de madurez digital y describir el fenómeno.', unsafe_allow_html=True)

        st.markdown(f'Este aplicativo es resultado de un <span style="color:#51C622">modelo de aprendizaje automático no supervisado</span> seleccionado de entre <span style="color:#51C622">450 modelos</span> y más de <span style="color:#51C622">un millón de iteraciones</span> para cada evaluación, con el fin de obtener una clasificación eficiente y precisa sin ningún criterio ajeno a las <span style="color:#51C622">181 características</span> medibles para cada municipio en México. Constituye un marco de referencia objetivo y preciso para ubicar al mununicipio de tu interés y compararlo con el total de municipios con miras a mejorar su madurez digital o conocer sus aptitudes para el desarrollo de negocios digitales. Asimismo, proporciona insights relevantes encuanto a la transición de un estado de madurez a otro y de las diferencias entre cada clasificación de municipios.', unsafe_allow_html=True)

        st.markdown(f'<div style="text-align: right;">Rodrigo Guarneros Gutiérrez<br><span style="color:#51C622">Ciudad de México, 20.12.2024</span></div>', unsafe_allow_html=True)

    with st.expander('¿Qué es la madurez digital?', expanded=False):
        st.markdown(f'En la inteligencia de negocios existen modelos de maduración para las organizaciones y empresas con el objeto de evaluar la toma decisiones basada en datos (Gartner 2004, AMR Research, Service Oriented Business Intelligence Maturirty Model (SOBIMM), entre otros descritos por <a href="https://aulavirtual.infotec.mx/pluginfile.php/115302/mod_label/intro/Medici%C3%B3n%20de%20Madurez%20en%20la%20Implementaci%C3%B3n%20de%20Inteligencia%20de%20Negocios.pdf" target="_blank"><b>Urbina Nájera y Medina-Barrera (2021)</b></a>), la Unión Europea desarrolló la metodología para evaluar la madurez digital de los gobiernos locales (<a href="https://data.europa.eu/en/news-events/news/lordimas-digital-maturity-assessment-tool-regions-and-cities" target="_blank"><b>LORDIMAS 2023, Digital Maturity Index for local governments</b></a>), no existe un enfoque único para evaluar la madurez digital de las regiones o localidades donde el ciudadano sea el objeto de estudio. No obstante, algunos países reconocen el papel de los servicios digitales y financieros como elementos fundamentales para hacer negocios y generar bienestar en una región. Por ello, han definido en sus estándares de desarrollo una canasta básica de bienes y servicios digitales.', unsafe_allow_html=True)

        st.markdown(f'Con base en los resultados del modelo de aprendizaje automático seleccionado para clasificar a los municipios, se identifican 4 etapas de madurez digital:', unsafe_allow_html=True)

        # Imagen
        st.image("fuentes/MDM_madurez1.png", caption="Modelo de Madurez Digital", use_column_width=True)

        st.markdown(f'<b style="color:#51C622">Etapa 1 (Inicial):</b> En esta etapa, los municipios tienen el desempeño más bajo en todas las variables relevantes identificadas.', unsafe_allow_html=True)
        
        st.markdown(f'<b style="color:#51C622">Etapa 2 (Desarrollo):</b> Los municipios tienen un avance en la dirección de más servicios digitales presentes con impacto en las variables de infraestructura, socio-económicos y demográficos.', unsafe_allow_html=True)

        st.markdown(f'<b style="color:#51C622">Etapa 3 (Definición):</b> Claramente se trata de municipios con una penetración promedio en los servicios digitales y un ecosistema financiero más vibrante.', unsafe_allow_html=True)

        st.markdown(f'<b style="color:#51C622">Etapa 4 (Optimización):</b> Los municipios alcanzan una mejor plenitud digital, se nota un balance en sus características que permiten mejor desempeño digital con beneficios tangibles para sus ciudadanos, generando un ecosistema propicio para los negocios digitales y el bienestar.', unsafe_allow_html=True)

    with st.expander('¿Cómo utilizar esta aplicación?', expanded=False):
        st.markdown(f'Como se puede ver, se cuenta con 5 secciones adicionales:', unsafe_allow_html=True)
        st.markdown(f'- <b style="color:#51C622">Municipio:</b> Una vez seleccionado el municipio, aquí encontrarás su ubicación geográfica, la distribución de las variables de interés y el ranking de ese municipio en el <b style="color:#51C622">Índice de madurez</b> construido con base en el modelo de aprendizaje automático.', unsafe_allow_html=True)
        st.markdown(f'- <b style="color:#51C622">Madurez digital:</b> Profundiza sobre lo que significa el ranking de madurez digital para el municipio seleccionado. Conoce cada uno de los componentes o índices que construyen el índice de madures digital y los principales patrones encontrados', unsafe_allow_html=True)
        st.markdown(f'- <b style="color:#51C622">Estadísticas por Grupo:</b> Esta sección presenta un análisis exploratorio de datos para cada clúster. Aprende más sobre las características de los otros clústers y las principales características del clúster del municipio que seleccionaste', unsafe_allow_html=True)
        st.markdown(f'- <b style="color:#51C622">Correlaciones:</b> ¿Te interesa conocer la relación líneal entre dos variables o características de tu municipio? Utiliza esta sección para profundizar en la relación de cada variable', unsafe_allow_html=True)
        st.markdown(f'- <b style="color:#51C622">Geografía:</b> ¿Qué hay de la consistencia geográfica? ¿Hace sentido la clasificación que nos proporciona el modelo? ¿Quiénes son los vecinos geográficos más cercanos al municipio de interés y de qué tipo son?', unsafe_allow_html=True)
        st.image("fuentes/como_utilizar_1.png", caption="Página de Inicio.", use_column_width=True)
        st.markdown(f'- <b style="color:#51C622">Barra de navegación:</b> Navega y selecciona el municipio de tu interés, las variables continuas y categóricas que quieres visualizar durante el análisis.', unsafe_allow_html=True)
        st.image("fuentes/como_utilizar_2.png", caption="Se pueden seleccionar dos variables para análisis correlacional y una variable categórica.", use_column_width=True)
        st.markdown(f'Conoce el enfoque de la programación orientada a objetos y detalles de la aplicación.', unsafe_allow_html=True)
        st.image("fuentes/como_utilizar_3.png", caption="Enfoque de la aplicación y fuentes de información.", use_column_width=True)        


    

with tab2:
    st.markdown(Titulo_dinamico2, unsafe_allow_html=True)
    
    # Expander con información adicional
    with st.expander('Descripción', expanded=False):
        st.markdown(f'Esta sección incluye cuatro visualizaciones relevantes para conocer mejor al municipio seleccionado y el lugar que tiene en la clasificación realizada por nuestra máquina de inferencia estadística. Se sugiere analizar en el siguiente orden:', unsafe_allow_html=True)
        st.markdown(f'- Conoce el índice de madurez digital del municipio seleccionado y comparalo con el del resto de los municipios de México con el Ranking presentado en la primera gráfica: <span style="color:#51C622"> Gráfica de barras con el Índice de Madurez por Municipio, que resalta en rojo el municipio y el lugar que ocupa en el Ranking.</span>', unsafe_allow_html=True)
        st.markdown(f'- Del lado derecho podrás encontrar el lungar del Municipio en el Ranking, la localización geográfica y el tipo de estado de madurez digital que tiene el municipio de acuerdo a su color: <span style="color:#51C622"> La geografía y sus vecinos cercanos es importante, profundiza más en la sección "Geografía" de esta aplicación.</span>.', unsafe_allow_html=True)
        st.markdown(f'- Justo después del mapa, podrás encontrar los estádisticos básicos de la distribución estadística del <span style="color:#51C622"> Índice de Madurez Digital.</span> Visita el área de análisis de esta gráfica para conocer más.', unsafe_allow_html=True)
        st.markdown(f'- Posteriormente, la siguiente gráfica: <span style="color:#51C622"> Histograma por variable</span>, te permite conocer la distribución de alguna variable de interés y combinarlo con las variables categóricas disponibles.', unsafe_allow_html=True)
        st.markdown(f'- Finalmente, ubica en qué lugar se encuentra tu municipio en esa variable de interés, comparado con los demás municipios: <span style="color:#51C622"> Diagrama de caja</span>, que permite revisar a profundidad cuál es el rezago del municipio de interés en esa métrica específica.', unsafe_allow_html=True)
    
    # Crear dos columnas principales con proporción 4:6 para dar más espacio al mapa
    col_izq, col_der = st.columns([6, 6])
    
    # Columna izquierda: solo el ranking
    with col_izq:
        st.plotly_chart(fig_ranking, width=400, use_container_width=True)

    # Columna derecha: mapa y gráficos en secuencia vertical
    with col_der:
        st.plotly_chart(cuadro_resumen, width=400, use_container_width=True)
        # Mapa ajustado al ancho de la columna
        folium_static(fig_municipio, width=455, height=180)  # Ajusta estos valores según necesites
        # Histograma después
        with st.expander('Análisis', expanded=False):
            st.markdown(f'Esta distribución bimodal sugiere dos grupos diferenciados en términos de madurez digital, una brecha digital significativa entre los municipios:', unsafe_allow_html=True)
            st.markdown(f'<b style="color:#51C622">- Un grupo grande con baja madurez digital (primera cresta)</b>. La cresta más alta alcanza aproximadamente 200 municipios, representa la mayor concentración de casos con 700 municipios. ', unsafe_allow_html=True)
            st.markdown(f'<b style="color:#51C622">- Un grupo más pequeño pero significativo con alta madurez digital (segunda cresta)</b>. Este grupo se concentra en el rango de 0.6 a 0.7, la cresta alcanza 150 municipios y en el acumulado son 450 casos.', unsafe_allow_html=True)
            st.markdown(f'<b style="color:#51C622">- Relativamente pocos casos en los niveles intermedios, lo que podría implicar una transición rápida una vez que incia el proceso de madurez digital.</b> Este valle entre los grupos sugiere a 500 municipios y representa una clara separación entre ambos grupos.', unsafe_allow_html=True)

        st.plotly_chart(fig_hist_index, use_container_width=True)
        st.plotly_chart(fig_hist, use_container_width=True)
        # Boxplot al final
        st.plotly_chart(fig_boxplot, use_container_width=True)
# 3D

with tab3:

    st.markdown(Titulo_dinamico3, unsafe_allow_html=True)
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
    col1, col2 = st.columns([1, 1])  # La columna 1 será más ancha que la columna 2

    with col1:
        # Gráfico 3D en toda la columna 1
        with st.expander('El significado de cada Componente Principal', expanded=False):            
            st.markdown(
                f'<span style="color:#51C622">Los componentes principales (PC1, PC2 y PC3) buscan maximizar la suma de las distancias al cuadrado entre los puntos proyectados y el origen</span>. Su resultado es una combinación lineal de todas las variables que los conforman. Así, la descomposición en valores singulares (SVD) nos permite visualizar en la gráfica la proyección de cada una de las combinaciones lineales en los municipios, representados en un espacio vectorial que va de -1 a 1 en cada eje del gráfico tridimensional.',
                unsafe_allow_html=True)
            
            st.markdown(
                f'Esta gráfica presenta los tres patrones más importantes encontrados en el análisis de componentes principales. Por el tipo de variables en cada componente principal y su peso relativo, se pueden identificar los siguientes patrones:',
                unsafe_allow_html=True)
            
            st.markdown(
                f'- <span style="color:#51C622">El componente principal primario (PC1)</span>, que explica el 48.23% de la varianza en todos los datos, puede considerarse como un <span style="color:#51C622">patrón o índice de actividad financiera</span>, asociado por orden de importancia a las siguientes características: (i) Ingresos promedio por vivienda; (ii) Terminales Punto de Venta (TPV); (iii) Transacciones con TPV de Banca Múltiple (BM); (iv) Transacciones en cajeros de BM; (v) Tarjetas de Débito; (vi) Ingresos promedio del sector comercial; (vii) Población Económicamente Activa (PEA); (viii) Cuentas Banca Popular; (ix) Cuentas de BM; (x) Transacciones N4 (personas de alto poder adquisitivo que prefieren servicios exclusivos sin límites de depósitos); (xi) Transacciones N3 (equivalentes a MX$81,112 pesos); (xii) Viviendas habitables, principalmente.',
                unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">Es significativo que el PC1 explique casi la mitad de la varianza total de los datos</span>, lo que sugiere que <b>la actividad financiera es el factor más diferenciador entre los municipios</b>.', unsafe_allow_html=True)
            
            st.markdown(
                f'- <span style="color:#51C622">El segundo componente (PC2)</span>, que explica el 15% de la varianza en el total de los datos, se considera un <span style="color:#51C622">patrón o índice de servicios digitales</span>. Está asociado por orden de importancia con las siguientes variables: (i) PEA; (ii) Ingresos promedio por vivienda; (iii) Viviendas habitables; (iv) Viviendas con TV; (v) Viviendas con celular; (vi) Viviendas con audio radiodifundido; (vii) Transacciones TPV BM; (viii) Ingresos promedio del sector comercial; (ix) Viviendas con TV de paga; (x) Viviendas con Internet; (xi) Ingresos promedio del sector manufacturero; (xii) Cuentas con capacidad móvil, entre otras.',
                unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">Es significativo que la PEA tenga el mayor de los pesos en el componente principal PCA2, sugiriendo <b>una fuerte relación entre la Población Económicamente Activa y los servicios digitiales</b></span>.', unsafe_allow_html=True)
            
            st.markdown(
                f'- <span style="color:#51C622">El tercer componente (PC3)</span>, que explica el 8.32% de la varianza total, se considera un <span style="color:#51C622">patrón o índice de adopción financiera</span>. Está asociado con las siguientes variables: (i) Transacciones TPV; (ii) Tarjetas de débito; (iii) Tarjetas de débito de Banca de Desarrollo; (iv) Cuentas de Banca Popular; (v) Cuentas de Cooperativas; (vi) PEA; (vii) Cuentas de Banca de Desarrollo; (viii) Cuentas N4; (ix) Cuentas de ahorro popular; (x) Cuentas de ahorro cooperativas; (xi) Viviendas habitables.',
                unsafe_allow_html=True)

            st.markdown(
                f'- Mientras PC1 se centra en la actividad financiera general, PC3 captura específicamente la adopción de servicios financieros más específicos (banca popular, cooperativas, desarrollo) <span style="color:#C2185B">La presencia de diferentes tipos de cuentas y servicios financieros sugiere efectivamente un patrón de adopción más que de uso intensivo</span>.', unsafe_allow_html=True)

            st.markdown(
                f'- <span style="color:#51C622">En conclusión, la visualización 3D nos permite ver que estos grupos no son completamente discretos sino que hay transiciones suaves entre ellos, lo que sugiere <b>una transición continua de desarrollo financiero-digital en los municipios mexicanos</b>.</span>', unsafe_allow_html=True)




        # Mostrar gráfico 3D
        st.plotly_chart(grafico3d, use_container_width=True, height=500)


        with st.expander('Patrones en los clústers', expanded=False):
            st.markdown(f'La separación entre clústers tiene mejor visibilidad en tres dimensiones, en general se puede decir que:', unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">El clúster de los municipios en desarrollo (color rojo) es el más numeroso y disperso.</span>', unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">Los clústers Inicial (turquesa) y Definición (morado) muestran una cohesión interna mucho mayor.</span>', unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">El clúster con los municipios en fase de Optimización (color verde) es el más compacto y diferenciado.</span>', unsafe_allow_html=True) 
        st.plotly_chart(grafico2d1, use_container_width=True, height=250)

    with col2:
        # Solo una columna dentro de col2 para los gráficos 2D

        with st.expander('Estructura de los clústers', expanded=False):
            st.markdown(f'Esta segmentación, resultado de las similitudes en las 81 características de los municipios que propone la reducción dimensional, sugiere una clara estratificación de los municipios basada principalmente en su nivel de desarrollo financiero y económico, con subdivisiones adicionales basadas en infraestructura y acceso a servicios financieros especializados.', unsafe_allow_html=True)
            st.markdown(f'En cuanto a la estructura de los clústers, se puede ver lo siguiente: <span style="color:#51C622">(i) Se identifican 4 grupos claramente diferenciados (clústers Inicio, En desarrollo, Definición y Optimización); (ii) la visualización en 2D y 3D muestra que estos grupos tienen fronteras relativamente bien definidas, y (iii) hay cierto solapamiento en las zonas de transición entre clústers, lo cual es natural en datos municipales que pueden compartir características</span>', unsafe_allow_html=True)
            st.markdown(f'La distribución espacial en los clústers es también importante: <span style="color:#51C622">(i) el PCA1 (eje horizontal) explica la mayor variación, abarcando aproximadamente de -0.6 a 0.8; (ii) el PCA2 muestra una dispersión menor, aproximadamente de -0.5 a 0.5, y (iii) el PCA3 añade una dimensión adicional que ayuda a separar mejor algunos grupos que parecían solapados en 2D </span>.', unsafe_allow_html=True)
        st.plotly_chart(grafico2d2, use_container_width=True, height=250)

        with st.expander('Perfil del municipio en cada clúster', expanded=False):
            st.markdown(f'El Clúster Inicial (turquesa) tiene las siguientes características:', unsafe_allow_html=True)

            st.markdown(
                f'- <span style="color:#51C622">Bajo en PC1 (actividad financiera): Se ubica en valores positivos altos.</span>', unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">Bajo/Medio en PC2 (servicios digitales): Valores negativos o neutros.</span>', unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">Bajo en PC3 (adopción financiera).</span>', unsafe_allow_html=True) 
            st.markdown(
                f'<b>Interpretación: Municipios con menor desarrollo financiero y digital, rurales o semi-urbanos con oportunidades de desarrollo en los tres aspectos. Cuenta con servicios financieros/comerciales en desarrollo y escasa infraestructura digital.</b></span>', unsafe_allow_html=True) 


            st.markdown(f'El Clúster en desarrollo (rojo) tiene las siguientes características:', unsafe_allow_html=True)

            st.markdown(
                f'- <span style="color:#51C622">Alto en PC1 (actividad financiera): Se ubica en valores positivos altos.</span>', unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">Bajo en PC2 (servicios digitales): Valores negativos o neutros.</span>', unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">Bajo/medio en PC3 (adopción financiera).</span>', unsafe_allow_html=True) 
            st.markdown(
                f'<b>Interpretación: Municipios con alta actividad financiera pero con brechas en infraestructura digital. Cuenta con servicios financieros/comerciales en desarrollo y escasa infraestructura digital.</b></span>', unsafe_allow_html=True) 

            st.markdown(f'El Clúster en la fase de definición (morado) tiene las siguientes características:', unsafe_allow_html=True)

            st.markdown(
                f'- <span style="color:#51C622">Valores medios en PC1 (actividad financiera): Se ubica en valores positivos altos.</span>', unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">Dispersión amplia en PC2 (servicios digitales): Valores negativos o neutros.</span>', unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">Variaciión en PC3 (adopción financiera).</span>', unsafe_allow_html=True) 
            st.markdown(
                f'<b>Interpretación: Municipios en transición, con niveles moderados de actividad financiera y desarrollo variable en servicios digitales.</b></span>', unsafe_allow_html=True) 



            st.markdown(f'El Clúster en la fase de optimización (verde) tiene las siguientes características:', unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">Alto en PC1 (actividad financiera): Se ubica en valores positivos altos.</span>', unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">Alto en PC2 (servicios digitales): Valores negativos o neutros.</span>', unsafe_allow_html=True)
            st.markdown(
                f'- <span style="color:#51C622">Medio/alto en PC3 (adopción financiera).</span>', unsafe_allow_html=True) 
            st.markdown(
                f'<b>Interpretación: Municipios urbanos y semi-urbanos altamente desarrollados con buena infraestructura digital y alto nivel de actividad financiera.</b></span>', unsafe_allow_html=True)

        st.plotly_chart(grafico2d3, use_container_width=True, height=250)

# El diagrama de caja
with tab4:
    st.markdown("¿Qué patrones se encuentran en cada clúster?")
    
    with st.expander('Recuento por nivel de madurez', expanded=False):
        # Crear las columnas
        col1, col2 = st.columns(2)
        
        # Columna 1: Recuento por nivel de madurez
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
            st.plotly_chart(recuento_clusters, use_container_width=True, height=250)
            
    # Mostrar las gráficas en orden vertical
    st.plotly_chart(boxplots_clusters, use_container_width=True)
    st.plotly_chart(histograma_por_clusters, use_container_width=True)

# La correlacion
with tab5:
    st.markdown(Titulo_dinamico, unsafe_allow_html=True)

    with st.expander('Análisis', expanded=False):
        # st.markdown(f'La población de <span style="color:#C2185B">{variable_seleccionada}</span> seguirá enfrentando cambios radicales. La tasa de crecimiento anual en <span style="color:#C2185B">{}</span> es de <span style="color:#C2185B">{calculos_df.Crecimiento.iloc[0]:,.1f}%</span>.', unsafe_allow_html=True)
        st.markdown(f'Los diagramas de dispersión permiten visualizar las relaciones lineales y no lineales de las variables.', unsafe_allow_html=True)
        st.markdown(f'<span style="color:#51C622">Se trata de un primer acercamiento <span style="color:#51C622">donde es importante recordar que una alta correlación no necesariamente implica causalidad.</span>', unsafe_allow_html=True)
        st.markdown(f'Vale la pena recordar que la R² ajustada se interpreta como el porcentaje de la varianza de la variable dependiente (eje de las Y) que es explicada por la variable independiente (eje de las X).  La R² ajustada es una medida de la bondad de ajuste de un modelo de regresión lineal. Representa el porcentaje de la varianza de la variable dependiente (eje Y) que es explicada por la variable independiente (eje X) después de ajustar el modelo para tener en cuenta el número de predictores en el modelo y el tamaño de la muestra. En otras palabras, la R² ajustada penaliza la inclusión de términos en el modelo que no mejoran significativamente la capacidad predictiva', unsafe_allow_html=True)
    st.plotly_chart(fig_scatter, use_container_width=True, height=500)

# El mapa final
with tab6:

    with st.expander('Análisis', expanded=False):
        st.markdown(f'La clasificación proporcionada por el aprendizaje automático no supervisado sugiere que <span style="color:#51C622"> la madurez digital de los municipios no es aleatoria, sino que sigue patrones relacionados con factores financieros, socio-económicos y geográficos</span>. Cuando se realizaba el entrenamiento de los modelos y se evaluaban, se revisaron los pesos de cada variable en cada componente principal; donde llama la atención que son estadísticamente relevantes variables geográficas como la latitud, longitud y el número de vecinos cercanos en un radio de 5 km. Sugiriendo que la proximidad geográfica entre los municipios influye en su madurez digital debido a la infraestructura compartida y la movilidad de sus factores productivos.', unsafe_allow_html=True)
        st.markdown(f'El mapa que se presenta en esta sección hace evidente que existe una <span style="color:#51C622">concentración de municipios con nivel de madurez óptima (color verde) al rededor de zonas metropolitanas y norte del país.</span>', unsafe_allow_html=True)
        st.markdown(f'Los municipios en desarrollo (color rojo) tienden a concentrarse más en <span style="color:#51C622">la región central y sur del país.</span>', unsafe_allow_html=True)
        st.markdown(f'Se puede ver una concentración significativa de municipios en fase de definición (color violeta) en la <span style="color:#51C622">península de Yucatán, formando un clúster definitivo</span>.', unsafe_allow_html=True)
        st.markdown(f'Los municipios en fase de definición (color violeta) se pueden ver en zonas periféricas a grandes centros urbanos <span style="color:#51C622">lo que sugiere un efecto de desbordamiento digital de los municipios más desarrollados a los menos desarrollados.</span> En general, esta fase sugiere que los municipios ya tienen una infraestructura digital básica y están formalizando sus procesos digitales.', unsafe_allow_html=True)
        st.markdown(f'Existen clústers claros en el nivel de madurez inicial (color azul turquesa).', unsafe_allow_html=True)
        st.markdown(f'Es posible observar <span style="color:#51C622">islas de desarrollo avanzado, correspondientes a centros urbanos importantes, rodeadas de zonas menos desarrolladas.</span>', unsafe_allow_html=True)
        st.markdown(f'Las disparidades regionales son evidentes y podrían requerir de <span style="color:#51C622">estrategias específicas para el despliegue de ofertas comerciales específicas o para el desarrollo digital de los municipios.</span>', unsafe_allow_html=True)
        st.markdown(f'En resumen, <span style="color:#51C622">existen zonas propicias para la comercialización de servicios digitales porque cuentan con infraestructura funcional y población familiarizada o con capacidad de utilizar los servicios digitales</span>, tales como: El corredor fronterizo del norte, la zona metropolitana del Valle de México, Guadalajara y su área de influencia, Monterrey y municipios circundantes.', unsafe_allow_html=True)
        st.markdown(f'Si quieres conocer más insights o realizar un análisis específico, [escríbeme](mailto:rodrigo.guarneros@gmail.com), con gusto te ayudo.', unsafe_allow_html=True)
    st.plotly_chart(fig_map_final, use_container_width=True, height=500)
