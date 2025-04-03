#############################
# Importar librerías
#############################
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
import pymongo
from pymongo import MongoClient
from gridfs import GridFS
from io import BytesIO
from dotenv import load_dotenv
import os
from bson import ObjectId
from concurrent.futures import ThreadPoolExecutor

#############################
# Configuración de la página
#############################
st.set_page_config(
    page_title="Aprendizaje Automático para los Municipios de México",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)
alt.themes.enable("dark")

#############################
# CSS styling
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

#############################################
# Funciones de ayuda y carga de datos
#############################################
def convert_objectid_to_str(document):
    for key, value in document.items():
        if isinstance(value, ObjectId):
            document[key] = str(value)
    return document

@st.cache_data
def get_collection_df(collection_name, strip_columns=False, process_str=True):
    mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
    client = MongoClient(mongo_uri)
    db = client['Municipios_Rodrigo']
    collection = db[collection_name]
    data = list(collection.find())
    df = pd.DataFrame(list(map(convert_objectid_to_str, data)))
    if process_str:
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].apply(lambda x: x.encode('Latin1').decode('Latin1') if isinstance(x, str) else x)
    if strip_columns:
        df.columns = df.columns.str.strip()
    return df

# Cargar datasets principales
datos = get_collection_df('datos_finales')
input_datos = datos.copy()
# Renombrar columnas y eliminar duplicados
datos['Operadores Escala Pequeña BAF'] = datos['operadores_escal_pequeña_baf']
datos.drop(columns=['operadores_escal_pequeña_baf'], inplace=True)
datos['Penetración BAF (Fibra)'] = datos['penetracion_baf_fibra']
datos.drop(columns=['penetracion_baf_fibra'], inplace=True)

dataset_complete = get_collection_df('completo', strip_columns=True)
df_entrenamiento = get_collection_df('X_for_training_normalizer', strip_columns=True)
df_normalizado = get_collection_df('df_pca_norm', strip_columns=True)

# Definir listas de variables (excluyendo columnas innecesarias)
cols_excl_numeric = ['Cluster2','Unnamed: 0', 'Unnamed: 0.2', 'cve_edo', 'cve_municipio', 'cvegeo',
                     'Estratos ICM', 'Estrato IDDM', 'Municipio', 'df1_ENTIDAD', 'df1_KEY MUNICIPALITY',
                     'df2_Clave Estado', 'df2_Clave Municipio', 'df3_Clave Estado', 'df3_Clave Municipio',
                     'df4_Clave Estado', 'df4_Clave Municipio']
cols_excl_categ = ['_id','Lugar', 'Estado2', 'df2_Región', 'df3_Región', 'df3_Tipo de población',
                   'df4_Región', 'Municipio']
variable_list_numerica = [col for col in list(input_datos.select_dtypes(include=['int64', 'float64']).columns) if col not in cols_excl_numeric]
variable_list_categorical = [col for col in list(input_datos.select_dtypes(include=['object', 'category']).columns) if col not in cols_excl_categ]
variable_list_municipio = sorted(list(input_datos['Lugar'].unique()))

#############################################
# Carga de GeoJSON y fusión con datos
#############################################
@st.cache_resource
def connect_to_mongo_resource(mongo_uri):
    client = MongoClient(mongo_uri)
    return client['Municipios_Rodrigo']

@st.cache_data
def get_geojson_bytes(db_resource):
    fs = GridFS(db_resource)
    file = fs.find_one({'filename': 'municipios.geojson'})
    return file.read() if file else None

def geojson_to_geodataframe(geojson_bytes):
    return gpd.read_file(BytesIO(geojson_bytes))

mongo_uri = st.secrets["MONGO"]["MONGO_URI"]
db_resource = connect_to_mongo_resource(mongo_uri)
geojson_bytes = get_geojson_bytes(db_resource)
geojson = geojson_to_geodataframe(geojson_bytes) if geojson_bytes else None

if geojson is not None:
    datos.rename(columns={'cvegeo': 'CVEGEO'}, inplace=True)
    datos['CVEGEO'] = datos['CVEGEO'].astype(str).str.zfill(5)
    geojson['CVEGEO'] = geojson['CVEGEO'].astype(str)
    dataset_complete_geometry = datos.merge(geojson[['CVEGEO', 'geometry']], on='CVEGEO', how='left')
else:
    dataset_complete_geometry = None

#############################################
# Incrementar contador de visitas
#############################################
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
        st.error(f"Error en base de datos: {e}")
        return "N/A"

contador_visitas = incrementar_contador_visitas()

#############################################
# Sidebar
#############################################
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
    variable_seleccionada_municipio = st.selectbox('Selecciona el municipio de tu interés:', variable_list_municipio)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("Análisis Estadístico por Variable:", unsafe_allow_html=True)
    variable_seleccionada_numerica = st.selectbox('Selecciona la variable numérica de interés:', sorted(variable_list_numerica))
    variable_seleccionada_categorica = st.selectbox('Selecciona la variable categórica de interés:', sorted(variable_list_categorical))
    variable_seleccionada_paracorrelacion = st.selectbox('Selecciona la variable que quieras correlaccionar con la primera selección:', sorted(variable_list_numerica))
    st.markdown("<hr>", unsafe_allow_html=True)
    with st.expander('Enfoque de esta aplicación', expanded=False):
        st.write('''
            - Se basa en un enfoque de <span style="color:#51C622">"Programación Orientada a Objetos"</span>.
            - Los 2,456 municipios se pueden modelar a partir de sus atributos y funciones para aprovechar la revolución digital. 
            - El principal objetivo es: <span style="color:#51C622">Ajustar un modelo de aprendizaje automático para clasificar a las localidades de México por su vocación para la transformación digital y despliegue de servicios TIC, en función de variables fundamentales de infraestructura, demográficas y socio-económicas.</span>
            - Este aplicativo incluye atributos a nivel municipal tales como:
                1. Número de viviendas. 
                2. Grado educativo (analfabetismo, % con educación básica, etc.).
                3. Edad promedio, 
                4. Penetración de Internet, entre otras.
            - Permite visualizar la distribución estadística, relaciones entre variables y la distribución geográfica.
        ''', unsafe_allow_html=True)
    with st.expander('Fuentes y detalles técnicos', expanded=False):
        st.write('''
            - Fuente: [Consejo Nacional de Población (CONAPO), consultado el 3 de febrero de 2024.](https://www.gob.mx/conapo).
            - Tecnologías: Python 3.10, Streamlit 1.30.0, CSS3, HTML5, Google Colab y GitHub. 
            - Autor: Rodrigo Guarneros ([LinkedIn](https://www.linkedin.com/in/guarneros/) y [X](https://twitter.com/RodGuarneros)).
            - Contacto: rodrigo.guarneros@gmail.com
        ''', unsafe_allow_html=True)
    st.image('fuentes/cc.png', caption='\u00A9 Copy Rights Rodrigo Guarneros, 2024', use_column_width=True)
    st.markdown("Esta aplicación se rige por [Creative Commons CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Si deseas adaptarla, [escríbeme](mailto:rodrigo.guarneros@gmail.com).", unsafe_allow_html=True)
    st.markdown(f"Visitas al sitio: **{contador_visitas}**", unsafe_allow_html=True)

#############################################
# Funciones de visualización (mapas, gráficos, etc.)
#############################################
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
            print(f"No se encontraron datos para: {lugar_a_buscar}")
            return None
        gdf = gdf_filtrado
    centro = gdf.geometry.centroid.iloc[0]
    m = folium.Map(location=[centro.y, centro.x], zoom_start=12, tiles="CartoDB dark_matter")
    bounds = gdf.geometry.total_bounds
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    mapa_colores = {'En desarrollo': '#D20103', 'Inicial': '#5DE2E7', 'Definición': '#CC6CE7', 'Optimización': '#51C622'}
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

fig_municipio = crear_mapa_choropleth2(dataset_complete_geometry, lugar=variable_seleccionada_municipio, municipio_inicial="Abalá, Yucatán")

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
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=0.5)
        ),
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
    height = max(400, len(plot_data) * 18)
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

fig_ranking = plot_bar_chart(datos, 'Lugar', 'Índice_Compuesto', variable_seleccionada_municipio)

def crear_display(data, lugar_seleccionado):
    mapa_colores = {'En desarrollo': '#D20103', 'Inicial': '#5DE2E7', 'Definición': '#CC6CE7', 'Optimización': '#51C622'}
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
        x=0.5, y=0.80,
        showarrow=False,
        font=dict(family="Arial", size=12, color="#050505"),
        align="center"
    )
    fig.add_annotation(
        text=str(int(lugar_ranking)),
        x=0.5, y=0.35,
        showarrow=False,
        font=dict(family="Arial", size=37, color="#050505"),
        align="center"
    )
    fig.update_layout(
        width=200, height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=color_rect,
        plot_bgcolor=color_rect,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1])
    )
    return fig

cuadro_resumen = crear_display(datos, variable_seleccionado_municipio)

def plot_histogram(df, numeric_column, categorical_column):
    color_map = {'En desarrollo': '#D20103', 'Inicial': '#5DE2E7', 'Definición': '#CC6CE7', 'Optimización': '#51C622'}
    fig = px.histogram(df, x=numeric_column, color=categorical_column,
                       color_discrete_map=color_map, opacity=0.6,
                       title=f'Histograma de la variable "{numeric_column}" y <br>la categoría "{categorical_column}"')
    fig.update_yaxes(title_text="Frecuencia absoluta")
    stats = {
        'Media': df[numeric_column].mean(),
        'Mediana': df[numeric_column].median(),
        'Moda': df[numeric_column].mode()[0],
        'Desviación estándar': df[numeric_column].std()
    }
    stats_text = "<br>".join([f"<b>{k}</b>: {v:.2f}" for k, v in stats.items()])
    counts_text = "<br>".join([f"<b>{cat}</b>: {cnt}" for cat, cnt in df[categorical_column].value_counts().items()])
    annotations_text = f"{stats_text}<br><br><b>Conteo por categoría:</b><br>{counts_text}"
    fig.update_layout(
        title_font=dict(color='#FFD86C', size=16),
        title_x=0.05,
        showlegend=True,
        width=1350,
        height=500,
        margin=dict(l=50, r=50, t=80, b=200),
        annotations=[dict(
            x=1.1, y=0.9, xref='paper', yref='paper',
            text=annotations_text,
            showarrow=False,
            font=dict(color='white', size=12),
            align='center',
            bgcolor='rgba(0, 0, 0, 0.7)',
            bordercolor='white',
            borderwidth=1,
            opacity=0.8
        )],
        legend=dict(orientation='h', yanchor='top', y=-0.3, xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title_font=dict(color='#FFD86C'), gridcolor='rgba(128, 128, 128, 0.2)'),
        yaxis=dict(title_font=dict(color='#FFD86C'), gridcolor='rgba(128, 128, 128, 0.2)')
    )
    return fig

fig_hist = plot_histogram(input_datos, variable_seleccionada_numerica, variable_seleccionada_categorica)

def plot_histogram_with_density(df, numeric_column, selected_value=None):
    fig = px.histogram(df, x=numeric_column, opacity=0.6,
                       title='Distribución del índice de madurez digital',
                       nbins=50,
                       labels={'x': 'Valores del Índice', 'y': 'Frecuencia'})
    fig.update_traces(marker_line_color='white', marker_line_width=1.5)
    hist_data = df[numeric_column].dropna().astype(float)
    kde = gaussian_kde(hist_data)
    density_x = np.linspace(hist_data.min(), hist_data.max(), 1000)
    density_y = kde(density_x)
    density_y_scaled = density_y * len(hist_data) * (hist_data.max() - hist_data.min()) / 50
    fig.add_trace(go.Scatter(x=density_x, y=density_y_scaled, mode='lines', line=dict(color='blue', width=2), name='Dens'))
    if selected_value is not None:
        try:
            sv = float(selected_value)
            fig.add_trace(go.Scatter(x=[sv], y=[0], mode='markers+text', marker=dict(color='red', size=10, line=dict(color='white', width=1)),
                                     text=f'{sv:.2f}', textposition='top center', name='Lugar seleccionado'))
        except ValueError:
            print(f"El valor {selected_value} no es numérico.")
    mean, median, mode_val, std = hist_data.mean(), hist_data.median(), hist_data.mode()[0], hist_data.std()
    annotation_text = f"<b>Estadísticos:</b><br>Media: {mean:.2f}<br>Mediana: {median:.2f}<br>Moda: {mode_val:.2f}<br>Desv. Est.: {std:.2f}"
    fig.add_annotation(dict(x=1, y=0.95, xref='paper', yref='paper', text=annotation_text, showarrow=False,
                            font=dict(size=12, color='white'), align='left',
                            bgcolor='rgba(0, 0, 0, 0.7)', bordercolor='rgba(255, 255, 255, 0.7)', borderwidth=2))
    fig.update_layout(title_font=dict(color='#FFD86C'), xaxis_title_font=dict(color='#FFD86C'),
                      yaxis_title_font=dict(color='#FFD86C'),
                      legend=dict(title_text='Leyenda', font=dict(color='#FFD86C')),
                      xaxis=dict(showgrid=False, title='Valores del Índice'),
                      yaxis=dict(showgrid=False, title='Frecuencia'),
                      plot_bgcolor='rgba(0, 0, 0, 0.1)')
    return fig

fig_hist_index = plot_histogram_with_density(input_datos, 'Índice_Compuesto', selected_value=variable_seleccionada_municipio)

def generate_boxplot_with_annotations(df, variable, lugar_seleccionado):
    stats = {
        'Media': np.mean(df[variable]),
        'Mediana': np.median(df[variable]),
        'Moda': df[variable].mode().iloc[0],
        'Desviación estándar': np.std(df[variable])
    }
    fig = px.box(df, y=variable, points=False, title=f'Diagrama para la variable<br>"{variable}"', template='plotly_dark')
    if lugar_seleccionado:
        df_lugar = df[df['Lugar'] == lugar_seleccionado]
        fig.add_scatter(x=[0]*len(df_lugar), y=df_lugar[variable], mode='markers',
                        marker=dict(color='rgba(0, 255, 0, 0.7)', size=10, line=dict(color='rgba(0, 255, 0, 1)', width=2)),
                        name=f'Lugar seleccionado: {lugar_seleccionado}',
                        hovertemplate='<b>%{customdata[0]}</b><br>'+variable+': %{y:.2f}<extra></extra>',
                        customdata=df_lugar[['Municipio']])
    df_rest = df[df['Lugar'] != lugar_seleccionado]
    fig.add_scatter(x=[0]*len(df_rest), y=df_rest[variable], mode='markers',
                    marker=dict(color='rgba(255, 165, 0, 0.5)', size=7, line=dict(color='rgba(255, 165, 0, 0.7)', width=1)),
                    name='Otros lugares',
                    hovertemplate='<b>%{customdata[0]}</b><br>'+variable+': %{y:.2f}<extra></extra>',
                    customdata=df_rest[['Municipio']])
    annotations_text = "<br>".join([f"<b>{k}</b>: {v:.2f}" for k,v in stats.items()])
    fig.update_layout(title_font=dict(color='#FFD86C', size=16), title_x=0.2, showlegend=True, width=1350, height=500,
                      margin=dict(l=55, r=55, t=80, b=200),
                      annotations=[dict(x=0.5, y=-0.3, xref='paper', yref='paper', text=annotations_text, showarrow=False,
                                          font=dict(color='white', size=12), align='center',
                                          bgcolor='rgba(0, 0, 0, 0.7)', bordercolor='white', borderwidth=2, opacity=0.8)],
                      legend=dict(orientation='h', yanchor='top', y=-0.3, xanchor='center', x=0.5, bgcolor='rgba(0,0,0,0)'),
                      yaxis=dict(title=variable, title_font=dict(color='#FFD86C'), showgrid=True, gridcolor='rgba(128, 128, 128, 0.2)'),
                      xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig

fig_boxplot = generate_boxplot_with_annotations(input_datos, variable_seleccionada_numerica, variable_seleccionada_municipio)

def generar_grafico_3d_con_lugar(df, df_normalizado, dataset_complete, lugar_seleccionado=None):
    color_map = {'Optimización': '#51C622', 'Definición': '#CC6CE7', 'En desarrollo': '#D20103', 'Inicial': '#5DE2E7'}
    df_pca = pd.DataFrame(df_normalizado.to_numpy()[:, 1:4], columns=['PCA1', 'PCA2', 'PCA3'])
    df_pca['Madurez'] = df['Etapa_Madurez']
    df_pca['Lugar'] = dataset_complete['Lugar']
    fig = px.scatter_3d(df_pca, x='PCA1', y='PCA2', z='PCA3', color='Madurez',
                         labels={'PCA1': 'Componente PC1', 'PCA2': 'Componente PC2', 'PCA3': 'Componente PC3'},
                         hover_data=['Lugar'],
                         category_orders={'Madurez': ['Optimización', 'Definición', 'En desarrollo', 'Inicial']},
                         color_discrete_map=color_map)
    if lugar_seleccionado:
        lugar_df = df_pca[df_pca['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            fig.add_trace(px.scatter_3d(lugar_df, x='PCA1', y='PCA2', z='PCA3', hover_data=['Lugar'],
                                        color_discrete_map={'Madurez': 'green'}).data[0])
            fig.update_traces(marker=dict(size=20, color='green', opacity=1), selector=dict(name=lugar_seleccionado))
    fig.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=0.02, color='gray')))
    fig.update_layout(title="Municipios por grado de madurez multidimensional", title_x=0.05,
                      showlegend=True, legend=dict(title=dict(text='Madurez'), itemsizing='constant', font=dict(color='white')),
                      scene=dict(xaxis_title="Componente PC1", yaxis_title="Componente PC2", zaxis_title="Componente PC3",
                                 xaxis=dict(titlefont=dict(color='white'), gridcolor='white', zerolinecolor='white'),
                                 yaxis=dict(titlefont=dict(color='white'), gridcolor='white', zerolinecolor='white'),
                                 zaxis=dict(titlefont=dict(color='white'), gridcolor='white', zerolinecolor='white'),
                                 bgcolor='rgb(0, 0, 0)', xaxis_showgrid=True, yaxis_showgrid=True, zaxis_showgrid=True),
                      font=dict(color='white'), paper_bgcolor='rgb(0, 0, 0)', plot_bgcolor='rgb(0, 0, 0)')
    return fig

grafico3d = generar_grafico_3d_con_lugar(datos, df_normalizado, dataset_complete, lugar_seleccionado=variable_seleccionada_municipio)

def generar_grafico_2d(df, df_normalizado, dataset_complete, lugar_seleccionado=None):
    df['Madurez'] = df['Madurez'].str.strip()
    df_pca = pd.DataFrame(df_normalizado.to_numpy()[:, 1:4], columns=['PCA1', 'PCA2', 'PCA3'])
    df_pca['Madurez'] = df['Madurez'].astype('category')
    df_pca['Lugar'] = dataset_complete['Lugar']
    color_map = {'Optimización': '#51C622', 'Definición': '#CC6CE7', 'En desarrollo': '#D20103', 'Inicial': '#5DE2E7'}
    fig = px.scatter(df_pca, x='PCA1', y='PCA2', color='Madurez',
                     labels={'PCA1': 'Componente PC1', 'PCA2': 'Componente PC2'},
                     hover_data=['Lugar'],
                     category_orders={'Madurez': ['Optimización', 'Definición', 'En desarrollo', 'Inicial']},
                     color_discrete_map=color_map)
    if lugar_seleccionado:
        lugar_df = df_pca[df_pca['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            fig.add_trace(px.scatter(lugar_df, x='PCA1', y='PCA2', hover_data=['Lugar'],
                                      color_discrete_map={'Madurez': 'green'}).data[0])
            fig.update_traces(marker=dict(size=10, color='green', opacity=1), selector=dict(name=lugar_seleccionado))
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.02, color='gray')))
    fig.update_layout(title="PC2 vs. PC1 (2D)", title_x=0.3, showlegend=True,
                      legend=dict(title=dict(text='Madurez'), itemsizing='constant', font=dict(color='white')),
                      font=dict(color='white'), paper_bgcolor='rgb(0, 0, 0)', plot_bgcolor='rgb(0, 0, 0)')
    return fig

grafico2d1 = generar_grafico_2d(datos, df_normalizado, dataset_complete, lugar_seleccionado=variable_seleccionada_municipio)

def generar_grafico_2d2(df, df_normalizado, dataset_complete, lugar_seleccionado=None):
    df['Madurez'] = df['Madurez'].astype('category')
    df_pca = pd.DataFrame(df_normalizado.to_numpy()[:, 1:4], columns=['PCA1', 'PCA2', 'PCA3'])
    df_pca['Etapa_Madurez'] = df['Madurez']
    df_pca['Lugar'] = dataset_complete['Lugar']
    color_map = {'Optimización': '#51C622', 'Definición': '#CC6CE7', 'En desarrollo': '#D20103', 'Inicial': '#5DE2E7'}
    fig = px.scatter(df_pca, x='PCA1', y='PCA3',
                     labels={'PCA1': 'Componente PC1', 'PCA3': 'Componente PC3'},
                     hover_data=['Lugar'], color='Etapa_Madurez',
                     color_discrete_map=color_map)
    if lugar_seleccionado:
        lugar_df = df_pca[df_pca['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            fig.add_trace(go.Scatter(x=lugar_df['PCA1'], y=lugar_df['PCA3'], mode='markers',
                                     marker=dict(size=12, color='orange', symbol='diamond'),
                                     name=f"Lugar: {lugar_seleccionado}"))
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.5, color='gray')))
    fig.update_layout(title="PC1 vs. PC3 (2D)", title_x=0.5, showlegend=True,
                      legend=dict(title=dict(text='Etapa de Madurez'), itemsizing='constant'),
                      paper_bgcolor='rgb(0, 0, 0)', plot_bgcolor='rgb(0, 0, 0)', font=dict(color='white'))
    return fig

grafico2d2 = generar_grafico_2d2(datos, df_normalizado, dataset_complete, lugar_seleccionado=variable_seleccionada_municipio)

def generar_grafico_2d3(df, df_normalizado, dataset_complete, lugar_seleccionado=None):
    df['Madurez'] = df['Madurez'].str.strip()
    df_pca = pd.DataFrame(df_normalizado.to_numpy()[:, 1:4], columns=['PCA1', 'PCA2', 'PCA3'])
    df_pca['Madurez'] = df['Madurez'].astype('category')
    df_pca['Lugar'] = dataset_complete['Lugar']
    color_map = {'Optimización': '#51C622', 'Definición': '#CC6CE7', 'En desarrollo': '#D20103', 'Inicial': '#5DE2E7'}
    fig = px.scatter(df_pca, x='PCA2', y='PCA3', color='Madurez',
                     labels={'PCA2': 'Componente PC2', 'PCA3': 'Componente PC3'},
                     hover_data=['Lugar'],
                     category_orders={'Madurez': ['Optimización', 'Definición', 'En desarrollo', 'Inicial']},
                     color_discrete_map=color_map)
    if lugar_seleccionado:
        lugar_df = df_pca[df_pca['Lugar'] == lugar_seleccionado]
        if not lugar_df.empty:
            fig.add_trace(px.scatter(lugar_df, x='PCA2', y='PCA3', hover_data=['Lugar'],
                                     color_discrete_map={'Madurez': 'green'}).data[0])
            fig.update_traces(marker=dict(size=10, color='green', opacity=1), selector=dict(name=lugar_seleccionado))
    fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=0.02, color='gray')))
    fig.update_layout(title="PC3 vs. PC2 (2D)", title_x=0.3, showlegend=True,
                      legend=dict(title=dict(text='Madurez'), itemsizing='constant', font=dict(color='white')),
                      font=dict(color='white'), paper_bgcolor='rgb(0, 0, 0)', plot_bgcolor='rgb(0, 0, 0)')
    return fig

grafico2d3 = generar_grafico_2d3(df, df_normalizado, dataset_complete, lugar_seleccionado=variable_seleccionada_municipio)

def boxplot_por_cluster(df, variable):
    color_map = {'Optimización': '#51C622', 'Definición': '#CC6CE7', 'En desarrollo': '#D20103', 'Inicial': '#5DE2E7'}
    stats = df.groupby('Madurez')[variable].agg(['mean', 'median', 'std']).reset_index()
    stats.rename(columns={'mean': f'mean_{variable}', 'median': f'median_{variable}', 'std': f'std_{variable}'}, inplace=True)
    df_stats = pd.merge(df, stats, on='Madurez', how='left')
    fig = px.box(df_stats, y=variable, points='all',
                 title=f'Diagrama de caja de la variable\n"{variable}"',
                 labels={variable: variable},
                 template='plotly_dark',
                 color='Madurez',
                 color_discrete_map=color_map,
                 hover_data={'Madurez': True, 'Lugar': True, f'mean_{variable}': True, f'median_{variable}': True, f'std_{variable}': True})
    fig.update_traces(marker=dict(opacity=0.6, line=dict(color='rgba(255, 165, 0, 0.5)', width=1)))
    return fig

boxplots_clusters = boxplot_por_cluster(datos, variable_seleccionada_numerica)

def recuento(df):
    total = len(df)
    counts = df['Madurez'].value_counts().reset_index()
    counts.columns = ['Madurez', 'Cantidad']
    counts['Frecuencia relativa'] = counts['Cantidad'] / total
    color_map = {'En desarrollo': '#D20103', 'Inicial': '#5DE2E7', 'Definición': '#CC6CE7', 'Optimización': '#51C622'}
    fig = px.bar(counts, x='Madurez', y='Frecuencia relativa',
                 title="Frecuencia relativa por nivel de madurez",
                 labels={'Frecuencia relativa': 'Frecuencia relativa', 'Madurez': 'Nivel de madurez'},
                 color='Madurez', color_discrete_map=color_map,
                 category_orders={'Madurez': ['Inicial', 'En desarrollo', 'Definición', 'Optimización']},
                 height=280)
    return fig

recuento_clusters = recuento(datos)

def titulo_dinamico(variable):
    return f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">La variable mostrada es: "{variable}".</span>'

Titulo_dinamico = titulo_dinamico(variable=variable_seleccionada_numerica)

def titulo_dinamico2(variable):
    return f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">Municipio de "{variable}".</span>'

Titulo_dinamico2 = titulo_dinamico2(variable=variable_seleccionada_municipio)

def titulo_dinamico3(variable):
    return f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">Análisis de Madurez Digital de "{variable}".</span>'

Titulo_dinamico3 = titulo_dinamico3(variable=variable_seleccionada_municipio)

def generate_scatter_with_annotations(df, x_var, y_var, cat_var):
    df_clean = df.dropna(subset=[x_var, y_var])
    color_map = {'En desarrollo': '#D20103', 'Inicial': '#5DE2E7', 'Definición': '#CC6CE7', 'Optimización': '#51C622'}
    fig = px.scatter(df_clean, x=x_var, y=y_var, hover_data={'Lugar': True, cat_var: True}, color=cat_var,
                     color_discrete_map=color_map)
    X = df_clean[[x_var]].values
    y = df_clean[y_var].values
    model = LinearRegression().fit(X, y)
    intercept, slope = model.intercept_, model.coef_[0]
    r2 = model.score(X, y)
    n, p = len(df_clean), 1
    r2_adj = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    eq = f"y = {slope:.2f}x + {intercept:.2f}"
    x_range = np.linspace(df_clean[x_var].min(), df_clean[x_var].max(), 100)
    y_pred = slope * x_range + intercept
    fig.add_scatter(x=x_range, y=y_pred, mode='lines', name='Regression Line',
                    line=dict(color='orange', dash='dash'))
    fig.update_layout(
        plot_bgcolor='rgb(30,30,30)',
        paper_bgcolor='rgb(30,30,30)',
        font_color='white',
        title=dict(text=f"Scatter Plot: '{x_var}' vs '{y_var}'", font=dict(color='white')),
        xaxis=dict(title=f"Variable: {x_var}", titlefont=dict(color='white'), tickfont=dict(color='white')),
        yaxis=dict(title=f"Variable: {y_var}", titlefont=dict(color='white'), tickfont=dict(color='white')),
        annotations=[dict(xref='paper', yref='paper', x=0.95, y=1.05,
                          text=f'R² Ajustada: {r2_adj:.4f}', showarrow=False, font=dict(color='orange')),
                     dict(xref='paper', yref='paper', x=0.05, y=1.05,
                          text=f'Regresión: {eq}', showarrow=False, font=dict(color='orange'))]
    )
    fig.update_traces(hovertemplate='<b>Municipio</b>: %{customdata[0]}<br>' +
                                    f'<b>{x_var}</b>: %{{x}}<br><b>{y_var}</b>: %{{y}}<br>')
    fig.update_traces(marker=dict(opacity=0.9, line=dict(color='rgba(255, 165, 0, 0.5)', width=1)))
    return fig

fig_scatter = generate_scatter_with_annotations(input_datos, variable_seleccionada_numerica, variable_seleccionada_paracorrelacion, variable_seleccionada_categorica)

def generar_mapa_con_lugar(df, lugar=None):
    color_map = {'En desarrollo': '#D20103', 'Inicial': '#5DE2E7', 'Definición': '#CC6CE7', 'Optimización': '#51C622'}
    df['Madurez'] = df['Madurez'].astype('category')
    fig = px.scatter_mapbox(df, lat="Latitud", lon="Longitud", color="Madurez", opacity=0.8,
                            hover_data=["Madurez", "Lugar"], zoom=4,
                            center={"lat": 23.6345, "lon": -102.5528},
                            title="Mapa de Clústers por Madurez Digital en México",
                            color_discrete_map=color_map)
    if lugar:
        lugar_df = df[df['Lugar'] == lugar]
        if not lugar_df.empty:
            fig.add_trace(px.scatter_mapbox(lugar_df, lat="Latitud", lon="Longitud",
                                            color_discrete_map={0: '#ffa500'}, size_max=10, size=[8],
                                            hover_data=["Madurez", "Lugar"]).data[0])
    fig.update_layout(mapbox_style="carto-darkmatter", height=600,
                      margin={"r": 0, "t": 50, "l": 0, "b": 0},
                      legend=dict(title="Nivel de Madurez", itemsizing="constant", traceorder="normal"))
    return fig

fig_map_final = generar_mapa_con_lugar(input_datos, lugar=variable_seleccionada_municipio)

#############################################
# Dashboard Main Panel (Tabs)
#############################################
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Presentación", "Municipio", "Madurez Digital", "Estadísiticas por Grupo", "Análisis Relacional", "Geografía"
])

with tab1:
    with st.expander('¿Para qué sirve esta aplicación?', expanded=False):
        st.markdown(
            'Provee un punto de referencia estadísticamente robusto, claro y preciso basado en aprendizaje automático para evaluar la madurez digital de los municipios de México.',
            unsafe_allow_html=True
        )
        st.markdown('Elementos motivadores:', unsafe_allow_html=True)
        st.markdown('- <span style="color:#51C622">La madurez digital</span> es multifactorial y requiere un punto de referencia para medir diferencias.',
                    unsafe_allow_html=True)
        st.markdown(
            '''
            <div style="text-align: center; padding-left: 40px;">
                Uno de mis libros favoritos, de <span style="color:#51C622">Antoine Augustin Cournot</span> (1897, pág. <span style="color:#51C622">24</span>)
                <a href="http://bibliotecadigital.econ.uba.ar/download/Pe/181738.pdf" target="_blank">
                    <em>Researches Into the Mathematical Principles of the Theory of Wealth Economic</em>
                </a>
                destaca la importancia de un punto de referencia.
            </div>
            ''', unsafe_allow_html=True)
        st.markdown('- La Inteligencia Artificial Generativa apoya la idea de que la ciencia necesita puntos de referencia para medir fenómenos.',
                    unsafe_allow_html=True)
        st.markdown('Esta aplicación es el resultado de un modelo no supervisado evaluado a partir de 181 características para clasificar municipios.',
                    unsafe_allow_html=True)
        st.markdown('<div style="text-align: right;">Rodrigo Guarneros Gutiérrez<br><span style="color:#51C622">Ciudad de México, 20.12.2024</span></div>',
                    unsafe_allow_html=True)
    with st.expander('¿Qué es la madurez digital?', expanded=False):
        st.markdown('Existen diversos modelos para evaluar la madurez digital. Esta aplicación clasifica municipios en 4 etapas: Inicial, Desarrollo, Definición y Optimización.',
                    unsafe_allow_html=True)
        st.image("fuentes/MDM_madurez1.png", caption="Modelo de Madurez Digital", use_column_width=True)
    with st.expander('¿Cómo puedes utilizar esta aplicación?', expanded=False):
        st.markdown('La aplicación se divide en secciones: Municipio, Madurez Digital, Estadísticas por Grupo, Correlaciones y Geografía.',
                    unsafe_allow_html=True)
        st.image("fuentes/como_utilizar_1.png", caption="Página de Inicio", use_column_width=True)
        st.markdown('Selecciona el municipio y las variables desde la barra lateral para visualizar los análisis correspondientes.',
                    unsafe_allow_html=True)

with tab2:
    st.markdown(Titulo_dinamico2, unsafe_allow_html=True)
    with st.expander('Descripción', expanded=False):
        st.markdown('Esta sección muestra el ranking, la ubicación geográfica y el análisis estadístico del municipio seleccionado.',
                    unsafe_allow_html=True)
    col_izq, col_der = st.columns([6, 6])
    with col_izq:
        st.plotly_chart(fig_ranking, width=400, use_container_width=True)
    with col_der:
        st.plotly_chart(cuadro_resumen, width=400, use_container_width=True)
        folium_static(fig_municipio, width=455, height=180)
        with st.expander('Análisis', expanded=False):
            st.markdown('La distribución bimodal sugiere dos grupos diferenciados en madurez digital.', unsafe_allow_html=True)
        st.plotly_chart(fig_hist_index, use_container_width=True)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.plotly_chart(fig_boxplot, use_container_width=True)

with tab3:
    st.markdown(Titulo_dinamico3, unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: justify;">
            Maximiza la página para visualizar los tres Componentes Principales (PC1, PC2 y PC3), los cuales representan:
            <br>- PC1: Actividad financiera.
            <br>- PC2: Servicios digitales.
            <br>- PC3: Adopción financiera.
        </div>
        """, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        with st.expander('Significado de cada Componente Principal', expanded=False):
            st.markdown('PC1 explica aproximadamente el 48.23% de la varianza y se asocia a la actividad financiera; PC2 y PC3 aportan dimensiones complementarias.',
                        unsafe_allow_html=True)
        st.plotly_chart(grafico3d, use_container_width=True, height=500)
        with st.expander('Patrones en los clústers', expanded=False):
            st.markdown('Se observan diferencias claras: el clúster en desarrollo es el más numeroso, mientras que Inicial y Definición muestran mayor cohesión.',
                        unsafe_allow_html=True)
        st.plotly_chart(grafico2d1, use_container_width=True, height=250)
    with col2:
        with st.expander('Estructura de los clústers', expanded=False):
            st.markdown('La visualización 2D complementa el análisis 3D mostrando la separación y solapamiento entre clústers.', unsafe_allow_html=True)
        st.plotly_chart(grafico2d2, use_container_width=True, height=250)
        with st.expander('Perfil del municipio', expanded=False):
            st.markdown('Se detalla la posición y características del municipio dentro de su clúster.', unsafe_allow_html=True)
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
        st.markdown('Los diagramas de dispersión muestran las relaciones entre variables y se complementan con la línea de regresión y R² ajustada.',
                    unsafe_allow_html=True)
    st.plotly_chart(fig_scatter, use_container_width=True, height=500)

with tab6:
    with st.expander('Análisis', expanded=False):
        st.markdown(
            'La clasificación del aprendizaje automático indica que la madurez digital no es aleatoria, sino que sigue patrones asociados a factores financieros, socio-económicos y geográficos.',
            unsafe_allow_html=True)
        st.markdown(
            'El mapa muestra una concentración de municipios con madurez óptima (verde) alrededor de zonas metropolitanas y en el norte, mientras que los municipios en desarrollo (rojo) se agrupan en la región central y sur.',
            unsafe_allow_html=True)
        st.markdown(
            'También se aprecian clústers en la península de Yucatán y zonas periféricas a grandes centros urbanos.',
            unsafe_allow_html=True)
        st.markdown(
            'Si deseas más detalles, [escríbeme](mailto:rodrigo.guarneros@gmail.com).',
            unsafe_allow_html=True)
    st.plotly_chart(fig_map_final, use_container_width=True, height=500)

# Pie de página
st.markdown("""
---
<p style="text-align: center; font-size: 14px;">
    © 2024 Rodrigo Guarneros. Todos los derechos reservados.<br>
    Para sugerencias o consultas, <a href="mailto:rodrigo.guarneros@gmail.com">escríbeme</a>.
</p>
""", unsafe_allow_html=True)
