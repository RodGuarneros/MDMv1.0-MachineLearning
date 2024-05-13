#######################
# Import libraries
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





# Page configuration
st.set_page_config(
    page_title="Aprendizaje Automático para los Municipios de México",
    page_icon="📱💻📶📊",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
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


# Load data
datos = pd.read_csv('Finale.csv', encoding='Latin1') # piramides
input_datos = datos
variable_list_numerica = list(input_datos.select_dtypes(include=['int64', 'float64']).columns)
variable_list_categoricala = list(input_datos.select_dtypes(include=['object']).columns)

columns_to_exclude_numeric = ['Unnamed: 0', 'cve_edo', 'cve_municipio', 'cvegeo', 'estrato_icm', 'estrato_iddm', 'muincipio']
columns_to_exclude_categorical = ['estado', 'lugar']

# Remove excluded columns from numeric and categorical lists
variable_list_numeric = [col for col in variable_list_numerica if col not in columns_to_exclude_numeric]
variable_list_categorical = [col for col in variable_list_categoricala if col not in columns_to_exclude_categorical]



# Path to the ZIP file
zip_file_path = 'to_mapping.zip'

# Name of the CSV file inside the ZIP archive
csv_file_name = 'to_mapping.geojson'

# Extract the CSV file from the ZIP archive
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extract(csv_file_name)

# Read the CSV file into a DataFrame
datos_map = gpd.read_file("to_mapping.geojson")

input_map = datos_map


# Sidebar
with st.sidebar:
    # st.title('Proyecto de Titulación <br> México')
    st.markdown("<h3 style='text-align: center;'>Aprendizaje Automático para Clasificar Municipios<br> por su Vocación Digital</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'> <hr>INFOTEC (Primera Etapa AED)<hr> Trabajo de Investigación presentado por Rodrigo Guarneros</h5>", unsafe_allow_html=True)
    st.sidebar.image("https://img.cdn-pictorem.com/uploads/collection/L/LD5RFK7TTK/900_Grasshopper-Geography_Elevation_map_of_Mexico_with_black_background.jpg", use_column_width=True)

    variable_seleccionada_numerica = st.selectbox('Selecciona la variable numérica de interés:', sorted(variable_list_numeric, reverse=False))
    variable_seleccionada_categorica = st.selectbox('Selecciona la variable categórica de interés:', sorted(variable_list_categorical, reverse=False))
    variable_seleccionada_paracorrelacion = st.selectbox('Selecciona la variable que quieras correlaccionar con la primera selección:', sorted(variable_list_numeric, reverse=False))

    # # Entidad
    # entidad_list = list(df_reshaped.ENTIDAD.unique())
    # selected_entidad = st.selectbox('Seleccione la entidad o República Mexicana:', sorted(entidad_list, reverse=False))
    # df_selected_entidad = df_reshaped[df_reshaped.ENTIDAD == selected_entidad]
    # input_entidad = df_selected_entidad
    # df_selected_entidad_sorted = df_selected_entidad.sort_values(by="POBLACION", ascending=False)

    # # Género
    # genero_list = list(df_reshaped.SEXO.unique())
    # selected_genero = st.selectbox('Seleccione por género o datos totales:', sorted(genero_list, reverse=True))
    # df_selected_genero = df_reshaped[df_reshaped.SEXO == selected_genero]
    # input_genero = df_selected_genero
    # df_selected_genero_sorted = df_selected_genero.sort_values(by="POBLACION", ascending=False)

    with st.expander('Enfoque del panel de control', expanded=False):
        st.write('''
            - Se basa en un enfoque de <span style="color:#C2185B">"Programación Orientada a Objetos"</span>.
            - La población se puede modelar a partir de sus atributos y funciones que en escencia definen sus características y capacidades para aprovechar la revolución digital, respectivamente. 
            - Este primer análisis exploratorio de la información disponible forma parte de un proyecto integral que busca: <span style="color:#C2185B">Ajustar un modelo de aprendizaje automático para clasificar a las localidades de México por su vocación para la transformación digital y despliegue de servicios TIC, en función de variables fundamentales de infraestructura, demográficas y socio-económicas.</span>
            - Este aplicativo incluye atributos a nivel municipal tales como:
                1. Número de viviendas. 
                2. Grado educativo (Analfabetismo, Porcentaje de personas con educación básica, etc.).
                3. Edad promedio, entre otas.
            - Con base en estas características, se pueden generar diferentes combinaciones y visualizaciones de interés para conocer mejor aspectos como:
                1. La distribución estadística de las variables. 
                2. Relación entre las variables. 
                3. La distribución geográfica de las variables.
            - La ventaja de un panel de control como este consiste en sus <span style="color:#C2185B">economías de escala y la capacidad que tiene para presentar insights más profundos respecto a la población y sus funciones o actividades, tales como capacidad adquisitiva, preferencias, crédito al consumo, acceso a servicios de conectividad, empleo, sequías y hasta modelos predictivos.</span> 
            ''', unsafe_allow_html=True)



    with st.expander('Fuentes y detalles técnicos', expanded=False):
        st.write('''
            - Fuente: [Consejo Nacional de Población (CONAPO), consultado el 3 de febrero de 2024.](https://www.gob.mx/conapo).
            - Tecnologías y lenguajes: Python 3.10, Streamlit 1.30.0, CSS 3.0, HTML5, Google Colab y GitHub. 
            - Autor: Rodrigo Guarneros ([LinkedIn](https://www.linkedin.com/in/guarneros/) y [X](https://twitter.com/RodGuarneros)).
            - Comentarios al correo electrónico rodrigo.guarneros@gmail.com
            ''', unsafe_allow_html=True)

##############
# Histograma #
##############

def plot_histogram(df, numeric_column, categorical_column):
    """
    Elaborada por Rodrigo Guarneros
    
    """
    # Map colors manually
    color_map = {'Sí': 'blue', 'No': 'red'}
    
    # Create the histogram
    fig = px.histogram(df, x=numeric_column, color=categorical_column,
                       color_discrete_map=color_map,
                       opacity=0.6,
                       title=f'Histograma de la variable "{numeric_column}" y la categoría "{categorical_column}"')
    
    # Update axis titles
    fig.update_xaxes(title_text="Rangos de valor")
    fig.update_yaxes(title_text="Frecuencia absoluta")
    
    # Add mean, std, median, and mode as annotations
    mean = df[numeric_column].mean()
    std = df[numeric_column].std()
    median = df[numeric_column].median()
    mode = df[numeric_column].mode()[0]
    
    # Add annotations to a list
    annotations = [
        dict(x=mean, y=0, xref='x', yref='y', text=f'Media: {mean:.2f}', showarrow=True, arrowhead=3, ax=300, ay=-59),
        dict(x=median, y=0, xref='x', yref='y', text=f'Mediana: {median:.2f}', showarrow=True, arrowhead=3, ax=300, ay=-69),
        dict(x=mode, y=0, xref='x', yref='y', text=f'Moda: {mode:.2f}', showarrow=True, arrowhead=3, ax=300, ay=-79),
        dict(x=mean + std, y=0, xref='x', yref='y', text=f'Desviación estándar: {std:.2f}', showarrow=True, arrowhead=3, ax=300, ay=-49)
    ]
    
    # Add total count of each category as annotations in the right top corner
    category_counts = df[categorical_column].value_counts()
    total_text = '<br>'.join([f'{category}: {count}' for category, count in category_counts.items()])
    fig.add_annotation(
        dict(x=.89, y=.89, xref='paper', yref='paper', text=total_text, showarrow=False,
             font=dict(size=13, color='white'), align='right', bgcolor='rgba(0, 0, 0, 0.5)', bordercolor='rgba(0, 0, 0, 0.5)', borderwidth=2)
    )
    
    # Add annotations to the layout
    for annotation in annotations:
        fig.add_annotation(annotation)
        
    return fig

# Example usage:
fig_hist = plot_histogram(input_datos, variable_seleccionada_numerica, variable_seleccionada_categorica)

######################
##### BOX PLOT #######
######################

def generate_boxplot_with_annotations(df, variable):
    # Calculate statistics
    mean_val = np.mean(df[variable])
    median_val = np.median(df[variable])
    mode_val = df[variable].mode().iloc[0]
    std_val = np.std(df[variable])
    
    # Convert numeric columns to string before concatenation
    df['muincipio'] = df['muincipio'].astype(str)
    df['estado'] = df['estado'].astype(str)

    df['lugar'] = df['lugar_1']

    # Create box plot
    fig = px.box(
        df,
        y=variable,
        points='all',
        title=f'Diagrama para la variable "{variable}"',
        labels={variable: variable},
        template='plotly_dark',
        hover_data={'lugar': True} 
    )

    # Add annotations
    annotations = [
        dict(
            x=0.5,
            y=mean_val,
            xref='paper',
            yref='y',
            text=f'Media: {mean_val:.2f}',
            showarrow=True,
            arrowhead=1,
            ax=80,
            ay=-20
        ),
        dict(
            x=0.5,
            y=median_val,
            xref='paper',
            yref='y',
            text=f'Mediana: {median_val:.2f}',
            showarrow=True,
            arrowhead=1,
            ax=80,
            ay=0
        ),
        dict(
            x=0.5,
            y=mode_val,
            xref='paper',
            yref='y',
            text=f'Moda: {mode_val}',
            showarrow=True,
            arrowhead=1,
            ax=80,
            ay=20
        ),
        dict(
            x=0.8,
            y=mean_val + std_val,
            xref='paper',
            yref='y',
            text=f'Desviación estándar: {std_val:.2f}',
            showarrow=True,
            arrowhead=1,
            ax=50,
            ay=10
        )
    ]

    # Update layout with annotations
    fig.update_layout(annotations=annotations)
    
    fig.update_traces(marker=dict(opacity=0.8, line=dict(color='rgba(255, 165, 0, 0.5)', width=1)))

    # Add 'Municipalidad' and variable values to the tooltip
    fig.update_traces(hovertemplate='<b>Municipality:</b> %{customdata[0]}<br><b>'+variable+':</b> %{y}')
    fig.update_layout(title_font=dict(color='#FFD86C'), xaxis_title_font=dict(color='#FFD86C'), yaxis_title_font=dict(color='#FFD86C'))    
    
    return fig
    
fig_boxplot = generate_boxplot_with_annotations(input_datos, variable_seleccionada_numerica)    

##############
### Scatter ##
##############

def generate_scatter_with_annotations(df, x_variable, y_variable, categorical_variable):
    # Create scatter plot with colored dots based on categorical variable
    fig = px.scatter(df, x=x_variable, y=y_variable, hover_data={'lugar': True, categorical_variable: True}, color=categorical_variable,
                     color_discrete_map={'No': 'pink', 'Sí':'blue'})

    # Calculate correlation coefficient
    correlation_coef = df[x_variable].corr(df[y_variable])

    # Calculate adjusted R2
    n = len(df)
    p = 1  # Assuming only one independent variable
    r_squared = correlation_coef ** 2
    r_squared_adj = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))

    # Update layout
    fig.update_layout(
        plot_bgcolor='rgb(30,30,30)',  # Set dark background color
        paper_bgcolor='rgb(30,30,30)',  # Set dark paper color
        font_color='white',  # Set font color to white
        title=dict(
            text=f"Diagrama de dispersión entre las variables '{x_variable}' y '{y_variable}'",
            font=dict(color='white')  # Set title font color to white
        ),
        xaxis=dict(
            title=f"Variable {x_variable}",  # Set x-axis title
            titlefont=dict(color='white'),  # Set x-axis title font color to white
            tickfont=dict(color='white')  # Set x-axis tick font color to white
        ),
        yaxis=dict(
            title=f"Variable {y_variable}",  # Set y-axis title
            titlefont=dict(color='white'),  # Set y-axis title font color to white
            tickfont=dict(color='white')  # Set y-axis tick font color to white
        ),
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=0.95,
                y=1.05,
                text=f'R² ajustada: {r_squared_adj:.4f}',
                showarrow=False,
                font=dict(color='orange')
            )
        ]
    )
    
    # Update hover template to include Municipio and categorical variable
    fig.update_traces(hovertemplate='<b>Municipio</b>: %{customdata[0]}<br>' +
                                    f'<b>{x_variable}</b>: %{{x}}<br>' +
                                    f'<b>{y_variable}</b>: %{{y}}<br>'
                     )
    fig.update_traces(marker=dict(opacity=0.6, line=dict(color='rgba(255, 165, 0, 0.5)', width=1)))

    return fig

# Call the function to generate scatter plot
fig_scatter = generate_scatter_with_annotations(input_datos, variable_seleccionada_numerica, variable_seleccionada_paracorrelacion, variable_seleccionada_categorica)

######################
######## MAPA ########
######################

# def map_municipios(input, variable_selected):
#     # Center the map on Mexico
#     mexico_center = [23.6345, -102.5528]  # Latitude and longitude of Mexico

#     # Create the map centered on Mexico with dark layout
#     municipios_map = folium.Map(location=mexico_center, zoom_start=5, tiles='CartoDB dark_matter')

#     # Calculate statistics for the selected variable
#     selected_variable_data = input[variable_selected]
#     variable_mean = selected_variable_data.mean()
#     variable_median = selected_variable_data.median()
#     variable_std = selected_variable_data.std()

#     variable_mean = '{:.2f}'.format(variable_mean)
#     variable_median = '{:.2f}'.format(variable_median)
#     variable_std = '{:.2f}'.format(variable_std)

#     # Add Choropleth layer to the map with tooltip
#     folium.GeoJson(
#         input,
#         name='choropleth',
#         style_function=lambda feature: {
#             'fillColor': color_producer(feature['properties'][variable_selected]),
#             'color': 'black',
#             'weight': 1,
#             'fillOpacity': 0.6,
#         },
#         highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.7},
#         tooltip=folium.features.GeoJsonTooltip(
#             fields=['lugar', variable_selected],
#             aliases=['Nombre del Municipio', f'{variable_selected}:'],
#             localize=True,
#             sticky=True,
#         )
#     ).add_to(municipios_map)

#     # Add notation for calculated fields on the top right corner
#     notation = f"<div style='color: white; font-family: Arial; font-size: 8pt'>Estadísticos Nacionales: <span style='color: green;'>{variable_selected}<br>Media:<span style='color: green;'>{variable_mean}</span><br>Mediana:<span style='color: green;'>{variable_median}<br>Desviación estándar:<span style='color: green;'>{variable_std}</div>"
#     folium.Marker(
#         [mexico_center[0]-4, mexico_center[1]-15],  # Adjust position as needed
#         icon=folium.DivIcon(html=notation),
#         tooltip=None
#     ).add_to(municipios_map)

#     # Return the map
#     return municipios_map


# # Assuming merged_gdf is your GeoDataFrame
# df = datos_map

# # Define a function to map values to colors
# def color_producer(value):
#     if value is None:
#         return '#808080'  # grey for NaN values
#     elif value < df[variable_seleccionada_numerica].quantile(0.33):
#         return '#ff222b'  # red for lower values
#     elif value < df[variable_seleccionada_numerica].quantile(0.66):
#         return '#fad8b1'  # yellow for middle values
#     else:
#         return '#3cf60e'  # green for higher values


# # Create the map
# fig_map = map_municipios(datos_map, variable_seleccionada_numerica)

# Convert the Folium map to HTML


#########################
### Título Dinámico #####
#########################
def titulo_dinamico(variable):

    # Set a yellow color for the title
    styled_title = f'<span style="color: #FFD86C; font-size: 30px; font-weight: bold;">La variable mostrada es: "{variable}".</span>'

    return styled_title

Titulo_dinamico = titulo_dinamico(variable_seleccionada_numerica)

# Dashboard Main Panel
st.markdown(Titulo_dinamico, unsafe_allow_html=True)
# calculos_df
# Define the tabs
tab1, tab2, tab3, tab4 = st.tabs(["Histograma","Gráfica de Caja", "Correlaciones","Mapa"])

# El histograma
with tab1:
    with st.expander('Análisis', expanded=False):
        # st.markdown(f'La población de <span style="color:#C2185B">{variable_seleccionada}</span> seguirá enfrentando cambios radicales. La tasa de crecimiento anual en <span style="color:#C2185B">{}</span> es de <span style="color:#C2185B">{calculos_df.Crecimiento.iloc[0]:,.1f}%</span>.', unsafe_allow_html=True)
        st.markdown(f'En esta visualización se pretenden calcular las estadísticas de tendencia central por variable disponible y la distribución ilustrada en un histograma donde se distinguen, en su caso, las categorías relacionadas con tres variables diferentes.', unsafe_allow_html=True)
        st.markdown(f'La primera vairable categórica es: <span style="color:#C2185B"> la disponibilidad de servicios de televisión o audio restringido (TAR), cuyas categorías posibles son: </span><span style="color:#C2185B">Sí tienen disponibilidad o No tienen disponibilidad</span>.', unsafe_allow_html=True)
        st.markdown(f'La segunda variable categórica es: <span style="color:#C2185B"> la isponibilidad de banda ancha fija móvil, con posibles valores: Sí y No</span>.', unsafe_allow_html=True)
        st.markdown(f'La tercera variable categórica es: <span style="color:#C2185B"> la disponibilidad de banda ancha fija alámbrica, con posibles valores: Sí y No</span>.', unsafe_allow_html=True)
    st.plotly_chart(fig_hist, use_container_width=True, height=500)

# El diagrama de caja
with tab2:
    with st.expander('Análisis', expanded=False):
        # st.markdown(f'La población de <span style="color:#C2185B">{variable_seleccionada}</span> seguirá enfrentando cambios radicales. La tasa de crecimiento anual en <span style="color:#C2185B">{}</span> es de <span style="color:#C2185B">{calculos_df.Crecimiento.iloc[0]:,.1f}%</span>.', unsafe_allow_html=True)
        st.markdown(f'Los diagramas de caja tienen la peculiaridad de visualizar claramente los cuartiles de la distribución y los datos aberrantes.', unsafe_allow_html=True)
        st.markdown(f'<span style="color:#C2185B">Se trata de un primera acercamiento <span style="color:#C2185B">donde es posible ver en qué variables tiene más brechas cada municipio, dónde son más similares, qué característica tiene mayor dispersión</span>.', unsafe_allow_html=True)
        
    st.plotly_chart(fig_boxplot, use_container_width=True, height=500)

# La correlacion
with tab3:
    with st.expander('Análisis', expanded=False):
        # st.markdown(f'La población de <span style="color:#C2185B">{variable_seleccionada}</span> seguirá enfrentando cambios radicales. La tasa de crecimiento anual en <span style="color:#C2185B">{}</span> es de <span style="color:#C2185B">{calculos_df.Crecimiento.iloc[0]:,.1f}%</span>.', unsafe_allow_html=True)
        st.markdown(f'Los diagramas de dispersión permiten visualizar las relaciones lineales y no lineales de las variables.', unsafe_allow_html=True)
        st.markdown(f'<span style="color:#C2185B">Se trata de un primera acercamiento <span style="color:#C2185B">donde es importante recordar que una alta correlación no necesariamente implica causalidad.</span>.', unsafe_allow_html=True)
        st.markdown(f'Vale la pena recordar que la R² ajustada se interpreta como el porcentaje de la varianza de la variable dependiente (eje de las Y) que es explicada por la variable independiente (eje de las X).  La R² ajustada es una medida de la bondad de ajuste de un modelo de regresión lineal. Representa el porcentaje de la varianza de la variable dependiente (eje Y) que es explicada por la variable independiente (eje X) después de ajustar el modelo para tener en cuenta el número de predictores en el modelo y el tamaño de la muestra. En otras palabras, la R² ajustada penaliza la inclusión de términos en el modelo que no mejoran significativamente la capacidad predictiva', unsafe_allow_html=True)
    st.plotly_chart(fig_scatter, use_container_width=True, height=500)

with tab4:
    with st.expander('Análisis', expanded=False):
        st.markdown(f'El mapa que aquí se presenta permite visualizar la distribución geográfica de cada variable para efectos de identificar efectos regionales.', unsafe_allow_html=True)
        st.markdown(f'<span style="color:#C2185B">Se trata de un primer acercamiento <span style="color:#C2185B">donde es importante recordar que este mapa es una representación visual que nos permite identificar tendencias relevantes a considerar para la construcción del modelo de aprendizaje automático predictivo.</span>', unsafe_allow_html=True)
    folium_map_html = fig_map._repr_html_()
    st.components.v1.html(folium_map_html, width=800, height=600)
