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



# Page configuration
st.set_page_config(
    page_title="Radiografía de la Transformación Digital",
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

# datos_map = pd.read_csv('for_map_df.csv', encoding='Latin1')
# input_map = datos_map
# df_reshaped = pd.read_csv('data/result_sorted_final.csv', encoding='utf-8')
# input_df = df_reshaped.sort_values(by="ENTIDAD", ascending=True)
# input_df = df_reshaped # cholopleth, ranking_edad evolucion_poblacion
# df_reshaped_2 = pd.read_csv('data/result_sorted2_hist.csv', encoding='Latin1') # histograma
# input_hist = df_reshaped_2
# df_calculos = pd.read_csv('calculos.csv', encoding='Latin1')


# Sidebar
with st.sidebar:
    # st.title('Proyecto de Titulación <br> México')
    st.markdown("<h1 style='text-align: center;'>Transformación Digital: <br>Un Problema de Clasificación Municipal</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'> <hr>INFOTEC <hr> Trabajo Presentado por Rodrigo Guarneros</h5>", unsafe_allow_html=True)
    st.sidebar.image("https://img.cdn-pictorem.com/uploads/collection/L/LD5RFK7TTK/900_Grasshopper-Geography_Elevation_map_of_Mexico_with_black_background.jpg", use_column_width=True)

    variable_seleccionada_numerica = st.selectbox('Selecciona la variable numérica de interés:', sorted(variable_list_numeric, reverse=False))
    variable_seleccionada_categorica = st.selectbox('Selecciona la variable categórica de interés:', sorted(variable_list_categorical, reverse=False))

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


# Histograma

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


######################
######## MAPA ########
######################

# # Load GeoJSON data into GeoDataFrame
# # Assuming merged_gdf is your GeoDataFrame
# def map_municipios(input_map, variable):

#     input_map[variable] = pd.to_numeric(input_map[variable], errors='coerce')

#     # Drop rows with NaN values in 'escolaridad_promedio' column
#     input_map = input_map.dropna(subset=[variable])

# # Define a function to map values to colors
#     def color_producer(value):
#         if value is None:
#             return '#808080'  # grey for NaN values
#         elif value < input_map[variable].quantile(0.33):
#             return 'red'  # red for lower values
#         elif value < input_map[variable].quantile(0.66):
#             return 'yellow'  # yellow for middle values
#         else:
#             return 'green'  # green for higher values

# # Center the map on Mexico
#     mexico_center = [23.6345, -102.5528]  # Latitude and longitude of Mexico

# # Create the map centered on Mexico with dark layout
#     municipios_map = folium.Map(location=mexico_center, zoom_start=5, tiles='CartoDB dark_matter')

# # Add Choropleth layer to the map
#     folium.GeoJson(
#         input_map,
#         name='choropleth',
#         style_function=lambda feature: {
#             'fillColor': color_producer(feature['properties'][variable]),
#             'color': 'black',
#             'weight': 1,
#             'fillOpacity': 0.6,
#         },
#         highlight_function=lambda x: {'weight': 3, 'fillOpacity': 0.7},
#         tooltip=folium.features.GeoJsonTooltip(fields=['lugar', variable],
#                                                 aliases=['Municipality', variable],
#                                                 labels=True,
#                                                 sticky=True)
#     ).add_to(municipios_map)

# # Display the map
#     return municipios_map

# fig_map = map_municipios(input_map, variable_seleccionada_numerica)


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

#     with chart2_col:
#         with st.expander('Perspectivas', expanded=False):
#             st.markdown(f'La edad promedio en <span style="color:#C2185B">{selected_entidad}</span> para el año <span style="color:#C2185B">{selected_year}</span> se registra en <span style="color:#C2185B">{calculos_df.loc[0, "Edad promedio"]:,.2f} años</span>.', unsafe_allow_html=True)
#             st.markdown(f'Claramente y sin excepción, las mujeres superan a los hombres en número.', unsafe_allow_html=True)
#             st.markdown(f'Todos los estados proyectan un giro en la pirámide poblacional donde las personas más jóvenes comienzan a reducirse año con año y la población adulta, incluidos los mayores de 65 años, comienza a aumentar, lo que <span style="color:#C2185B">incrementa la tasa de dependencia (número de personas que no trabaja y tiene más de 65 años, comparada con aquellos que están en edad de trabajar)</span>.', unsafe_allow_html=True)
        
#         st.plotly_chart(fig_piramide, use_container_width=True, height=500)

#     with chart3_col:
#         with st.expander('Perspectivas', expanded=False):
#             st.markdown(f'La edad promedio en <span style="color:#C2185B">{selected_entidad}</span> para el año <span style="color:#C2185B">{selected_year}</span> es de <span style="color:#C2185B">{calculos_df.loc[0, "Edad promedio"]:,.2f} años</span>. Se trata de un estadístico de tendencia central útil. No obstante, ante la existencia de datos aberrantes, se sugiere la mediana de la edad disponible en las última sección de este tablero, cuya cualidad es que es menos sensible a los datos extremos.', unsafe_allow_html=True)
#             st.markdown(f'Si bien excedemos el objetivo de esta app, vale la pena señalar que la distribución por edad tiende a reducir su sesgo y comportarse como una distribución normal en periodos posteriores a 2030. Lo anterior es atribuible a factores tales como: <span style="color:#C2185B">(i) Reducción de las tasas de nacimiento; (ii) Incremento en la expectativa de vida; (iii) Reducción de las tasas de mortalidad; (iv) Factores sociales y económicos; (v) Impacto migratorio</span>.', unsafe_allow_html=True)

        
#         st.plotly_chart(fig_hist, use_container_width=True, height=500)

# # Define the content for tab2
# with tab2:
#     with st.expander('Perspectivas', expanded=False):
#         st.write('''
#                  - En 1970, las cinco entidades federativas más pobladas fueron: Ciudad de México (3.5 M), Estado de México (2.08 M), Veracruz (2.06 M), Jalisco (1.7 M) y Puebla (1.4 M).
#                  - En 2024, la lista de las entidades federativas más pobladas es la siguiente: Estado de México (8.5 M), Ciudad de México (4.4 M), Jalisco (4.3 M), Veracruz (3.9 M) y Puebla (3.4 M).
#                  - Para 2050, las trayectorias poblacionales sugieren que la lista será encabezada por: Estado de México (18.1 M), Jalisco (10.05 M), Nuevo León (8.4 M), Puebla (8.3 M) y Ciudad de México (8.01 M).
#                  - Si nos preguntamos cuál debería ser la tasa de crecimiento anual promedio que cada estado debería experimentar en su población para alcanzar las predicciones de los próximos 26 años, la respuesta es la siguiente: Estado de México (2.9%), Jalisco (3.3%), Nuevo León (1.11%), Puebla (3.5%) y Ciudad de México (2.3%).
#                  - Estas tasas de crecimiento poblacionales son considerablemente altas si se comparan con la media de la tasa de crecimiento anual a nivel mundial, que se espera sea del 1% durante el mismo período.                    
#                  ''')

#     chart1_col, chart2_col = st.columns((1, 1))  # Two columns for Tab2

#     with chart1_col:
#         st.plotly_chart(mapa_poblacion_render2, use_container_width=True, height=500)

#     with chart2_col:
#         st.plotly_chart(fig_ranking2, use_container_width=True)

# # Define the content for tab3
# with tab3:
#     with st.expander('Perspectivas', expanded=False):
#         st.write('''
#                  - La mediana de la edad en 2050, a nivel mundial, se estima en 41 años.  
#                  - En México, en 1970, las cinco entidades federativas con la mediana de edad más alta registrada son: Yucatán (23.4 años), Ciudad de México (22.7 años), Tlaxcala (22.5 años), Nuevo León (22.4 años) y Tamaulipas (22.3 años).
#                  - En 2024, la lista de las entidades federativas con mayor mediana de edad es la siguiente: Ciudad de México (37.3 años), Veracruz (34.1 años), Morelos (33.6 años), Colima (33.6 años) y Tamaulipas (33.3 años).
#                  - Para 2050, las predicciones poblacionales sugieren que la lista estará encabezada por: Ciudad de México (47.8 años), Colima (43.8 años), Veracruz (43.5 años), Morelos (43 años) y Yucatán (43.6 años), mientras que la mediana de la población en todo el país será de 40.9 años.
#                  ''')

#     chart1_col, chart2_col = st.columns((1, 1))  # Two columns for Tab3

#     with chart1_col:
#         st.plotly_chart(mapa_poblacion_render, use_container_width=True, height=500)

#     with chart2_col:
#         st.plotly_chart(fig_ranking, use_container_width=True)