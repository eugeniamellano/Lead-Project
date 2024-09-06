import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_lottie import st_lottie
import numpy as np
from PIL import Image
import requests
import json
import openpyxl
import streamlit as st
import json
# CSS para centrar el título y estilizarlo
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-family: 'Arial Black', Gadget, sans-serif;
        font-size: 48px;
        color: #FF0000;
        animation: pulse 2s infinite;
    }
    
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
    """, unsafe_allow_html=True)

# Título con clase CSS personalizada
st.markdown('<h1 class="title">FRAUD DASHBOARD</h1>', unsafe_allow_html=True)


# Función para cargar una animación Lottie desde un archivo local
def load_lottie_file(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

# Cargar las animaciones desde archivos locales
lottie_1 = load_lottie_file("Animation - 1725018690774.json")
lottie_2 = load_lottie_file("Animation - 1724966619023.json")
lottie_3 = load_lottie_file("Animation - 1724967109842.json")
lottie_4 = load_lottie_file("Animation - 1724966741508.json")

st_lottie(lottie_1, key="animation4", height=200)


# Cargar el archivo JSON local
##with open('Animation - 1725018690774.json', 'r') as f:
#    lottie_animation = json.load(f)

# Usar st_lottie con el archivo cargado
#st_lottie(lottie_animation, key="animation1", height=200)

#st_lottie("https://lottie.host/bac139a9-4838-4f18-8e6e-b36a44e5cac0/ASbLTPuSoK.json")
# Data URL
DATA_URL = 'fraud_data.csv'
DATA_COMPLETE = 'fraudTest.csv'



### App


st.subheader("""
   In this dashboard, you will be able to explore different characteristics of fraudulent transactions and view the results of fraud alerts.
""")


# Data loading
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    return data

@st.cache_data
def load_complete_data():
    data = pd.read_csv(DATA_COMPLETE)
    return data

df_fraud = load_data()

df_fraud_complete = load_complete_data()


# Displaying the first few rows of data
st.subheader("🔸 Visualization of Fraud Transactions")
st.markdown('')
df_example = df_fraud
df_example['prediction_result'] = df_example['prediction_result'].replace(0, 'FRAUD')
st.write(df_fraud.head())

# Título para la sección de chequeo de transacciones
st.subheader("🔸 Check the Transaction")

# Input para que el usuario ingrese el número de transacción
trans_num_input = st.text_input("Enter transaction number:")

# Verificación cuando se ingresa un número de transacción
if trans_num_input:
    # Convertir todos los IDs a string para asegurar la comparación
    df_fraud_complete['trans_num'] = df_fraud_complete['trans_num'].astype(str)
    df_fraud['transaction_id'] = df_fraud['transaction_id'].astype(str)
    trans_num_input = str(trans_num_input)
    
    # Buscar en df_fraud_complete
    transaction_complete = df_fraud_complete[df_fraud_complete['trans_num'] == trans_num_input]
    
    # Buscar en df_fraud si no se encontró en df_fraud_complete
    transaction_fraud = df_fraud[df_fraud['transaction_id'] == trans_num_input]
    
    # Verificar si la transacción existe en df_fraud_complete
    if not transaction_complete.empty:
        if transaction_complete['is_fraud'].values[0] == 0:
            st.error("FRAUD")
        else:
            st.success("NO FRAUD")
    
    # Si no se encuentra en df_fraud_complete, verificar en df_fraud
    elif not transaction_fraud.empty:
        if transaction_fraud['prediction_result'].values[0] == "FRAUD":
            st.error("FRAUD")
        else:
            st.success("NO FRAUD")
    
    # Si no se encuentra en ninguno de los DataFrames
    else:
        st.warning("THIS TRANSACTION DOES NOT EXIST")



st.markdown('If there is a fraudulent transaction an email is going to be sent')
# Función para cargar la animación Lottie desde una URL
#def load_lottie_url(url):
#    return url

# Crear una fila con tres columnas
col1, col2, col3 = st.columns(3)

# Mostrar cada animación en su respectiva columna
with col1:
    st_lottie(lottie_2, key="animation1", height=200)

with col2:
    st_lottie(lottie_3, key="animation2", height=200)

with col3:
    st_lottie(lottie_4, key="animation3", height=200)

# Cargar las animaciones
#lottie_1 = load_lottie_url("https://lottie.host/74ff7936-bedf-4be8-8a25-3027d3ef1a26/ItZn7RRYeZ.json")
#lottie_2 = load_lottie_url("https://lottie.host/20c7e3b4-10f8-4b40-b499-dbcbb5431d5a/i5Ko1NtoRw.json")
#lottie_3 = load_lottie_url("https://lottie.host/dd722392-64d2-4bda-9681-3e7ca465547c/UEq83rQt1z.json")




# Crear una fila con tres columnas
#col1, col2, col3 = st.columns(3)

# Mostrar cada animación en su respectiva columna
#with col1:
#    st_lottie(lottie_1, key="animation2", height=200)

#with col2:
#    st_lottie(lottie_2, key="animation3", height=200)

#with col3:
#    st_lottie(lottie_3, key="animation4", height=200)

# Gráfico de distribución de fraudes con colores personalizados
fig = px.histogram(df_fraud_complete, 
                   x=df_fraud_complete['is_fraud'].map({0: 'No Fraud', 1: 'Fraud'}), 
                   title='Faude Distribution',
                   text_auto=True,
                   barmode='group',
                   color=df_fraud_complete['is_fraud'].map({0: 'No Fraud', 1: 'Fraud'}),
                   color_discrete_map={'No Fraud': 'green', 'Fraud': 'red'})

fig.update_layout(xaxis_title=None, yaxis_title='Frecuencia')

# Mostrar gráfico en Streamlit
st.plotly_chart(fig)


# Crear columnas para organizar los gráficos
col1, col2 = st.columns(2)

# Primer gráfico: Distribución de AMT para transacciones fraudulentas y no fraudulentas
with col1:
    fig1 = px.histogram(df_fraud_complete[df_fraud_complete.amt <= 1200], 
                        x='amt', 
                        color='is_fraud', 
                        nbins=45, 
                        barmode='overlay', 
                        histnorm='percent',
                        color_discrete_map={0: 'green', 1: 'red'},  # Asignar colores específicos
                        title="Amount Distribution for Fraudulent and Non-Fraudulent Transactions")

    fig1.update_layout(xaxis_title="Amount", yaxis_title="Percentage")
    st.plotly_chart(fig1)

# Segundo gráfico: Box plot de AMT con límite superior en el eje Y
with col2:
    fig2 = px.box(df_fraud_complete, y='amt', title='Box Plot of AMT')
    fig2.update_yaxes(range=[0, 250])
    st.plotly_chart(fig2)


# Nuevo gráfico: Distribución de categorías por estado de fraude
st.subheader("Category Distribution by Fraud Status")
fig3 = px.histogram(df_fraud_complete, 
                    x='category', 
                    color=df_fraud_complete['is_fraud'].map({0: 'No Fraud', 1: 'Fraud'}), 
                    title='Category Distribution by Fraud Status',
                    color_discrete_map={'No Fraud': 'green', 'Fraud': 'red'},
                    barmode='stack',  # barras se apilan una sobre otra
                    text_auto=True    # valores en las barras
                   )

fig3.update_layout(xaxis_title='Category', yaxis_title='Frequency')
st.plotly_chart(fig3)




st.markdown("Created by Eugenia M")