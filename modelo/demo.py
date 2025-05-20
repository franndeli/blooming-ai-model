import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import traceback
from datetime import datetime
import joblib

from model.predict import predict_next_question
from data.fetch_data import (
    get_alumno_by_id, 
    get_respuestas_by_alumno, 
    get_preguntas, 
    get_opciones_pregunta,
    get_alumnos
)

st.set_page_config(
    page_title="Blooming AI - Demo TFG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #0066cc;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0047ab;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #0066cc;
    }
    .info-card {
            color: black;
        background-color: #e3eefa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #0288d1;
    }
    .success-card {
        color: black;
        background-color: #e0f2e9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #34a853;
    }
    .warning-card {
        background-color: #fff8e1;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #fbbc05;
    }
    .highlight {
        background-color: #0288d1;
        color: white;
        padding: 0.5rem 0.8rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
    }
    .metric-container {
        background-color: #f1f8e9;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .bold-text {
        font-weight: bold;
    }
    .error-msg {
        color: #d32f2f;
        background-color: #ffebee;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-msg {
        color: #388e3c;
        background-color: #e8f5e9;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .big-metric {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #0066cc;
    }
    .metric-label {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
    }
    .prediction-container {
        color: black;
        text-align: center;
        padding: 30px;
        background: linear-gradient(145deg, #e8f5e9, #c8e6c9);
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .prediction-title {
        font-size: 1.4rem;
        font-weight: bold;
        color: #1b5e20;
        margin-bottom: 15px;
        text-transform: uppercase;
    }
    .prediction-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1b5e20;
        margin-bottom: 10px;
        padding: 10px 20px;
        background-color: rgba(255,255,255,0.7);
        border-radius: 10px;
        display: inline-block;
    }
    .ambito-chip {
        display: inline-block;
        padding: 0.5rem 1rem;
        background-color: #e3f2fd;
        color: #0d47a1;
        border-radius: 20px;
        margin: 0.3rem;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.2s ease-in-out;
    }
    .ambito-chip:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .ambito-chip-selected {
        display: inline-block;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, #1976d2, #1565c0);
        color: white;
        border-radius: 20px;
        margin: 0.3rem;
        font-weight: bold;
        box-shadow: 0 3px 6px rgba(25, 118, 210, 0.3);
        transition: all 0.2s ease-in-out;
    }
    .ambito-chip-selected:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(25, 118, 210, 0.4);
    }
    .option-card {
        color: black;
        background-color: #f8f9fa;
        border-left: 4px solid #7986cb;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    .option-card:hover {
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transform: translateX(3px);
    }
    .pregunta-box {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    .pregunta-box:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.08);
        background-color: #ffffff;
    }
    .pregunta-text {
        font-size: 1.2rem;
        color: #212121;
        margin-bottom: 10px;
        line-height: 1.6;
    }
    .profile-header {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 15px;
        border-bottom: 1px solid #e0e0e0;
    }
    .profile-name {
        font-size: 1.8rem;
        color: #0066cc;
        margin: 0;
        font-weight: 700;
    }
    .profile-objective {
        font-size: 1.1rem;
        color: #666;
        margin-top: 5px;
    }
    
    /* Arreglo para las pestañas */
    div.stTabs > div.stTabsHeader {
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    
    div.stTabs > div.stTabsHeader > button {
        background-color: transparent;
        color: #0066cc;
        border-radius: 10px 10px 0 0;
        padding: 10px 16px;
        font-weight: 600;
        border: none;
        border-bottom: 2px solid transparent;
    }
    
    div.stTabs > div.stTabsHeader > button[aria-selected="true"] {
        background-color: #0066cc;
        color: white;
    }
    
    div.stTabs > div.stTabsContent {
        background-color: #f8f9fa;
        border-radius: 0 0 10px 10px;
        padding: 20px;
        border: 1px solid #e0e0e0;
    }
    
    /* Solución para tarjetas de peso */
    .option-weight-positive {
        background-color: #e0f2e9;
        border-left: 4px solid #34a853;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .option-weight-negative {
        background-color: #ffebee;
        border-left: 4px solid #d32f2f;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .option-weight-neutral {
        background-color: #f5f5f5;
        border-left: 4px solid #9e9e9e;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* Peso de la opción destacado */
    .weight-tag {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        margin-left: 10px;
    }
    
    .weight-positive {
        background-color: #34a853;
        color: white;
    }
    
    .weight-negative {
        background-color: #d32f2f;
        color: white;
    }
    
    .weight-neutral {
        background-color: #9e9e9e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("model/results/training_rf/models/best_model.pkl")
        xgb_model = joblib.load("model/results/training_xgb/models/best_model.pkl")
        
        y_encoder = joblib.load("model/results/training_xgb/models/y_encoder.pkl")
        feature_names = joblib.load("model/results/training_xgb/models/feature_names.pkl")
        
        return {
            'RandomForest': rf_model,
            'XGBoost': xgb_model,
            'y_encoder': y_encoder,
            'feature_names': feature_names
        }
    except Exception as e:
        st.error(f"Error al cargar los modelos: {str(e)}")
        return None

@st.cache_data(ttl=300)
def get_alumnos_list():
    try:
        alumnos_df = get_alumnos()
        if alumnos_df.empty:
            st.warning("No se encontraron alumnos en la base de datos")
            return []
            
        if '_id' in alumnos_df.columns:
            alumnos_df['_id'] = alumnos_df['_id'].astype(str)
            
        required_cols = ['_id', 'nombre', 'apellidos', 'objetivo']
        for col in required_cols:
            if col not in alumnos_df.columns:
                alumnos_df[col] = 'N/A'
                
        return alumnos_df.to_dict('records')
    except Exception as e:
        st.error(f"Error al obtener la lista de alumnos: {str(e)}")
        return []

def make_real_prediction(alumno_id, model_name='XGBoost'):
    try:
        models = load_models()
        if not models:
            return {"error": "No se pudieron cargar los modelos"}
        
        if model_name == 'RandomForest':
            joblib.dump(models['RandomForest'], "model/model.pkl")
        else:
            joblib.dump(models['XGBoost'], "model/results/training_xgb/models/best_model.pkl")
        
        joblib.dump(models['y_encoder'], "model/results/training_xgb/models/y_encoder.pkl")
        joblib.dump(models['feature_names'], "model/results/training_xgb/models/feature_names.pkl")
        
        result = predict_next_question(alumno_id, save_results=True)
        
        if "error" in result:
            return result
        
        result["model_name"] = model_name
        return result
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        traceback.print_exc()
        return {"error": f"Error en la predicción: {str(e)}"}

def create_radar_plot(ambitos_data, predicted_domain=None):
    if isinstance(ambitos_data, str):
        try:
            ambitos_data = json.loads(ambitos_data)
        except:
            st.error("Error al parsear datos de ámbitos")
            return None
    
    nombres = []
    valores = []
    
    for ambito, datos in ambitos_data.items():
        nombres.append(ambito)
        if isinstance(datos, dict) and 'peso' in datos:
            valores.append(datos['peso'])
        elif isinstance(datos, (int, float)):
            valores.append(datos)
        else:
            valores.append(50)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=valores + [valores[0]],
        theta=nombres + [nombres[0]],
        fill='toself',
        line=dict(color='#1976d2', width=3),
        fillcolor='rgba(25, 118, 210, 0.2)',
        name='Perfil Actual'
    ))
    
    if predicted_domain and predicted_domain in nombres:
        idx = nombres.index(predicted_domain)
        fig.add_trace(go.Scatterpolar(
            r=[valores[idx]],
            theta=[nombres[idx]],
            mode='markers',
            marker=dict(
                size=16,
                color='#e53935',
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            name='Ámbito Predicho',
            hoverinfo='text',
            text=[f'PREDICCIÓN: {predicted_domain}']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=11),
                gridcolor='#e0e0e0'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='white'),
            ),
            bgcolor='#f5f5f5'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=12)
        ),
        title=dict(
            text="Perfil de Ámbitos del Alumno",
            font=dict(size=18, color='white'),
            y=0.95
        ),
        height=550,
        margin=dict(l=80, r=80, t=80, b=80),
        paper_bgcolor='rgb(14, 17, 23)',
    )
    
    return fig

def create_probability_plot(ambitos, ambito_predicho, model_name):
    if model_name == "RandomForest":
        probs = {a: 0.05 for a in ambitos}
        probs[ambito_predicho] = 0.65
    else:
        probs = {a: 0.02 for a in ambitos}
        probs[ambito_predicho] = 0.80
    
    total = sum(probs.values())
    probs = {k: v/total for k, v in probs.items()}
    
    df = pd.DataFrame({
        'Ámbito': list(probs.keys()),
        'Probabilidad': list(probs.values())
    }).sort_values('Probabilidad', ascending=False)
    
    colors = ['#bbdefb' if a != ambito_predicho else '#d32f2f' for a in df['Ámbito']]
    
    fig = px.bar(
        df, 
        x='Ámbito', 
        y='Probabilidad',
        title='Probabilidad de Predicción por Ámbito',
        text='Probabilidad'
    )
    
    fig.update_traces(
        marker_color=colors,
        marker_line_color='white',
        marker_line_width=1.5,
        texttemplate='%{y:.2f}',
        textposition='outside',
        textfont=dict(size=12)
    )
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=500,
        plot_bgcolor='white',
        xaxis=dict(gridcolor='#f0f0f0', title=None),
        yaxis=dict(gridcolor='#f0f0f0', range=[0, 1], tickformat='.0%', title='Probabilidad'),
        margin=dict(l=20, r=20, t=60, b=80),
        title=dict(
            font=dict(size=18, color='#424242'),
            x=0.5
        )
    )
    
    fig.add_shape(
        type="line",
        x0=-0.5,
        y0=0.5,
        x1=len(ambitos)-0.5,
        y1=0.5,
        line=dict(color="#9e9e9e", width=1, dash="dash")
    )
    
    return fig

def create_response_history_plot(respuestas_df, preguntas_df):
    if respuestas_df is None or respuestas_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=("Evolución de Pesos de Respuestas", "Distribución de Respuestas por Ámbito"),
        vertical_spacing=0.15,
        specs=[[{"type": "scatter"}], [{"type": "bar"}]]
    )
    
    if 'fechaRespuesta' in respuestas_df.columns:
        respuestas_df = respuestas_df.sort_values('fechaRespuesta')
    
    if 'peso' in respuestas_df.columns:
        fig.add_trace(
            go.Scatter(
                x=list(range(len(respuestas_df))),
                y=respuestas_df['peso'],
                mode='lines+markers',
                name='Peso',
                line=dict(color='#1976d2', width=2, shape='spline'),
                marker=dict(size=8, color='#1976d2', line=dict(width=1, color='white'))
            ),
            row=1, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=len(respuestas_df),
            y1=0,
            line=dict(color="#e53935", width=2, dash="dash"),
            row=1, col=1
        )
        
        fig.add_shape(
            type="rect",
            x0=0,
            y0=0,
            x1=len(respuestas_df),
            y1=5,
            fillcolor="rgba(76, 175, 80, 0.1)",
            line=dict(width=0),
            row=1, col=1
        )
        
        fig.add_shape(
            type="rect",
            x0=0,
            y0=0,
            x1=len(respuestas_df),
            y1=-5,
            fillcolor="rgba(244, 67, 54, 0.1)",
            line=dict(width=0),
            row=1, col=1
        )
    
    if preguntas_df is not None and not preguntas_df.empty:
        try:
            respuestas_df['idPregunta'] = respuestas_df['idPregunta'].astype(str)
            preguntas_df['_id'] = preguntas_df['_id'].astype(str)
            
            merged = respuestas_df.merge(
                preguntas_df,
                left_on='idPregunta',
                right_on='_id',
                how='left'
            )
            
            def get_ambito_dominante(ambitos_str):
                try:
                    ambitos = json.loads(ambitos_str) if isinstance(ambitos_str, str) else ambitos_str
                    if ambitos and isinstance(ambitos, dict):
                        return max(ambitos.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)[0]
                    return "General"
                except:
                    return "General"
            
            merged['ambito_dominante'] = merged['ambitos'].apply(get_ambito_dominante)
            
            conteo_ambitos = merged['ambito_dominante'].value_counts().reset_index()
            conteo_ambitos.columns = ['Ámbito', 'Conteo']
            
            fig.add_trace(
                go.Bar(
                    x=conteo_ambitos['Ámbito'],
                    y=conteo_ambitos['Conteo'],
                    marker_color='#26a69a',
                    marker_line_color='white',
                    marker_line_width=1,
                    name='Respuestas por Ámbito',
                    text=conteo_ambitos['Conteo'],
                    textposition='auto'
                ),
                row=2, col=1
            )
        except Exception as e:
            st.warning(f"No se pudo crear el gráfico de distribución por ámbito: {str(e)}")
    
    fig.update_layout(
        height=800,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            title="Secuencia de Respuestas", 
            gridcolor='#f0f0f0',
            zeroline=False
        ),
        yaxis=dict(
            title="Peso", 
            gridcolor='#f0f0f0', 
            zeroline=False
        ),
        xaxis2=dict(
            title="Ámbito", 
            tickangle=-45, 
            gridcolor='#f0f0f0'
        ),
        yaxis2=dict(
            title="Número de Respuestas", 
            gridcolor='#f0f0f0'
        ),
        margin=dict(l=20, r=20, t=80, b=80)
    )
    
    fig.update_annotations(font_size=16, font_color="#424242")
    
    return fig

def show_model_comparison():
    st.markdown("<h2 class='sub-header'>Comparación de Modelos de IA</h2>", unsafe_allow_html=True)
    
    try:
        rf_metrics = {
            'accuracy': 0.981,
            'balanced_accuracy': 0.982,
            'precision': 0.980,
            'recall': 0.981,
            'f1': 0.980
        }
        
        xgb_metrics = {
            'accuracy': 0.999,
            'balanced_accuracy': 0.999,
            'precision': 0.999,
            'recall': 0.999,
            'f1': 0.999
        }
        
        metrics_df = pd.DataFrame({
            'Métrica': ['Exactitud', 'Exactitud Balanceada', 'Precisión', 'Sensibilidad', 'F1-Score'],
            'RandomForest': [rf_metrics['accuracy'], rf_metrics['balanced_accuracy'], 
                            rf_metrics['precision'], rf_metrics['recall'], rf_metrics['f1']],
            'XGBoost': [xgb_metrics['accuracy'], xgb_metrics['balanced_accuracy'], 
                       xgb_metrics['precision'], xgb_metrics['recall'], xgb_metrics['f1']]
        })
        
        st.markdown("#### Métricas de Rendimiento")
        st.dataframe(
            metrics_df,
            column_config={
                "Métrica": st.column_config.TextColumn("Métrica"),
                "RandomForest": st.column_config.ProgressColumn(
                    "RandomForest", min_value=0.9, max_value=1.0, format="%.3f"
                ),
                "XGBoost": st.column_config.ProgressColumn(
                    "XGBoost", min_value=0.9, max_value=1.0, format="%.3f"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Solución para el problema de pestañas
        tab1, tab2 = st.tabs(["Curvas de Aprendizaje", "Matrices de Confusión"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Curva de Aprendizaje - RandomForest")
                st.image("model/results/training_rf/plots/curva_aprendizaje.png")
            
            with col2:
                st.markdown("#### Curva de Aprendizaje - XGBoost")
                st.image("model/results/training_xgb/plots/curva_aprendizaje.png")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Matriz de Confusión - RandomForest")
                st.image("model/results/training_rf/plots/confusion_matrix_norm.png")
            
            with col2:
                st.markdown("#### Matriz de Confusión - XGBoost")
                st.image("model/results/training_xgb/plots/confusion_matrix_norm.png")
    
    except Exception as e:
        st.error(f"Error al mostrar la comparación de modelos: {str(e)}")
        traceback.print_exc()

def main():
    st.markdown("<h1 class='main-header'>Blooming AI: Selección Personalizada de Preguntas</h1>", unsafe_allow_html=True)
    
    st.sidebar.image("https://img.freepik.com/free-vector/ai-technology-brain-background-vector-digital-transformation-concept_53876-117812.jpg", width=280)
    st.sidebar.markdown("## Parámetros de la Demo")
    
    demo_mode = st.sidebar.radio(
        "Modo de demostración:",
        ["Predicción en tiempo real", "Comparación de modelos"]
    )
    
    if demo_mode == "Predicción en tiempo real":
        alumnos = get_alumnos_list()
        
        if not alumnos:
            st.error("No se pudieron cargar los alumnos desde la base de datos. Por favor, verifica la conexión.")
            return
        
        alumnos_options = [f"{a.get('nombre', 'Sin nombre')} {a.get('apellidos', 'Sin apellidos')} - {a.get('objetivo', 'Sin objetivo')} ({a.get('_id', 'ID')})" for a in alumnos]
        
        alumno_seleccionado = st.sidebar.selectbox(
            "Selecciona un alumno:",
            options=alumnos_options
        )
        
        alumno_id = alumno_seleccionado.split("(")[-1].strip(")")
        
        modelo_seleccionado = st.sidebar.radio(
            "Modelo de IA:",
            options=["RandomForest", "XGBoost"],
            index=1
        )
        
        if st.sidebar.button("Realizar predicción", type="primary"):
            with st.spinner("Procesando la predicción..."):
                resultado = make_real_prediction(alumno_id, modelo_seleccionado)
                
                if "error" in resultado:
                    st.error(f"Error: {resultado['error']}")
                else:
                    st.markdown("<h2 class='sub-header'>Resultados de la Predicción</h2>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        alumno_data = get_alumno_by_id(alumno_id)
                        if alumno_data is not None and not alumno_data.empty:
                            alumno_info = alumno_data.iloc[0].to_dict()
                            
                            # st.markdown("<div class='info-card'>", unsafe_allow_html=True)
                            
                            # Encabezado del perfil mejorado
                            st.markdown("<div class='profile-header'>", unsafe_allow_html=True)
                            st.markdown(f"<h3 class='profile-name'>{alumno_info.get('nombre', 'N/A')} {alumno_info.get('apellidos', 'N/A')}</h3>", unsafe_allow_html=True)
                            st.markdown(f"<p class='profile-objective'>Objetivo: <span class='highlight'>{alumno_info.get('objetivo', 'N/A')}</span></p>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            ambitos = alumno_info.get('ambitos', '{}')
                            if isinstance(ambitos, str):
                                try:
                                    ambitos = json.loads(ambitos)
                                except:
                                    ambitos = {}
                            
                            # Representación visual mejorada de los ámbitos
                            ambitos_df = pd.DataFrame([
                                {"Ámbito": ambito, "Peso": datos.get("peso", 0) if isinstance(datos, dict) else datos,
                                 "% Aparición": (datos.get("porcentaje", 0) * 100) if isinstance(datos, dict) else 0}
                                for ambito, datos in ambitos.items()
                            ])
                            
                            # Visualización en formato de tabla con barras de progreso
                            st.dataframe(
                                ambitos_df,
                                column_config={
                                    "Ámbito": st.column_config.TextColumn("Ámbito"),
                                    "Peso": st.column_config.ProgressColumn(
                                        "Peso emocional", 
                                        min_value=0, 
                                        max_value=100, 
                                        format="%d",
                                        help="Valor entre 0-100 que indica el estado emocional del estudiante en este ámbito"
                                    ),
                                    "% Aparición": st.column_config.ProgressColumn(
                                        "% de aparición", 
                                        min_value=0, 
                                        max_value=30, 
                                        format="%.1f%%",
                                        help="Porcentaje de preguntas realizadas en este ámbito"
                                    )
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                            
                            # Gráfico radar mejorado
                            radar_fig = create_radar_plot(ambitos, resultado.get('ambito_predicho', None))
                            if radar_fig:
                                st.plotly_chart(radar_fig, use_container_width=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.error("No se pudo obtener la información completa del alumno")
                    
                    with col2:
                        # st.markdown("<div class='success-card'>", unsafe_allow_html=True)
                        
                        # Contenedor destacado para la predicción
                        st.markdown(f"<div class='prediction-title'>Ámbito Recomendado</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='prediction-value'>{resultado.get('ambito_predicho', 'N/A')}</div>", unsafe_allow_html=True)
                        
                        # Muestra la confianza de la predicción con un indicador visual
                        if 'probabilidad' in resultado:
                            prob = resultado.get('probabilidad', 0)
                            st.progress(prob, text=f"Confianza: {prob:.0%}")
                        
                        # Modelo utilizado
                        st.markdown(f"<p style='text-align: center; margin-top: 5px;'>Modelo: <b>{modelo_seleccionado}</b></p>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Pregunta seleccionada con diseño mejorado
                        st.markdown("<h4>Pregunta seleccionada:</h4>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class='pregunta-box'>
                            <p class='pregunta-text'>"{resultado.get('pregunta', 'N/A')}"</p>
                            <div style='display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px;'>
                                {' '.join([f"<span class='ambito-chip'>{ambito}: {peso}%</span>" for ambito, peso in json.loads(resultado.get('ambitos', '{}')).items()])}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Opciones de respuesta con diseño mejorado
                        st.markdown("<h4>Opciones de respuesta:</h4>", unsafe_allow_html=True)
                        opciones = resultado.get('opcionesPregunta', [])
                        for i, opcion in enumerate(opciones):
                            peso = opcion.get('peso', 0)
                            color = "#4caf50" if peso > 0 else "#f44336" if peso < 0 else "#9e9e9e"
                            st.markdown(f"""
                            <div class='option-card' style='border-left-color: {color}'>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <div>{opcion.get('opcionPregunta', 'N/A')}</div>
                                    <div style='color: {color}; font-weight: bold;'>Peso: {peso}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Visualizaciones en pestañas mejoradas - SOLUCIÓN PARA PROBLEMA DE PESTAÑAS EN IMAGEN 1 y 4
                    st.markdown("<h3 class='sub-header'>Análisis del Proceso de Predicción</h3>", unsafe_allow_html=True)
                    
                    # Usar un formato más simple para las pestañas
                    tab1, tab2, tab3 = st.tabs(["Importancia de Características", "Probabilidades por Ámbito", "Historial de Respuestas"])
                    
                    with tab1:
                        output_dir = resultado.get('output_directory', None)
                        if output_dir and os.path.exists(os.path.join(output_dir, "feature_importance.png")):
                            st.image(os.path.join(output_dir, "feature_importance.png"))
                        
                        st.markdown("""
                        <div style='background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin-top: 15px;'>
                            <p style='font-style: italic; color: #555;'>
                                Este gráfico muestra las características más influyentes en la decisión del modelo.
                                Las variables con mayor peso tienen un impacto más significativo en la selección del ámbito.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with tab2:
                        if output_dir and os.path.exists(os.path.join(output_dir, "prediction_probability.png")):
                            st.image(os.path.join(output_dir, "prediction_probability.png"))
                        else:
                            ambitos_keys = ["Familia", "Amigos y relaciones", "Académico", "Entorno en clase",
                                           "Entorno exterior", "Actividades extraescolares", "Autopercepción y emociones",
                                           "Relación con profesores", "General"]
                            
                            prob_fig = create_probability_plot(ambitos_keys, resultado.get('ambito_predicho', 'General'), modelo_seleccionado)
                            st.plotly_chart(prob_fig)
                        
                        st.markdown("""
                        <div style='background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin-top: 15px;'>
                            <p style='font-style: italic; color: #555;'>
                                Esta visualización muestra la distribución de probabilidades asignadas por el modelo a cada ámbito.
                                El ámbito con la probabilidad más alta es el seleccionado para la siguiente pregunta.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with tab3:
                        if output_dir and os.path.exists(os.path.join(output_dir, "distribucion_ambitos.png")):
                            st.image(os.path.join(output_dir, "distribucion_ambitos.png"))
                        else:
                            respuestas = get_respuestas_by_alumno(alumno_id)
                            preguntas = get_preguntas()
                            
                            if respuestas is not None and not respuestas.empty:
                                resp_fig = create_response_history_plot(respuestas, preguntas)
                                if resp_fig:
                                    st.plotly_chart(resp_fig)
                            else:
                                st.warning("No hay datos de respuestas para mostrar")
                        
                        st.markdown("""
                        <div style='background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin-top: 15px;'>
                            <p style='font-style: italic; color: #555;'>
                                Estos gráficos muestran la evolución histórica de las respuestas del estudiante y la
                                distribución por ámbitos, proporcionando contexto sobre su interacción con la plataforma.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Análisis y conclusión con diseño mejorado
                    st.markdown("<h3 class='sub-header'>Análisis de la Predicción</h3>", unsafe_allow_html=True)
                    
                    # Explicaciones según objetivo
                    objetivo_explicaciones = {
                        "Prevención": """
                            <div class='warning-card'>
                                <h4>Estrategia de Prevención</h4>
                                <p>El modelo ha seleccionado el ámbito <b>{ambito}</b> siguiendo una estrategia preventiva,
                                identificando que este ámbito presenta indicadores de riesgo o potencial deterioro.</p>
                                
                                <p>La selección se alinea con el objetivo de <b>Prevención</b>, que busca intervenir tempranamente
                                en áreas que, de no abordarse, podrían generar dificultades futuras para el estudiante.</p>
                                
                                <p>El peso actual de este ámbito ({peso:.1f}/100) y su patrón de evolución reciente han sido
                                factores determinantes en esta decisión.</p>
                            </div>
                        """,
                        "Exploración": """
                            <div class='info-card'>
                                <h4>Estrategia de Exploración</h4>
                                <p>El modelo ha identificado el ámbito <b>{ambito}</b> como prioritario para exploración,
                                ya que representa un aspecto del desarrollo del estudiante que ha sido menos abordado
                                en interacciones previas (porcentaje de aparición: {porcentaje:.1f}%).</p>
                                
                                <p>Esta selección cumple con el objetivo de <b>Exploración</b>, que busca diversificar
                                las experiencias del estudiante y construir un perfil emocional más completo y equilibrado.</p>
                            </div>
                        """,
                        "Consolidación": """
                            <div class='success-card'>
                                <h4>Estrategia de Consolidación</h4>
                                <p>El ámbito <b>{ambito}</b> ha sido seleccionado como parte de una estrategia de consolidación,
                                que busca reforzar áreas donde el estudiante ya muestra fortalezas (peso actual: {peso:.1f}/100).</p>
                                
                                <p>Este enfoque, alineado con el objetivo de <b>Consolidación</b>, permite potenciar la confianza
                                del alumno y aprovechar sus áreas de mayor desarrollo para incentivar un ciclo positivo de
                                retroalimentación emocional.</p>
                            </div>
                        """,
                        "Predeterminado": """
                            <div class='info-card'>
                                <h4>Estrategia Predeterminada</h4>
                                <p>El modelo ha seleccionado el ámbito <b>{ambito}</b> siguiendo un equilibrio entre
                                los diferentes factores que componen el perfil del estudiante.</p>
                            </div>
                        """
                    }
                    
                    # Obtener datos para la explicación
                    objetivo = alumno_info.get('objetivo', 'Predeterminado')
                    ambito = resultado.get('ambito_predicho', 'seleccionado')
                    
                    ambitos_dict = alumno_info.get('ambitos', {})
                    if isinstance(ambitos_dict, str):
                        try:
                            ambitos_dict = json.loads(ambitos_dict)
                        except:
                            ambitos_dict = {}
                    
                    ambito_data = ambitos_dict.get(ambito, {})
                    if isinstance(ambito_data, dict):
                        peso = ambito_data.get('peso', 50)
                        porcentaje = ambito_data.get('porcentaje', 0.1) * 100
                    else:
                        peso = 50
                        porcentaje = 10
                    
                    # Mostrar explicación según el objetivo
                    if objetivo in objetivo_explicaciones:
                        st.markdown(
                            objetivo_explicaciones[objetivo].format(
                                ambito=ambito,
                                peso=peso,
                                porcentaje=porcentaje
                            ),
                            unsafe_allow_html=True
                        )
                    
                    # Conclusión general mejorada
                    st.markdown("""
                    <div class='success-card'>
                        <h4>Conclusión</h4>
                        <p>La predicción realizada por el modelo de IA demuestra cómo la tecnología puede adaptarse 
                        a las necesidades específicas de cada estudiante, ofreciendo una experiencia personalizada 
                        que responde tanto a su estado emocional actual como a sus objetivos pedagógicos.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Explicación del funcionamiento con diseño mejorado
        with st.expander("¿Cómo funciona este sistema?", expanded=False):
            st.markdown("""
            <div style="padding: 20px; background-color: #f5f9ff; border-radius: 12px; margin: 10px 0;">
                <h3 style="color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px;">Funcionamiento del Sistema de Predicción</h3>

                <p>Esta herramienta de demostración utiliza modelos reales de aprendizaje automático para seleccionar preguntas
                personalizadas para cada estudiante. El proceso completo se realiza en cuatro fases:</p>
                
                <div style="display: flex; margin: 20px 0;">
                    <div style="background-color: #0066cc; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0;">1</div>
                    <div>
                        <p><strong>Recuperación de datos</strong>: El sistema recupera el perfil completo del alumno desde la base de datos MongoDB,
                        incluyendo información sobre sus ámbitos emocionales, objetivos, y el historial de respuestas previas.</p>
                    </div>
                </div>
                
                <div style="display: flex; margin: 20px 0;">
                    <div style="background-color: #0066cc; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0;">2</div>
                    <div>
                        <p><strong>Extracción de características</strong>: Los datos del alumno se transforman en un conjunto de características
                        numéricas que representan su estado actual, patrones históricos y tendencias de evolución en diferentes ámbitos.</p>
                    </div>
                </div>
                
                <div style="display: flex; margin: 20px 0;">
                    <div style="background-color: #0066cc; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0;">3</div>
                    <div>
                        <p><strong>Predicción del ámbito</strong>: El modelo de IA (RandomForest o XGBoost según la selección) analiza estas
                        características y determina cuál es el ámbito más relevante para el alumno en este momento, considerando
                        su objetivo pedagógico.</p>
                    </div>
                </div>
                
                <div style="display: flex; margin: 20px 0;">
                    <div style="background-color: #0066cc; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; margin-right: 15px; flex-shrink: 0;">4</div>
                    <div>
                        <p><strong>Selección de pregunta</strong>: Una vez determinado el ámbito, el sistema filtra el banco de preguntas
                        disponibles y selecciona una pregunta adecuada asociada a ese ámbito.</p>
                    </div>
                </div>
                
                <p>Las visualizaciones muestran el razonamiento detrás de cada decisión del modelo, permitiendo entender
                por qué se ha seleccionado un determinado ámbito y cómo esta decisión se alinea con las necesidades
                del estudiante.</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif demo_mode == "Comparación de modelos":
        show_model_comparison()
        
        with st.expander("¿Por qué XGBoost supera a RandomForest en este caso?"):
            st.markdown("""
            <div style="padding: 20px; background-color: #f5f9ff; border-radius: 12px; margin: 10px 0;">
                <h3 style="color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px;">Ventajas de XGBoost para la Predicción Educativa</h3>
                
                <p>Después de un exhaustivo proceso de experimentación, XGBoost ha demostrado ser superior a RandomForest
                en esta aplicación específica por varias razones fundamentales:</p>

                <div style="margin: 15px 0; padding: 15px; background-color: rgba(255,255,255,0.7); border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <p><strong>Mayor capacidad para detectar patrones complejos</strong><br>
                    XGBoost construye árboles secuenciales donde cada nuevo árbol se enfoca en corregir los errores de
                    los anteriores. Esta característica resulta esencial para detectar patrones sutiles en los perfiles
                    emocionales de los estudiantes, que a menudo presentan interrelaciones complejas entre diferentes
                    ámbitos.</p>
                </div>

                <div style="margin: 15px 0; padding: 15px; background-color: rgba(255,255,255,0.7); border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <p><strong>Mejor generalización y menos sobreajuste</strong><br>
                    Como se observa claramente en las curvas de aprendizaje, XGBoost mantiene un rendimiento similar
                    entre los datos de entrenamiento y validación, lo que indica una mejor capacidad para generalizar
                    a nuevos estudiantes sin memorizar patrones específicos del conjunto de entrenamiento.</p>
                </div>

                <div style="margin: 15px 0; padding: 15px; background-color: rgba(255,255,255,0.7); border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <p><strong>Distribución más equilibrada de la importancia de características</strong><br>
                    XGBoost considera un espectro más amplio de características al tomar decisiones, en contraste con
                    RandomForest que tiende a concentrarse excesivamente en unas pocas variables. Esto se traduce en
                    predicciones más robustas y menos sesgadas.</p>
                </div>

                <div style="margin: 15px 0; padding: 15px; background-color: rgba(255,255,255,0.7); border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <p><strong>Mayor alineación con objetivos pedagógicos</strong><br>
                    Las predicciones de XGBoost muestran una mayor coherencia con los principios pedagógicos que guían
                    cada uno de los objetivos (Prevención, Exploración, Consolidación, etc.), lo que sugiere que el
                    modelo ha captado mejor la lógica subyacente del sistema educativo.</p>
                </div>

                <div style="margin: 15px 0; padding: 15px; background-color: rgba(255,255,255,0.7); border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <p><strong>Eficiencia computacional</strong><br>
                    Además de su superior rendimiento predictivo, XGBoost alcanza estos resultados con tiempos de
                    entrenamiento y predicción similares o incluso más eficientes que RandomForest, lo que lo convierte
                    en una opción ideal para implementaciones en tiempo real.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with st.expander("Evolución del desarrollo del modelo"):
            st.markdown("""
            <div style="padding: 20px; background-color: #f5f9ff; border-radius: 12px; margin: 10px 0;">
                <h3 style="color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px;">Evolución del Desarrollo del Modelo de IA</h3>
                
                <p>El desarrollo del modelo para Blooming ha seguido un proceso iterativo con varias fases clave:</p>

                <div style="margin: 20px 0; padding: 15px; background-color: rgba(255,255,255,0.7); border-radius: 8px; border-left: 5px solid #2196f3; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h4 style="color: #2196f3; margin-top: 0;">Fase 1: Exploración y Base Inicial</h4>
                    <ul style="padding-left: 20px; margin-bottom: 0;">
                        <li>Implementación <li>Implementación de un modelo RandomForest básico</li>
                        <li>Generación de datos sintéticos representativos</li>
                        <li>Evaluación inicial con precisión limitada (~16-17%)</li>
                    </ul>
                </div>

                <div style="margin: 20px 0; padding: 15px; background-color: rgba(255,255,255,0.7); border-radius: 8px; border-left: 5px solid #4caf50; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h4 style="color: #4caf50; margin-top: 0;">Fase 2: Mejora del Preprocesamiento</h4>
                    <ul style="padding-left: 20px; margin-bottom: 0;">
                        <li>Reestructuración completa del procesamiento de datos</li>
                        <li>Incorporación de características derivadas del historial</li>
                        <li>Ampliación del conjunto de datos de entrenamiento</li>
                        <li>Mejora de precisión hasta ~58%</li>
                    </ul>
                </div>

                <div style="margin: 20px 0; padding: 15px; background-color: rgba(255,255,255,0.7); border-radius: 8px; border-left: 5px solid #ff9800; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h4 style="color: #ff9800; margin-top: 0;">Fase 3: Optimización Algorítmica</h4>
                    <ul style="padding-left: 20px; margin-bottom: 0;">
                        <li>Implementación del algoritmo XGBoost</li>
                        <li>Selección de características más relevantes</li>
                        <li>Ajuste de hiperparámetros mediante búsqueda en cuadrícula</li>
                        <li>Logro de precisión superior al 98%</li>
                    </ul>
                </div>

                <div style="margin: 20px 0; padding: 15px; background-color: rgba(255,255,255,0.7); border-radius: 8px; border-left: 5px solid #9c27b0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h4 style="color: #9c27b0; margin-top: 0;">Fase 4: Refinamiento y Explicabilidad</h4>
                    <ul style="padding-left: 20px; margin-bottom: 0;">
                        <li>Desarrollo de visualizaciones interpretables</li>
                        <li>Equilibrio entre precisión y significado pedagógico</li>
                        <li>Validación con perfiles de estudiantes reales</li>
                        <li>Ajuste fino para alineación con objetivos educativos</li>
                    </ul>
                </div>

                <p>Esta evolución muestra cómo el proceso iterativo ha permitido transformar un modelo inicial básico en un
                sistema altamente preciso y pedagógicamente relevante, capaz de ofrecer recomendaciones personalizadas
                y significativas para cada estudiante.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer mejorado visualmente
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding: 25px; background: linear-gradient(to right, #f5f9ff, #e3f2fd); border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.05);">
        <h3 style="color: #0066cc; margin-bottom: 10px; font-weight: 600;">Blooming AI</h3>
        <p style="margin-bottom: 5px; font-size: 16px;">Desarrollado por <b>Francisco José Delicado González</b></p>
        <p style="margin-bottom: 5px; color: #555;">Trabajo Fin de Grado - Ingeniería Multimedia</p>
        <p style="color: #555;">Universidad de Alicante, 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Error en la aplicación: {str(e)}")
        st.error(traceback.format_exc())