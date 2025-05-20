import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
import os
import random
from datetime import datetime
from data.fetch_data import (
    get_document_by_id, 
    get_preguntas, 
    get_opciones_pregunta,
    get_alumno_by_id,
    get_respuestas_by_alumno
)
from data.preprocess import extract_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_prediction_directory():
    prediction_dir = "predictions"
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(prediction_dir, f"prediction_{timestamp}")
    os.makedirs(run_dir)
    return run_dir

def get_feature_importance(model, feature_names):
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
    else:
        classifier = model
    
    if hasattr(classifier, 'feature_importances_'):
        importances = classifier.feature_importances_
        
        if hasattr(model, 'named_steps') and 'feature_selector' in model.named_steps:
            selector = model.named_steps['feature_selector']
            if hasattr(selector, 'get_support'):
                selected_indices = selector.get_support(indices=True)
                if len(feature_names) > len(selected_indices):
                    feature_names = [feature_names[i] for i in selected_indices]
        
        if len(feature_names) > len(importances):
            feature_names = feature_names[:len(importances)]
        elif len(feature_names) < len(importances):
            feature_names = list(feature_names) + [f"Feature {i+len(feature_names)}" 
                                              for i in range(len(importances) - len(feature_names))]
        
        feature_importance = {name: float(importance) for name, importance in zip(feature_names, importances)}
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        return feature_importance
    else:
        logger.warning("El modelo no tiene el atributo feature_importances_")
        return {}

def plot_ambitos_radar(ambitos_alumno, predicted_domain, output_dir):
    if isinstance(ambitos_alumno, str):
        try:
            ambitos_alumno = json.loads(ambitos_alumno)
        except:
            logger.error("Error al parsear ambitos_alumno como JSON")
            return

    ambitos = []
    valores = []
    
    for ambito, valor in ambitos_alumno.items():
        ambitos.append(ambito)
        if isinstance(valor, dict) and 'peso' in valor:
            valores.append(valor['peso'])
        elif isinstance(valor, (int, float)):
            valores.append(valor)
        else:
            valores.append(50)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    ambitos_circular = ambitos + [ambitos[0]]
    valores_circular = valores + [valores[0]]
    
    angles = np.linspace(0, 2*np.pi, len(ambitos), endpoint=False).tolist()
    angles += angles[:1]
    
    ax.plot(angles, valores_circular, 'o-', linewidth=2)
    ax.fill(angles, valores_circular, alpha=0.25)
    
    for i, ambito in enumerate(ambitos):
        if ambito == predicted_domain:
            ax.plot(angles[i], valores[i], 'ro', markersize=10)
            ax.annotate(f'PREDICCIÓN: {ambito}',
                      xy=(angles[i], valores[i]),
                      xytext=(angles[i], valores[i] + 20),
                      arrowprops=dict(facecolor='red', shrink=0.05))
    
    ax.set_thetagrids(np.degrees(angles[:-1]), ambitos)
    ax.set_ylim(0, 100)
    ax.set_title("Perfil de Ámbitos del Alumno", size=15, y=1.1)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ambitos_radar.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(feature_importance, output_dir, top_n=15):
    if not feature_importance:
        logger.warning("No hay datos de importancia de características para graficar")
        return
    
    df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['importance'])
    df = df.reset_index().rename(columns={'index': 'feature'})
    df = df.sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    
    sns.barplot(x='importance', y='feature', data=df, palette='viridis')
    
    for i, v in enumerate(df['importance']):
        plt.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    plt.title('Importancia de Características en la Predicción', size=16)
    plt.xlabel('Importancia', size=14)
    plt.ylabel('Características', size=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_prediction_probability(probabilities, class_names, predicted_class_idx, output_dir):
    plt.figure(figsize=(12, 6))
    
    df = pd.DataFrame({
        'Ámbito': class_names,
        'Probabilidad': probabilities
    })
    
    df = df.sort_values('Probabilidad', ascending=False)
    
    colors = ['#1f77b4'] * len(df)
    for i, ambito in enumerate(df['Ámbito']):
        if ambito == class_names[predicted_class_idx]:
            colors[i] = '#d62728'
    
    ax = sns.barplot(x='Ámbito', y='Probabilidad', data=df, palette=colors)
    
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0)
    
    plt.title('Probabilidad de Predicción por Ámbito', size=16)
    plt.xlabel('Ámbito', size=14)
    plt.ylabel('Probabilidad', size=14)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_probability.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_respuestas_historicas(respuestas_alumno, output_dir):
    if respuestas_alumno is None or respuestas_alumno.empty:
        logger.warning("No hay respuestas históricas para graficar")
        return
    
    logger.info("Hola caracola", respuestas_alumno.columns)
    
    if 'fechaRespuesta' in respuestas_alumno.columns and 'peso' in respuestas_alumno.columns:
        try:
            logger.info("Creando gráfico de evolución de pesos de respuestas")
            if respuestas_alumno['fechaRespuesta'].dtype == 'object':
                respuestas_alumno['fecha'] = pd.to_datetime(respuestas_alumno['fechaRespuesta'])
            else:
                respuestas_alumno['fecha'] = respuestas_alumno['fechaRespuesta']
            
            respuestas_ord = respuestas_alumno.sort_values('fecha')
            
            if len(respuestas_ord) > 1:
                plt.figure(figsize=(12, 6))
                plt.plot(range(len(respuestas_ord)), respuestas_ord['peso'], 'o-', linewidth=2)
                plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                plt.grid(True, alpha=0.3)
                plt.title('Evolución de Pesos de Respuestas', size=16)
                plt.xlabel('Secuencia de Respuestas', size=14)
                plt.ylabel('Peso', size=14)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "evolucion_pesos.png"), dpi=300)
                plt.close()
        except Exception as e:
            logger.error(f"Error al crear gráfico de evolución de pesos: {e}")
    
    try:
        from data.fetch_data import get_preguntas
        preguntas_df = get_preguntas()
        
        if preguntas_df is not None and not preguntas_df.empty:
            preguntas_df['_id'] = preguntas_df['_id'].astype(str)
            respuestas_alumno['idPregunta'] = respuestas_alumno['idPregunta'].astype(str)
            
            merged = respuestas_alumno.merge(
                preguntas_df,
                left_on='idPregunta',
                right_on='_id',
                how='left',
                suffixes=('', '_preg')
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
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Ámbito', y='Conteo', data=conteo_ambitos, palette='viridis')
            plt.title('Distribución de Respuestas por Ámbito', size=16)
            plt.xlabel('Ámbito', size=14)
            plt.ylabel('Número de Respuestas', size=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "distribucion_ambitos.png"), dpi=300)
            plt.close()
    except Exception as e:
        logger.error(f"Error al crear gráfico de distribución por ámbito: {e}")

def predict_next_question(alumno_id, save_results=True):
    logger.info(f"Iniciando predicción para el alumno {alumno_id}")
    
    try:
        model = joblib.load("model/results/training_xgb/models/best_model.pkl")
        y_encoder = joblib.load("model/results/training_xgb/models/y_encoder.pkl")
        feature_names = joblib.load("model/results/training_xgb/models/feature_names.pkl")
        logger.info("Modelo y recursos cargados correctamente")
    except Exception as e:
        logger.error(f"Error al cargar recursos: {e}")
        return {"error": f"Error al cargar recursos del modelo: {e}"}
    
    df_alumno = get_alumno_by_id(alumno_id)
    if df_alumno is None or df_alumno.empty:
        logger.error(f"No se encontró el alumno con ID {alumno_id}")
        return {"error": "No se encontró el alumno."}
    
    alumno_info = df_alumno.iloc[0].to_dict()
    
    respuestas_alumno = get_respuestas_by_alumno(alumno_id)
    
    output_dir = None
    if save_results:
        output_dir = create_prediction_directory()
        logger.info(f"Guardando resultados en {output_dir}")
    
    ambitos = alumno_info.get("ambitos", "{}")
    
    if isinstance(ambitos, str):
        try:
            ambitos_dict = json.loads(ambitos)
        except json.JSONDecodeError:
            logger.error("Error al decodificar ambitos como JSON")
            ambitos_dict = {}
    else:
        ambitos_dict = ambitos
    
    if output_dir:
        with open(os.path.join(output_dir, "ambitos_raw.json"), "w") as f:
            json.dump(ambitos_dict, f, indent=2)
    
    X_features = extract_features(ambitos, respuestas_alumno)
    logger.info(f"Características extraídas: {len(X_features)}")
    logger.info(f"Valores de características: {X_features}")
    
    X = np.array([X_features])
    
    try:
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            logger.info(f"El modelo espera {expected_features} características")
            
            if X.shape[1] < expected_features:
                X_padded = np.zeros((X.shape[0], expected_features))
                X_padded[:, :X.shape[1]] = X
                X = X_padded
                logger.info(f"Características rellenadas a {X.shape}")
            elif X.shape[1] > expected_features:
                X = X[:, :expected_features]
                logger.info(f"Características truncadas a {X.shape}")
        
        predicted_class_idx = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0] if hasattr(model, 'predict_proba') else []
        
        predicted_domain = y_encoder.inverse_transform([predicted_class_idx])[0]
        logger.info(f"Ámbito predicho: {predicted_domain} (índice: {predicted_class_idx})")
        
    except Exception as e:
        logger.error(f"Error durante la predicción: {e}")
        return {"error": f"Error durante la predicción: {e}"}
    
    feature_importance = get_feature_importance(model, feature_names)
    
    if save_results and output_dir:
        plot_ambitos_radar(ambitos_dict, predicted_domain, output_dir)
        
        plot_feature_importance(feature_importance, output_dir)
        
        class_names = y_encoder.classes_
        if len(probabilities) > 0:
            plot_prediction_probability(probabilities, class_names, predicted_class_idx, output_dir)
        
        if respuestas_alumno is not None and not respuestas_alumno.empty:
            plot_respuestas_historicas(respuestas_alumno, output_dir)
    
    preguntas_df = get_preguntas()
    if preguntas_df is None or preguntas_df.empty:
        logger.error("No se pudieron obtener las preguntas")
        return {"error": "Error al obtener las preguntas."}
    
    preguntas_df["_id"] = preguntas_df["_id"].astype(str)
    
    preguntas_filtradas = []
    for _, pregunta in preguntas_df.iterrows():
        try:
            ambitos_pregunta = json.loads(pregunta["ambitos"]) if isinstance(pregunta["ambitos"], str) else pregunta["ambitos"]
            if predicted_domain in ambitos_pregunta:
                preguntas_filtradas.append(pregunta)
        except:
            continue
    
    if not preguntas_filtradas:
        logger.warning(f"No se encontraron preguntas para el ámbito {predicted_domain}")
        preguntas_filtradas = preguntas_df.to_dict('records')
    
    if preguntas_filtradas:
        pregunta_seleccionada = random.choice(preguntas_filtradas)
    else:
        logger.error("No hay preguntas disponibles")
        return {"error": "No se encontraron preguntas para el dominio predicho."}

    opciones_df = get_opciones_pregunta()
    if opciones_df is None or opciones_df.empty:
        logger.error("No se pudieron obtener las opciones de pregunta")
        return {"error": "Error al obtener las opciones."}
    
    opciones_df["_id"] = opciones_df["_id"].astype(str)
    opciones_df["idPregunta"] = opciones_df["idPregunta"].astype(str)
    
    opciones_filtradas = opciones_df[opciones_df["idPregunta"] == pregunta_seleccionada["_id"]]
    opciones_lista = opciones_filtradas[["_id", "opcionPregunta", "peso"]].to_dict(orient="records")
    
    top_ambitos = {}
    if ambitos_dict:
        pesos_ambitos = {}
        for ambito, valor in ambitos_dict.items():
            if isinstance(valor, dict) and 'peso' in valor:
                pesos_ambitos[ambito] = valor['peso']
            elif isinstance(valor, (int, float)):
                pesos_ambitos[ambito] = valor
        
        suma_pesos = sum(pesos_ambitos.values())
        if suma_pesos > 0:
            pesos_porcentaje = {ambito: (peso / suma_pesos) * 100 for ambito, peso in pesos_ambitos.items()}
            top_ambitos = dict(sorted(pesos_porcentaje.items(), key=lambda x: x[1], reverse=True)[:3])
    
    model_info = {
        "feature_importance": feature_importance,
        "top_ambitos": top_ambitos,
        "total_respuestas": len(respuestas_alumno) if respuestas_alumno is not None else 0
    }
    
    result = {
        "alumno": alumno_info,
        "ambito_predicho": predicted_domain,
        "probabilidad_prediccion": float(probabilities[predicted_class_idx]) if len(probabilities) > 0 else None,
        "confianza_prediccion": model.predict_proba(X)[0].max() if hasattr(model, 'predict_proba') else None,
        "pregunta": pregunta_seleccionada,
        "opciones": opciones_lista,
        "feature_importance": feature_importance,
        "timestamp": datetime.now().isoformat()
    }
    
    if save_results and output_dir:
        with open(os.path.join(output_dir, "prediccion_completa.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"Resultados guardados en {output_dir}")
    
    api_response = {
        "idPregunta": pregunta_seleccionada["_id"],
        "pregunta": pregunta_seleccionada["pregunta"],
        "ambitos": pregunta_seleccionada["ambitos"],
        "ambito_predicho": predicted_domain,
        "probabilidad": float(probabilities[predicted_class_idx]) if len(probabilities) > 0 else None,
        "objetivo": alumno_info.get("objetivo", ""),
        "opcionesPregunta": opciones_lista,
        "output_directory": output_dir if save_results else None
    }
    
    return api_response

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        alumno_id = sys.argv[1]
    else:
        alumno_id = input("Introduce el ID del alumno: ")
    
    result = predict_next_question(alumno_id)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("\n=== PREDICCIÓN COMPLETADA ===")
        print(f"Ámbito predicho: {result['ambito_predicho']}")
        print(f"Pregunta: {result['pregunta']}")
        print(f"Opciones: {len(result['opcionesPregunta'])}")
        print(f"Resultados guardados en: {result['output_directory']}")