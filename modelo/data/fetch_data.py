from pymongo import MongoClient
import pandas as pd
from bson.objectid import ObjectId
import json
import logging
import numpy as np
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client.tfg_database
    logger.info("Conexión a MongoDB establecida correctamente")
except Exception as e:
    logger.error(f"Error al conectar a MongoDB: {e}")
    db = None

def get_collection_as_dataframe(collection_name):
    if db is None:
        logger.error("No hay conexión a la base de datos")
        return pd.DataFrame()
    
    try:
        collection = list(db[collection_name].find())
        df = pd.DataFrame(collection)
        logger.info(f"Obtenidos {len(df)} registros de {collection_name}")
        return df
    except Exception as e:
        logger.error(f"Error al obtener datos de {collection_name}: {e}")
        return pd.DataFrame()

def get_document_by_id(collection_name, doc_id):
    if db is None:
        logger.error("No hay conexión a la base de datos")
        return None
    
    try:
        oid = ObjectId(doc_id)
        doc = db[collection_name].find_one({'_id': oid})
        return pd.DataFrame([doc]) if doc else None
    except Exception as e:
        logger.error(f"Error al obtener documento {doc_id} de {collection_name}: {e}")
        return None

def get_user_states():
    logger.info("Iniciando obtención de estados de usuario para entrenamiento")
    
    respuestas = get_collection_as_dataframe("respuestas")
    alumnos = get_collection_as_dataframe("alumnos")
    preguntas = get_collection_as_dataframe("preguntas")
    opciones_pregunta = get_collection_as_dataframe("opcionespregunta")
    
    if respuestas.empty or alumnos.empty or preguntas.empty or opciones_pregunta.empty:
        logger.warning("Una o más colecciones están vacías, no se puede proceder con el entrenamiento")
        return pd.DataFrame()

    logger.info("Preparando datos para procesamiento")
    
    respuestas["idAlumno"] = respuestas["idAlumno"].astype(str)
    alumnos["_id"] = alumnos["_id"].astype(str)
    respuestas["idPregunta"] = respuestas["idPregunta"].astype(str)
    preguntas["_id"] = preguntas["_id"].astype(str)
    respuestas["idOpcionPregunta"] = respuestas["idOpcionPregunta"].astype(str)
    opciones_pregunta["_id"] = opciones_pregunta["_id"].astype(str)
    opciones_pregunta["idPregunta"] = opciones_pregunta["idPregunta"].astype(str)

    def parse_json_safely(json_str):
        if not isinstance(json_str, str):
            return json_str
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Error decodificando JSON: {json_str[:50]}...")
            return {}
    
    alumnos['ambitos_dict'] = alumnos['ambitos'].apply(parse_json_safely)
    preguntas['ambitos_dict'] = preguntas['ambitos'].apply(parse_json_safely)
    
    logger.info("Realizando operaciones de merge")
    
    merged = respuestas.merge(
        opciones_pregunta, 
        left_on="idOpcionPregunta", 
        right_on="_id", 
        how="left",
        suffixes=("", "_opcion")
    )
    
    merged = merged.merge(
        alumnos, 
        left_on="idAlumno", 
        right_on="_id", 
        how="left",
        suffixes=("", "_alum")
    )
    
    merged = merged.merge(
        preguntas, 
        left_on="idPregunta", 
        right_on="_id", 
        how="left",
        suffixes=("", "_preg")
    )
    
    logger.info(f"Datos unidos correctamente. Registros totales: {len(merged)}")
    
    result = []
    ambitos_keys = [
        "Familia", "Amigos y relaciones", "Académico", "Entorno en clase",
        "Entorno exterior", "Actividades extraescolares", "Autopercepción y emociones",
        "Relación con profesores", "General"
    ]
    
    if merged.empty:
        logger.warning("No hay datos después de unir las colecciones")
        return pd.DataFrame()
    
    logger.info("Procesando características por registro")
    
    respuestas_por_alumno = {}
    for idx, row in merged.iterrows():
        id_alumno = row.get('idAlumno')
        if id_alumno not in respuestas_por_alumno:
            respuestas_por_alumno[id_alumno] = []
        respuestas_por_alumno[id_alumno].append(row)
    
    for idx, row in merged.iterrows():
        try:
            ambitos_alumno = row.get('ambitos_dict', {})
            if not isinstance(ambitos_alumno, dict):
                ambitos_alumno = {}
            
            ambitos_pregunta = row.get('ambitos_dict_preg', {})
            if not isinstance(ambitos_pregunta, dict):
                ambitos_pregunta = {}
                
            if not ambitos_pregunta:
                logger.warning(f"Pregunta sin ámbitos definidos, ID: {row.get('idPregunta', 'desconocido')}")
                ambitos_pregunta = {"General": 100}
            
            try:
                ambito_dominante = max(ambitos_pregunta.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)
                ambito_siguiente = ambito_dominante[0]
            except ValueError:
                logger.warning("No se pudo determinar el ámbito dominante, usando 'General'")
                ambito_siguiente = "General"
            
            objetivo = row.get('objetivo', 'Exploración')
            
            features = {
                'objetivo': objetivo,
            }
            
            for ambito in ambitos_keys:
                if ambito in ambitos_alumno:
                    ambito_info = ambitos_alumno[ambito]
                    if isinstance(ambito_info, dict):
                        features[f'{ambito}_peso'] = ambito_info.get('peso', 50)
                        features[f'{ambito}_porcentaje'] = ambito_info.get('porcentaje', 0)
                    else:
                        features[f'{ambito}_peso'] = ambito_info if isinstance(ambito_info, (int, float)) else 50
                        features[f'{ambito}_porcentaje'] = 0
                else:
                    features[f'{ambito}_peso'] = 50
                    features[f'{ambito}_porcentaje'] = 0
            
            peso_respuesta = row.get('peso', 0)
            features['peso_respuesta'] = peso_respuesta if isinstance(peso_respuesta, (int, float)) else 0
            
            if 'fechaRespuesta' in row and pd.notna(row['fechaRespuesta']):
                try:
                    features['fecha_respuesta'] = pd.to_datetime(row['fechaRespuesta'])
                except:
                    features['fecha_respuesta'] = pd.to_datetime('now')
            
            id_alumno = row.get('idAlumno')
            historial_alumno = respuestas_por_alumno.get(id_alumno, [])
            
            conteo_respuestas_por_ambito = {ambito: 0 for ambito in ambitos_keys}
            suma_pesos_por_ambito = {ambito: 0 for ambito in ambitos_keys}
            ultima_interaccion_por_ambito = {ambito: None for ambito in ambitos_keys}
            
            fecha_actual = pd.to_datetime('now')
            if 'fechaRespuesta' in row and pd.notna(row['fechaRespuesta']):
                fecha_actual = pd.to_datetime(row['fechaRespuesta'])
            
            for resp_hist in historial_alumno:
                if resp_hist.get('_id') == row.get('_id'):
                    continue
                
                ambitos_hist = resp_hist.get('ambitos_dict_preg', {})
                if not isinstance(ambitos_hist, dict) or not ambitos_hist:
                    ambitos_hist = {"General": 100}
                
                try:
                    ambito_dom_hist = max(ambitos_hist.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0)[0]
                except ValueError:
                    ambito_dom_hist = "General"
                
                conteo_respuestas_por_ambito[ambito_dom_hist] += 1
                
                peso_hist = resp_hist.get('peso', 0)
                if isinstance(peso_hist, (int, float)):
                    suma_pesos_por_ambito[ambito_dom_hist] += peso_hist
                
                if 'fechaRespuesta' in resp_hist and pd.notna(resp_hist['fechaRespuesta']):
                    fecha_resp = pd.to_datetime(resp_hist['fechaRespuesta'])
                    if ultima_interaccion_por_ambito[ambito_dom_hist] is None or fecha_resp > ultima_interaccion_por_ambito[ambito_dom_hist]:
                        ultima_interaccion_por_ambito[ambito_dom_hist] = fecha_resp
            
            for ambito in ambitos_keys:
                features[f'{ambito}_conteo_resp'] = conteo_respuestas_por_ambito[ambito]
                
                if conteo_respuestas_por_ambito[ambito] > 0:
                    features[f'{ambito}_prom_peso'] = suma_pesos_por_ambito[ambito] / conteo_respuestas_por_ambito[ambito]
                else:
                    features[f'{ambito}_prom_peso'] = 0
                
                if ultima_interaccion_por_ambito[ambito] is not None:
                    dias_desde_ultima = (fecha_actual - ultima_interaccion_por_ambito[ambito]).days
                    features[f'{ambito}_dias_ultima'] = dias_desde_ultima
                else:
                    features[f'{ambito}_dias_ultima'] = 365
            
            total_respuestas = len(historial_alumno)
            features['total_respuestas'] = total_respuestas
            
            if total_respuestas > 0:
                respuestas_positivas = sum(1 for r in historial_alumno if r.get('peso', 0) > 0)
                features['ratio_respuestas_positivas'] = respuestas_positivas / total_respuestas
            else:
                features['ratio_respuestas_positivas'] = 0.5
            
            features['next_ambito'] = ambito_siguiente
            
            # Añadir secundario y terciario ámbitos de la pregunta actual
            if len(ambitos_pregunta) > 1:
                sorted_ambitos = sorted(ambitos_pregunta.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)
                if len(sorted_ambitos) > 1:
                    features['ambito_secundario'] = sorted_ambitos[1][0]
                    features['peso_ambito_secundario'] = sorted_ambitos[1][1]
                if len(sorted_ambitos) > 2:
                    features['ambito_terciario'] = sorted_ambitos[2][0]
                    features['peso_ambito_terciario'] = sorted_ambitos[2][1]
            
            # Añadir ámbito con mayor y menor peso del alumno
            if ambitos_alumno:
                ambs_peso = {}
                for amb, info in ambitos_alumno.items():
                    if isinstance(info, dict) and 'peso' in info:
                        ambs_peso[amb] = info['peso']
                    elif isinstance(info, (int, float)):
                        ambs_peso[amb] = info
                
                if ambs_peso:
                    max_amb = max(ambs_peso.items(), key=lambda x: x[1])
                    min_amb = min(ambs_peso.items(), key=lambda x: x[1])
                    features['ambito_max_peso'] = max_amb[0]
                    features['valor_max_peso'] = max_amb[1]
                    features['ambito_min_peso'] = min_amb[0]
                    features['valor_min_peso'] = min_amb[1]
            
            # Característica para calcular el desequilibrio en los pesos de los ámbitos
            pesos_ambitos = []
            for ambito in ambitos_keys:
                if ambito in ambitos_alumno:
                    info = ambitos_alumno[ambito]
                    if isinstance(info, dict) and 'peso' in info:
                        pesos_ambitos.append(info['peso'])
                    elif isinstance(info, (int, float)):
                        pesos_ambitos.append(info)
            
            if pesos_ambitos:
                features['desviacion_pesos'] = np.std(pesos_ambitos)
                features['rango_pesos'] = max(pesos_ambitos) - min(pesos_ambitos)
            
            # Relación entre el objetivo y los pesos actuales
            if objetivo == "Consolidación":
                # Para consolidación, calcular el promedio de los pesos altos
                pesos_altos = [p for p in pesos_ambitos if p > 50]
                features['media_pesos_altos'] = np.mean(pesos_altos) if pesos_altos else 50
            elif objetivo == "Prevención":
                # Para prevención, calcular el promedio de los pesos bajos
                pesos_bajos = [p for p in pesos_ambitos if p < 50]
                features['media_pesos_bajos'] = np.mean(pesos_bajos) if pesos_bajos else 50
            
            result.append(features)
            
        except Exception as e:
            logger.error(f"Error procesando registro {idx}: {e}")
            continue
    
    logger.info(f"Procesados {len(result)} registros con éxito")
    
    df_result = pd.DataFrame(result)
    
    if df_result.empty:
        logger.warning("No se pudo extraer ninguna característica válida")
        return df_result
    
    if 'fecha_respuesta' in df_result.columns:
        df_result['hora_del_dia'] = df_result['fecha_respuesta'].dt.hour
        df_result['dia_semana'] = df_result['fecha_respuesta'].dt.dayofweek
        df_result['es_fin_de_semana'] = df_result['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)
        df_result['periodo_dia'] = df_result['hora_del_dia'].apply(
            lambda h: 'mañana' if 5 <= h < 12 else ('tarde' if 12 <= h < 18 else 'noche')
        )
        df_result.drop('fecha_respuesta', axis=1, inplace=True)
    
    # Factor de tiempo desde últimas interacciones (promedio para todos los ámbitos)
    dias_ultimas = [col for col in df_result.columns if col.endswith('_dias_ultima')]
    if dias_ultimas:
        df_result['promedio_dias_ultima_interaccion'] = df_result[dias_ultimas].mean(axis=1)
    
    # Relación entre el objetivo y los conteos de respuestas
    df_result['ratio_max_min_conteo'] = df_result.apply(
        lambda row: max([row.get(f'{amb}_conteo_resp', 0) for amb in ambitos_keys]) / 
                   (min([row.get(f'{amb}_conteo_resp', 1) for amb in ambitos_keys]) or 1),
        axis=1
    )
    
    # Indicador de si el ámbito dominante en la pregunta coincide con el de mayor peso
    df_result['coincidencia_amb_dom_max_peso'] = df_result.apply(
        lambda row: 1 if row.get('ambito_max_peso') == row.get('next_ambito') else 0,
        axis=1
    )
    
    logger.info(f"Dataset final: {df_result.shape[0]} filas, {df_result.shape[1]} columnas")
    logger.info(f"Distribución de clases objetivo: {df_result['next_ambito'].value_counts().to_dict()}")
    
    return df_result

def get_alumnos():
    return get_collection_as_dataframe("alumnos")

def get_preguntas():
    return get_collection_as_dataframe("preguntas")

def get_respuestas():
    return get_collection_as_dataframe("respuestas")

def get_opciones_pregunta():
    return get_collection_as_dataframe("opcionespregunta")

def get_alumno_by_id(id_alumno):
    return get_document_by_id("alumnos", id_alumno)

def get_pregunta_by_id(id_pregunta):
    return get_document_by_id("preguntas", id_pregunta)

def get_respuestas_by_alumno(id_alumno):
    if db is None:
        logger.error("No hay conexión a la base de datos")
        return pd.DataFrame()
    
    try:
        oid = ObjectId(id_alumno)
        respuestas = list(db.respuestas.find({'idAlumno': oid}))
        return pd.DataFrame(respuestas)
    except Exception as e:
        logger.error(f"Error al obtener respuestas del alumno {id_alumno}: {e}")
        return pd.DataFrame()