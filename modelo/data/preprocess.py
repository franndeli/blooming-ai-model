import json
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_features(ambitos_str, respuestas_alumno=None):
    try:
        ambitos = json.loads(ambitos_str) if isinstance(ambitos_str, str) else ambitos_str
        
        if not isinstance(ambitos, dict):
            logger.warning(f"Formato de ámbitos inválido: {type(ambitos)}")
            return [50] * 9
    except Exception as e:
        logger.warning(f"Error procesando ámbitos: {e}")
        return [50] * 9
    
    ambitos_ordenados = [
        "Familia", "Amigos y relaciones", "Académico", "Entorno en clase",
        "Entorno exterior", "Actividades extraescolares", 
        "Autopercepción y emociones", "Relación con profesores", "General"
    ]
    
    features = []
    
    # Pesos y porcentajes de cada ámbito (prioridad alta)
    for dominio in ambitos_ordenados:
        info_dominio = ambitos.get(dominio, {})
        
        if isinstance(info_dominio, (int, float)):
            peso = info_dominio
            porcentaje = 0
        else:
            peso = info_dominio.get("peso", 50)
            porcentaje = info_dominio.get("porcentaje", 0)
        
        features.append(peso / 100)  # Normalizado
        features.append(porcentaje)
    
    # Estadísticas de los pesos (prioridad media)
    pesos = []
    for dominio in ambitos_ordenados:
        info_dominio = ambitos.get(dominio, {})
        if isinstance(info_dominio, (int, float)):
            pesos.append(info_dominio)
        elif isinstance(info_dominio, dict):
            pesos.append(info_dominio.get("peso", 50))
        else:
            pesos.append(50)
    
    if pesos:
        features.append(np.mean(pesos) / 100)
        features.append(np.std(pesos) / 100)
        features.append(max(pesos) / 100)
        features.append(min(pesos) / 100)
        
        # Entropía de los pesos (distribución)
        pesos_norm = np.array(pesos) / sum(pesos)
        pesos_norm = pesos_norm[pesos_norm > 0]
        if len(pesos_norm) > 0:
            entropia = -np.sum(pesos_norm * np.log2(pesos_norm))
        else:
            entropia = 0
        features.append(entropia / 3)
        
        # Rango de pesos (indica disparidad)
        features.append((max(pesos) - min(pesos)) / 100)
    else:
        features.extend([0.5, 0, 0.5, 0.5, 0, 0])
    
    # Historial de respuestas (prioridad alta)
    if respuestas_alumno is not None and not respuestas_alumno.empty:
        total_respuestas = len(respuestas_alumno)
        
        # Ratio de respuestas positivas
        if total_respuestas > 0:
            respuestas_positivas = sum(1 for _, row in respuestas_alumno.iterrows() if row.get('peso', 0) > 0)
            features.append(respuestas_positivas / total_respuestas)
        else:
            features.append(0.5)
        
        # Número total de respuestas (normalizado)
        features.append(min(total_respuestas / 200, 1.0))
        
        # Tendencia reciente (últimas 5 respuestas)
        if total_respuestas >= 5:
            ultimas_resp = respuestas_alumno.sort_values('fechaRespuesta', ascending=False).head(5)
            pesos_recientes = [row.get('peso', 0) for _, row in ultimas_resp.iterrows()]
            features.append(np.mean(pesos_recientes) / 5)  # Media normalizada
        else:
            features.append(0.5)
    else:
        features.extend([0.5, 0, 0.5])
    
    # Ámbito con mayor peso como one-hot (prioridad alta)
    if ambitos:
        max_dominio = max(ambitos.items(), key=lambda x: 
                         x[1].get("peso", 0) if isinstance(x[1], dict) else 
                         (x[1] if isinstance(x[1], (int, float)) else 0))
        max_dominio_index = ambitos_ordenados.index(max_dominio[0]) if max_dominio[0] in ambitos_ordenados else -1
        
        for i in range(len(ambitos_ordenados)):
            features.append(1.0 if i == max_dominio_index else 0.0)
    else:
        features.extend([0.0] * len(ambitos_ordenados))
    
    # Ámbito con menor peso como one-hot (prioridad media)
    if ambitos:
        min_dominio = min(ambitos.items(), key=lambda x: 
                         x[1].get("peso", 0) if isinstance(x[1], dict) else 
                         (x[1] if isinstance(x[1], (int, float)) else 0))
        min_dominio_index = ambitos_ordenados.index(min_dominio[0]) if min_dominio[0] in ambitos_ordenados else -1
        
        for i in range(len(ambitos_ordenados)):
            features.append(1.0 if i == min_dominio_index else 0.0)
    else:
        features.extend([0.0] * len(ambitos_ordenados))
    
    # Variaciones entre ámbitos (prioridad media-alta)
    for i in range(len(ambitos_ordenados)-1):
        for j in range(i+1, len(ambitos_ordenados)):
            dominio1 = ambitos.get(ambitos_ordenados[i], {})
            dominio2 = ambitos.get(ambitos_ordenados[j], {})
            
            peso1 = dominio1.get("peso", 50) if isinstance(dominio1, dict) else (dominio1 if isinstance(dominio1, (int, float)) else 50)
            peso2 = dominio2.get("peso", 50) if isinstance(dominio2, dict) else (dominio2 if isinstance(dominio2, (int, float)) else 50)
            
            # Diferencia normalizada
            features.append((peso1 - peso2) / 100)
    
    return features

def preprocess_data(df, balance_classes=True, feature_selection=True):
    logger.info("Iniciando preprocesamiento de datos")
    
    if df.empty:
        logger.warning("DataFrame vacío recibido para preprocesamiento")
        return [], [], None, []
    
    all_feature_names = list(df.columns)
    
    # Separar etiquetas y características
    y = df['next_ambito'].values
    X_df = df.drop('next_ambito', axis=1)
    
    feature_names = list(X_df.columns)
    
    logger.info(f"Distribución original de clases: {pd.Series(y).value_counts().to_dict()}")
    
    # Identificar columnas categóricas y numéricas
    cat_cols = [col for col in X_df.columns if X_df[col].dtype == 'object']
    num_cols = [col for col in X_df.columns if col not in cat_cols]
    
    logger.info(f"Columnas categóricas: {len(cat_cols)}")
    logger.info(f"Columnas numéricas: {len(num_cols)}")
    
    # Verificar y gestionar valores atípicos
    for col in num_cols:
        X_df[col] = X_df[col].replace([np.inf, -np.inf], np.nan)
        
        missing_pct = X_df[col].isna().mean() * 100
        if missing_pct > 0:
            logger.info(f"Columna {col}: {missing_pct:.2f}% de valores faltantes")
    
    # Crear transformadores para preprocesamiento
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='desconocido')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_transformer, cat_cols),
            ('num', num_transformer, num_cols)
        ],
        remainder='passthrough'
    )
    
    logger.info("Aplicando transformaciones de preprocesamiento")
    
    X_transformed = preprocessor.fit_transform(X_df)
    
    transformed_feature_names = []
    
    if cat_cols:
        oh_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        for i, col in enumerate(cat_cols):
            categories = oh_encoder.categories_[i]
            for cat in categories:
                transformed_feature_names.append(f"{col}_{cat}")
    
    transformed_feature_names.extend(num_cols)
    
    # Codificar etiquetas
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)
    
    logger.info(f"Clases codificadas: {dict(zip(y_encoder.classes_, range(len(y_encoder.classes_))))}")
    
    # Aplicar selección de características
    if feature_selection and X_transformed.shape[1] > 10:
        logger.info("Aplicando selección de características")
        
        base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        n_features_to_select = max(15, int(X_transformed.shape[1] * 0.6))
        
        selector_f = SelectKBest(f_classif, k=n_features_to_select)
        X_selected_f = selector_f.fit_transform(X_transformed, y_encoded)
        
        selector_mi = SelectKBest(mutual_info_classif, k=n_features_to_select)
        X_selected_mi = selector_mi.fit_transform(X_transformed, y_encoded)
        
        selector_rfe = RFE(estimator=base_clf, n_features_to_select=n_features_to_select, step=0.1)
        X_selected_rfe = selector_rfe.fit_transform(X_transformed, y_encoded)
        
        from sklearn.model_selection import cross_val_score
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        score_f = np.mean(cross_val_score(clf, X_selected_f, y_encoded, cv=5, scoring='balanced_accuracy'))
        score_mi = np.mean(cross_val_score(clf, X_selected_mi, y_encoded, cv=5, scoring='balanced_accuracy'))
        score_rfe = np.mean(cross_val_score(clf, X_selected_rfe, y_encoded, cv=5, scoring='balanced_accuracy'))
        
        logger.info(f"Puntuaciones de selección de características:")
        logger.info(f"f_classif: {score_f:.4f}")
        logger.info(f"mutual_info_classif: {score_mi:.4f}")
        logger.info(f"RFE: {score_rfe:.4f}")
        
        # Seleccionar el mejor método
        best_score = max(score_f, score_mi, score_rfe)
        if best_score == score_f:
            X_selected = X_selected_f
            selected_indices = selector_f.get_support(indices=True)
            method_name = "f_classif"
        elif best_score == score_mi:
            X_selected = X_selected_mi
            selected_indices = selector_mi.get_support(indices=True)
            method_name = "mutual_info_classif"
        else:
            X_selected = X_selected_rfe
            selected_indices = selector_rfe.get_support(indices=True)
            method_name = "RFE"
        
        logger.info(f"Mejor método de selección: {method_name} con puntuación {best_score:.4f}")
        
        # Actualizar nombres de características seleccionadas
        if len(transformed_feature_names) >= len(selected_indices):
            selected_feature_names = [transformed_feature_names[i] for i in selected_indices]
            logger.info(f"Seleccionadas {len(selected_indices)} características de {len(transformed_feature_names)}")
        else:
            logger.warning(f"Discrepancia: {len(transformed_feature_names)} nombres vs {len(selected_indices)} índices")
            selected_feature_names = transformed_feature_names
            X_selected = X_transformed
        
        # Guardar importancia de características si se usó RFE
        if method_name == "RFE" and hasattr(selector_rfe, 'estimator_') and hasattr(selector_rfe.estimator_, 'feature_importances_'):
            importances = selector_rfe.estimator_.feature_importances_
            feature_ranks = selector_rfe.ranking_
            
            # Solo crear DataFrame si las longitudes coinciden
            if len(transformed_feature_names) == len(importances) == len(feature_ranks):
                feature_importance_df = pd.DataFrame({
                    'Feature': transformed_feature_names,
                    'Importance': importances,
                    'Rank': feature_ranks
                })
                
                feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
                feature_importance_df.to_csv("feature_importance_selection.csv", index=False)
                logger.info("Guardado CSV con importancia de características")
            else:
                logger.warning("No se pudo crear DataFrame de importancia: longitudes inconsistentes")
        
        X_final = X_selected
        final_feature_names = selected_feature_names
    else:
        logger.info("Saltando selección de características")
        X_final = X_transformed
        final_feature_names = transformed_feature_names
    
    # Balanceo de clases
    if balance_classes:
        class_counts = np.bincount(y_encoded)
        logger.info(f"Distribución de clases inicial: {dict(zip(range(len(class_counts)), class_counts))}")
        
        min_samples = np.min(class_counts)
        
        if min_samples >= 5 and len(np.unique(y_encoded)) > 1:
            logger.info("Aplicando técnicas de balanceo de clases")
            
            class_ratio = np.max(class_counts) / np.min(class_counts)
            
            try:
                if class_ratio > 10:
                    logger.info(f"Desbalance extremo detectado (ratio {class_ratio:.2f}). Usando combinación de técnicas.")
                    
                    rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
                    X_under, y_under = rus.fit_resample(X_final, y_encoded)
                    
                    smote = SMOTE(sampling_strategy='auto', random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X_under, y_under)
                else:
                    logger.info(f"Desbalance moderado (ratio {class_ratio:.2f}). Usando SMOTE.")
                    smote = SMOTE(random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X_final, y_encoded)
                
                logger.info(f"Distribución después de balanceo: {np.bincount(y_resampled)}")
                
                return X_resampled, y_resampled, y_encoder, final_feature_names
            
            except Exception as e:
                logger.warning(f"Error aplicando técnicas de balanceo: {e}. Continuando sin balanceo de clases.")
                return X_final, y_encoded, y_encoder, final_feature_names
        else:
            logger.warning("No hay suficientes muestras para aplicar técnicas de balanceo. Continuando sin balanceo de clases.")
    
    logger.info("Preprocesamiento completado")
    return X_final, y_encoded, y_encoder, final_feature_names