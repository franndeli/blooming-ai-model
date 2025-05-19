import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import logging
from collections import Counter
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report, 
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)

from model.pipeline import create_pipeline, create_explainable_pipeline
from data.fetch_data import get_user_states
from data.preprocess import preprocess_data

import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def create_results_directory():
    results_dir = "model/results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_dir, f"training_{timestamp}")
    os.makedirs(run_dir)
    
    os.makedirs(os.path.join(run_dir, "plots"))
    os.makedirs(os.path.join(run_dir, "models"))
    os.makedirs(os.path.join(run_dir, "reports"))
    
    return run_dir

def plot_confusion_matrix(cm, classes, normalize=False, title='Matriz de Confusión', output_path=None, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=16)
    plt.ylabel('Etiqueta Verdadera', fontsize=14)
    plt.xlabel('Etiqueta Predicha', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_importance(feature_names, importances, output_path=None, top_n=15):
    if len(feature_names) != len(importances):
        logger.warning(f"Longitudes diferentes - features: {len(feature_names)}, importances: {len(importances)}")
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    
    indices = np.argsort(importances)[::-1]
    
    if len(indices) > top_n:
        indices = indices[:top_n]
    
    feature_importance_df = pd.DataFrame({
        'Característica': [feature_names[i] for i in indices],
        'Importancia': importances[indices]
    })
    
    feature_importance_df = feature_importance_df.sort_values('Importancia', ascending=False)
    
    if output_path:
        csv_path = os.path.splitext(output_path)[0] + ".csv"
        feature_importance_df.to_csv(csv_path, index=False)
    
    plt.figure(figsize=(12, max(8, len(indices) * 0.4)))
    
    ax = sns.barplot(y='Característica', x='Importancia', data=feature_importance_df, palette='viridis')
    
    for i, v in enumerate(feature_importance_df['Importancia']):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')
    
    plt.xlabel("Importancia", fontsize=14)
    plt.title("Importancia de las Características para Predicción de Ámbitos", fontsize=16)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    logger.info("\nIMPORTANCIA DE CARACTERÍSTICAS (TOP {}):".format(len(feature_importance_df)))
    logger.info("="*60)
    for i, row in feature_importance_df.iterrows():
        logger.info(f"{i+1:2d}. {row['Característica']:<40} {row['Importancia']:.4f}")

def plot_class_distribution(y_true, y_pred, class_labels, output_path=None):
    counter_true = Counter(y_true)
    counter_pred = Counter(y_pred)
    
    if not isinstance(y_true[0], (int, np.integer)):
        class_to_idx = {cls: i for i, cls in enumerate(class_labels)}
        counter_true = Counter([class_to_idx.get(cls, -1) for cls in y_true])
        counter_pred = Counter([class_to_idx.get(cls, -1) for cls in y_pred])
    
    all_classes = range(len(class_labels))
    
    true_counts = [counter_true.get(cls_idx, 0) for cls_idx in all_classes]
    pred_counts = [counter_pred.get(cls_idx, 0) for cls_idx in all_classes]
    
    df = pd.DataFrame({
        'Clase': np.repeat(class_labels, 2),
        'Conteo': true_counts + pred_counts,
        'Tipo': ['Real'] * len(class_labels) + ['Predicha'] * len(class_labels)
    })
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Clase', y='Conteo', hue='Tipo', data=df, palette=['#3498db', '#e74c3c'])
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Ámbitos", fontsize=14)
    plt.ylabel("Número de muestras", fontsize=14)
    plt.title("Distribución de Ámbitos: Reales vs. Predichos", fontsize=16)
    plt.legend(title='', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), output_path=None):
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, train_sizes=train_sizes, scoring='balanced_accuracy',
        n_jobs=-1, shuffle=True, random_state=42)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Precisión de entrenamiento')
    plt.plot(train_sizes, test_mean, 'o-', color='red', label='Precisión de validación')
    
    plt.xlabel('Tamaño del conjunto de entrenamiento', fontsize=14)
    plt.ylabel('Precisión balanceada', fontsize=14)
    plt.title('Curva de Aprendizaje para Predicción de Ámbitos', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_correlation_matrix(X, feature_names, output_path=None):
    if not isinstance(X, pd.DataFrame):
        X_df = pd.DataFrame(X, columns=feature_names)
    else:
        X_df = X
    
    corr = X_df.corr()
    
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, 
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    plt.title('Matriz de Correlación entre Características', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_roc_curves(model, X_test, y_test, class_labels, output_path=None):
    y_score = model.predict_proba(X_test)
    
    n_classes = len(class_labels)
    
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    plt.figure(figsize=(12, 8))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, 
                 label=f'ROC {class_labels[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos', fontsize=14)
    plt.ylabel('Tasa de Verdaderos Positivos', fontsize=14)
    plt.title('Curvas ROC para Clasificación de Ámbitos', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_objective_influence(df, output_path=None):
    if 'objetivo' not in df.columns or 'next_ambito' not in df.columns:
        logger.warning("No se pueden crear gráficos de influencia de objetivo: faltan columnas")
        return
    
    plt.figure(figsize=(14, 10))
    
    cross_tab = pd.crosstab(df['objetivo'], df['next_ambito'], normalize='index')
    sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='.2f')
    
    plt.title('Influencia del Objetivo en la Selección de Ámbitos', fontsize=16)
    plt.xlabel('Ámbito Seleccionado', fontsize=14)
    plt.ylabel('Objetivo del Alumno', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_ambitos_weights_distribution(df, output_path=None):
    ambitos = ["Familia", "Amigos y relaciones", "Académico", "Entorno en clase",
               "Entorno exterior", "Actividades extraescolares", 
               "Autopercepción y emociones", "Relación con profesores", "General"]
    
    peso_cols = [f"{ambito}_peso" for ambito in ambitos if f"{ambito}_peso" in df.columns]
    
    if not peso_cols:
        logger.warning("No se encontraron columnas de peso de ámbitos para visualizar")
        return
    
    plt.figure(figsize=(12, 8))
    
    peso_df = df[peso_cols]
    
    rename_dict = {col: col.replace('_peso', '') for col in peso_cols}
    peso_df = peso_df.rename(columns=rename_dict)
    
    peso_df_long = pd.melt(peso_df, var_name='Ámbito', value_name='Peso')
    
    sns.violinplot(x='Ámbito', y='Peso', data=peso_df_long, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribución de Pesos por Ámbito', fontsize=16)
    plt.ylabel('Peso', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def train_model(explainable=False):
    results_dir = create_results_directory()
    logger.info(f"Resultados del entrenamiento se guardarán en: {results_dir}")
    
    df_user_states = get_user_states()
    logger.info(f"Dimensiones del dataset: {df_user_states.shape}")
    
    if df_user_states.empty:
        logger.error("No hay datos disponibles para el entrenamiento.")
        return

    df_user_states.to_csv(os.path.join(results_dir, "datos_originales.csv"), index=False)
    
    columns_info = pd.DataFrame({
        'Columna': df_user_states.columns,
        'Tipo': df_user_states.dtypes,
        'Valores_Nulos': df_user_states.isnull().sum(),
        'Valores_Únicos': [df_user_states[col].nunique() for col in df_user_states.columns]
    })
    
    columns_info.to_csv(os.path.join(results_dir, "columnas_info.csv"), index=False)
    logger.info("Información de columnas guardada en columnas_info.csv")
    
    if 'objetivo' in df_user_states.columns and 'next_ambito' in df_user_states.columns:
        objetivo_counts = df_user_states['objetivo'].value_counts()
        logger.info(f"Distribución de objetivos: {objetivo_counts.to_dict()}")
        
        plot_objective_influence(df_user_states, os.path.join(results_dir, "plots", "objetivo_influencia.png"))
    
    plot_ambitos_weights_distribution(df_user_states, os.path.join(results_dir, "plots", "distribucion_pesos_ambitos.png"))
    
    X, y, y_encoder, feature_names = preprocess_data(df_user_states)
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    logger.info(f"Número de características después de preprocesamiento: {len(feature_names)}")
    
    with open(os.path.join(results_dir, "feature_names.json"), 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    with open(os.path.join(results_dir, "class_labels.json"), 'w') as f:
        json.dump(y_encoder.classes_.tolist(), f, indent=2)
    
    plt.figure(figsize=(10, 6))
    counts = Counter(y)
    df_counts = pd.DataFrame({'Clase': y_encoder.classes_, 'Conteo': [counts.get(i, 0) for i in range(len(y_encoder.classes_))]})
    sns.barplot(x='Clase', y='Conteo', data=df_counts, palette='viridis')
    plt.title('Distribución de Ámbitos en el Dataset', fontsize=16)
    plt.xlabel('Ámbito', fontsize=14)
    plt.ylabel('Número de Muestras', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plots", "distribucion_ambitos_inicial.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"Distribución de ámbitos: {counts}")
    
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(y, list):
        y = np.array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                 random_state=42, stratify=y)
    logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    if explainable:
        pipeline = create_explainable_pipeline()
        logger.info("Pipeline explícito creado con énfasis en ámbitos y objetivos")
    else:
        pipeline = create_pipeline(model_type='xgb');
        logger.info("Pipeline estándar creado con configuración optimizada")
    
    logger.info("Evaluando el modelo con validación cruzada...")
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='balanced_accuracy')
    logger.info(f"Puntuaciones de validación cruzada: {cv_scores}")
    logger.info(f"Precisión balanceada media en CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    logger.info("Generando curva de aprendizaje...")
    plot_learning_curve(pipeline, X_train, y_train, cv=5, 
                        output_path=os.path.join(results_dir, "plots", "curva_aprendizaje.png"))
    
    if explainable:
        param_grid = {
            'feature_selector__k': ['all', 15, 20],
            'classifier__n_estimators': [100, 150],
            'classifier__max_depth': [10, 15, None],
            'classifier__min_samples_leaf': [1, 2]
        }
    else:
        param_grid = {
            'feature_selector__k': ['all', 15, 20, 25],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 15, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
    
    logger.info("Iniciando GridSearchCV con énfasis en parámetros que afectan la importancia de ámbitos y objetivos...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    logger.info("Mejores parámetros encontrados:")
    logger.info(grid_search.best_params_)
    
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_results.to_csv(os.path.join(results_dir, "reports", "grid_search_results.csv"), index=False)
    
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    logger.info("\n--- Métricas del Modelo en Conjunto de Prueba ---")
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"Balanced Accuracy: {bal_acc:.4f}\n")
    
    class_labels = y_encoder.classes_
    class_report = classification_report(y_test, y_pred, target_names=class_labels, zero_division=0)
    logger.info("Classification Report:\n" + class_report)
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    
    report_df = pd.DataFrame({
        'Clase': class_labels,
        'Precisión': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Soporte': support
    })
    report_df.to_csv(os.path.join(results_dir, "reports", "classification_report.csv"), index=False)
    
    cm = confusion_matrix(y_test, y_pred)
    logger.info("Matriz de Confusión:\n" + np.array2string(cm))
    
    plot_confusion_matrix(cm, classes=class_labels, normalize=True, 
                         output_path=os.path.join(results_dir, "plots", "confusion_matrix_norm.png"))
    
    plot_confusion_matrix(cm, classes=class_labels, normalize=False, 
                         output_path=os.path.join(results_dir, "plots", "confusion_matrix.png"))
    
    classifier = best_model.named_steps['classifier']
    importances = classifier.feature_importances_
    
    if 'feature_selector' in best_model.named_steps:
        feature_selector = best_model.named_steps['feature_selector']
        if hasattr(feature_selector, 'get_support'):
            mask = feature_selector.get_support()
            selected_feature_names = [f for f, m in zip(feature_names, mask) if m]
        else:
            selected_feature_names = feature_names
    else:
        selected_feature_names = feature_names
    
    plot_feature_importance(selected_feature_names, importances, 
                          output_path=os.path.join(results_dir, "plots", "feature_importance.png"))
    
    plot_class_distribution(y_test, y_pred, class_labels=class_labels,
                           output_path=os.path.join(results_dir, "plots", "class_distribution.png"))
    
    plot_roc_curves(best_model, X_test, y_test, class_labels,
                   output_path=os.path.join(results_dir, "plots", "roc_curves.png"))
    
    joblib.dump(best_model, os.path.join(results_dir, "models", "best_model.pkl"))
    joblib.dump(y_encoder, os.path.join(results_dir, "models", "y_encoder.pkl"))
    joblib.dump(selected_feature_names, os.path.join(results_dir, "models", "feature_names.pkl"))
    
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/model.pkl")
    joblib.dump(y_encoder, "model/y_encoder.pkl")
    joblib.dump(selected_feature_names, "model/feature_names.pkl")
    
    logger.info(f"Modelo entrenado y guardado. Resultados completos en: {results_dir}")
    return results_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelo de predicción de ámbitos')
    parser.add_argument('--explainable', action='store_true', 
                        help='Usar pipeline explicable (más simple) para facilitar interpretación')
    
    args = parser.parse_args()
    
    results_dir = train_model(explainable=args.explainable)
    
    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print(f"Resultados guardados en: {results_dir}")
    print("Reporte detallado disponible en el directorio 'reports'")
    print("Visualizaciones disponibles en el directorio 'plots'")
    print("Modelo guardado en: model/model.pkl")