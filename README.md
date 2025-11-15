# Blooming AI 🧠

> Personalized Question Selection System for Educational Well-being Assessment

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Latest-green.svg)](https://www.mongodb.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic-yellow.svg)]()

Blooming AI is an intelligent system that uses machine learning to provide personalized question recommendations for students based on their emotional well-being profiles across multiple life domains. Developed as a Final Year Project (TFG) at the University of Alicante.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Generation](#data-generation)
- [Model Training](#model-training)
- [Demo Application](#demo-application)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

## 🎯 Overview

Blooming AI addresses the challenge of personalizing educational interventions for student well-being by intelligently selecting questions that target the most relevant emotional domains for each individual student. The system considers:

- **9 Life Domains**: Family, Friends & Relationships, Academic, Classroom Environment, External Environment, Extracurricular Activities, Self-perception & Emotions, Teacher Relationships, and General
- **4 Pedagogical Objectives**: Prevention, Exploration, Consolidation, and Predetermined
- **Historical Response Patterns**: Learning from past interactions to optimize future recommendations

The AI model achieves **99.9% accuracy** using XGBoost, demonstrating exceptional capability in understanding student profiles and making pedagogically meaningful recommendations.

## ✨ Key Features

### Intelligent Question Selection
- ML-powered domain prediction based on student emotional profiles
- Adaptive learning from historical response patterns
- Alignment with pedagogical objectives

### Multi-Model Support
- **RandomForest**: 98.1% accuracy with robust feature importance
- **XGBoost**: 99.9% accuracy with superior generalization
- Comparative analysis tools for model evaluation

### Comprehensive Student Profiling
- Track emotional well-being across 9 life domains
- Monitor domain weights (0-100 scale)
- Analyze interaction frequency per domain
- Temporal pattern recognition

### RESTful API Backend
- Flask-based REST API for data management
- MongoDB integration for scalable data storage
- Resource endpoints for students, questions, responses, and options

### Interactive Demo Interface
- Streamlit-based visualization dashboard
- Real-time predictions with explainability
- Feature importance analysis
- Historical response tracking
- Radar charts for student profiles

### Synthetic Data Generation
- Realistic student profile simulation
- Configurable emotional profiles (Positive, Neutral, Negative)
- Response pattern generation aligned with pedagogical objectives
- Temporal distribution modeling

## 🏗️ System Architecture

```
┌─────────────────┐
│   Streamlit UI  │  ← Interactive Demo & Visualization
└────────┬────────┘
         │
┌────────▼────────┐
│  Prediction     │  ← ML Models (XGBoost/RandomForest)
│  Engine         │
└────────┬────────┘
         │
┌────────▼────────┐
│  Flask API      │  ← RESTful Backend
└────────┬────────┘
         │
┌────────▼────────┐
│    MongoDB      │  ← Data Persistence
└─────────────────┘
```

### Component Breakdown

**Frontend Layer**
- Streamlit demo application (`modelo/demo.py`)
- Interactive visualizations with Plotly
- Real-time prediction interface

**Model Layer**
- Feature extraction and preprocessing (`modelo/data/preprocess.py`)
- Training pipeline with hyperparameter optimization (`modelo/model/train.py`)
- Prediction service with explainability (`modelo/model/predict.py`)

**API Layer**
- Flask RESTful API (`backend/app.py`)
- Resource management for all entities
- PyMongo integration

**Data Layer**
- MongoDB database (tfg_database)
- Collections: alumnos, preguntas, respuestas, opcionespregunta, ambitos, objetivos

## 🛠️ Technologies Used

### Backend
- **Python 3.12**: Core programming language
- **Flask 3.0**: RESTful API framework
- **Flask-RESTful**: REST API extensions
- **PyMongo**: MongoDB driver
- **Flask-PyMongo**: Flask-MongoDB integration

### Machine Learning
- **scikit-learn 1.5.2**: ML algorithms and preprocessing
- **XGBoost 2.1.1**: Gradient boosting implementation
- **imbalanced-learn 0.12.3**: Class balancing techniques
- **NumPy 1.26.4**: Numerical computations
- **pandas 2.2.3**: Data manipulation

### Visualization
- **Streamlit**: Interactive web application
- **Plotly**: Advanced interactive plots
- **Matplotlib 3.10.0**: Static visualizations
- **Seaborn 0.13.2**: Statistical graphics

### Database
- **MongoDB**: NoSQL document database
- **pymongo 4.11**: Python MongoDB driver

## 📦 Installation

### Prerequisites
- Python 3.12+
- MongoDB (local installation or remote instance)
- Git

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd blooming-ai
```

### Step 2: Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n blooming python=3.12
conda activate blooming

# Or using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Backend dependencies
pip install -r requirements.txt

# Additional backend packages
cd backend
pip install flask flask-restful flask-pymongo
```

### Step 4: Start MongoDB
```bash
# On macOS/Linux
sudo systemctl start mongod

# On Windows
net start MongoDB

# Verify connection
mongo --eval "db.version()"
```

### Step 5: Initialize Database
```bash
# Import seed data or generate synthetic data (see Data Generation section)
python generate_data.py
```

## ⚙️ Configuration

### Backend Configuration (`modelo/config.py`)
```python
# API Endpoints
BACKEND_URL = "http://localhost:5000"
RESPUESTAS_ENDPOINT = f"{BACKEND_URL}/respuestas"
PREGUNTAS_ENDPOINT = f"{BACKEND_URL}/preguntas"
ALUMNOS_ENDPOINT = f"{BACKEND_URL}/alumnos"

# Database
SQLALCHEMY_DATABASE_URI = "mysql+pymysql://root@localhost/tfg_database"
MODEL_PATH = "model/modelo_final.pkl"
```

### MongoDB Connection (`modelo/data/fetch_data.py`)
```python
client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
db = client.tfg_database
```

### Application Settings (`backend/app.py`)
```python
app.config["MONGO_URI"] = "mongodb://localhost:27017/tfg_database"
```

## 🚀 Usage

### Starting the Backend API
```bash
cd backend
python app.py
# API will be available at http://localhost:5000
```

### API Endpoints

**Students**
- `GET /alumnos` - List all students
- `GET /alumnos/<alumno_id>` - Get student by ID

**Questions**
- `GET /preguntas` - List all questions

**Responses**
- `GET /respuestas` - List all responses with joined data

**Domains & Objectives**
- `GET /ambitos` - List all life domains
- `GET /objetivos` - List all pedagogical objectives

**Question Options**
- `GET /opciones_pregunta` - List all answer options

### Running the Demo Application
```bash
cd modelo
streamlit run demo.py
# Access the interface at http://localhost:8501
```

### Making Predictions Programmatically
```python
from modelo.model.predict import predict_next_question

# Get recommendation for a student
result = predict_next_question(alumno_id="<student_id>", save_results=True)

print(f"Predicted Domain: {result['ambito_predicho']}")
print(f"Recommended Question: {result['pregunta']}")
print(f"Available Options: {result['opcionesPregunta']}")
```

## 🎲 Data Generation

The system includes a sophisticated synthetic data generator that creates realistic student profiles and response patterns.

### Running Data Generation
```bash
python generate_data.py
```

### Configuration Parameters
```python
N_STUDENTS = 200                  # Number of students to generate
RESPONSES_PER_STUDENT = 200       # Responses per student
BASE_DATE = datetime(2024, 1, 1)  # Starting date for responses

# Emotional profile distribution
EMOTIONAL_WEIGHTS = {
    "Positivo": 0.4,    # 40% positive students
    "Neutro": 0.4,      # 40% neutral students
    "Negativo": 0.2     # 20% negative students
}
```

### Generated Output
- `generated_data/alumnos.json` - Student profiles with domain weights
- `generated_data/respuestas.json` - Simulated responses
- `generated_data/generate_data.log` - Generation process log

### Features of Synthetic Data
- Realistic emotional profiles across 9 domains
- Response patterns aligned with pedagogical objectives
- Temporal distribution over 60-day period
- Weighted question selection based on student profiles
- Option selection considering emotional tendencies

## 🧠 Model Training

### Training Process
```bash
cd modelo
python -m model.train
```

### Training Pipeline

1. **Data Extraction**
   - Fetches student profiles, questions, and responses
   - Constructs feature vectors from historical data
   - Includes temporal features and interaction patterns

2. **Preprocessing**
   - Handles missing values with median imputation
   - Standardizes numerical features
   - One-hot encodes categorical variables
   - Applies feature selection (SelectKBest, RFE)

3. **Class Balancing**
   - Uses SMOTE for minority class oversampling
   - Random undersampling for extreme imbalances
   - Ensures balanced representation across domains

4. **Model Training**
   - RandomForest with 200 estimators
   - XGBoost with optimized hyperparameters
   - 5-fold cross-validation
   - Grid search for hyperparameter tuning

5. **Evaluation**
   - Accuracy, precision, recall, F1-score
   - Confusion matrices (normalized and absolute)
   - Learning curves
   - Feature importance analysis

### Model Outputs
```
model/results/
├── training_rf/
│   ├── models/
│   │   ├── best_model.pkl
│   │   ├── y_encoder.pkl
│   │   └── feature_names.pkl
│   ├── plots/
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix_norm.png
│   │   ├── curva_aprendizaje.png
│   │   └── feature_importance.png
│   └── metrics/
│       └── classification_report.txt
├── training_xgb/
│   └── ... (same structure)
```

## 🎨 Demo Application

The Streamlit demo provides an intuitive interface for exploring the system's capabilities.

### Features

**Real-time Prediction Mode**
- Select student from database
- Choose AI model (RandomForest/XGBoost)
- View predicted domain with confidence scores
- Examine recommended question and answer options
- Analyze feature importance
- Review probability distributions
- Explore response history

**Model Comparison Mode**
- Side-by-side performance metrics
- Learning curve comparison
- Confusion matrix analysis
- Feature importance differences

### Visualizations

- **Radar Charts**: Student emotional profile across domains
- **Bar Charts**: Probability distribution for domain predictions
- **Line Graphs**: Response weight evolution over time
- **Heatmaps**: Confusion matrices for model evaluation
- **Feature Importance**: Top contributing factors in predictions

## 📁 Project Structure

```
blooming-ai/
│
├── backend/                      # Flask REST API
│   ├── app.py                   # Main application entry point
│   ├── db.py                    # Database configuration
│   ├── develop.sh               # Development startup script
│   ├── requirements.txt         # Backend dependencies
│   └── resources/               # API resource definitions
│       ├── alumnos.py          # Student endpoints
│       ├── preguntas.py        # Question endpoints
│       ├── respuestas.py       # Response endpoints
│       ├── ambitos.py          # Domain endpoints
│       ├── objetivos.py        # Objective endpoints
│       └── opcionespregunta.py # Option endpoints
│
├── modelo/                       # ML model and frontend
│   ├── config.py                # Configuration settings
│   ├── demo.py                  # Streamlit demo application
│   ├── main.py                  # CLI interface
│   │
│   ├── data/                    # Data management
│   │   ├── fetch_data.py       # MongoDB data retrieval
│   │   └── preprocess.py       # Feature engineering
│   │
│   └── model/                   # ML models
│       ├── train.py            # Training pipeline
│       ├── predict.py          # Prediction service
│       └── results/            # Training outputs
│           ├── training_rf/    # RandomForest results
│           └── training_xgb/   # XGBoost results
│
├── generated_data/              # Synthetic data outputs
│   ├── alumnos.json
│   ├── respuestas.json
│   └── generate_data.log
│
├── generate_data.py             # Synthetic data generator
├── requirements.txt             # Project dependencies
└── README.md                    # This file
```

## 🤝 Contributing

This is an academic project developed as part of a Final Year Project. While direct contributions may not be accepted, feedback and suggestions are welcome.

### Reporting Issues
If you find bugs or have suggestions:
1. Check existing issues
2. Create a new issue with detailed description
3. Include steps to reproduce (if applicable)

## 👨‍💻 Author

**Francisco José Delicado González**

- Final Year Project (TFG)
- Multimedia Engineering
- University of Alicante
- Year: 2025
- Program: BEng in Software Engineering, University of Maastricht

## 📄 License

This project is developed for academic purposes as part of a university degree program. All rights reserved.
